"""Collect calibration-source and target quality evidence for Task103 data governance."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row['id'], int(row['finger_index'])


def review_sources(source, refit, plan, targets):
    old = {r['id']: r for r in read_json(source/'checks/calibration_quality.json')['records']}
    new = {r['id']: r for r in read_json(refit/'checks/calibration_quality.json')['records']}
    assert set(new) == {rid for rid, row in plan.items() if row['stage'] == 'mix-test-5'}
    fit = {key(r): r for r in read_csv(source/'analysis_output/accepted_calibration_diagnostics.csv')}
    cv = {key(r): r for r in read_csv(source/'analysis_output/calibration_press_cv.csv')}
    assert fit.keys() == cv.keys()
    assert fit.keys() == {(rid, int(finger)) for rid, r in new.items() for finger in r['channels']}
    applied = Counter(row['calibration_source_id'] for row in targets if row['status']=='retained')
    source_rows, repeat_rows = [], []
    aggregate = Counter()
    for rid, current in sorted(new.items()):
        previous = old[rid]
        assert {k: v for k, v in current.items() if k!='reason'} == {
            k: v for k, v in previous.items() if k!='reason'}
        for folder, extension in [('calibration', 'json'), ('events', 'json'), ('routes', 'npz')]:
            path = f'cache/{folder}/{rid}.{extension}'
            a, b = source/path, refit/path
            assert a.is_file() == b.is_file(), path
            if a.is_file():
                assert a.read_bytes() == b.read_bytes(), path
        aggregate['records_with_identical_refit_artifacts'] += current['status']=='prepared'
        aggregate['unavailable_records'] += current['status']!='prepared'
        aggregate['usable_records'] += bool(current['channels'])
        aggregate['prepared_no_curve_records'] += current['status']=='prepared' and not current['channels']
        counts = Counter()
        for finger in current['channels']:
            k = (rid, int(finger))
            d, c = fit[k], cv[k]
            counts['historical_review_curves'] += bool(d['review_flags'])
            counts['single_press_curves'] += c['status']=='one_press_no_independent_repeat'
            counts['no_scorable_repeat_curves'] += c['status']=='no_scorable_repeat_fold'
            if c['status'] != 'scored':
                continue
            coverage, error, in_fit, span = map(float, (c['coverage'], c['point_equal_MAE_N'],
                                                        c['source_fit_MAE_N'], d['fit_range_N']))
            counts['repeat_coverage_below_half_curves'] += coverage < .5
            if coverage < .5 or error <= max(3, .2*span, 3*in_fit):
                continue
            counts['repeat_disagreement_review_curves'] += 1
            counts['new_review_curves'] += not d['review_flags']
            repeat_rows.append(dict(
                id=rid, recording_id=plan[rid]['recording_id'], object_key=plan[rid]['key'],
                finger_index=k[1], presses=c['presses'], scored_folds=c['scored_folds'], coverage=coverage,
                source_fit_mae_N=in_fit, held_press_mae_N=error, fit_range_N=span,
                old_review_flags=d['review_flags'], used_by_recordings=applied[rid], disposition='review'))
        aggregate.update(counts)
        aggregate['accepted_curves'] += len(current['channels'])
        aggregate['rejected_curve_fit_attempts'] += len(current.get('problems', {}))
        source_rows.append(dict(
            id=rid, recording_id=plan[rid]['recording_id'], object_key=plan[rid]['key'],
            status=current['status'], reason=current.get('reason', ''),
            fitted_channels=len(current['channels']), rejected_fit_attempts=len(current.get('problems', {})),
            used_by_recordings=applied[rid], usable_channels=';'.join(current['channels']),
            channel_rejections=json.dumps(current.get('problems', {}), sort_keys=True),
            rejected_press_reasons=json.dumps(current.get('rejected_event_reasons', {}), sort_keys=True),
            **{name: counts[name] for name in (
                'historical_review_curves', 'single_press_curves', 'no_scorable_repeat_curves',
                'repeat_coverage_below_half_curves', 'repeat_disagreement_review_curves', 'new_review_curves')}))
    aggregate['sources_used'] = len(applied)
    aggregate['new_review_records'] = len({r['id'] for r in repeat_rows if not r['old_review_flags']})
    repeat_rows.sort(key=lambda r: (-r['held_press_mae_N'], r['id'], r['finger_index']))
    return source_rows, repeat_rows, dict(aggregate)


def main(source, parent, run, output):
    plan = {r['id']: r for r in read_json(source/'plan.json')['rows']}
    targets = read_csv(run/'checks/calibration_target_coverage.csv')
    retained = [r for r in targets if r['status']=='retained']
    source_rows, repeat_rows, source_summary = review_sources(source, run/'quality_refit', plan, targets)
    target_findings = []
    points = Counter()
    for row in targets:
        if row['status'] != 'retained':
            target_findings.append({**row, 'quality_finding': row['status']})
            continue
        points.update({k: int(row[k]) for k in ('raw_valid_points','cal_valid_points',
                      'missing_curve_points','out_of_domain_points','invalid_ad_points')})
        flags = []
        if float(row['retained_fraction']) < .5:
            flags.append('retains_less_than_half_native_points')
        eligible = int(row['raw_valid_points'])-int(row['missing_curve_points'])
        if int(row['out_of_domain_points']) > .2*eligible:
            flags.append('over_20pct_AD_out_of_domain_on_available_channels')
        if flags:
            target_findings.append({**row, 'quality_finding': ';'.join(flags)})
    exclusions = read_json(parent/'prepared/exclusions.json')
    extraction = [dict(id=rid, recording_id=plan[rid]['recording_id'], object_key=plan[rid]['key'],
                       **row) for rid, row in sorted(exclusions.items())]
    coverage = read_json(run/'prepared/subject_split.json')['counts']
    coverage_reviews = [r for r in target_findings if r['status']=='retained']
    worst_retained = min(coverage_reviews, key=lambda r: float(r['retained_fraction']))
    worst_domain = max(coverage_reviews, key=lambda r: int(r['out_of_domain_points']) /
                       max(1, int(r['raw_valid_points'])-int(r['missing_curve_points'])))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'test5_source_review.csv', source_rows)
    write_csv(output/'test5_repeat_disagreement_review.csv', repeat_rows)
    write_csv(output/'calibrated_target_findings.csv', target_findings)
    write_csv(output/'all_stage_raw_extraction_findings.csv', extraction)
    summary = dict(
        sources_scanned=len(source_rows), source_qc=source_summary,
        repeat_review_rule='Held-press coverage >=0.5; held-press MAE > max(3 N, 0.2*fit span, 3*source fit MAE). Review only.',
        target_status=dict(Counter(r['status'] for r in targets)), retained_source_coverage=dict(points),
        target_findings=dict(Counter(flag for row in target_findings for flag in row['quality_finding'].split(';'))),
        raw_extraction_status=dict(Counter(r['reason'] for r in extraction)), calibrated_split=coverage,
        stage_missing_reason='No test5 in same person/date/PC/boot group',
        example_repeat_review=next(r for r in repeat_rows if not r['old_review_flags']),
        example_missing_channels=worst_retained, example_out_of_domain=worst_domain,
        retained_recordings=len(retained))
    (output/'calibrated_subject_data_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'parent', 'run', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.parent, args.run, args.output)

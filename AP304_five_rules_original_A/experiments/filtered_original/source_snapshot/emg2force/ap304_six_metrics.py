"""Recoverable Corr/MAPE/MAE under the six requested aggregation hierarchies.

All arithmetic is float64 in calibrated N. Primary reports pool five fingers;
per-finger reports use the same hierarchy independently, then mean valid fingers.
Undefined Corr/MAPE are null, excluded from that metric's mean, and counted.
Selection loss remains the separately recorded legacy trial-equal L_std.
"""
from collections import defaultdict
import math

import numpy as np

METRICS = ('corr', 'mape_percent', 'mae_n')
METHODS = {
    '1_trial_equal': '所有trial等权',
    '2_stage_trial_equal': 'stage内trial等权，再stage等权',
    '3_subject_trial_equal': '每人内部trial等权，再逐人等权',
    '4_subject_stage_trial_equal': '每人内部stage等权，再逐人等权（stage内trial等权）',
    '5_subject_pooled': '每人合并全部trial后计算，再逐人等权',
    '6_subject_stage_pooled': '每人按stage合并后计算，再stage等权、逐人等权',
}


def moments(prediction, target, threshold=.5):
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(target, np.float64).reshape(-1)
    if p.shape != y.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('invalid prediction/target arrays')
    mp, my = p.mean(), y.mean()
    dp, dy = p-mp, y-my
    error = np.abs(p-y); active = np.abs(y) >= threshold
    return dict(n=int(len(p)), mean_pred=float(mp), mean_target=float(my),
        m2_pred=float(dp@dp), m2_target=float(dy@dy), cross=float(dp@dy),
        absolute_error=float(error.sum()), active_count=int(active.sum()),
        ape_sum=float((error[active]/np.abs(y[active])).sum()),
        squared_error=float(error@error), target_absolute_sum=float(np.abs(y).sum()))


def merge(a, b):
    if a is None: return dict(b)
    n = a['n']+b['n']; factor = a['n']*b['n']/n
    dp, dy = b['mean_pred']-a['mean_pred'], b['mean_target']-a['mean_target']
    result = dict(n=n, mean_pred=a['mean_pred']+dp*b['n']/n,
        mean_target=a['mean_target']+dy*b['n']/n,
        m2_pred=a['m2_pred']+b['m2_pred']+dp*dp*factor,
        m2_target=a['m2_target']+b['m2_target']+dy*dy*factor,
        cross=a['cross']+b['cross']+dp*dy*factor)
    for key in ('absolute_error', 'active_count', 'ape_sum', 'squared_error', 'target_absolute_sum'):
        result[key] = a[key]+b[key]
    return result


def scores(s):
    return dict(corr=max(-1., min(1., s['cross']/math.sqrt(s['m2_pred']*s['m2_target'])))
        if s['m2_pred'] > 0 and s['m2_target'] > 0 else None,
        mae_n=s['absolute_error']/s['n'],
        mape_percent=100*s['ape_sum']/s['active_count'] if s['active_count'] else None)


def average(values):
    values = list(values); result = dict(units=len(values), valid_units={})
    for key in METRICS:
        valid = [x[key] for x in values if x[key] is not None]
        result[key] = math.fsum(valid)/len(valid) if valid else None
        result['valid_units'][key] = len(valid)
    return result


def _hierarchies(rows):
    by_stage, by_subject, by_person_stage = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_stage[row['stage']].append(row)
        by_subject[row['subject']].append(row)
        by_person_stage[(row['subject'], row['stage'])].append(row)
    trial = {x['trial_key']: scores(x['statistics']) for x in rows}
    stage_means = {s: average(trial[x['trial_key']] for x in rs) for s, rs in by_stage.items()}
    subject_means = {s: average(trial[x['trial_key']] for x in rs) for s, rs in by_subject.items()}
    cells, person_cells, pooled_person_cells = [], defaultdict(list), defaultdict(list)
    for (subject, stage), rs in sorted(by_person_stage.items()):
        stat = None
        for row in rs: stat = merge(stat, row['statistics'])
        trial_mean = average(trial[x['trial_key']] for x in rs)
        pooled = scores(stat)
        person_cells[subject].append(trial_mean); pooled_person_cells[subject].append(pooled)
        cells.append(dict(subject=subject, stage=stage, trials=len(rs), statistics=stat,
            trial_equal=trial_mean, pooled=pooled))
    people = []
    for subject, rs in sorted(by_subject.items()):
        stat = None
        for row in rs: stat = merge(stat, row['statistics'])
        people.append(dict(subject=subject, trials=len(rs), stages=sorted({x['stage'] for x in rs}),
            statistics=stat, trial_equal=subject_means[subject], pooled=scores(stat),
            stage_trial_equal=average(person_cells[subject]),
            stage_pooled_equal=average(pooled_person_cells[subject])))
    results = {
        '1_trial_equal': average(trial.values()),
        '2_stage_trial_equal': average(stage_means.values()),
        '3_subject_trial_equal': average(subject_means.values()),
        '4_subject_stage_trial_equal': average(x['stage_trial_equal'] for x in people),
        '5_subject_pooled': average(x['pooled'] for x in people),
        '6_subject_stage_pooled': average(x['stage_pooled_equal'] for x in people),
    }
    for key, value in results.items(): value['name'] = METHODS[key]
    return dict(methods=results, per_subject=people, per_subject_stage=cells,
        per_stage_trial_equal=[dict(stage=s, **x) for s,x in sorted(stage_means.items())])


def summarize_six(records):
    if not records or len({x['trial_key'] for x in records}) != len(records):
        raise ValueError('empty or duplicate trial cohort')
    pooled = _hierarchies(records)
    finger_results = []
    for finger in range(5):
        rows = [dict(trial_key=x['trial_key'], subject=x['subject'], stage=x['stage'],
            statistics=x['per_finger_statistics'][finger]) for x in records]
        finger_results.append(dict(finger_index=finger, **_hierarchies(rows)))
    finger_mean = {key: average(x['methods'][key] for x in finger_results) for key in METHODS}
    return dict(schema='ap304-six-aggregation-v1', trials=len(records),
        subjects=len(pooled['per_subject']), observed_stages=sorted({x['stage'] for x in records}),
        pooled_five_fingers=pooled, per_finger=finger_results, per_finger_then_mean=finger_mean,
        policy=dict(primary='pooled_five_fingers', statistics_dtype='float64',
            undefined='null, excluded per metric; valid units recorded at each averaging layer',
            missing_subject_stage='absent cells excluded; no zero fill',
            stage_scope='all stages observed in evaluated split; no stage exclusions',
            units=dict(corr='dimensionless', mape_percent='percent', mae_n='N'),
            pooled_Corr='Chan merge with between-trial mean terms, never average trial Corr',
            fingers='thumb,index,middle,ring,little; no force-amplitude weighting',
            recoverability='Corr/MAE/MAPE recoverable at fixed active threshold; pooled Kendall tau requires predictions'))

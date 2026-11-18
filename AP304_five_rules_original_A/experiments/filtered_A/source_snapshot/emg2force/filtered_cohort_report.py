"""Report a frozen subset from identical complete-trial predictions."""
import json
from pathlib import Path
from .ap304_rank_data import atomic_json, sha
from .ap304_six_metrics import summarize_six

def report_filtered(c, split, label, out, binding, records, summarize):
    spec = c['filtered_evaluation'][split]
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Filtered manifest changed')
    rows = json.loads(Path(spec['path']).read_text())[split]
    keys = {r['trial_key'] for r in rows}
    selected = [r for r in records if r['trial_key'] in keys]
    if not keys or len(keys) != len(rows) or len(selected) != len(keys):
        raise ValueError('Filtered cohort mismatch')
    target = out.with_name(label + '_filtered')
    target.mkdir(exist_ok=True)
    (target / 'COMPLETE.json').unlink(missing_ok=True)
    binding = dict(binding, trial_count=len(selected), cohort='five-rule filtered',
                   manifest_sha256=spec['sha256'], parent_evaluation=str(out))
    atomic_json(target / 'protocol.json', binding)
    six = summarize_six(selected)
    six['binding'] = binding
    atomic_json(target / 'six_aggregations.json', six)
    result = summarize(selected)
    result.update(split=split, label=label + '_filtered',
                  six_aggregations=six['pooled_five_fingers']['methods'])
    atomic_json(target / 'summary.json', result)
    with (target / 'trials.jsonl').open('w') as handle:
        for record in selected:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
    files = ('protocol.json', 'six_aggregations.json', 'summary.json', 'trials.jsonl')
    atomic_json(target / 'COMPLETE.json', dict(binding=binding, trials=len(selected),
                files={name: sha(target / name) for name in files}))
    return result

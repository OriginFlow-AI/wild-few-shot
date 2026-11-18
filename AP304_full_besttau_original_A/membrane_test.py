"""Summarize an existing Test evaluation with per-finger sufficient statistics."""
import argparse
import json
from pathlib import Path
import runpy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evaluation', type=Path)
parser.add_argument('--exclude45', action='store_true')
args = parser.parse_args()
metrics = runpy.run_path(str(Path(__file__).parent/'metrics/finger_stage_subject_metrics.py'))
files = [args.evaluation/'trials.jsonl']
if not files[0].is_file():
    files = sorted(args.evaluation.glob('rank*_trials.jsonl'))
trials = []
seen = set()
for path in files:
    with path.open() as lines:
        for line in lines:
            r = json.loads(line)
            if r['trial_key'] in seen:
                raise ValueError('Duplicate trial: '+r['trial_key'])
            seen.add(r['trial_key'])
            if r['layout']=='1x5x1' and 1 <= int(r['stage']) <= 15:
                if not args.exclude45 or int(r['stage']) not in (4,5):
                    trials.append(metrics['fingers'](r))
summary = json.loads((args.evaluation/'summary.json').read_text())
if len(seen) != summary['trials'] or not trials:
    raise ValueError('Incomplete source evaluation or empty membrane cohort')
overall, cells, people = metrics['hierarchy'](trials)
print(json.dumps(dict(layout='1x5x1', trials=len(trials), subjects=len(people),
                      stages=sorted({int(t['stage']) for t in trials}), **overall), indent=2))

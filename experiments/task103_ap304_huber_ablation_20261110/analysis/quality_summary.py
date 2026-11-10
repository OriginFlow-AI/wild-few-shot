"""Summarize completed calibration QA without evaluating neural Test accuracy."""
import argparse
import collections
import json
from pathlib import Path

def main(root):
    root=Path(root);q=json.loads((root/'checks/calibration_quality.json').read_text())['records']
    t5=[r for r in q if r['stage']=='mix-test-5']
    a=dict(calibration_versions=sorted({r.get('calibration_version','legacy') for r in q}),records=len(q),
        statuses=dict(collections.Counter(r['status'] for r in q)),test5=len(t5),
        usable_test5=sum(bool(r['channels']) for r in t5),accepted_finger_models=sum(len(r['channels']) for r in t5),
        partial_test5=sum(0<len(r['channels'])<10 for r in t5),
        channel_failures=dict(collections.Counter(x for r in q for x in r.get('problems',{}).values())),
        unavailable=[{k:r.get(k) for k in ['id','subject','stage','reason']} for r in q if r['status']!='prepared'],
        zero_channel_test5=[{k:r.get(k) for k in ['id','subject','reason','events','problems','rejected_event_reasons']} for r in t5 if not r['channels']])
    (root/'checks/full_calibration_quality_summary.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in a.items() if k not in ['unavailable','zero_channel_test5']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

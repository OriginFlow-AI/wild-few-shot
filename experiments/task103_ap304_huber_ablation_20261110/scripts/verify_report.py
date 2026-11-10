"""Synthetic end-to-end report check, including rejection of unequal Test masks."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from common import BRANCHES, atomic, load
from train import measures
from report import main as report

def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='report_fixture_',dir=output) as directory:
        root=Path(directory)
        plan=[dict(id='training',split='train',active=True),dict(id='validation',split='val',active=True),
              dict(id='fixture5',split='test',active=True),dict(id='fixture7',split='test',active=True)]
        atomic(root/'plan.json',dict(rows=plan))
        atomic(root/'prepared/dataset.json',dict(entries=plan,windows={'train':1,'val':1,'test':2},exclusions=[]))
        atomic(root/'checks/calibration_quality.json',dict(records=[dict(id='cal',subject='fixture',stage='mix-test-5',status='prepared',channels={str(j):{} for j in range(10)})]))
        mask=np.ones((10,5),bool);target=np.arange(50,dtype=float).reshape(10,5)/10
        finger=np.broadcast_to(np.arange(5),(10,5));t=np.broadcast_to(np.arange(10)[:,None]*20_000_000,(10,5))
        for branch in BRANCHES:
            folder=root/'runs'/branch;(folder/'test_predictions').mkdir(parents=True)
            rows=[]
            for id,stage in [('fixture5','mix-test-5'),('fixture7','mix-test-7')]:
                pred=target+.2;raw=target+.3;cal=target+.1
                metrics={k:measures(pred,a,mask) for k,a in [('raw',raw),('cal',cal),('bsq',target)]}
                rows.append(dict(id=id,subject='fixture',stage=stage,metrics=metrics,
                    by_finger={str(j):metrics['bsq'] for j in range(10)},
                    BSQ_force_bands={b:metrics['bsq'] for b in ['0-2','2-10','10-25','25+']}))
                np.savez_compressed(folder/'test_predictions'/(id+'.npz'),pred=pred,raw=raw,cal=cal,bsq=target,mask=mask,finger=finger,t_ns=t)
            atomic(folder/'final_test.json',dict(rows=rows));atomic(folder/'DONE.json',dict(phase='complete'))
        report(root)
        result=load(root/'results/FINAL_REPORT.json')
        assert len(result['experiments'])==4 and result['point_and_ground_truth_identity_checked']
        assert len(result['by_stage'])==8
        assert (root/'results/report.html').stat().st_size>1000
        # Reject a changed point mask instead of publishing a misleading comparison.
        path=root/'runs/raw_original/test_predictions/fixture5.npz'
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        a['mask'][0,0]=False;np.savez_compressed(path,**a)
        try:report(root)
        except ValueError as exc:assert 'Scoring points or GT differ' in str(exc)
        else:raise AssertionError('Changed Test denominator was accepted')
    atomic(output/'report_software_test.json',dict(status='PASS_SYNTHETIC_REPORT_AND_POINT_IDENTITY_GUARD',
        real_Test_metrics_read=False,completed_model_claim=False))
    print('PASS: four comparisons, stage/finger/force-band summaries, HTML/plot, unequal-point rejection')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)

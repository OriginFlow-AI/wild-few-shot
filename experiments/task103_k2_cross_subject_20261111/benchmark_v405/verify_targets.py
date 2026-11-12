"""Compare reconstructed calibrated targets with the frozen v4.05 sufficient statistics."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_test import HANDS, atomic, read, sha


def main(root, reference):
    old=read(reference)
    source=root/'signal_receipt.json'
    receipt=read(source)
    assert not receipt['errors'] and len(receipt['recordings'])==109
    assert old['evaluation_token_count']==receipt['owned_points']==500040
    assert old['evaluation_token_sha256']==receipt['identity_sha256']
    by_subject_hand=defaultdict(lambda: np.zeros((5,5),dtype=np.float64))
    for row in receipt['recordings']:
        path=root/'signals'/f"{row['trial_id']}.npz"
        assert sha(path)==row['cache_sha256']
        with np.load(path,allow_pickle=False) as z:
            for hand in HANDS:
                idx=z[f'{hand}_score_indices']
                y=z[f'force_{hand}_calibrated_N'][idx].astype(np.float64)
                assert len(y)==row['owned_points'][hand]
                if len(y):
                    by_subject_hand[f"{row['subject_id']}/{hand}"]+=np.stack([
                        y.sum(axis=0), (y**2).sum(axis=0), abs(y).sum(axis=0),
                        (abs(y)>.5).sum(axis=0),np.full(5,len(y))])
    frozen=old['sufficient_statistics_by_subject_hand']
    fields=['sum_target','sum_target_squared','sum_abs_target','active_count','count']
    ids=[old['sufficient_statistics_rows'].index(k) for k in fields]
    assert set(frozen)==set(by_subject_hand)
    diffs=[]
    for key,current in by_subject_hand.items():
        before=np.asarray(frozen[key])[ids]
        diffs.append(abs(before-current))
        np.testing.assert_allclose(current,before,rtol=1e-10,atol=1e-8,err_msg=key)
    maximum=dict(zip(fields,np.max(diffs,axis=(0,2)).tolist()))
    atomic(root/'target_parity.json',dict(
        historical_metrics=str(reference), historical_metrics_sha256=sha(reference), signal_receipt_sha256=sha(source),
        subjects=6,subject_hands=len(frozen),timepoints=500040,identity_sha256=receipt['identity_sha256'],
        fields=fields,max_absolute_difference=maximum))
    print('v4.05 targets match historical subject×hand×finger statistics:',maximum,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.reference)

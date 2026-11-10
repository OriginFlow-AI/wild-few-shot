"""Audit actual indices, split isolation and contracts without touching training."""
import argparse
import collections
from pathlib import Path
import sys
import numpy as np


def main(root):
    root=Path(root)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from common import BRANCHES, EXPERIMENT, atomic, load, sha, source_binding, windows
    dataset=load(root/'prepared/dataset.json');plan=load(root/'plan.json')
    assert dataset['source_code']==source_binding()
    assert dataset['config_sha256']==sha(EXPERIMENT/'config.json')
    assert dataset['plan_sha256']==sha(root/'plan.json')
    entries=dataset['entries'];split_ids={};indices={}
    assert all(r['source_id']!=r['id'] for r in entries)
    for split in ['train','val','test']:
        path=root/'prepared'/(split+'_windows.npy');pairs=np.load(path)
        assert len(np.unique(pairs,axis=0))==len(pairs)==dataset['windows'][split]
        for entry,start in pairs:
            row=entries[int(entry)]
            assert row['split']==split and 0<=start and start+4000<=row['samples']
        split_ids[split]={r['id'] for r in entries if r['split']==split}
        indices[split]=dict(windows=len(pairs),recordings=len(split_ids[split]),sha256=sha(path))
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not split_ids[a]&split_ids[b]
    trained=[]
    for branch in BRANCHES:
        folder=root/'runs'/branch;contract=load(folder/'CONTRACT.json')
        assert contract['dataset_sha256']==sha(root/'prepared/dataset.json')
        assert contract['source_code']==source_binding()
        assert contract['config_sha256']==dataset['config_sha256']
        assert contract['common_initial_sha256']==sha(root/'assets/common_stage2_initial_state_dict.pt')
        assert contract['effective_microbatch']==16
        kind='cal' if branch=='calibrated_huber' else 'raw'
        assert contract['force_norm']==dataset['force_norm'][kind]
        trained.append(dict(branch=branch,status=load(folder/'STATUS.json'),
            checkpoint_bytes=(folder/'last.pt').stat().st_size,contract_sha256=sha(folder/'CONTRACT.json')))
    included=set.union(*split_ids.values());excluded={r['id'] for r in dataset['exclusions']}
    empty=[]
    for r in plan['rows']:
        if not r['active'] or r['id'] in included or r['id'] in excluded:continue
        details={}
        with np.load(root/'cache/labels'/(r['id']+'.npz')) as z:
            meta=load(root/'cache/emg'/r['id']/'done.json')
            for side in ['left','right']:
                ev=np.load(root/'cache/emg'/r['id']/(side+'_valid.npy'),mmap_mode='r')
                prefix=np.r_[0,np.cumsum(~ev)];starts=windows(meta['hands'][side]['samples'])
                details[side]=dict(native_common_force_points=int(z[side+'_valid'].sum()),
                    full_EMG_valid_windows=sum(prefix[s+4000]==prefix[s] for s in starts))
        reason='no_common_native_force_points' if not any(x['native_common_force_points'] for x in details.values()) else 'no_full_EMG_valid_windows' if not any(x['full_EMG_valid_windows'] for x in details.values()) else 'no_common_force_at_full_valid_EMG_token_queries'
        empty.append(dict(id=r['id'],subject=r['subject'],split=r['split'],reason=reason,hands=details))
    atomic(root/'checks/prepared_launch_audit.json',dict(status='PASS',indices=indices,
        calibration_version=plan['calibration_version'],source_binding_verified=True,
        record_splits_disjoint=True,common_initialization_and_data_verified=True,
        branches=trained,empty_window_recordings=empty,
        exceptions=len(dataset['exclusions']),empty_window_reason_counts=dict(collections.Counter(r['reason'] for r in empty)),
        note='Startup/data audit only. Final-model completion and Test accuracy are not claimed. Test5 BSQ is provided offline calibration input to other records, not untouched truth.'))
    print(indices,'empty-window recordings',len(empty))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

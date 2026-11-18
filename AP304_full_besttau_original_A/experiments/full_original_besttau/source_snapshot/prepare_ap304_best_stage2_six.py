"""Freeze the authorized best-tau stage-2 experiment without GPU activity."""
from pathlib import Path
import csv
import json
import os
import shutil
import sys

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/src'))
from emg2force.ap304_rank_data import sha,atomic_json
from emg2force.ap304_stage2_data import preparation_binding
from emg2force.ap304_six_metrics import METHODS

OLD=ROOT/'runs/ap304_full_calibrated_stage2_lambda0_seed43_20261028'
OUT=ROOT/'runs/ap304_full_calibrated_stage2_besttau_lambda0_six_seed43_20261029'


def main():
    os.nice(19);torch.set_num_threads(1)
    resolved=OUT/'resolved_config.yaml'
    if resolved.exists():
        print('FROZEN_CONFIG_ALREADY_EXISTS',resolved);return
    OUT.mkdir(parents=True,exist_ok=True);inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    c=yaml.safe_load((OLD/'resolved_config.yaml').read_text())
    source=ROOT/'runs/ap304_full_raw_rank_seed43_20261027/checkpoints/best_tau.pt'
    ck=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    expected=dict(epoch=1,next_update=22661,global_update=67982)
    assert all(ck['state'][k]==v for k,v in expected.items())
    assert ck['output_units']=='dimensionless_ranking_score'
    with (source.parent.parent/'epoch_metrics.csv').open() as f:
        history=[x for x in csv.DictReader(f) if x['split']=='val' and x['label'].startswith('update')]
    best=max(history,key=lambda x:float(x['tau_b']))
    assert int(best['global_update'])==67982
    selection=dict(original_checkpoint=str(source),checkpoint_sha256=sha(source),state=expected,
        selected_by='maximum full Val stage-macro trial-finger Kendall tau-b',validation=best,
        original_history_sha256=sha(source.parent.parent/'epoch_metrics.csv'))
    del ck
    target=inputs/'ranking_best.pt'
    if not target.exists():os.link(source,target)
    assert sha(target)==selection['checkpoint_sha256']
    atomic_json(OUT/'ranking_best_selection.json',selection)
    for key,name in [('index','split.json'),('emg_norm','emg_train_norm.npz')]:
        destination=inputs/name;shutil.copyfile(c[key],destination);c[key]=str(destination)
    calibration=inputs/'calibration';calibration.mkdir(exist_ok=True)
    original_cal=Path(c['calibration_package'])
    for name in ('calibration_tools.py','calibration_lookup.json','calibration_parameters.json','LABEL_POLICY.json'):
        shutil.copyfile(original_cal/name,calibration/name)
    shutil.copytree(original_cal/'calibration_assets',calibration/'calibration_assets',dirs_exist_ok=True)
    c['calibration_package']=str(calibration)
    snapshot=OUT/'source_snapshot';package=snapshot/'emg2force';package.mkdir(parents=True,exist_ok=True)
    for name in ('__init__.py','config.py','losses.py','ap304_rank_data.py','ap304_stage2_data.py',
                 'ap304_six_metrics.py','ap304_stage2_six_train.py'):
        shutil.copyfile(ROOT/'code/src/emg2force'/name,package/name)
    shutil.copyfile(c['model_source'],snapshot/'model_source.py')
    c.update(run_root=str(OUT),model_source=str(snapshot/'model_source.py'),source_checkpoint=str(target),
        source_checkpoint_sha256=selection['checkpoint_sha256'],source_checkpoint_state=expected,
        initialization='best_stage1_Val_tau_checkpoint_except_reinitialized_output_head',
        experiment_id=OUT.name,evaluation_aggregation_methods=METHODS,
        reporting_primary_finger_rule='pool all five fingers; separate per-finger alternative also saved',
        reporting_undefined_metrics='null/exclude with counts; legacy trial-equal selection loss unchanged',
        checkpoint_policy='every epoch before validation and after validation, plus best and periodic last',
        replaced_experiment=str(OLD))
    for key in ('stop_receipt','source_receipt_sha256'):c.pop(key,None)
    for name in ('PREPARED.json','train_norm.npz'):
        shutil.copyfile(OLD/name,OUT/name)
    shutil.copytree(OLD/'prepared',OUT/'prepared',dirs_exist_ok=True)
    ready=json.loads((OUT/'PREPARED.json').read_text())
    assert ready['binding']==preparation_binding(c)
    assert sha(OUT/'train_norm.npz')==c['force_norm_sha256']==ready['normalization_sha256']
    resolved.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
    config=ROOT/'configs/experiments/ap304_full_calibrated_stage2_besttau_lambda0_six_seed43.yaml'
    shutil.copyfile(resolved,config)
    hashes={str(p.relative_to(OUT)):sha(p) for p in snapshot.rglob('*.py')}
    hashes.update({str(p.relative_to(OUT)):sha(p) for p in calibration.rglob('*') if p.is_file()})
    for p in (resolved,inputs/'split.json',inputs/'emg_train_norm.npz',target,OUT/'train_norm.npz',OUT/'PREPARED.json',OUT/'ranking_best_selection.json'):
        hashes[str(p.relative_to(OUT))]=sha(p)
    atomic_json(OUT/'source_hashes.json',hashes)
    with np.load(OUT/'train_norm.npz',allow_pickle=False) as z:
        normalization={k:z[k].tolist() for k in ('emg_mean','emg_std','force_mean','force_std')}
    atomic_json(OUT/'experiment_parameters.json',dict(config=c,config_sha256=sha(resolved),
        source_selection=selection,normalization=normalization,
        normalization_provenance=dict(reused_from=str(OLD),train_only=True,
            SHA256=ready['normalization_sha256'],same_split_calibration_and_raw_file_bindings=True),
        loss='(1-Corr)+0.3*MAPE_ratio+0.03*MAE_N; lambda_ord=0',
        optimizer=dict(name='AdamW',betas=[.9,.999],epsilon=1e-8,reset=True),
        head=dict(reinitialized=True,gate_bias=4.,all_other_parameters_loaded=True),
        score_grid=dict(input_samples=4000,first_target=470,target_stride=40,targets_per_window=89,
            reset_every_window=True,tail_deduplicated=True,extra_prefix=0,MA7=False),
        six_methods=METHODS,primary_finger_rule=c['reporting_primary_finger_rule'],
        stage_scope='all stages present in each split; no exclusions',
        selection='original full-Val trial-equal minimum L_std; reports do not change selection',
        metric_statistics='float64 centered moments with between-trial merge terms; absolute error sum, active APE sum/count; per trial and finger',
        saved_aggregates='six methods, per subject, per subject-stage, per stage and per finger',
        undefined_metrics=c['reporting_undefined_metrics'],
        Test_policy='only after early stop/max epochs, using best Val checkpoint',
        Kendall_tau_policy='per-trial per-finger tau-b saved; six reaggregation reports cover Corr/MAPE/MAE'))
    atomic_json(OUT/'status.json',dict(phase='prepared_waiting_for_CPU_checks',source_global_update=67982,
        source_validation_tau=float(best['tau_b']),lambda_ord=0,six_methods=list(METHODS)))
    print(json.dumps(dict(prepared=str(OUT),source_global_update=67982,source_tau=float(best['tau_b']),
        config=str(resolved),GPU_used=False),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

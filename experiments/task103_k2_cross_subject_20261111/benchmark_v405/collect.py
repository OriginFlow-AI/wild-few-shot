"""Verify v4.05 model reports and export overall and stage comparison tables."""
import argparse
import csv
from pathlib import Path

from prepare_test import atomic, read, sha


BENCH=('benchmark_corr','benchmark_mae_N','benchmark_mape_gt_0p5N_percent','benchmark_mape_gt_1N_percent')
TASK=('corr','mae_N','mae_below_1N','mape_ge_0p5N_percent','mape_ge_1N_percent','kendall_tau_b')
DISPLAY={
    'A_ep1':'A_ep1',
    'ap304_within_person_calibrated_huber':'AP304 同人 · 标定 Huber',
    'ap304_within_person_raw_huber':'AP304 同人 · 原始 Huber',
    'ap304_within_person_raw_original':'AP304 同人 · 原始力原损失',
    'k2_person_restricted_calibrated_huber':'K2 跨人指定 stage · 标定 Huber',
    'k2_person_restricted_raw_huber':'K2 跨人指定 stage · 原始 Huber',
    'k2_person_restricted_raw_original':'K2 跨人指定 stage · 原始力原损失',
    'k2_person_all_raw_original':'K2 全 stage 跨人 · 原损失',
    'ap304_person_all_raw_huber':'AP304 全 stage 跨人 · Huber',
    'ap304_person_all_raw_original':'AP304 全 stage 跨人 · 原损失',
    'ap304_person_cal_calibrated_original':'AP304 跨人 · 标定力调整系数',
    'ap304_rank_best':'AP304 一阶段排序',
}


def markdown(path, rows, participant_audit):
    lines=[
        '# Benchmark v4.05 Test · EMG-only',
        '',
        '6 人、109 段、500,040 个时间点，每点五指。109 段 RRD 的大小及 checker 哈希已核对。'
        '固定计分身份与官方清单一致，重建后的标定标签按人、手、指与历史统计交叉核验。',
        '',
        '各模型用配套归一化，以 FP32 Online W100 推理。EMG 使用 RRD 交付值，因果对齐到 2 kHz，'
        '50 Hz 的预测保持到下次更新并对齐至原生力时间戳。真值为原始 CMCU 前五指 mN 转 N 后应用一次 v4.05 LUT。',
        '',
        '下表使用 v4.05 的人员等权、力幅加权 Corr 和 MAPE，MAE 为人员与手指等权。'
        '低力 MAE 和 Kendall τ-b 用 Task103 的 recording 后按人等权口径。排序模型的绝对力指标留空。',
        '',
        '| 模型 | 权重轮次 | Corr | MAE (N) | MAE <1 N (N) | MAPE >0.5 N | MAPE >1 N | Kendall τ-b |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    fields=('benchmark_corr','benchmark_mae_N','mae_below_1N','benchmark_mape_gt_0p5N_percent',
            'benchmark_mape_gt_1N_percent','kendall_tau_b')
    for r in rows:
        if r['stage']!='all':
            continue
        values=['—' if r[k] is None else f'{r[k]:.2f}%' if 'mape' in k else f'{r[k]:.3f}' for k in fields]
        lines.append('| '+' | '.join([DISPLAY[r['model']],str(r['epoch'] or '—'),*values])+' |')
    lines.extend(['','[逐 stage 和两套指标口径](metrics.csv) · [权重和计分凭据](audit.json) · [标签复建核验](target_parity.json)',''])
    if participant_audit:
        lines.extend(['按人员标识核对，v4.05 六位 Test 受试者与本次 Task103 训练集和 AP304 一阶段历史 Train 的交集均为零。'
                      'A_ep1 的完整回归训练划分未独立核验。[人员重叠核验](subject_overlap_audit.json)',''])
    path.write_text('\n'.join(lines),encoding='utf-8')


def main(root,registry,output):
    source=root/'signal_receipt.json'
    receipt=read(source)
    parity=read(root/'target_parity.json')
    assert parity['signal_receipt_sha256']==sha(source)
    assert parity['identity_sha256']==receipt['identity_sha256'] and parity['timepoints']==500040
    expected={r['trial_id']:(r['subject_id'],r['stage'],sum(r['owned_points'].values())) for r in receipt['recordings']}
    assert len(expected)==109 and len({r[0] for r in expected.values()})==6
    rows,proof=[],[]
    for item in read(registry):
        model=item['name']
        dest=root/'results'/model
        report=read(dest/'report.json')
        assert report['model']==model and report['signal_receipt_sha256']==sha(source)
        assert report['evaluation_token_identity_sha256']==receipt['identity_sha256']
        assert report['mask_sha256']==receipt['mask_sha256']
        assert report['qualifier']['reset_bitwise_equal'] and report['max_hold_s']<=.023
        with (dest/'per_recording.csv').open(newline='') as f:
            records={r['trial_id']:(r['subject_id'],r['stage'],int(r['timepoints'])) for r in csv.DictReader(f)}
        assert records==expected
        assert len(report['metrics'])==len({r[1] for r in expected.values()})+1
        assert {r['stage'] for r in report['metrics']}=={'all',*(r[1] for r in expected.values())}
        for r in report['metrics']:
            scope=expected.values() if r['stage']=='all' else [v for v in expected.values() if v[1]==r['stage']]
            assert r['timepoints']==sum(v[2] for v in scope) and r['recordings']==len(scope)
            rows.append(dict(model=model,family=report['family'],epoch=report['epoch'],unit=report['output_unit'],**r))
        proof.append({k:v for k,v in report.items() if k!='metrics'})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    atomic(output/'audit.json',dict(
        frozen_mask_sha256=receipt['mask_sha256'],frozen_index_sha256=receipt['identity_sha256'],
        trials=109,subjects=6,timepoints=500040,models=proof))
    atomic(output/'target_parity.json',parity)
    participant_audit=root/'subject_overlap_audit.json'
    if participant_audit.is_file():
        atomic(output/participant_audit.name,read(participant_audit))
    markdown(output/'README.md',rows,participant_audit.is_file())
    print((output/'README.md').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','registry','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    args=parser.parse_args()
    main(args.root,args.registry,args.output)

"""Produce the four-way comparison only after all three selected models finish."""
import argparse
import collections
import csv
import base64
import html
import json
import os
from pathlib import Path
import numpy as np
from common import BRANCHES, atomic, load
from train import measures

def aggregate_by_subject(records):
    groups=collections.defaultdict(list)
    for subject,metric in records:
        if metric is not None:groups[subject].append(metric)
    result=dict(subjects=len(groups),scored_units=sum(map(len,groups.values())))
    for k in ['mae_N','mape_percent','corr','huber_N2']:
        means=[]
        for rr in groups.values():
            values=[r[k] for r in rr if r[k] is not None]
            if values:means.append(float(np.mean(values)))
        result[k]=float(np.mean(means)) if means else None
        result['valid_subjects_'+k]=len(means)
    return result

def quality_summary(root):
    plan=load(root/'plan.json')['rows'];dataset=load(root/'prepared/dataset.json')
    quality=load(root/'checks/calibration_quality.json')['records']
    problems=collections.Counter(reason for r in quality for reason in r.get('problems',{}).values())
    failures=[dict(id=r['id'],subject=r['subject'],stage=r['stage'],reason=r.get('reason')) for r in quality if r['status']!='prepared']
    zero=[dict(id=r['id'],subject=r['subject'],stage=r['stage'],events=r['events'],reasons=r.get('rejected_event_reasons'))
          for r in quality if r['stage']=='mix-test-5' and r['status']=='prepared' and not r['channels']]
    selection={s:dict(all_recordings=sum(r['split']==s for r in plan),
                      metadata_active=sum(r['split']==s and r['active'] for r in plan),
                      prepared_recordings=len({r['id'] for r in dataset['entries'] if r['split']==s}),
                      windows=dataset['windows'][s]) for s in ['train','val','test']}
    extraction_errors=[]
    for p in (root/'checks').glob('extraction_*.json'):
        # Progress files carry counters, final extraction files carry errors.
        a=load(p)
        if isinstance(a.get('errors'),list):extraction_errors.extend(a['errors'])
    diagnostics=root/'analysis_output/accepted_calibration_diagnostics.json'
    diagnostics=load(diagnostics) if diagnostics.exists() else None
    if diagnostics and diagnostics['calibration_version']!=load(root/'plan.json')['calibration_version']:
        raise ValueError('Stale source calibration diagnostics')
    return dict(selection=selection,channel_QC_failures=dict(problems),record_QC_failures=failures,
        accepted_curve_review_diagnostics=diagnostics,
        test5_without_any_valid_channel=zero,extraction_errors=extraction_errors,
        dataset_exclusion_reasons=dict(collections.Counter(r['reason'] for r in dataset['exclusions'])),
        partial_test5_records=sum(r['stage']=='mix-test-5' and 0<len(r['channels'])<10 for r in quality),
        experience=[
            '先核实BSQ的N、CMCU的mN及同步时钟；不要排序时间戳后掩盖逆序或跨时钟异常。',
            '最大AD不一定是受测指，test7的支撑拇指尤其容易抢占；歧义指身份应屏蔽。',
            'BSQ零点应固定在姿态稳定空载段；整段均值、翻转前首秒都可能引入系统偏差。',
            '没有完整按压、AD变化不足、单调性弱和拟合力范围坍缩分别记录；缺指不等于整条硬件损坏。',
            '同一AD下参考力明显不一致时，单调相关也可能通过。源拟合残差P90同时超过10N和拟合力范围50%的指通道已屏蔽；较轻残差复核仅作标记，通过基础规则不等于独立精度合格。',
            '单调拟合不能解决饱和、迟滞、佩戴变化或域外数据。域外点必须计入覆盖限制，不能静默夹端点。',
            'test5按原始最近test5口径使用自身整次曲线；源内拟合符合度不等于独立曲线精度。本方案使用现场BSQ作为自身及其他记录的标定输入，不是BSQ完全未参与过标定的评估。',
            '不按模型Test误差挑源或改剔除阈值；最近源失效不静默回退。'])

def main(root):
    root=Path(root);data={}
    for b in BRANCHES:
        if load(root/'runs'/b/'DONE.json')['phase']!='complete':raise ValueError('Incomplete branch')
        data[b]={r['id']:r for r in load(root/'runs'/b/'final_test.json')['rows']}
    ids=set(data[BRANCHES[0]])
    if any(set(v)!=ids for v in data.values()):raise ValueError('Test record denominators differ')
    checked=0;sensor_baselines={'raw':[],'cal':[]}
    for id in sorted(ids):
        paths=[root/'runs'/b/'test_predictions'/(id+'.npz') for b in BRANCHES]
        with np.load(paths[0]) as z:reference={k:z[k].copy() for k in ['mask','raw','cal','bsq','finger','t_ns']}
        for p in paths[1:]:
            with np.load(p) as z:
                if not all(np.array_equal(reference[k],z[k]) for k in reference):raise ValueError('Scoring points or GT differ: '+id)
        for label in sensor_baselines:
            sensor_baselines[label].append((data[BRANCHES[0]][id]['subject'],
                measures(reference[label],reference['bsq'],reference['mask'])))
        checked+=1
    experiments=[(1,'calibrated_huber','cal'),(2,'calibrated_huber','bsq'),(3,'raw_huber','bsq'),(4,'raw_original','bsq')]
    rows=[];people={};by_stage=[];by_finger=[];by_band=[]
    for n,b,target in experiments:
        subject=collections.defaultdict(list);stage=collections.defaultdict(list)
        for r in data[b].values():
            subject[r['subject']].append(r['metrics'][target]);stage[r['stage']].append(r['metrics'][target])
        def avg(records,key):
            values=[r[key] for r in records if r is not None and r[key] is not None]
            return float(np.mean(values)) if values else None
        by_person={s:{k:avg(rr,k) for k in ['mae_N','mape_percent','corr','huber_N2']} for s,rr in subject.items()}
        people[n]=by_person
        row=dict(experiment=n,training_branch=b,test_label=target,recordings=len(ids),subjects=len(subject))
        for k in ['mae_N','mape_percent','corr','huber_N2']:
            row['subject_equal_'+k]=avg(list(by_person.values()),k)
            row['trial_equal_'+k]=avg([r['metrics'][target] for r in data[b].values()],k)
            row['valid_subjects_'+k]=sum(r[k] is not None for r in by_person.values())
        rows.append(row)
        for stage_name in sorted(stage):
            metric=aggregate_by_subject([(r['subject'],r['metrics'][target]) for r in data[b].values() if r['stage']==stage_name])
            by_stage.append(dict(experiment=n,stage=stage_name,test_label=target,**metric))
        if target=='bsq':
            for j in range(10):
                metric=aggregate_by_subject([(r['subject'],r['by_finger'][str(j)]) for r in data[b].values()])
                by_finger.append(dict(experiment=n,finger_index=j,**metric))
            for band in ['0-2','2-10','10-25','25+']:
                metric=aggregate_by_subject([(r['subject'],r['BSQ_force_bands'][band]) for r in data[b].values()])
                by_band.append(dict(experiment=n,BSQ_band_N=band,**metric))
    paired=[]
    for a,b,name in [(2,3,'calibration: experiment2 minus3'),(3,4,'loss: experiment3 minus4')]:
        for s in sorted(set(people[a])&set(people[b])):
            paired.append(dict(comparison=name,subject=s,mae_difference_N=people[a][s]['mae_N']-people[b][s]['mae_N']))
    out=root/'results';out.mkdir(exist_ok=True)
    for name,rr in [('four_experiments.csv',rows),('paired_person_differences.csv',paired),
                    ('by_stage.csv',by_stage),('by_finger.csv',by_finger),('by_force_band.csv',by_band)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    quality=quality_summary(root);atomic(out/'calibration_quality_summary.json',quality)
    sensor_summary={label:aggregate_by_subject(rr) for label,rr in sensor_baselines.items()}
    atomic(out/'FINAL_REPORT.json',dict(status='COMPLETE',experiments=rows,per_person=people,
        by_stage=by_stage,by_finger_BSQ=by_finger,by_force_band_BSQ=by_band,quality=quality,
        sensor_vs_BSQ_same_scoring_points=sensor_summary,
        common_test_recordings=checked,point_and_ground_truth_identity_checked=True,
        calibration_role='all complete eligible presses; test5 uses its own whole curve, other stages use nearest test5; provided offline calibration input',
        caveats=['Experiment1 measures agreement with calibrated CMCU. Experiments2/3/4 use BSQ truth.',
                 'Metrics cover the common in-domain, routed, synchronized, EMG-valid subset; exclusions must be read with dataset and calibration QC.',
                 'This is person-disjoint model evaluation with offline reference-assisted calibration. Held-out people do not contribute to model fitting or normalization. Test5 BSQ remains available as input to per-record calibration and inclusion masks.',
                 'A scored test5 uses its own complete calibration curve. This is calibration-assisted evaluation, not independent curve validation. For other stages, nearest test5 may be before or after the target.']))
    lines=['Task103 K2 5.56M person-disjoint four-way comparison', '', 'Subject-equal metrics on identical scoring points:',
           'experiment\ttest label\tMAE_N\tMAPE_%\tCorr']
    def fmt(value,digits=6):return 'NA' if value is None else f'{value:.{digits}f}'
    for r in rows:lines.append(f"{r['experiment']}\t{r['test_label']}\t{fmt(r['subject_equal_mae_N'])}\t{fmt(r['subject_equal_mape_percent'])}\t{r['subject_equal_corr']}")
    lines.extend(['','See FINAL_REPORT.json and paired_person_differences.csv.',
                  'Read calibration_quality.json and dataset exclusions before interpreting retained-point accuracy.'])
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    mpl=root/'tmp/mpl';mpl.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(mpl)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.4))
    for i,stage_name in enumerate(['mix-test-5','mix-test-7']):
        subset=[r for r in by_stage if r['experiment'] in [2,3,4] and r['stage']==stage_name]
        if not subset:continue
        ax.bar(np.array([r['experiment'] for r in subset])+(.18 if i else -.18),
               [r['mae_N'] for r in subset],width=.35,label=stage_name)
    ax.set(xticks=[2,3,4],xlabel='Experiment (BSQ truth)',ylabel='Subject-equal MAE (N)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'BSQ_MAE_by_stage.png',dpi=180);plt.close(fig)
    image=base64.b64encode((out/'BSQ_MAE_by_stage.png').read_bytes()).decode()
    header='<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in ['实验','Test标签','人员等权MAE_N','人员等权MAPE_%','人员等权Corr'])+'</tr>'
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['experiment'],r['test_label'],
              fmt(r['subject_equal_mae_N'],4),fmt(r['subject_equal_mape_percent'],3),r['subject_equal_corr']])+'</tr>' for r in rows)
    experiences=''.join('<li>'+html.escape(x)+'</li>' for x in quality['experience'])
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Task103四组实验</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:16px/1.65 system-ui;padding:0 20px;color:#1c2433}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%;max-width:850px}pre{white-space:pre-wrap;background:#f3f5f8;padding:16px}</style>'
        '<h1>Task103 K2 5.56M 跨人四组实验</h1><p>三套独立训练，完整Val选模后最终Test。第1组衡量与标定CMCU标签的符合程度，第2/3/4组用同一BSQ真值与计分点比较。</p><p>Train、Val、Test 人员互斥，使用现场 BSQ 作为离线标定输入。test5计分使用自身本次完整曲线；其他stage的最近test5可以在目标之前或之后，其BSQ作为自身及其他记录的标定输入，不能称为BSQ完全未参与过标定的评估。</p>'
        '<table>'+header+body+'</table><img alt="BSQ MAE by stage" src="data:image/png;base64,'+image+'">'
        '<h2>同一计分点上的标签质量</h2><pre>'+html.escape(json.dumps(sensor_summary,ensure_ascii=False,indent=2))+'</pre>'
        '<h2>覆盖与排除</h2><p>下面数量用于约束精度结论。分数仅覆盖共同域内、可路由、同步及EMG有效的子集，不能等同全部原始数据精度。</p><pre>'
        +html.escape(json.dumps(quality['selection'],ensure_ascii=False,indent=2))+'</pre>'
        '<h2>标定检查经验</h2><ul>'+experiences+'</ul><pre>'
        +html.escape(json.dumps({k:v for k,v in quality.items() if k in ['channel_QC_failures','dataset_exclusion_reasons','partial_test5_records']},ensure_ascii=False,indent=2))
        +'</pre><h2>已接受源曲线的残差复核</h2><pre>'
        +html.escape(json.dumps(quality['accepted_curve_review_diagnostics'],ensure_ascii=False,indent=2))
        +'</pre><p>源拟合残差不是独立精度；复核标记不改变训练mask。完整统计见 FINAL_REPORT.json、by_stage.csv、by_finger.csv、by_force_band.csv 和 paired_person_differences.csv。未用神经网络Test误差反向挑选曲线或调整阈值。</p></html>')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

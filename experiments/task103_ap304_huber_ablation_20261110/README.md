# Task103 AP304 四组实验

三套二阶段训练覆盖四组比较。正式结果以三个训练分支的完整Val选模、最终Test和同点核对全部完成为准；进度快照不代表完成。

| 实验 | Train/Val标签 | Test真值 | 训练loss | 分支 |
|---|---|---|---|---|
| 1 | 标定CMCU | 标定CMCU | Huber，δ=1N | calibrated_huber |
| 2 | 标定CMCU | BSQ | 同实验1，共享模型 | calibrated_huber |
| 3 | 原始设备CMCU force | BSQ | Huber，δ=1N | raw_huber |
| 4 | 原始设备CMCU force | BSQ | (1−Corr)+0.3×MAPE+0.03×MAE | raw_original |

## 一阶段权重与二阶段训练

使用`.21:/data/zzy_data/emg2force_k2_stage_training_ap304_ex45/runs/ap304_full_raw_rank_seed43_20261027/checkpoints/best_tau.pt`，global update67982、完整Val stage-macro τ-b=0.4342530492。原checkpoint SHA256为`85445c11c7b20764ddb04bbf7e4baedfb6ff269ac0c3ff608b84e6e7be15d22a`，导出state_dict SHA为`f6dfe88f7e25d5be934ee0b065ae76d94aa89edc88340164244ecd149fa89928`。不使用旧epoch004_half或ZIP的二阶段成品继续训练。

AP304 width304、9 encoder、4 decoder，18,357,945参数，原生输出`[B,5,89]`。保留243个非输出头tensor，seed43重置两项输出头tensor、gate bias=4，全部参数可训练，重置优化器。三套都加载同一份`common_stage2_initial_state_dict.pt`，SHA为`f1fb53b6abeaecbabfea746a27a6cc41a831c5153a902f67a031aab394c8675b`。

EMG沿用一期norm（SHA `7ee5781e253ca45c2f31176e11ec5f065fd529df8afd9dc80bde5e3fec9d2de6`）；raw/cal force分别在共同有效的Train原生点上统计mean/std，原始loss与raw-Huber共享raw统计。Huber在输出反归一化后的N域计算，δ=1N；MAPE只取`abs(GT)>=0.5N`，在loss里使用比例值。无活动点的诊断MAPE记null；原loss训练/选模保留MAPE=0的有限约定。

AdamW：body2e-5、head1e-4，weight_decay1e-4，bias/norm不衰减；warmup1000 updates，10epoch跨度cosine到0.1倍，BF16前向、FP32 loss、clip1、global batch256。三套固定逻辑loss组16，按有效标量点数累积；原loss的Corr在每个16窗组的五指和时间点一起计算，不等于对256窗计算一次全局Corr。max10 epochs、Val patience3、min_delta1e-4，每轮完整Val按trial-equal分支目标选best，训练结束后最终Test。共同前向计算调度为前3轮16窗、第4轮起64窗；合批后仍分别计算每个16窗loss，Val/Test前向保持16窗。计算调度的上线证据见后文。

ZIP的pure-Huber旧对照固定1epoch；本实验沿用主二阶段max10/patience3政策，三套预算一致。`reference/PROVENANCE.json`保存来源及各loss追溯。

## 数据、划分与标定来源

冻结清单合并7份已有元数据，2026-10-19至11-10，共10,576条、109个subject、110个subject-day、约352小时；不是实时对象存储重枚举。普通stage为mix-1至mix-18，共8,163条；`mix-test-5` 1,977条、`mix-test-7` 436条。另29条元数据解析异常单列。

普通stage归Train。test5按subject/date/PC/boot分层，以seed43稳定哈希按完整recording划分：Val988、Test989；test7全部归Test。本轮二阶段Train没有test5/test7的EMG。三套模型共用记录、窗口及有效掩码，避免把额外训练数据误计为非标定方案优势。

按原始“最近一次test5标定全部CMCU”口径，**test5使用自身本次完整曲线**；普通stage及test7在同人、同日、同PC/boot中取墙钟录制中点绝对时间最近的test5，可在目标之前或之后，并用同步时钟核对连续性。PC/boot只是佩戴/硬件连续性的代理，不能排除未记录的换膜或重戴。不跨人借曲线，不静默绕过坏最近源。

普通stage有5,767条候选；2,380条同人同日无test5（来自10月19–23日），另16条时钟不一致。缺标定记录和坏通道明确排除，不用他人曲线补齐。最终Train5,702条、680,815窗；Val983条、40,447窗；Test1,406条、53,421窗。实际索引、逐记录/标签哈希和Train-only norm见`results/dataset_manifest.json`及`results/window_indices`。

这是**提供现场BSQ标定输入的离线个体评估**：test5的BSQ用于自身及其他记录的CMCU标定，不是BSQ完全未参与过标定、跨人或未见标定泛化基准。实验1衡量与标定CMCU标签的符合程度，自身曲线符合度不代表独立标定精度；实验2/3/4用相同BSQ、同受测指和同计分点比较神经模型。

## 标定质量与参考路由

完整test5每指全部合格按压的quiet/core点用于median-per-AD、count-weighted PAVA。至少20点、8种AD、AD跨度100、Spearman>=0.2、拟合力跨度>=1N；源拟合残差P90同时>10N和>拟合力范围50%的明显失配指通道屏蔽。仅在标定AD域内预测，不夹端点或外推；缺指和局部故障不自动等同整条硬件损坏。

1,977条test5中1,966条至少一个可用指，16,307条可拟合指曲线；1,124条部分指可用。239个指通道弱单调、2个力范围坍缩、88个严重残差失配。另4,040条较轻可疑曲线只作复核标记，不按神经Test误差删改。`results/calibration_quality_examples.png`与`accepted_calibration_diagnostics.*`提供示例和完整表。

跨按压质检：13,212条曲线有重复合格按压，13,206条可计分，81名人员，留出参考点覆盖79.20%；同批整次源拟合MAE中位0.544N、留一按压MAE中位1.441N、曲线MAE的90分位4.517N。另3,095条仅一次合格按压，无法重复复核。该诊断以整次源曲线先通过基础QC和固定指路由为条件，tare共享，衡量同次重复稳定性，不能代替跨次、跨人或模型Test。

BSQ是单通道参考，不能复制为十指真值。test7用活动手四个非拇指路由，避免支撑拇指抢占；指身份不明确则屏蔽。357条已有稳定姿态零位沿用记录锚点，余下要求>=0.4秒低位稳定空载段；已知正放例外保留记录标识。时钟逆序、参考量纲/恒定、零位不稳定、域外和缺帧分别留痕。

## 手动预处理与窗口

从MCAP流式读取双手16通道uV EMG、CMCU AD/force与BSQ，核对同步时间戳、逆序/重复、单位和有限值，CMCU mN转N。视频与whole MCAP不落盘。

EMG重采样2000Hz，每个连续有效段一次50Hz谐波陷波Q30、四阶40–850Hz因果SOS带通、首1000点均值初始化；>5ms缺口屏蔽并重置状态，段首0.5秒warmup无效。原生力标签不另加低通/MA7、不截负BSQ/预测，不按Test调时间平移。

4000点reset窗口、stride4000，补完整末窗；输出坐标`start+470+40*k,k=0..88`。CMCU线性对齐且不跨掩码或>40ms缺口，BSQ直接从原生50Hz插值到token查询时刻；Val/Test重叠末窗按真实时间保留新点。训练遍历全部保留窗，三套seed/逻辑顺序相同。

## GPU、续跑与口径修正审计

最多三路独立实验并行，按三次显存/利用率观测自动选共享卡。保留4GiB外部余量、2GiB CUDA开销后向下取4GiB档，allocator上限最高32GiB；固定8GiB上限已取消。初始真实Train benchmark：micro16最慢目标约101窗/秒、峰值7.1GiB；micro32约137窗/秒、峰值13.9GiB。初期可用卡条件优先三路同时运行，三套冻结逻辑loss组16。共享余量不能保证外部任务后续突增时完全无OOM；控制器保存完整model/Adam/RNG，每25更新last，支持重试和恢复。

正式训练最初给test5额外用了“最近另一条”的曲线。该约束不是原始“最近test5标定全部”的必要条件；已在首次完整Val之前纠正为自身曲线。修正前模型/Adam/游标/CPU-CUDA RNG完整保存；`analysis/migrate_validation_only_binding.py`要求普通stage的全部逻辑Train窗、输入/标签哈希、norm和训练算法**完全一致**后才重绑Val/Test数据与配置。修正保留2,575/1,275/1,375次已保存更新，不从一期重训。证据为`results/validation_only_protocol_migration.json`；原方案归档于服务器`archive/nearest_other_test5_before_self_correction`。

口径敏感性附件说明：在相同基础QC、路由和raw-force守卫的原生BSQ配对点上，自身曲线覆盖97.94%，最近另一条覆盖80.53%；无EMG/token筛选，不是模型Test指标，也未据此按神经Test误差选曲线。最早首压/前次试跑也归档，均不混为最终方法或结果。

独立`analysis/archive_epoch_candidates.py`在完整epoch边界用同一文件描述符复制model/Adam/RNG候选，保存到`archive/epoch_candidates/<branch>/<dataset_sha_prefix>/epochNNN.pt`。最终导出脚本在三套最终评测、同点核对完成后输出纯state_dict和推理参数，不上传Adam/RNG大断点。

## 运行与产物

所有新增服务器文件位于`.1:/pub_data/zjl/task103_ap304_huber_calibration_ablation_20261110`。可用`TASK103_CREDENTIAL_FILE`指向库外已有私密文件，或使用标准AWS环境变量；密钥不放Git。

```bash
git lfs pull
python -m pip install -r requirements.txt
export TASK103_REQUIRE_PUB_DATA=1
export TMPDIR=/pub_data/zjl/.tmp_task103_huber_1110
mkdir -p "$TMPDIR"
ROOT=/pub_data/zjl/task103_ap304_huber_calibration_ablation_20261110
EXP=experiments/task103_ap304_huber_ablation_20261110
python "$EXP/scripts/bootstrap.py" --root "$ROOT"
python "$EXP/scripts/verify_protocol.py"
python "$EXP/scripts/pipeline.py" --root "$ROOT" --workers 32
```

进度看`STATUS.json`、`runs/<branch>/STATUS.json`、`checks/epoch_archive_status.json`。最终为三个`runs/<branch>/DONE.json`及`results/FINAL_REPORT.json`、四组CSV、逐stage/指/力段/人差异、质量统计、PNG和HTML；未核验这些不能视为完成。

目标北京时间2026-11-11 08:00前，属于尽力目标。启动共享负载下三路约94/44/47窗每秒，raw分支单轮约4小时；不因截止时间缩减既定max10/patience3、完整Val或最终Test。Goal保持active直至完整结果与Git上传验收。

首个真实完整Val已完成：calibrated_huber第1轮覆盖全部983条Val记录，trial-equal Huber loss=7.375427957466372，保存第1轮best后进入第2轮。`results/epoch001_Val_and_state_audit.json`核对了完整记录集合、真实epoch候选与best，并验证完整Val前后model/Adam/CPU-CUDA RNG逐项不变。对应逐记录指标在`results/validation/calibrated_huber_epoch001_val.json`；这是Val检查，最终Test和四组对比仍未完成。

raw_huber第1轮完整Val也已完成：全部983条记录，raw标签trial-equal Huber loss=11.25892579844423，保存best并进入第2轮。`results/raw_huber_epoch001_Val_and_state_audit.json`核对完整记录集合、真实候选/best、有限参数，以及完整Val前后model/Adam/CPU-CUDA RNG不变；两个Huber分支逐记录计分点数一致。Val目标标签不同，不能直接把两个Val loss之差当作标定收益；最终收益比较使用同点BSQ Test。

raw_original第1轮完整Val覆盖全部983条记录，trial-equal原始loss=1.315550807049214，保存第1轮best。`analysis/audit_first_validation.py`在真实候选和best之间逐项核对model/Adam/CPU-CUDA RNG不变、参数有限，并确认三路完整Val记录集合及raw/cal/BSQ逐记录计分点数、MAPE有效点数全部一致。证据为`results/raw_original_epoch001_Val_and_state_audit.json`，逐记录指标在`results/validation/raw_original_epoch001_val.json`。原始loss与Huber数值单位不同，不能直接比较大小；三套最终Test均仍待训练结束。

北京时间11月11日02:16，GPU1/2三次新采样有足够显存且利用率低于60%，将raw_original从GPU7移至GPU1、calibrated_huber从GPU6移至GPU2。`analysis/migrate_shared_gpus.py`只操作与原始receipt、命令、UID、进程启动时刻一致的自身控制器/训练进程及其子进程；保留原始断点文件，分别从global update2825/3975的model/Adam/CPU-CUDA RNG和游标恢复，重放11/16个尚未保存更新。raw_huber原PID不变。真实GPU UUID、续跑更新、Adam step、参数有限性和训练绑定审计通过；短时吞吐约49.45→93.28及46.78→91.69窗/秒，后续仍受共享负载影响。证据为`results/shared_GPU_migration_20261111.json`与`results/shared_GPU_migration_progress_20261111.json`；微批16、global256、数据、loss、norm和max10/patience3均不变。

02:20，GPU3也通过三次新采样，将raw_huber从GPU4迁至GPU3，从global update3925的完整断点续跑，重放15个未保存更新；另两路不中断。`analysis/audit_gpu_migration.py`验证实际GPU UUID、首个恢复更新3926、后续已保存更新3950、全部Adam step与global update一致、参数和loss/梯度有限、数据和训练源码绑定不变，短时吞吐53.48→94.86窗/秒。证据为`results/raw_huber_GPU_migration_20261111.json`及`results/raw_huber_GPU_migration_progress_20261111.json`。三路当前分占GPU1/2/3，allocator上限32GiB，仍与既有任务共卡。

03:27前，两路Huber均完成第2轮全部983条Val并进入第3轮：calibrated_huber的trial-equal目标loss为6.704771246797424（第1轮7.375427957466372），raw_huber为10.842374263981393（第1轮11.25892579844423）。两路第2轮best均为global update5320、bad_epochs=0。`analysis/audit_selected_validation.py`验证真实第2轮候选与新best的model/Adam/CPU-CUDA RNG逐项相等、Adam step与global update一致、参数有限、实际数据/配置/源码绑定不变，以及完整Val记录和计分点数一致。逐记录结果与审计在`results/validation/*_epoch002_val.json`和`results/*_epoch002_Val_and_state_audit.json`；仍未读取最终Test指标。

迁移后约93–96窗/秒，单轮约2小时。两路第2轮仍刷新best，依patience3至少还需3轮才可能早停；按03:27的实测吞吐，08:00前完成全部训练及最终评测已不现实。保留既定训练政策并继续完成；资源有实质改善时再更新估计。

04:15，raw_original也完成第2轮全部983条Val，原始目标loss为1.2639655736797213（第1轮1.315550807049214），保存global update5320的新best、bad_epochs=0并进入第3轮。`results/raw_original_epoch002_Val_and_state_audit.json`验证真实轮末候选与新best的model/Adam/CPU-CUDA RNG逐项相等，Adam step与global update一致、参数有限、完整Val记录和计分点数一致、实际数据/配置/源码绑定不变。逐记录指标在`results/validation/raw_original_epoch002_val.json`。三套第2轮best均已验收；最终Test仍待正常训练结束，不以这些不同目标的Val loss推断标定收益。

更大显存余量下准备共同计算调度：三路都在完成global update7980（第3轮末）后，首次于optimizer update7981改用64窗物理前向，仍分成每16窗计算原目标并累积到256窗。`analysis/bench_forward_grouping.py`使用独立Train探针，在共享GPU0上测得约67.18→97.58窗/秒、峰值27.36GiB；无dropout的BF16分组比较中预测最大差约0.0201N、梯度相对L2差约0.154%。合批改变具体dropout掩码及随机数推进，因此按三路共同更新点切换，不按各自墙钟时刻即时切换。架构、dropout率、loss组、数据、norm、优化器、LR和预算不变，Val/Test计算方式不变。独立候选runtime的三种真实Train GPU smoke均通过（峰值27.19GiB），CPU目标/掩码/来源检查及边界7980/7981检查通过；未读Test指标。正式上线与完整保存前缀证明以`forward_schedule_migration`审计为准。

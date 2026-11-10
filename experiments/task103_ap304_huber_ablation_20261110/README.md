# Task103 AP304 四组实验

本目录是独立的训练实验，不是固定排序模型后拟合输出曲线的旧实验。正式结果要等三个训练分支的完整 Val 选模和最终 Test 完成后生成。

## 权重、模型和损失

- 一阶段来源为 `.21:/data/zzy_data/emg2force_k2_stage_training_ap304_ex45/runs/ap304_full_raw_rank_seed43_20261027/checkpoints/best_tau.pt`，global update 67982，完整 Val stage-macro τ-b=0.4342530492。
- 原始 checkpoint SHA256：`85445c11c7b20764ddb04bbf7e4baedfb6ff269ac0c3ff608b84e6e7be15d22a`。导出 state_dict SHA256：`f6dfe88f7e25d5be934ee0b065ae76d94aa89edc88340164244ecd149fa89928`。
- width304、9 encoder、4 decoder、18,357,945 参数，原生 `[B,5,89]` 输出。保留 243 个非输出头 tensor，seed43 重置输出头，gate bias=4；所有参数可训练；不恢复一期优化器。
- 两种 Huber 训练与原 loss 训练都加载仓库提供的同一份 `common_stage2_initial_state_dict.pt`。不从 ZIP 的二阶段成品继续训练。
- 纯 Huber 在**输出反归一化后的 N 域**计算，δ=1 N，对有效标量点平均。原 loss 为 `(1−Corr)+0.3×MAPE+0.03×MAE`；MAPE 只取 `abs(GT)>=0.5 N`，在 loss 内是比例值。
- EMG 沿用一期 norm（SHA `7ee5781e253ca45c2f31176e11ec5f065fd529df8afd9dc80bde5e3fec9d2de6`）。两种 force 标签分别在 task103 Train 的共同有效原生点上计算 mean/std；raw-Huber 与 raw-original 使用同一份统计。
- AdamW，主体2e-5、新头1e-4、weight_decay1e-4，bias/norm不衰减；warmup1000 updates，10 epoch跨度 cosine 到0.1倍；BF16、FP32 loss、clip1，global batch256。max10 epochs，Val patience3、min_delta1e-4。
- 原 ZIP 的 pure-Huber 是固定1 epoch的旧数据 loss 对照。本实验沿用主二阶段 max10/patience3政策，三套训练预算一致，并非声称复现其同一训练时长。

## 数据范围与划分

`manifests/sources.json` 合并7份现有冻结元数据，按对象key去重；不是实时对象存储的新枚举。2026-10-19至11-10：10,576条、109个subject、约352小时。普通stage 8,163条、`mix-test-5` 1,977条、`mix-test-7` 436条。另29个元数据解析异常对象单独列出，未伪装成可用数据。

按subject/date/PC/boot分层，以seed43稳定哈希分配test5完整recording：val988、test989；test7全部归test。普通stage归train。raw_huber、raw_original、calibrated_huber共用同一套记录和有效点，避免把更多训练数据误计成未标定方案优势。此设定评测同人的新动作/标定stage，不是task103跨人泛化。

标定源取同人、同日、同PC/boot内，目标开始前已完成的最近test5，不取未来源、不跨人、不静默跳过坏源回退。PC/boot只是佩戴/硬件连续性代理；无法排除同一boot内未记录的换膜、重戴。

仅元数据阶段，普通stage中5,691条有候选前次源、2,439条无同域前次源、33条时钟间隔不一致。候选尚需信号QC。加上全部test5和有前次源的test7，共8,089条进入候选准备；最终可用数由运行时QC明确报告。

### 标定与计分隔离

每条test5每指**首次完整、可明确路由的按压**作为现场标定角色，静息/按压点用于拟合；该按压及全部拟合点从val/test计分中排除。其余合格按压才是计分角色。即使首次标定失败，也不根据后续计分误差挑更好的一次。

目标test5本身也使用前一次完整test5的曲线来生成标定CMCU；不以同条拟合回放充当独立test。普通stage使用前次test5的标定角色；网络训练不读取任何test5的EMG窗口。这个设定允许现场采集的BSQ标定作为个体适配输入，独立计分按压的BSQ没有用于拟合映射。

标定采用逐指 median-per-AD + count-weighted PAVA，只在校准AD域内预测，不做夹端点或域外外推。校准角色需至少20个有效点、8个不同AD、AD跨度100、Spearman>=0.2、拟合力跨度>=1 N。上述规则固定，不按Test误差剔除。

缺少指、域外点、弱/歧义手指身份、时钟异常、BSQ异常和缺帧分别记录与屏蔽；不把缺缓存直接判成硬件故障。原有严重异常及待复核案例见 `manifests/cached_calibration_review_flags.csv`；新运行的完整依据为 `checks/calibration_quality.json`、`prepared/dataset.json` 中的exclusions。

### Test7与BSQ

BSQ是单通道参考，不能复制为十指真值。test7按活动手的四个非拇指路由，避免支撑拇指抢占真值；身份不明确不强配。357条已有稳定姿态零位沿用已记录锚点，余下记录要求可找到>=0.4秒低位稳定空载段，未通过则排除并记原因。已知正放例外保留其记录标识，不声称所有test7姿态相同。

第2/3/4组在相同受测指、同一50Hz计分时刻和共同有效掩码下比较。第1组只表示与标定CMCU标签的符合程度；真实力值收益看BSQ的第2/3/4组。报告test5/test7、逐人、逐指、力段与排除数；保留点精度不能代替全部数据的覆盖表现。

## 手动预处理与窗口

从原MCAP流式读取双手16通道uV EMG、CMCU AD/force和单通道BSQ。显式检查同步时间戳、逆序/重复、单位、有限值与流缺失；CMCU mN转N。whole MCAP与视频不落盘。

EMG重采样2000Hz，对每个连续有效段施加一次50Hz谐波陷波Q30和四阶40–850Hz带通，因果SOS、首1000点均值初始化。>5ms缺口屏蔽并重置滤波状态，段首0.5秒warmup不用于模型窗口。原生力标签不另加低通/MA7、不对负BSQ或预测截零、不按Test寻找最优时间平移。

4000点reset窗口、stride4000、完整末窗补充；输出坐标 `start+470+40*k,k=0..88`，不把89个预测插值到2000Hz制造重复loss点。标签线性对齐且不得跨掩码或>40ms缺口，val/test末窗按真实时间去重。训练每轮遍历全部保留下来的窗口，三个训练分支使用相同seed/顺序。

Huber按有效点数归一化。第4组Corr在每个microbatch的五指与时间点一起计算，再按有效点数梯度累积至256窗；这不等于对256窗计算一次全局Corr。自动调优只在正式训练前进行，三套训练冻结同一个microbatch。

## GPU共用及续跑

最多三个实验分支并行，各占一张可与已有任务共用的GPU。三次采样显存/利用率，保留8GiB外部余量及2GiB CUDA额外开销后，将剩余空间向下取4GiB档作为本进程allocator上限，最高32GiB。torch allocator限制和预留降低干扰，不能保证其他任务后续显存突增时绝无OOM。

用真实Train窗对8/16/32/64微批依次测试三种目标，保存各档时间和峰值显存。单卡最快为32（最慢目标约137窗/秒、峰值13.9GiB）；为在当前空余显存下同时安排三张卡，固定三套训练共用microbatch16（最慢目标约101窗/秒、峰值7.1GiB）。这优先缩短三个实验全部完成的时间；不是各自选择不同微批。测试更新为一次性检查，不改变正式初始模型。

保存完整model/Adam/RNG/epoch/update，默认每25更新last、每轮完整Val与best。恢复绑定配置、数据集和共同初始化哈希。按各分支完整Val的trial-equal目标选best，只在该分支训练完成后评一次最终Test。三套训练完成后严格核对预测文件的时间、指、GT和mask一致，再生成四组结果。

## 复现

```bash
git lfs pull
python -m pip install -r requirements.txt
export TASK103_S3_ENDPOINT='YOUR_AUTHORIZED_ENDPOINT'
export AWS_ACCESS_KEY_ID='YOUR_KEY'
export AWS_SECRET_ACCESS_KEY='YOUR_SECRET'
export TASK103_S3_BUCKET=tes-bucket
# .1新增文件全部放/pub_data/zjl；短TMPDIR防止Linux多进程Unix socket路径超长。
export TASK103_REQUIRE_PUB_DATA=1
export TMPDIR=/pub_data/zjl/.tmp_task103_huber_1110
mkdir -p "$TMPDIR"
ROOT=/pub_data/zjl/task103_ap304_huber_calibration_ablation_20261110
EXP=experiments/task103_ap304_huber_ablation_20261110
python "$EXP/scripts/bootstrap.py" --root "$ROOT"
python "$EXP/scripts/verify_protocol.py"
python "$EXP/scripts/pipeline.py" --root "$ROOT" --workers 8
```

也可用 `TASK103_CREDENTIAL_FILE` 指向权限允许的**库外**三行 `endpoint: ... / ak: ... / sk: ...`文件。密钥不放Git，不复制到本实验配置。

检查进度：`STATUS.json`、`checks/extraction_progress.json`、`checks/gpu_tuning.json`、`runs/<branch>/STATUS.json`。正式结果为 `results/FINAL_REPORT.json`、`four_experiments.csv`、`paired_person_differences.csv`及`REPORT.txt`；未出现完成凭据不能视为完成。

目标为北京时间2026-11-11 08:00前完成，属于尽力目标。保留既定max10/patience3、完整Val选模和最终Test，不按截止时间伪装完成或临时削减训练预算。进度预测要以全量有效窗口数及实际并行吞吐为准。

准备阶段默认保留100GiB磁盘余量。错误记录、质量掩码和运行缓存仅落运行目录，Git只保存可复现源码、冻结清单、源权重与紧凑检查结果。

# Task103 K2 5.56M 跨人四组实验

四组正式 Test 与核验已于北京时间 2026-11-12 01:53 完成。标定 Huber 跑满 5 轮并选第 3 轮，两套未标定模型按 patience=2 在第 4 轮结束并各选第 2 轮。全部 Val 使用同一批 293 条记录、724,829 个有效点。四组 Test 使用 12 位留出人员的 240 条记录、654,635 个共同有效点。

[最终指标表](results/force_threshold_metrics.md)给出人员等权 Corr、MAE、MAE <1 N 和两档 MAPE。[分 stage 明细](results/by_stage.csv)、[完整报告](results/report.html)、[核验记录](results/final_run_audit.json)和[推理权重](../../models/task103_k2_cross_subject_20261111)一并保存。

在这次 Test 划分上，标定 Huber 相比未标定 Huber 的 BSQ MAE 低 0.055 N，Corr 低 0.056，两档 MAPE 分别高 0.54 和 0.74 个百分点。原损失相比未标定 Huber 的 Corr 高 0.035，MAE 高 0.437 N，两档 MAPE 分别高 1.55 和 2.43 个百分点。

模型为历史训练代码 `k2_cross_subject_dg_v1/model.py` 中的 `k2_ring_transformer_affine_5p56m`。宽度 224、8 个注意力头、5 层 encoder、2 层 decoder，共 5,562,617 个参数。已逐类和逐函数对比网络 AST。历史模型源 SHA256 为 `6df60f9aba00223753997b9a81a5d2b613bae040b64571be3331a036d72b45db`，本目录的独立网络去掉了旧训练框架的包装。输入为每手 16 通道、2 kHz、4,000 点窗口，输出为五指各 89 个力点。

| 实验 | Train 和 Val 标签 | Test 标签 | 训练损失 |
|---|---|---|---|
| 1 | 标定 CMCU | 标定 CMCU | Huber δ=1 N |
| 2 | 标定 CMCU | BSQ | 与实验 1 共用模型 |
| 3 | 原始 CMCU force | BSQ | Huber δ=1 N |
| 4 | 原始 CMCU force | BSQ | (1−Corr)+0.3×MAPE+0.03×MAE |

按人员标识分组，在 test5/test7 可用性分层内固定 SHA256 排序，seed 43。三组人员互斥。原始 recording ID、可用对象标识和预处理 EMG 内容哈希也逐组核对。用于标定的 test5 与对应记录始终属于同一人，训练不会引用 Val 或 Test 人员的标定。划分见 [人员清单](results/subject_split.json) 和 [recording 清单](results/recording_split.csv)。

| 集合 | 人数 | 记录 | 窗口 | 来源 |
|---|---:|---:|---:|---|
| Train | 58 | 4,160 | 496,726 | 训练人员的普通 stage |
| Val | 12 | 293 | 10,753 | 验证人员的 test5 |
| Test | 12 | 240 | 9,554 | 测试人员的 188 条 test5 和 52 条 test7 |

复用前一轮 Task103 冻结的手动预处理、逐指质检、标定来源及有效窗，只变更人员划分。三种训练在相同的记录和掩码下进行。缺少可用同人标定的原始 CMCU 数据不纳入这次配对比较。留出人的 test5 BSQ 仍用于自己的离线标定及共同计分点筛选，test7 使用同人最近 test5。这里验证模型跨人泛化，评估口径保留离线标定辅助。

三套从 seed 42 生成的同一个随机初始化开始，不读取一期权重。EMG 与力归一化只统计 Train 人员，原始和标定力分别计算。按上一轮四组口径，以物理 N 域分支损失选最小完整 Val，MAPE 仅用 `|GT|≥0.5 N`。AdamW 峰值 LR 3e-4、权重衰减 1e-4、梯度裁剪 1、BF16，线性 warmup 后余弦衰减。每次更新 256 个窗，物理前向 64 个窗、损失按 16 窗计算并按有效标量点数加权。每轮将连续 16 窗组成块后按固定 seed 打乱，访问全部 Train 窗。初次跨人基线最多 5 轮，patience 2，完整 Val 选模后才开始正式 Test。

服务器产物目录为 `.1:/pub_data/zjl/task103_k2_cross_subject_20261111`。将本目录下的 `scripts/`、`analysis/`、`vendor/`、`config.json`、`launch_parallel.py` 放到运行目录的 `code/`。在运行目录下将 `cache/emg`、`cache/labels` 和 `checks/calibration_quality.json` 链接到原 Task103 缓存，再运行：

```sh
python -B code/launch_parallel.py --root "$K2_RUN_ROOT" --source "$AP304_RUN_ROOT"
```

启动器根据三次显存和利用率采样选共享 GPU，逐分支实际跑一批验证显存，再并行训练。训练状态见 `STATUS.json` 与各分支的 `runs/*/STATUS.json`，所有最终指标、同点核验、权重导出就绪后标记完成。

共享负载变化后，未标定 Huber 在第 1,500 步保存 checkpoint，随后从 GPU 6 迁到 GPU 2，并从第 1,501 步继续。迁移后短时实测约 285 窗／秒，迁移前约 137 窗／秒。启动器通过 PID 和命令识别并继续监控另外两路运行中的进程。证据见 [迁移核验](results/raw_huber_gpu_migration_audit.json)。

## 留出人员的普通 stage 补评

[普通 stage 指标](results/ordinary_stages/ordinary_stage_metrics.csv)包括总体和逐 stage 的 Corr、MAE、MAE <1 N、MAPE ≥0.5 N、MAPE ≥1 N。正式导出的三份权重在上一轮 Val、Test 人员各自的 16 个普通 stage 上推理，分别对原始和标定 CMCU 打分。Val 有 872 条 recording，Test 有 670 条。每个集合在三份权重间共享时间点、手指、目标和掩码，逐点哈希一致。统计先逐条 recording，再按人等权。补评沿用冻结的成对有效窗和原有训练归一化。

## 全 stage 跨人基线

[正式 Val、Test 汇总](results/all_stages/README.md)包含五项总体指标和逐 stage 表。

保留此前 82 人的划分，将剩余人员按 seed 43 排序后补进三个集合。划分以人为单位，每人的全部合格记录进入同一个集合。20 个 stage 均可参与 Train、Val、Test。全 stage 的共同参考是设备原始 CMCU force，单位为 N，测试数据也使用同一口径。

沿用实验 4 的 K2 结构、相同的 seed 42 初始化、EMG 预处理、训练超参及 `(1−Corr)+0.3×MAPE+0.03×MAE`。在新 Train 数据上重算归一化，在完整 Val 上选模型，最后在 Test 上算五项指标。缺少标定、BSQ 或标定超出范围的记录可以进入原始 CMCU 基线。要求双手 CMCU 时钟一致、EMG 窗口有效，参与打分的 CMCU 点有限且位于 0–100 N。

服务器运行目录为 `.1:/pub_data/zjl/task103_k2_all_stages_cross_subject_20261112`。将本目录的 `scripts/`、`analysis/`、`vendor/`、两个 launcher 放入运行目录的 `code/`，并将 `config_all_stages.json` 复制为 `code/config.json`。用 AP304 的 `scripts/fetch_missing_task103.py` 补取缺失的缓存后，运行以下命令。

```sh
python -B code/launch_all_stages.py \
  --root "$K2_ALL_STAGE_ROOT" --source "$AP304_RUN_ROOT" --prior-root "$K2_PRIOR_ROOT"
```

划分清单和排除理由保存在运行目录的 `prepared/`。启动器自动选择共享 GPU 并执行真实批次检查，评测完成后将指标写入 `results/all_stage_metrics.csv`，将推理权重和归一化参数写入 `exported_models/raw_original/`。

实际可用数据为 Train 75 人、7,659 条 recording、916,855 窗，Val 15 人、1,405 条、168,100 窗，Test 16 人、1,464 条、175,390 窗。三组均覆盖 20 个 stage。原始清单里的另外三人没有通过数据准备的 recording。

### 同数据的 AP304 二阶段

AP304 18.36M 从历史 `best_tau` 保留 243 项主干张量，并复用前一次 Task103 二阶段重置输出头后的初始化。两个全参数微调分支分别使用 Huber δ=1 N 和 `(1−Corr)+0.3×MAPE+0.03×MAE`。AP304 沿用一期 EMG 归一化，三套模型共用逐手 recording、原始 CMCU N、有效窗口和 Train 力值归一化。历史一期和本次数据的人员标识交集见 [一期人员核对](results/all_stages/ap304_stage1_subject_audit.json)。

AP304 每路使用两卡 DDP。物理前向 64 窗，按每 16 窗单独计算 loss，每次更新保留 256 个窗的有效点加权。末批空窗权重为零。学习率、warmup 和 10 轮／patience 3 沿用前次二阶段配置。完整 Val 按 recording 等权 loss 选模，最终报告按人与 recording 分层等权汇总五项指标。

服务器运行目录为 `.1:/pub_data/zjl/task103_ap304_stage2_all_stages_cross_subject_20261112`。部署共享 `scripts/` 和 `analysis/`、两个 AP304 所需 launcher，以及前次 AP304 的 `vendor/model_source.py` 和 `vendor/force_losses.py`。将 `config_ap304_all_stages.json` 复制为 `code/config.json` 后运行。

```sh
python -B code/launch_ap304_all_stages.py \
  --root "$AP304_ALL_STAGE_ROOT" --dataset-root "$K2_ALL_STAGE_ROOT" \
  --previous-ap304 "$AP304_RUN_ROOT"
```

最终汇总由 `analysis/collect_all_stage_results.py` 核验共享输入、选模凭据和逐 recording 计分点数后生成。

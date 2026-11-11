# Task103 K2 5.56M 跨人四组实验

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

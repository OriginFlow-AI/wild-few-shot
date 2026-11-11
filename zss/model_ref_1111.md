# Chord 3.6 ep005 模型参考与五指力试跑

Chord 3.6 ep005 从 16 通道、2 kHz EMG 预测 20 维手指角度和 2 维腕角。它与此前的 K2 五指力模型使用相近的 Ring encoder，主要差异在解码和训练。此分支提供正式权重的姿态推理，以及用冻结 encoder 接一个五指 Fz 线性头的示例。

- [准备脚本](../experiments/chord36_ep005/prepare.py)按固定提交取回源码和推理权重，验证发布清单和 SHA-256 后组装推理包。
- [试跑脚本](../experiments/chord36_ep005/run.py)用合成 EMG 验证 Regression 和 Tracking，也可以用自备的 Train／Val／Test 数据拟合并评估五指力头。
- 版本对应 Chord 3.6 的固定 scaling 快照 `ep005.pt`，该次 EMG 训练已完成六轮、27,960 次优化器更新。发布的基准结果使用上游原生评测配置。[固定版发布说明](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/README.md)

## 先跑通正式姿态模型

从仓库根目录运行。准备步骤需要 GitHub 上游仓库的读取权限，以及能访问交付 GitLab 的 SSH 环境。以下命令使用 Python 3.11。上游推理依赖为 NumPy 和 PyTorch 2.8 或以上。[交付文档](https://code.infra.originflowai.com/ai-infra/pulse-software-delivery/-/blob/eb66168b1adafa2ac15dab62306bdea8d0c61ab4/docs/chord36_ep005.md)

```bash
git clone --branch zss/model_ref_1111 https://github.com/OriginFlow-AI/wild-few-shot.git
cd wild-few-shot
python3.11 experiments/chord36_ep005/prepare.py
python3.11 -m venv .chord36-work/venv
source .chord36-work/venv/bin/activate
python -m pip install -r .chord36-work/source/baselines/chord36_ep005/requirements.txt
python experiments/chord36_ep005/run.py smoke
```

已有发布包时，在准备命令上加 `--asset-dir /path/to/Chord36_MuonWD005_ep005` 即可使用该目录的 `weights.pt` 和 `tokenizer.pt`。两份文件仍会按发布清单验哈希。原生训练 checkpoint `ep005.pt` 需经上游导出流程生成便携 `weights.pt`。默认工作目录 `.chord36-work/` 供本地源码、权重、环境和实验输出使用。

`smoke` 使用正式权重和一段合成 EMG，检查两种模式的输出形状、数值有效性和 Tracking 起点。运行自己的数据时可直接调用上游 NPZ 接口。

```bash
PYTHONPATH="$PWD/.chord36-work/source" \
  python -m baselines.chord36_ep005.infer \
  --model-dir .chord36-work/bundle \
  --input .chord36-work/pose_windows.npz \
  --output .chord36-work/pose_predictions.npz \
  --task regression --device cpu
```

| 字段 | 格式与含义 |
| --- | --- |
| `emg` | 浮点 `[B,11790,16]`，2 kHz、时间在前，与上游发布 EMG 的通道顺序、量纲和预处理一致 |
| `initial_pose` | 仅 Tracking 需要，浮点 `[B,22]`，弧度，对应每窗的输入索引 1790 |
| `anchor_valid` | 仅 Tracking 需要，布尔 `[B,22]`，手指 20 维必须有效，缺失的腕角可设为 `False` |
| `pose` | 输出 `[B,11790,22]`，弧度，手指 20 维之后是腕屈伸和桡偏代理角；前 1790 帧为上下文占位，只使用后 10000 帧 |

Tracking 将 `--task` 改为 `tracking`，并提供上述两个锚点字段。它每窗使用一次初始姿态。模型原生输出是 250 帧、50 Hz，上游接口将评分段插值回 2 kHz。[上游 NPZ 接口](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/infer.py)

## 架构与 K2 对照

```mermaid
flowchart LR
    emg[16 通道 EMG] --> patch[双尺度 Patch MLP]
    patch --> encoder[共享 Ring encoder]
    encoder --> views[3 组固定特征掩码]
    views --> decode[每路 K=2 池化与稀疏专家]
    decode --> coarse[第一阶段 RQ 粗姿态]
    coarse --> fine[历史粗姿态与增量作为第二阶段条件]
    fine --> average[3 路最终姿态平均]
```

| 部分 | K2 五指力基线 | Chord 3.6 ep005 |
| --- | --- | --- |
| 输入与 patch | 16 通道、2 kHz，感受野 40／128，步长 40 | 相同，原生时序频率均为 50 Hz |
| Encoder | 5 层、224 维、8 头，环形位置编码，空间 MHA 与因果时间衰减注意力 | 同类主干，原生 CUDA 路径使用 FLA 加速 |
| 池化与解码 | K=2，224 维单套解码器，两层因果注意力 | 3 套 128 维解码器，每路仍为 K=2，使用相对时间偏置与两阶段自条件解码 |
| 专家 | 无 | 每路在 3 个位置配置 32 个专家的库，按特征选择 Top-8 |
| 输出 | 直接预测门控递推参数，输出五指 Fz | 软 RQ 权重经冻结姿态模块输出 20 个手指角和 2 个腕角 |
| 原生参数量 | 5,562,617 | 16,129,132 |

三路解码共享同一次 encoder 计算。每路固定屏蔽 224 个隐藏特征中的 16 个，三个屏蔽集合互不重叠。第一阶段估计粗姿态，第二阶段接收上一帧粗姿态及其变化量。每个 RQ 读出包含五个手指组和一个腕部组，每组有两级码本，每级 128 个 8 维码向量。用 softmax 加权码向量后可生成连续角度。冻结的时序模块在当前观测与上一帧状态之间做门控递推，两阶段共享参数并维护各自的状态。[三视图源码](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/native/warm_source/src/emg2pose/emg2pose/triple_rq_view.py) · [Tokenizer 源码](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/native/wrist22_core.py)

K2 对照采用 `k2_ring_transformer_affine_5p56m`，即此前 C7 五指 Fz 实验的版本。它使用 AdamW、有效 batch 1024、峰值 LR `3e-4`、WD `1e-4`，训练目标为加权相关损失加 `0.3 × MAPE` 与 `0.03 × MAE`。这是架构和训练设置的对照，两个模型的原始成绩分别使用角度与力标签。

## 训练与评估

1. 从 Meta 姿态 tokenizer 初始化，在 Chord 3.6 Train 上适配 22 维，训练 30,000 步后冻结。随后 EMG 网络学习预测码本权重，沿用冻结的码本、姿态 decoder 和时序模块。
2. EMG 模型迁移发布包 ep019 的兼容权重并扩展腕部输出，优化器重新初始化。同一批 EMG 同时进入 Regression 和 Tracking，损失权重分别为 0.125 和 0.875。有效 batch 为 256 个窗口，8 卡、每卡 micro batch 16、累积 2 次。
3. 主要矩阵使用 Muon，指定参数如路由和 RQ 输出头使用 AdamW。基础学习率分别为 `1e-3` 和 `1e-4`，专家和路由组为 `1e-3`。可训练组使用 WD `0.05`。手指监督为角度 MAE、指尖位置误差和一／二阶差分辅助损失。训练姿态标签使用 10 Hz 低通，手指一阶差分目标额外使用 SG5，腕角按有效标签掩码评分。

训练数据含 437 个身份组、30 个 stage、1,192,866 个窗口，留出 stage 18／21／24／25。下表为手指 20 维 MAE，Tracking 的每窗起始锚点是额外输入。腕角使用单独指标。[训练与评估定义](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/config/baseline.json) · [指标明细](https://github.com/OriginFlow-AI/FM_PULSE/blob/f6116893d894d7ca133e57445886072b11c75d26/baselines/chord36_ep005/config/metrics_summary.json)

| Test 口径 | Regression | Tracking |
| --- | ---: | ---: |
| 普通 pooled，micro | 8.169° | 5.555° |
| 跨用户，用户与 stage 双平衡 | 7.250° | 4.897° |
| 跨 stage，用户与 stage 双平衡 | 8.510° | 5.756° |
| 同时跨用户和 stage，双平衡 | 9.379° | 6.073° |
| OOD pooled，micro | 12.364° | 8.113° |

## 在自己的五指 Fz 任务上试跑

线性探针先将共享 encoder 的 16 个电极特征取均值，得到每帧 224 维表示，再在 Train 上拟合五个岭回归头。它测量预训练 encoder 特征对力的可读出程度，使用自备 Fz 标签。每个数据集准备一个 NPZ，窗口内 EMG 和 Fz 须对齐在相同的 2 kHz 时间轴上。

| 字段 | 格式与含义 |
| --- | --- |
| `emg` | 同姿态输入，浮点 `[B,11790,16]` |
| `force_n` | 数值 `[B,11790,5]`，牛顿，三份数据使用同一指序与符号约定 |
| `subject_id` | 字符串 `[B]`，同一受试者在所有文件中使用相同 ID，Train、Val 和 Test 按受试者隔离 |
| `valid` | 可选布尔 `[B,11790,5]`，无效标签置 `False`；非有限的力标签也视为无效 |

自采数据先确认采样率、通道环向顺序、单位、去偏置和标签同步。Task103 的原始 MCAP 存在明显直流偏置，姿态和力示例均接收已处理的浮点 EMG。Task103 的 BSQ 与 CMCU 标签来源、去皮与独立评价口径可沿用对应实验文档。[Task103 实验说明](../codex/task103-cmcu-calibration-20261109.md)

先用 Train／Val 选择正则强度，指定新的输出目录。确定配置后可追加 `--test .chord36-work/data/test.npz`，同时换一个新的输出目录。

```bash
python experiments/chord36_ep005/run.py force-probe \
  --train .chord36-work/data/train.npz \
  --val .chord36-work/data/val.npz \
  --ridge 0.01 \
  --output .chord36-work/probe-val
```

输出包括 `head.npz`（`weight[224,5]`、`bias[5]`）、`metrics.json`、`val_predictions.npz` 及可选的 `test_predictions.npz`。预测为 `[B,250,5]`，与输入窗口顺序一致。`sample_offset` 记录相对窗口起点的 2 kHz 采样偏移 `1809.5 + 40k`，每帧标签取对应的两个中心采样点平均，两点都有效才评分。指标先在每人内按可评分手指等权平均，再对人员等权平均。MAPE 只在 `|Fz| > 0.5 N` 上计算，Pearson 仅使用至少两个有效点且预测、真值均有方差的手指。

若要和 K2 比较迁移收益，使用同一批窗口、Fz 真值、对齐方式和评分代码，给两套 encoder 接相同的线性读出。若要继续做 Task103 的 few-shot 标定，可将此探针作为全局力模型起点，再用每人的 test5 拟合标定参数，在独立 test7 评价。此分支的验证使用发布权重和合成输入，Task103 的实际迁移指标待跑同口径实验取得。

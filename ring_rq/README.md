# Ring-RQ 旧架构：训练、推理与本地数据读取

本目录整理原 Force3 标定版 **ex45 W304 / E9 / D4 Ring-RQ**，独立于正在开发的三视图架构。模型主体、损失、采样、读取和评估函数来自旧实验的实际执行源码；逐文件来源、原 SHA256 和整理改动见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。

本分支只发布代码。实际人员 split、逐条路径清单、录制时钟、标定表、norm、RQ 码本和训练权重需要通过项目授权渠道另行提供。这里的合成测试不代表已重跑真实数据训练或 Test。

## 代码导航

| 文件 | 内容 |
|---|---|
| `model.py`, `source/ring_model.py` | W304、9层 Encoder、4层 Decoder、2条 ring stream；最终 head 使用 Ring-RQ 软读出 |
| `source/rq_model.py` | 五指独立 tokenizer；2级残差码本，每级128项、latent8；训练后冻结 |
| `prepare_assets.py` | 有效区间索引、固定 Train 面板 norm、RQ 拟合；不会读取模型 Test |
| `train.py`, `controller.py` | 两阶段 Kendall/A-loss、7.5%梯度比校准、八卡、完整断点与恢复资格 |
| `losses.py`, `source/ordinal.py` | 力幅加权 Corr/MAPE、物理N MAE、成对排序代理损失 |
| `io_local.py`, `batches.py`, `fast_batches.py` | 三类本地数据、时钟插值、有序预读、分 stage/人员/手侧采样与增强 |
| `source/processed_mcap_fast.py` | 已 processed 的 Task43 MCAP 解码与 schema/单位/来源核验 |
| `label_policy.py`, `label_calibration/` | 源域显式路由；膜式旧ex45 LUT恰好一次、gen3原Fz恒等 |
| `dev_metrics.py`, `fast_metrics.py`, `val_pipeline.py` | 完整 Val、精确 Kendall tau-b、分力段/逐人/逐stage与原聚合 |
| `infer.py` | 旧模型离线推理 API/CLI；独立2秒窗口 reset |

## 模型与时间合同

- 输入：已按原处理链滤波的 EMG，16通道、2kHz、4000点；原尺度单位 uV，先用配套 norm 标准化。
- 原生输出：89个五指 normalized Force token。软max码字概率加权两级码本，再通过冻结逐指 MLP 读出。这里不是 Affine 输出头。
- 旧版监督/评估：`89 -> 3572` 线性插值，`align_corners=True`；对应输入索引428至3999。反向保留原 CPU ATen 有序插值梯度。
- 预测用配套 `force_std/mean` 还原为 N；不再对预测应用膜式 LUT，不截负值、不附加平滑。
- **此包使用旧版3572点口径**。三视图消融中两臂统一的55个时刻是另一份协议，本目录没有混入该数据流或三视图模型。

## 环境

训练使用 Linux、8张 CUDA GPU；控制器和训练入口使用 `fcntl` 与 NCCL。CPU 推理与合成单测也可在 Windows 运行。

```bash
python -m pip install -r ring_rq/requirements.txt
python -m unittest discover -s ring_rq/tests -v
```

原训练为 FP32、TF32关闭、确定性算法，不启用 AMP。`requirements.txt` 给出接口依赖范围，不保证不同 PyTorch/CUDA/GPU 版本逐位相同；严格复现应复用原执行环境，并重新通过八卡恢复资格。无需三视图的 FLA、稀疏专家或 tokenizer 依赖。

## 本地数据资产

`init_run.py` 接受单独的私有资产目录，将清单和小资产复制到**新建**运行目录。大缓存留在原位只读，路径/文件身份由清单约束；不会下载 S3、改写缓存、自动迁移路径或启动训练。

必须提供：

```text
private_assets/
  data/FULL_SPLIT.json.gz                  # revision=2, splits=train/val/test
  data/TASK43_LOCAL_INVENTORY.json.gz       # 原缓存身份与文件路径
  data/CLOCK_BINDINGS.json                 # rows[id]={path,sha256}
  data/LABEL_BINDINGS.json.gz              # rows[id]，源域/布局/单位
  data/...                                # CLOCK_BINDINGS引用的全部时钟小文件
  LABEL_POLICY.json                       # 包含 frozen_ex45_assets 校验绑定
  label_calibration/LABEL_POLICY.json
  label_calibration/calibration_assets/coefficients.json
  label_calibration/calibration_assets/tables.npz
```

可选配套资产：`data/INDEX.json.gz`、**成对**提供的 `data/norm.npz` 和 `data/RQ_FROZEN.pt`；使用 Task43 processed MCAP 时还需要原 `MCAP_FAST_QUALIFIED.json`。清单中每条记录至少包含 `id/subject/session/split/stage/component/source_uri/local_files/hands` 及相应身份与来源信息；细节见 [DATA_SCHEMA.md](DATA_SCHEMA.md)。不得用“文件里看起来是N”推断已经标定。

```bash
python ring_rq/init_run.py --assets /secure/private_assets --run /scratch/ringrq_run
cd /scratch/ringrq_run
# 如果尚未提供norm和RQ，先在已批准的本地清单上拟合；不覆盖已有资产。
python prepare_assets.py --build-index
# 已有配套norm/RQ与INDEX时跳过上一步。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python controller.py all
```

必须由数据负责人提供正确、获准使用的 split、interval、时钟和标定来源绑定。代码检查人员隔离、源文件身份、单位、shape与区间，但不代替上游数据质量筛选。缓存迁移后 inode/stat 已变化时应重新审计清单，不关闭身份检查。

`controller.py qualify` 只运行资格；`controller.py formal` 在资格通过后运行/接续正式两阶段。资格各阶段比较连续2步与跨进程1+1的模型、全部Adam、scheduler、8份RNG、游标、输入与局部梯度，另执行小型 Train→Val 检查。正式阶段二在真实一期 Val-best 上重新标定梯度系数，绝不复用资格父态的系数。活动进程修改前应正常停止并保存完整断点。

## 训练合同

- 每卡 logical micro256，global2048；各卡完整微批统计 loss，再按 rank0至7顺序求和梯度，统一裁剪/AdamW更新。
- 采样：stage均匀 → 人员均匀 → 手侧均匀 → 按合格整数起点数加权录制，再抽有效区间内起点；16窗块复用录制。属于有放回采样。
- 增强顺序：EMG norm → 每通道gain U(.8,1.25) → 5%概率置零一通道 → std .02高斯噪声。
- A-loss：`0.5*(1-Corr_w)+0.1*MAPE_ratio_w+0.15*MAE_N`。训练MAPE阈值和Val阈值不同，保留原实现；不将百分数直接代入训练公式。
- 一期：排序为主 `L_ord + epsilon*L_A`；历史六轮上限、min5/pat5，完整常规stage Val精确tau-b最大选模、负tau调度。
- 二期：整网一期best初始化，fresh Adam/scheduler，`L_A + lambda*L_ord`；max50/pat10，完整常规stage Val宏A最小选模。
- 7.5%指固定Train面板上“加权辅助项/主项”的全可训参数梯度L2范数比中位数；每阶段初始化校准一次，不是每批动态控制，也不是loss占比。
- 每16次更新及停止/epoch边界保存完整断点；last/previous/best、每个完整epoch独立权重、累计best与完整高低力分段/逐人/stage明细均保留。模型Test读取保持关闭。

详细公式和配置见 [PROTOCOL.json](PROTOCOL.json)。新的独立包有自己的代码SHA，**不接管或自动恢复服务器历史运行**；其训练资格必须在本包和目标环境中重做。

## 离线推理

```bash
python ring_rq/infer.py \
  --checkpoint /secure/best.pt \
  --norm /secure/norm.npz --rq /secure/RQ_FROZEN.pt \
  --emg /secure/filtered_windows.npy \
  --output /scratch/force_predictions.npz --device cuda:0 --batch-size 1
```

输入严格为 `float32 [B,4000,16]`，已经滤波、尚未标准化。输出 `native_normalized/native_N` 为 `[B,5,89]`，`dense_N` 为 `[B,5,3572]`，另带dense对应输入索引。API为 `RingRQInference(...).predict(emg)`。支持原完整断点和含`model`字段的独立权重；存在checkpoint norm/RQ绑定时会核验。只加载可信checkpoint。

这是固定窗口离线回放接口；不声称逐token在线状态接口、跨窗连续KV缓存或历史ex45 Affine生产接口等价。

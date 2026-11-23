# Ring-RQ 旧架构：训练、推理与本地数据读取

## Test 与基线对比：同人口、同 W100

固定权重为**标定 Ring-RQ：一期 EP6 → 二期 EP1**，由完整 Val A-loss 选出。基线为旧 **ex45 W304/E9/D4 Affine A-loss EP1**（权重SHA前缀 `e7110ae4`），不是尚无完整Val的新架构消融基线。

| Test范围 | 模型 | 参考A-loss† ↓ | MAE N ↓ | MAPE % ↓ | Corr ↑ | Kendall τ_b ↑ | R²宏平均 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Task43共同Test：1,395条 / 6人 | 旧 Affine A-loss EP1 | 0.20676 | 0.47048 | 46.82 | 0.82128 | 0.56109 | -1106.71 |
| Task43共同Test：1,395条 / 6人 | **本分支 Ring-RQ 二期EP1** | 0.22416 | 0.53775 | 47.64 | 0.80829 | 0.55721 | -2249.25 |
| Task83 Test：3,379条 / 33人 | 旧 Affine A-loss EP1 | 0.39471 | 0.39096 | 80.05 | 0.48797 | 0.29304 | -8809.85 |
| Task83 Test：3,379条 / 33人 | **本分支 Ring-RQ 二期EP1** | 0.34148 | 0.38079 | 65.82 | 0.56292 | 0.37214 | -18154.04 |
| 合计：4,774条 / 39人 | 旧 Affine A-loss EP1 | 0.36579 | 0.40319 | 74.94 | 0.53925 | 0.33428 | -7624.75 |
| 合计：4,774条 / 39人 | **本分支 Ring-RQ 二期EP1** | 0.32343 | 0.40493 | 63.02 | 0.60067 | 0.40061 | -15707.15 |

合计相对旧基线：Corr **+0.06142**、Kendall τ **+0.06633**、MAPE **降低11.916个百分点**；MAE **增加0.432%**，基本持平。Task43共同Test退步，Task83的MAE/MAPE/Corr/τ改善；R²宏平均受极端负值影响，原样保留。该模型没有在所有来源或指标上同时胜出。

†参考A-loss是将本表宏Corr、MAPE比例、MAE代入 `0.5*(1-Corr)+0.1*MAPE_ratio+0.15*MAE_N`，**不是训练history的production L**。聚合为录制逐指 → 同人/stage/手/指内录制等权 → 有效手侧×手指等权 → stage等权 → 人等权；MAPE仅在 |GT|≥0.5N 计算，各指标独立排除null，不做力幅加权。τ使用完整录制逐指tau-b。

Task43仅保留在旧权威split中同为Test的6人；Task83的33人在该旧索引中没有Train/Val登记。此曝光核验限于该冻结索引。两组人员无交集，合计中Task83占33/39人口权重。两模型训练人口及训练协议不同，这不是仅替换解码头的受控架构消融。

以上均复用已经完成的固定权重评估，不重新读取Test，不用Test选权重或调参。[完整精度、分母与证据SHA](EVALUATION.json)。

本目录整理原 Force3 标定版 **ex45 W304 / E9 / D4 Ring-RQ**，独立于正在开发的三视图架构。模型主体、损失、采样、读取和评估函数来自旧实验的实际执行源码；逐文件来源、原 SHA256 和整理改动见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。

本分支包含代码及已授权发布的最佳权重、配套norm和RQ码本。实际人员split、路径清单、录制时钟、标定表和波形数据不在仓库内；真实训练数据仍需单独授权提供。

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
| `infer_w100.py`, `streaming/` | 已发表Test所用W100连续状态、固定256 Encoder Graph和真实尾块 |
| `assets/phase2_ep1/` | 最佳权重与配套norm/RQ，以及逐文件SHA清单 |

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
  --checkpoint ring_rq/assets/phase2_ep1/best.pt \
  --norm ring_rq/assets/phase2_ep1/norm.npz --rq ring_rq/assets/phase2_ep1/RQ_FROZEN.pt \
  --emg /secure/filtered_windows.npy \
  --output /scratch/force_predictions.npz --device cuda:0 --batch-size 1
```

输入严格为 `float32 [B,4000,16]`，已经滤波、尚未标准化。输出 `native_normalized/native_N` 为 `[B,5,89]`，`dense_N` 为 `[B,5,3572]`，另带dense对应输入索引。API为 `RingRQInference(...).predict(emg)`。支持原完整断点和含`model`字段的独立权重；存在checkpoint norm/RQ绑定时会核验。只加载可信checkpoint。

上面是固定窗口离线回放接口，与下述W100连续推理分别保留。它不等价于历史ex45 Affine生产接口。

## 固定权重的完整 Val / Test 表现

这份权重已有独立的W100连续推理评估：Encoder递归状态连续，Decoder KV包含当前共100个token，绝对RoPE；每条录制或审计技术缺口reset，2秒评分块之间不reset。原生50Hz网格每个冻结2秒评分窗取89帧，位置为470+40j（j=0…88）。不使用MA7，不截负值。

下表沿原评估器：人内拼接，再按人等权；Corr/MAPE为五指力幅加权。**与开头基线对照表的逐指层级聚合不同，不可直接混合比较。**

| 范围 | 人数 / 录制数 | production A-loss ↓ | MAE N ↓ | MAPE % ↓ | Corr ↑ | R² ↑ | 窗口Kendall τ ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| W100 Val：常规24stage | 87 / 15154 | 0.25691 | 0.45680 | 53.99815 | 0.73122 | 0.02890 | 0.47711 |
| W100 Test：常规24stage | 64 / 7557 | 0.25584 | 0.43469 | 54.77619 | 0.72828 | 0.30487 | 0.46349 |
| W100 Test：全部29stage | 64 / 8613 | 0.26049 | 0.41874 | 55.54982 | 0.71573 | 0.33426 | 0.45713 |
| W100 Test：留出5stage | 30 / 1056 | 0.29699 | 0.37769 | 64.37732 | 0.64807 | -3.58111 | 0.37721 |

完整Val为87人/15,739条/929,029个评分窗；完整Test为64人/8,613条/505,633窗。留出Test的5个stage为mix-13/14/15/16/18，30人/1,056条；各范围人员可重叠，人数不能相加。完整结果的聚合与分母见[EVALUATION.json](EVALUATION.json)。

选择该权重时使用的历史Val是每2秒reset、3572点dense评分：A=0.2682736925、MAE=0.481413N、MAPE=53.7106%、Corr=0.715298、R²=0.047662。它不是本表W100 Test的同模式Val，不能直接相减解释泛化差距。

## 已发布权重与 W100 加载推理

三个文件均直接随Git分支提供，无需Git LFS。克隆分支即可获得；`best.pt`是约75.15MB的独立模型权重，不包含Adam/RNG，不是训练恢复断点。

```bash
git clone --branch syh/structure/ring-rq --single-branch https://github.com/OriginFlow-AI/wild-few-shot.git
cd wild-few-shot
python -m pip install -r ring_rq/requirements.txt
# 和Test相同的W100模式；输入为一条连续、已滤波但未norm的float32[T,16] EMG，单位uV。
python -m ring_rq.infer_w100 --emg /path/filtered_segment.npy --output /path/force_w100.npz --device cuda:0
```

```python
import numpy as np
from ring_rq.infer_w100 import RingRQW100Inference

model = RingRQW100Inference(device="cuda:0")  # 默认加载本分支配套三件套
emg = np.load("filtered_segment.npy", allow_pickle=False)
result = model.predict(emg)
force_N = result["force_N"]                 # [有效原生帧数,5]，拇指到小指
sample_indices = result["input_sample_indices"]  # 470+40*k，输入2kHz网格坐标
```

每次`predict`对应一条完整连续的录制/技术区间，调用之间reset；不要将连续录制拆成2秒重复调用。W100后端使用CUDA FP32、TF32关闭、固定256-token Encoder Graph、真实尾块和KV100；不需要FLA。输出所有有效原生帧，表中指标另按私有冻结评分清单选89点评分窗。原始录制有缺口时，应由上游按审计区间分开调用。预测仅反归一化，不应用标签LUT，不截负值。

若只需要CPU或独立2秒窗口，可使用前节`infer.py`或`from ring_rq.infer import RingRQInference`，它返回native89和dense3572，**不用于复现上面的W100成绩**。

| 文件 | 大小 | SHA256 |
|---|---:|---|
| [`best.pt`](assets/phase2_ep1/best.pt) | 75,150,907 bytes | `3f2fe00f22d05e328e3ae21da760a6c6686fdbfbefa6ada632da693c0daaa416` |
| [`norm.npz`](assets/phase2_ep1/norm.npz) | 1,186 bytes | `2b961d7a90b33e9e5d63561add0a22f67ffde229fd42c822f1a8fac1e667ad18` |
| [`RQ_FROZEN.pt`](assets/phase2_ep1/RQ_FROZEN.pt) | 127,108 bytes | `bea49219886588dd5592c77e97eb2d267d54b02baa45e05b9a85c1a4bf1b1ca6` |

权重、norm和RQ必须成套使用；加载器核验checkpoint的norm/RQ绑定以及冻结readout与码本的一致性。[资产清单](assets/phase2_ep1/MANIFEST.json)。

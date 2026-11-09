# AP304 全量 raw 排序模型使用包

本包固定导出**已完成首阶段**的完整 AP304 Encoder、Decoder 与 affine head，提供配套归一化、原始模型定义和可独立运行的推理程序。后续追加3轮仍在运行，本包不代表尚未结束的续训结果。

**输出是无量纲排序分数，不是N、不是经过标定的force，也不是force z-score。** 较高分数表示模型预测同一手指的相对力较大。不得套用ex45的force mean/std或正反标定曲线，不保证不同人、手指或session的分数尺度可直接比较。

## 直接开始

需要Python 3.10及以上、NumPy和兼容的PyTorch。打包时验证环境为Python3.10.21、NumPy1.26.4、PyTorch2.11.0+cu130；CPU推理无需CUDA。`requirements.txt`是依赖清单，不会自动选择适合你设备的CUDA安装包。

在解压目录中运行：

```bash
python self_check.py
python make_example.py --output synthetic_emg.npy
python predict.py --input synthetic_emg.npy --output synthetic_scores.npz --device cpu
```

示例输入由固定随机种子生成，不是采集数据，不带force真值，不用于判断精度。自检默认在CPU分别加载best/last并执行短合成推理、原生时间对齐和末窗去重检查；`python self_check.py --no-forward`只核验文件与严格加载。

使用自己的数据：

```python
import numpy as np
from ap304_rank import RankPredictor

# emg.npy: [T,16]，float32，2000Hz，T>=4000；一条独立trial。
emg = np.load('emg.npy', allow_pickle=False)
predictor = RankPredictor(variant='best', device='cpu')
result = predictor.predict_trial(emg, batch_size=4)
scores = result['scores']                # [N_out,5]，无量纲
indices = result['sample_indices']       # 对应原始EMG的零起始采样索引
time_s = result['time_seconds']          # indices / 2000
```

CUDA可用时，可用 `--device cuda:0 --precision bf16 --batch-size 8`。正式训练评估采用BF16；便于移植的默认CPU路径使用FP32。不同硬件、精度和批量可能产生浮点差异，不承诺跨设备逐位复现指标。

## 文件与模型选择

| 文件 | 用途 |
|---|---|
| `ap304_rank_best_model_state_dict.pt` | 默认模型：按完整Val tau-b选出的最佳权重，update67,982，第2轮50% |
| `ap304_rank_last_model_state_dict.pt` | 首阶段停止时最新权重，update113,303，第3轮50% |
| `train_norm.npz` | 与两份权重配对的完整Train EMG mean/std |
| `load_model.py`、`ap304_rank/` | 独立加载、归一化、2秒窗与整条trial推理接口 |
| `predict.py` | 输入EMG `.npy`，输出分数和时间轴 `.npz` |
| `MODEL_CONFIG.json` | 网络全部结构参数与输入输出契约 |
| `FORMULAS.md`、`LABEL_POLICY.json` | 排序损失、原始标签使用方式和输出尺度 |
| `calibration_parameters.json` | 明确记录本模型未做force校准，不含伪造的标定系数 |
| `training_config_reference.json`、`training_reference/` | 实际训练设置与冻结算法源码，供审阅或移植 |
| `WEIGHTS_PROVENANCE.json`、`SOURCE_PROVENANCE.json` | 权重/源码/归一化来源与哈希 |
| `MANIFEST.json`、`SHA256SUMS.txt` | 包内文件完整性校验 |
| `SELF_CHECK.json` | 打包时的CPU自检结果 |
| `EXPORT_PARITY.json` | 与原checkpoint及生产原生推理公式的导出一致性核验 |
| `ASK_CHATGPT.txt` | 可直接使用的辅助阅读提示 |

两份 `.pt` 都是245个张量的完整原始网络state_dict，参数量18,357,945；不是裸Encoder，也不包含优化器、RNG或原始训练断点。默认`best`用于固定模型推理；`last`可作为后续微调的起点。两者不能直接作为N值回归模型使用。

## 输入要求与预处理

1. 输入是一条独立trial的原始EMG，形状 `[T,16]`，2000Hz，保留训练时的16通道电极顺序、单位、增益和极性。不能仅凭shape相同就认为传感器协议一致；本包不自动重采样、换单位或重排通道。
2. 使用本包完整Train统计做 `(raw_emg-emg_mean)/emg_std`，以NumPy float32先减后除。统计来自188,777段Train、44,990,082,249个时间点，不使用Val/Test拟合。
3. 不做EMG滤波、镜像或数据增强。不要把已经标准化的EMG再交给`RankPredictor`重复归一化。
4. force不作为推理输入，不需要提供旧标签，也不应用任何force归一化或ex45标定。

低层接口：

```python
import numpy as np
import torch
from load_model import load_model
from ap304_rank import Normalization

net = load_model(variant='best', device='cpu')
raw_windows = np.zeros((1,4000,16), dtype=np.float32)  # 仅为形状演示
x = Normalization().normalize_emg(raw_windows)
x = torch.from_numpy(np.ascontiguousarray(x.transpose(0,2,1)))
with torch.inference_mode():
    z = net(x)  # [B,5,89]，已包含native_output/0.01，不要再除一次
```

## 时间轴与推理模式

本包实现生产排序训练及验证使用的**4000点、2秒独立重置窗**：常规起点0、4000、8000……；末尾不足一个新窗时，追加起点 `T-4000` 的最后完整窗。短于4000点直接报错，不补零。

每窗有11个上下文token，原生输出89个五指分数。第k个输出的raw标签/时间位置为 `start+470+40*k`，k=0..88；最后一个是 `start+3990`。参数中的left_context=428不等于原生首输出时间470。手指顺序固定为拇指、食指、中指、无名指、小指。

每个窗口重置encoder、decoder和affine状态，不共享KV，不追加历史1秒，不做MA7。它与旧AP304/ex45的W100连续流式模式、11790点离线模式不是同一个已验证接口；本包不把这些模式当作等价替换。

同一trial的输出按真实起点拼接。追加末窗只保留晚于已有最后时间点的输出，避免重叠计分。**每个窗内是50Hz，但整条拼接时间轴含上下文缺口、尾窗也可能改变间隔，因此必须使用返回的sample_indices/time_seconds，不能直接当作均匀50Hz数组。**

例如T=8010，窗口起点为0、4000、4010；拼接后179点，首索引470、末索引8000。本包不插值回2kHz、不填补缺口、不把缺口当零分数。若下游需要重采样，请保留原生预测，并明确记录自己的插值策略。

## 算法与后续微调

训练使用全部Train，每轮无放回遍历11,422,364个真实窗口；每轮重新打乱并在同trial内组成双窗组。只在同trial、同手指内比较力大小，同窗与跨窗排序权重分别0.8/0.2，另加0.001的分数L2正则。完整公式见 `FORMULAS.md`。

`load_model`返回可训练参数、默认eval模式的包装网络。其 `.network` 是原始AP304本体；导出的state_dict对应 `.network`，不是包装器新增`network.`前缀后的state_dict。需要新数据微调时可调用 `.train()` 并建立自己的优化器；如目标是准确N值，需要可靠新标签和明确的回归输出/标定设计，不能把分数直接重命名为force_N。

`training_reference/`保留原算法源码，但本包不含数据索引、原始EMG/force或优化器状态，不是开箱即用的服务器训练复现包。应在自有数据上重建索引、划分和路径，再移植训练逻辑；推理与CPU自检完全不依赖该目录或原服务器。

## 来源与边界

首阶段使用全量Train、随机初始化、全参数训练；最大3轮，实际约2.5轮时因连续4次Val检查未满足tau-b改善门槛而停止。最佳模型位于第2轮50%，而不是停止时权重。这个历史配置在本包如实记录；后续追加训练的patience10不追写到本阶段。

本包不包含追加训练中不断变化的checkpoint、原始录制、被试级结果、Test预测数组或访问凭据；分享重点是模型与算法使用。选模只依据Val，不用Test挑权重。无量纲分数上的MAPE不等于物理力误差；本包没有声称绝对N值准确或新数据微调一定获益。

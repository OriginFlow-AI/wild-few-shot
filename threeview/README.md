# 三视图集成：训练、推理和数据读取

本分支 `syh/structure/threeview` 整理当前 Force3 架构消融中实际运行的三视图候选，包括两阶段训练、上游 tokenizer 拟合、三路集成推理和本地数据读取。Ring-RQ 已在独立分支 [syh/structure/ring-rq](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/ring-rq) 发布；该分支的旧3572点评估、W100成绩和权重不能当作本架构的结果。

本分支提供两份三视图模型权重。训练恢复需在新运行目录完成资格验证。

## 模型权重

- `threeview-phase1-end-ep1.pt` 是一期 EP1 的 Val-best，也是二期初始化权重。
- `threeview-phase2-ep1.pt` 是二期完成 EP1 后的权重。

在 [GitHub Release](https://github.com/OriginFlow-AI/wild-few-shot/releases/tag/threeview-weights-20261124) 下载全部分卷、清单和合并脚本。脚本自动解压并验证两份 `.pt` 的 SHA256。

```bash
gh release download threeview-weights-20261124 --repo OriginFlow-AI/wild-few-shot --dir threeview-weights
python threeview-weights/assemble_weights.py
```

两份权重包含冻结 tokenizer，经逐张量核对与原快照一致。使用 `threeview/infer.py` 时，把 `--checkpoint` 指向合并后的权重，把 `--norm` 指向配套的原 `norm.npz`。模型与 norm 的 SHA256 见 [CHECKPOINTS.json](https://github.com/OriginFlow-AI/wild-few-shot/releases/download/threeview-weights-20261124/CHECKPOINTS.json)。

## 架构与时间支持

- 共享 Package Encoder：W304、E9、16路环形电极、多尺度因果patch、频域缩放、动态衰减时间注意力。
- 三个独立 W128/D2 读出分支：各自固定屏蔽24个Encoder特征维，三个屏蔽集合不重叠；分支参数独立。
- 每个分支包含双ring流、因果卷积、相对lag注意力与Top8/32稀疏专家。先由自身coarse预测形成因果历史，再输出fine；没有GT历史输入。
- 共享冻结 tokenizer：五指各2级128项、latent8码本，逐指LeakyReLU MLP与宽128的自主Motion解码器。frame和Motion仅用Train梯度训练、各按Val normalized MAE选择best后冻结；不是沿用旧Ring-RQ tokenizer。
- 总参数 **26,106,189**，可训练 **25,986,494**，冻结tokenizer **119,695**；可训练张量1,783个。
- 输入为已滤波、原尺度uV的EMG，16通道、2kHz，每个窗口4000点。每次独立窗口reset；输出55个50Hz时刻，对应原输入索引 `1829 + 40*k`，k=0…54。
- 最终预测为三路fine的算术平均。训练对三路分别求A-loss与排序loss后平均，不能改成对集成预测求训练loss。

## 代码导航

| 文件 | 内容 |
|---|---|
| `model.py`, `source_candidate/model.py`, `source_candidate/vendor/` | 完整三视图模型、固定mask、冻结tokenizer及实际稀疏专家kernels |
| `tokenizer_fit.py`, `source/rq_model.py` | frame EMA RQ与Motion上游训练、Val-best冻结与资产绑定 |
| `prepare_assets.py` | 已审计有效区间的索引、原固定Train面板norm；不再拟合本模型不用的旧Ring-RQ码本 |
| `train.py`, `controller.py`, `losses.py`, `source/ordinal.py` | 八卡两阶段、7.5%梯度比校准、完整断点、资格及选模 |
| `optimizations.py`, `streams.py`, `ordered_reducer.py` | 五指concat、双CUDA stream、有序rank0…7 FP32梯度求和 |
| `infer.py` | checkpoint+norm加载、成员与集成预测，API/CLI |
| `io_local.py`, `batches.py`, `fast_batches.py`, `source/processed_mcap_fast.py` | 旧processed NPY、processed Task43 MCAP、滤波Task83 NPZ读取、时钟、分段采样与增强 |
| `label_policy.py`, `label_calibration/` | 旧ex45膜式LUT恰好一次、gen3恒等；标定数值表单独提供 |
| `dev_metrics.py`, `fast_metrics.py`, `val_pipeline.py` | 完整Val、精确tau-b、逐人/stage、力段/分母/低力误报和累计best |

## 环境与快速自检

训练/加速推理需Linux CUDA；CPU/reference推理和数据单测可在Windows运行。

```bash
git clone --single-branch --branch syh/structure/threeview https://github.com/OriginFlow-AI/wild-few-shot.git
cd wild-few-shot
python -m pip install -r threeview/requirements.txt
python -m unittest discover -s threeview/tests -v
# CUDA环境：先安装适用于GPU的PyTorch，再安装实际使用的加速依赖。
python -m pip install -r threeview/requirements-cuda.txt
```

实际加速依赖为 `fla-core 0.5.2 / Triton 3.7.1 / einops 0.8.1`；FP32、TF32关闭、确定性算法、IEEE Triton。缺少CUDA依赖时明确失败，不静默降级训练后端。CPU/reference是同架构的诊断执行，不能承诺与加速后端或不同硬件逐位相同。执行环境及验证范围见 [VALIDATION.json](VALIDATION.json)。

## 私有资产和数据

先由数据负责人提供批准使用、人员隔离且符合五规则的split和缓存身份；本包不从S3下载、不筛选或删除源数据。清单与单位见 [DATA_SCHEMA.md](DATA_SCHEMA.md)。

```text
private_assets/
  data/FULL_SPLIT.json.gz                 # revision2，含train/val/test元数据
  data/TASK43_LOCAL_INVENTORY.json.gz
  data/CLOCK_BINDINGS.json                # 以及引用的全部data/时钟小文件
  data/LABEL_BINDINGS.json.gz
  LABEL_POLICY.json
  label_calibration/LABEL_POLICY.json
  label_calibration/calibration_assets/coefficients.json
  label_calibration/calibration_assets/tables.npz
  # 可选：data/INDEX.json.gz、data/norm.npz、MCAP_FAST_QUALIFIED.json
  # 若复用已拟合tokenizer，必须配套norm并成对提供：
  tokenizer/TOKENIZER_FROZEN.pt
  tokenizer/TOKENIZER_FROZEN.json
```

```bash
python threeview/init_run.py --assets /secure/private_assets --run /scratch/threeview_run
cd /scratch/threeview_run
# 没有INDEX时生成；不会覆盖norm。
python prepare_assets.py --index-only
# 仅当没有配套norm时，拟合原Train-only面板norm。
python prepare_assets.py
# 仅当没有配套冻结tokenizer时执行，max50/pat10分别训练frame和Motion。
CUDA_VISIBLE_DEVICES=0 python tokenizer_fit.py
# 准备就绪后，显式启动八卡资格和两阶段正式训练。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python controller.py all
```

已有配套资产时跳过相应拟合命令。初始化拒绝覆盖运行目录，只复制代码和明确列出的资产，不复制任意私有文件；TB级缓存保持原位只读。源文件dev/inode/大小/mtime与清单不匹配即拒绝；数据迁移后应重新审计绑定，不关闭检查。

Task83 NPZ中EMG和Force仍按独立原生时钟保存。读取先在原生Force网格做膜式LUT/或gen3恒等，再插值到2kHz目标网格，最后选55点；禁止外推、跨审计缺口拼接或再次滤波。旧Task43缓存已processed，不把它当raw再滤波。已标定缓存必须带准确域和策略SHA，避免重复标定。

`LocalReader`与批读取器拒绝模型Test信号；Test清单元数据只用于人员隔离。API示例在**新运行目录**执行：

```python
from io_local import LocalReader, closed_rows
reader = LocalReader()
row = closed_rows()["train"][0]
start = reader.segments(row)[0][0]
emg_uV, calibrated_force_N = reader.window(row, start)  # [4000,16], [4000,5]
```

## 训练合同

- logical micro256/卡，global2048；物理16窗checkpoint/双stream后拼接所有预测，完整256窗计算统计，不能平均小块loss。
- seed42，有放回：stage→人员→手侧均匀、按合法整数起点数加权录制与区间；16窗块复用录制。norm之后gain U(.8,1.25)、5%概率置零一通道、std.02高斯噪声。
- `A = .5*(1-Corr_w) + .1*MAPE_ratio_w + .15*MAE_N`，保留原训练/Val阈值和五指力幅加权，MAPE比例不当作百分数代入。
- 一期固定6轮：`L_ord + epsilon*A`，完整常规Val A最小选交接best；调度仍按负精确tau-b。
- 二期从真实一期Val-A-best初始化整网，fresh Adam/scheduler：`A + lambda*L_ord`，max50/pat10，完整Val A最小选模/早停。
- 每阶段在固定Train面板重新校准：辅助项/主项全可训参数梯度L2比中位数7.5%。它不是loss占比，也不是每步动态控制；二期不沿用资格父态的系数。
- 各GPU梯度按原rank0至7顺序FP32求和，统一clip1/AdamW；没有使用不同归约顺序的all_reduce替代梯度汇总。
- 每16更新及安全停止/epoch边界存完整model、Adam、scheduler、八卡RNG、游标、history和best；每完整epoch另留独立权重。
- 完整Val维持所有固定窗口，按常规Train stage选模，其他Val stage另列诊断；保存L/MAE/MAPE/Corr/R²/tau、力段、人员/stage、有效分母、bias、低力误报及逐轮累计best。

`controller.py qualify --phase 1` 可只做一期资格；`controller.py formal` 在一期资格完成后启动。阶段二资格必须等待真实一期best。资格严格比较连续2更新与跨进程1+1的完整状态、输入和局部梯度；正式phase2系数/资格在真实父态上执行。配置见 [PROTOCOL.json](PROTOCOL.json)。

## 推理

```bash
python -m threeview.infer \
  --checkpoint /secure/threeview_best.pt --norm /secure/matching_norm.npz \
  --emg /secure/filtered_windows.npy --output /scratch/force55.npz \
  --device cuda:0 --batch-size 16
```

```python
import numpy as np
from threeview.infer import ThreeViewInference
runtime = ThreeViewInference("/secure/best.pt", "/secure/norm.npz", "cuda:0")
result = runtime.predict(np.load("/secure/filtered_windows.npy", allow_pickle=False))
force_N = result["ensemble_N"]                 # [B,55,5] 拇指至小指
members_N = result["members_N"]                # [3,B,55,5]
sample_indices = result["input_sample_indices"] # 1829+40*k
```

输入严格为finite float32 `[B,4000,16]`、已滤波、尚未norm的uV。checkpoint可为原三视图完整断点或独立epoch权重，须包含`model`和匹配`binding.norm_sha256/arm=candidate`，冻结tokenizer已嵌入模型。加载时核norm、固定mask、时间网格及完整state_dict；不另需外部RQ文件。只加载可信checkpoint。

输出只反归一化为N，不对预测再做标签LUT，不截负值、不附加平滑或dense插值。每个2秒窗口独立reset，没有生产流式状态接口；不能拿此接口声称复现Ring-RQ的W100连续推理。CPU诊断可指定 `--device cpu --backend reference`。

## 发布与验证范围

来源、逐文件SHA和整理差异见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)。保留相同参数名/状态布局和数值函数，主要整理为独立路径、单候选控制器、显式资产复制及公开推理API；移除仅用于两臂绑定、并不参与三视图计算的旧RQ文件依赖。

当前服务器训练未停止或热改。本发布包不带历史完整训练状态，新的训练绑定有独立代码SHA，不能直接恢复历史服务器optimizer状态；须新建运行目录重新资格。代码验证使用合成输入及只读历史权重，未打开真实Test。GPU验证是有限单卡代码可移植性检查，不能替代目标环境正式训练前的八卡恢复资格。没有在发布过程中重新运行整轮Val或Test。

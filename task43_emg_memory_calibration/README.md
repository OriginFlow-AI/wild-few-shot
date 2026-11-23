# Task43 EMG标定记忆：A / LoRA / F

本目录提供三臂的训练、数据读取、完整断点恢复、Val统计和推理代码。基座为 **A-loss A_ep1、W304/E9/D4**；解码采用Affine递推，`K2RingNetwork`是源码中的历史类名。

|臂|个人标定输入|训练范围|适配参数|
|---|---|---|---:|
|A|无|固定基座，只评估和推理|0|
|B_L|无|19处rank16 LoRA **及完整Affine输出头**；Encoder冻结|354783|
|F|本人近期Stage3前60秒|基座全部冻结，只训PrefixMemory|450101|

F读取正确的本人标定，B_L不读取个人标定；两者共享查询数据、逐轮抽样和更新预算。A的上游预训练/回归不在此适配包中，可参照仓库的[上游A模型分支](https://github.com/OriginFlow-AI/wild-few-shot/tree/codex/ap304-original-a-20261118)。

## 目录与外部资产

- `runtime/`：模型、记忆、LoRA、原训练器/loss/sampler、数据集、mmap/S3读取、静态缓存、原生与CUDA Graph推理。
- `tools/experiment.py`：准备、物化、标签/支持库、共享采样、恢复资格、训练、Val和预测入口。
- `configs/`：网络、LoRA范围、训练配置及路径绑定模板。
- `tests/`：纯合成CPU测试，以及显式启用的GPU集成测试。
- `PROVENANCE.json`：原实验源码SHA、历史checkpoint标识及外部资产SHA。
- `VALIDATION.md`：本次代码发布的实际验证范围。

**本仓库不含权重、归一化数值、标定LUT、真实清单/个人信息、波形、预测缓存或凭据。** 运行前需向资产所有者取得下列匹配文件；本分支不能单凭代码重现历史成绩。

```text
external/
  A_ep1.pt                         # 原始base state_dict
  train_norm.npz                   # A_ep1对应Train-only norm
  calibration/
    LABEL_POLICY.json
    calibration_assets/coefficients.json
    calibration_assets/tables.npz
  frozen_train_val_PLAN.json
  F_val_best.pt                    # 仅历史F推理需要
  B_L_val_best.pt                  # 仅历史LoRA推理需要
```

准备工具会复制小资产到新运行目录，并核对固定base/norm/LUT的SHA。凭据保持外置，大波形保持原位只读。历史推理绑定为F的Val-best EP8/6016更新和B_L的Val-best EP1/752更新，具体文件SHA见`PROVENANCE.json`。没有把最近覆盖选窗实验的EP4权重混入此包。

## 协议

- 原生输入为FP32、2kHz、16通道EMG；每窗4000点/2秒，输出5指×89帧、50Hz物理N。原生监督位置为`470 + 40*k`，`k=0..88`。通道顺序为拇/食/中/环/小指。
- F支持来自同一原始受试者名、同手侧、同日、查询前已结束的最近Stage3；起点间隔≤7200秒，支持≥60秒。最近支持不合格时不向更早支持回退。没有可靠的重新佩戴字段，连续佩戴仍是使用假设。
- 此处Stage3指旧Task43/benchmark的权威Stage3（`wzswl-3`），不按文件夹数字或Task103的stage名称猜测。
- 支持前60秒建立每指128条记忆。每条包含304维融合特征、16维EMG RMS、256点/128ms力历史及幅值摘要。训练时K/V随记忆参数更新；只有冻结特征/力历史可以跨epoch缓存。
- F修正Affine命令后再做递推，不是强行把预测替换成最近邻力值。无合格支持时，F严格回退A。
- CMCU API成员为原始mN，先转N，再按冻结LUT标定一次。本地`.fz.npy`必须已是**原始N**，尚未经过LUT；模型输出已经是目标N域，不再套LUT。EMG按配套norm处理。
- Stage3只作支持；Train排除Stage4/5，Val沿冻结人口保留并另报排4/5。训练只接收Train/Val，不自动打开Test。
- 查询采样75%均匀、12.5%高力、12.5%静息/释放，先按人/stage/录制分层；两训练臂使用同一批预先冻结的采样计划。
- 原预算752更新/轮、最多24轮、logical256×4/global1024；FP32/TF32off，AdamW、weight decay1e-4、梯度裁剪1。F/LoRA峰值LR3e-4，LoRA完整输出头3e-5；首轮半LR。按完整Val `L_A`最小选best、调度和patience10早停。
- 报告口径为`L_A = .5*(1-Corr) + .1*MAPE比例 + .15*MAE_N`。同时保存MAE/MAPE/Corr/R²、逐轮累计best，以及≤0.5、(0.5,2)、[2,5)、≥5、≥10N分段、偏差/误报/分母和逐人逐stage明细。低力误报为参考绝对力≤0.5N时预测>1N。

## 安装与数据清单

参考训练环境是Linux、Python3.12、Torch2.12.1+cu130、NumPy1.26.4、RTX5090；复用兼容环境即可。训练器使用`fcntl`、`/proc`和`nvidia-smi`，不承诺Windows训练。原生CPU推理和数据读取可单独使用。

```bash
pip install -r task43_emg_memory_calibration/requirements.txt
python -m unittest discover -s task43_emg_memory_calibration/tests -v
```

数据所有者需提供已审计冻结的Benchmark2.0清单，schema为`stage3-prefix60-crossstage-v1`。如果原清单还含Test角色，先仅导出Train/Val元数据；不读取任何Test信号、不重划分或移动人员：

```bash
python task43_emg_memory_calibration/tools/experiment.py export-plan \
  --input /private/original_PLAN.json --output /private/train_val_PLAN.json
```

导出后会得到新文件SHA，但Train/Val查询顺序、支持映射和数据人口不变。配置中的`plan_sha256`使用这个实际文件的SHA。清单字段见[数据接口](docs/DATA.md)；最近支持选择依赖上游完整元数据，校验一个已选子集无法替代完整源清单的准入审计。

## 准备和训练

复制`configs/bindings.example.json`到仓库外的私有配置，填写资产、清单、本地只读cache和可选S3配置。运行目录必须在Git checkout外：

```bash
CLI=task43_emg_memory_calibration/tools/experiment.py
RUN=/data/my_experiments/task43_emg_memory_run
python "$CLI" prepare --config /private/bindings.json --workdir "$RUN" --gpu GPU-YOUR-UUID
# 可选：只物化EMG/Fz，不读视频；新增缓存只写RUN/waveform_cache。
python "$CLI" materialize --workdir "$RUN" --roles train val --fields emg fz
python "$CLI" targets --workdir "$RUN"
python "$CLI" banks --workdir "$RUN"
python "$CLI" plans --workdir "$RUN"
python "$CLI" qualify --workdir "$RUN"
python "$CLI" train --workdir "$RUN" --arm F --run formal
python "$CLI" train --workdir "$RUN" --arm B_L --run formal
```

`plans`只生成一次24轮共同采样，避免两臂并发生成同名计划。上面两条train命令可在独立终端运行；要分卡并行，在**资格验证之前**分别配置`CONFIG_F.json`、`CONFIG_B_L.json`中的GPU UUID。每个进程绑定一张物理卡，动态KV不跨卡共享。64路预读不是固定最优，应按整机CPU/RAM/网络和其他任务负载取舍；改变配置后重新资格，不热改已有进程。

`qualify`在真实global1024的smoke批次上比较连续2步与跨进程1+1步，核对完整模型、Adam、RNG、采样游标、scheduler、history/best及Train→Val。资格时仅将墙钟统计置0便于完整状态对拍，正式训练使用真实时钟。smoke取清单前两条查询，不是完整Val；资格更新不计入正式预算。

断点恢复保留原配置与数据绑定：

```bash
python "$CLI" train --workdir "$RUN" --arm F --run formal --resume
```

每次保存含`last.pt`、旋转`previous.pt`、Adam、Python/NumPy/CPU/CUDA RNG、epoch/游标/计划、scheduler及累计best。每完整epoch保存history/力段，并保留`best.pt`。`STOP_REQUEST`放在该臂运行目录可在更新边界完整保存退出；工具不会自动移除人工停止标记。历史服务器checkpoint可用于推理；跨目录完整续训涉及配置/路径绑定迁移，不能删除严格config校验来强行恢复。

## 推理

先在仓库外bindings中填写可选`checkpoints`，再执行prepare；或者在已准备好的`INFERENCE.json`登记自己的Val-best路径及SHA：

```json
{"checkpoints":{"F":{"path":"checkpoints/F_val_best.pt","sha256":"<file-sha256>","epoch":8},"B_L":{"path":"checkpoints/B_L_val_best.pt","sha256":"<file-sha256>","epoch":1}}}
```

只推理A不需要F/LoRA checkpoint，A使用匹配的base和norm。输入`.npy`为`float32 [windows,4000,16]`，输出`.npz`的`A/B_L/F`为`[windows,5,89]`。不静默拼接波形或补真实序列尾帧；需要按原窗口规则显式准备完整2秒窗。

若使用本次新训练的best，显式登记；工具核对正式Val-best的epoch/step和数据清单，并用独立文件名保存，不覆盖旧checkpoint：

```bash
python "$CLI" use-best --workdir "$RUN" --arm F --run formal
python "$CLI" use-best --workdir "$RUN" --arm B_L --run formal
```

```bash
python "$CLI" predict --workdir "$RUN" --arms A --device cpu \
  --input /private/query_windows.npy --output /private/A_prediction.npz
export CUDA_VISIBLE_DEVICES=GPU-YOUR-UUID
python "$CLI" predict --workdir "$RUN" --device cuda:0 --backend cuda-graph --batch 64 \
  --input /private/query_windows.npy --query-metadata /private/query_record.json \
  --support-catalog /private/stage3_records.json --output /private/three_arm_prediction.npz
```

元数据入口自动执行最近已完成Stage3准入；无支持时F逐位等于A。也可给`--support /private/support_record.json`使用调用方已明确绑定的支持，或做显式错配对照；该模式会标记为`explicit_caller_binding`，不是重新证明了时间/人员准入。

加速版沿已验证实现：**原生Torch FP32、固定64窗CUDA Graph、A/F/LoRA共享冻结Encoder、LoRA合并、冻结推理KV缓存**。尾批只在独立窗口的batch维补dummy并裁回真实数量。默认不启用FLA；本次三臂主结果未采用FLA候选。独立2秒窗口每次reset，**不是W100连续历史或逐帧20ms服务接口**。原样保留signed N，不加ReLU、MA7或额外插值。

完整Val：

```bash
python "$CLI" evaluate --workdir "$RUN" --arm A --output /private/A_val.json
python "$CLI" evaluate --workdir "$RUN" --arm F --output /private/F_val.json
```

单模型Val沿冻结完整Val集合，使用原统计实现；不得用smoke成绩冒充完整Dev，也不根据Test选择best。A/F无标定回退及Graph重绑定检查见[验证记录](VALIDATION.md)。

## 代码来源与范围

模型、记忆、LoRA、训练器、loss、sampler和Val统计取自原冻结实验。发布修改集中在路径/外部资产绑定、数据读取配置、按清单计算静态库人口，以及推理loader；网络与训练数值逻辑保留。没有上传旧的一次性SSH迁移/启动/修复调度器，没有随代码启动正式训练。代码发布不改变已停止实验的状态。

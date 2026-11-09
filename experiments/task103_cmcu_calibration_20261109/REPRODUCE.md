# 数据获取与复现

## 保存了什么

- `report.html`：单文件详细报告，图像已嵌入，无 CDN/网络依赖；下载后浏览器直接打开。GitHub 默认会展示 HTML 源码。
- `results/`：分日期/方案、逐人、逐 trial×指指标，bootstrap 差值区间、配对清单、路由/去皮审计及完整两层标定映射。大 CSV 使用 gzip，`pandas.read_csv()` 可直接读取 `.csv.gz`。
- `examples/`：六个固定选择的 test7×指 `.npz`，包含完整原生时间戳、BSQ 标签、模型分数、三种输出标定预测和事件 ID。每一天按配对 MAE 差选择最大改善、中位数、最大退化，未只展示成功例子。
- `models/ap304_rank/`：本次实际使用的 best 权重及配套归一化、原模型源码和来源哈希。权重 73,534,168 bytes，用 Git LFS 保存；未保存不用的 last。
- 不保存原始 MCAP、EMG、视频和全量中间预测。可按冻结清单重新读取。临时缓存默认放在被 Git 忽略的 `work/`。

## 数据来源与权限

冻结数据为 task103 的 `2026-11-07`、`2026-11-09` 两天：28 人，465 条 `mix-test-5`，225 条 `mix-test-7`。不是普通动作 `mix-5`。完整 690 条对象 key、ID、字节数、ETag、stage、人员、设备与时钟审计在 [manifests/sources.json](manifests/sources.json)。原 MCAP 对象总大小约 66.1 GB。

对象存储 bucket 为 `tes-bucket`，key 形如：

```text
raw/103/2026-11-07/<subject>/<device>/DTS-103-..._h265.mcap
```

这些是组织内部数据，不提供匿名公网下载地址。请向 task103 数据维护者获取有权限的 S3 endpoint 和只读账号，使用标准 AWS 环境变量/凭据文件/profile；仓库不包含 endpoint 凭据、access.json、令牌或视频。S3 读取附带冻结 ETag 的 `IfMatch`，对象变化时应重新核验，不能静默替换。

```bash
# 任选一种标准认证方式；凭据只在本机/环境中配置，不写进仓库。
export TASK103_S3_ENDPOINT='https://<your-authorized-s3-endpoint>'
export AWS_PROFILE='your-readonly-profile'
# 或用 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN。
```

也可完全不使用 S3。将有权限获得的 MCAP 放在同一个本地根目录，并保持清单中的 key 层级，例如 `/datasets/task103/raw/103/2026-11-07/...`，再传 `--mcap-dir /datasets/task103`。本地模式检查对象大小；应由数据提供方确认这些文件对应冻结清单。

`manifests/side_rest_tare.json` 保存 225 条 test7 的侧放后稳定空载零位及候选起点（由原生 BSQ 稳定段与已有视频复核得到）。这是本批数据的冻结评价辅助信息，**不能直接用于其他录制**。新数据需要重新确认侧放后的空载段。

原实验保留的完整缓存曾位于内部计算节点 `10.20.20.2:/data/lr_data/zjl/task103_rank_cmcu_ablation_20261109/filtered_run`。该路径仅用于有权限人员找回历史运行，不是脚本依赖，也不保证永久保留。

## 环境与最短运行方式

Python 3.10+，先装适合本机的 PyTorch，再安装仓库依赖。本次结果环境：H200、PyTorch 2.3.1+cu121，固定 best、BF16、每批最多 64 个窗。不同 GPU/批量/精度可能有浮点差异；不保证逐位相同。

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/task103_cmcu_calibration_20261109/scripts/verify_artifacts.py

# S3：从标准 AWS 配置获得凭据
python experiments/task103_cmcu_calibration_20261109/scripts/run_experiment.py \
  --run-root work/task103 --device cuda:0 --precision bf16 --workers 8

# 或本地 MCAP，完全无需 AWS 凭据
python experiments/task103_cmcu_calibration_20261109/scripts/run_experiment.py \
  --run-root work/task103-local --mcap-dir /datasets/task103 \
  --device cuda:0 --precision bf16
```

输出为 `<run-root>/report.html`，内嵌全部图像；统计 CSV、原生预测在同一运行目录。建议预留约 100 GB 磁盘，批量读取与推理耗时取决于存储和 GPU。CPU 可传 `--device cpu --precision fp32`，但全量会慢很多。不要对同一个运行目录同时启动多个完整流水线。

脚本参数可覆盖 endpoint、bucket、profile、工作进程数、输出目录、设备、精度、批量。`TASK103_MODEL_DIR` 可指向同一经过哈希验证的模型包。更换模型、输入处理或精度时使用新的运行目录，避免复用旧预测。

## 分阶段运行与缓存

主命令支持 `--stage extract|prepare|infer|score|report|all`。依赖顺序：

1. `extract`：只读五个原生 topic，解码 BSQ、CMCU 和双手 16 通道 EMG；EMG 2000 Hz 时间规整后只做一次标准滤波；保存 `native/`、`emg/`。
2. `prepare`：按冻结侧放零位路由按压、生成第一 3/4 秒及侧放后两套参考力，配对 test7 前后的 test5。保存 `prepared/`、`pairs.csv`、`quality.csv`、`events.csv`。
3. `infer`：固定 best，2 秒独立重置窗，原生输出位置 `start+470+40*k`，末窗去重，保留上下文缺口。保存 `predictions/<id>.npz`。
4. `score`：两层标定与独立评价，四个 CPU 分片；合并指标并做人员配对 bootstrap，渲染图表。`--score-workers` 可调整。
5. `report`：使用已有指标/主方案评分点重新汇总与渲染，包括不限制 25 N 上界的补充；不需要重新读 MCAP。

只验证数据访问/模型接口时，可先抽一条，无需跑完全部：

```bash
export TASK103_RANK_ROOT="$PWD/work/task103-smoke"
python experiments/task103_cmcu_calibration_20261109/scripts/extract.py \
  --id ad78177c0c58a0ffd0fe --workers 1
python experiments/task103_cmcu_calibration_20261109/scripts/infer.py \
  --id ad78177c0c58a0ffd0fe --device cuda:0 --precision bf16
```

中间格式：`native/<id>.npz` 存 `bsq_t/bsq_force`、双手 `*_t/*_ad/*_force`；时间单位 ns，力单位 N。`emg/<id>/<side>.npy` 为训练格式 uV，`*_valid.npy` 标记缺口。`predictions/<id>.npz` 为双手 `*_t` 与 `*_scores [N,5]`，输出无量纲。`points/<test7>_<finger_index>.npz` 为主方案评分点，键 `ref/ref3/refstable` 对应前三种去皮，三条预测键为 `raw_cmcu/bsq_calibrated_cmcu/direct_bsq`。

## 评价口径与不能忽略的边界

- A/B 两组都用 test5 拟合分数→N。A 用默认 CMCU force；B 先在两个 test5 内分别拟合 AD→去皮 BSQ，再合并各自的模型输出/重标定标签点拟合分数→N。另列直接 BSQ 标定分数的参照。无模型训练或 test7 参数拟合。
- 主方案 `both` 使用 test7 前后最近、同人同日同设备/boot 的两个 test5，属于离线双侧标定；`previous` 才是只使用已采集数据的方案。不能把 `both` 作为在线结果。
- test7 拇指为支撑，只评价共同八个非拇指。按压身份由原生 AD 路由，不用模型误差选择。已知正放 test7 `ecc7e67d1e6aa84e533e` 单列。其余不因误差大小人工剔除。
- 文件前 3/4 秒中位数在本批数据恰好相同，但部分录制当时尚未侧放。主结论同时核对侧放后稳定空载去皮。按压路由固定用侧放后局部零位，避免负的静息读数漏检。
- 同步阈值：模型原生输出→BSQ 最近点 ≤12 ms，CMCU→BSQ ≤20 ms。`recv-sync` 的 EMG 批量传输延迟不用于移动信号时间轴。超过 5 ms EMG 采样缺口的窗剔除。
- 主输出映射为单调保序，域外不外推；另列非负斜率 affine 与端点饱和策略。两组使用完全相同的输出标定输入点和评分域。
- MAE 先按压等权，再 trial→指→人等权。28 人、210 次 test7、1196 个共同可评分 trial×指；单元可用率约 69%，已有模型单元的人员等权域内点覆盖约 97%。缺失单元不计零误差。0–25 N 与完整非负力范围分别报告。
- 配对 bootstrap 为固定 seed=20261109、10,000 次人员重采样。排序 τ-b 不能当绝对力误差。校准仍有约 3.3 N MAE、高力段误差，不能据改善比例宣称已达到部署精度。

## 输入处理核验

模型包写的“不滤波”指模型接口内部不再处理已转换的训练格式输入。直接将带数万 uV 直流偏置的 MCAP 值送入模型，会使预测近乎不变。此实验依据既有数据入口补齐 50 Hz 谐波陷波（Q=30）与四阶 40–850 Hz 带通，因果 SOS，前 1000 点均值初始化，随后仅使用模型配套的 Train mean/std。

`provenance/raw_to_training*.py` 记录核对的训练数据入口，`results/preprocessing_parity.json` 记录 30,000×16 个值逐样本一致、最大差 0；`results/model_parity.json` 核对模型原生接口时间位置。原始未滤波版只作输入问题诊断，没有作为最终精度。测试集已在诊断中查看，因此本次为探索性评测，严格盲测泛化仍需新数据。

## 仓库移植验证

[results/repository_validation.json](results/repository_validation.json) 记录了本次检查：690 条数据准备与历史结果一致；重算 `summary.csv`（900 行）、`paired_effect.csv`（300 行）、`per_subject.csv`（8400 行）及传感器汇总，全部数值最大差为 **0**。使用真实录制缓存构造 6 秒 MCAP 回放，BSQ/CMCU 标签与时间戳、滤波后 EMG、推理时间戳均通过核对；BF16 分数按浮点容差核对。原临时 S3 凭据已清除，仓库移植阶段未重新在线获取原始 MCAP，回放不冒充新的 S3 全链路验收。

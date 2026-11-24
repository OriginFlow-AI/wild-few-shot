# BSQ 标定 CMCU 与质量过滤

本分支提供独立 Python 包和批处理入口。输入本地 MCAP 或已提取的原生 NPZ，输出逐指标定力、有效掩码、标定来源和质量检查明细。

## 安装与运行

Python 3.10+，支持 Linux/macOS。NPZ 入口依赖 NumPy、SciPy。读取 MCAP 时额外安装 protobuf、mcap 和解压依赖。

```bash
python -m pip install -e '.[mcap]'
python -m cmcu_calibration --manifest recordings.json --out work/calibration
```

`recordings.json` 支持数组，或含 `records` 数组的对象。`path` 相对清单所在目录解析，也可为绝对路径。

```json
[
  {
    "id": "source_test5",
    "subject": "subject01",
    "stage": "mix-test-5",
    "start": "2026-11-18T09:00:00+08:00",
    "end": "2026-11-18T09:02:00+08:00",
    "pc": "device01",
    "boot": "session01",
    "path": "data/source_test5.mcap"
  },
  {
    "id": "ordinary_stage",
    "subject": "subject01",
    "stage": "mix-17",
    "start": "2026-11-18T09:05:00+08:00",
    "end": "2026-11-18T09:07:00+08:00",
    "pc": "device01",
    "boot": "session01",
    "path": "data/ordinary_stage.npz"
  }
]
```

`id`、`subject`、`start`、`path` 必填。`stage` 省略时按 `mix-test-5` 处理。`date` 可省略，按 `start` 日期推导。`pc`、`boot` 可省略；填写后，来源与目标的这些字段也需相同。时间采用统一的本地 ISO 时间或统一带时区的 ISO 时间。`end` 可省略，此时按原生信号跨度估计完成时间。

## 标定

1. 仅用 `mix-test-5` 建立来源曲线。BSQ 默认按整条录制低十分位附近的读数中位数去皮，沿用 `fixed_bsq_zero`。清单内可提供 `zero_N` 覆盖自动零点。
2. 以 BSQ >0.3 N 的完整按压分段，需达到 3 N 才尝试识别指别。默认 `ad_delta` 路由先扣动作前静息 AD，再比较高力阶段的增量，主指响应至少 100 计数且至少是次指的 3 倍。静息不足、响应太弱或指别不明确的事件记录原因。
3. 在同步误差 ≤20 ms、BSQ 0–25 N 的路由按压点中逐指取原始 AD 与参考力。加入首秒 BSQ 距零 ≤0.05 N 的同步 AD 零点锚。AD 保留原始绝对计数。
4. 相同 AD 的 BSQ 取中位数，以出现次数为权重做单调保序回归，再在节点间线性插值。固定原始 AD 等于 0 时输出 0 N。拟合域内的负输入也输出 0 N，以保持力值非负和曲线单调。建表仍需至少 8 个原始评分点、两个不同 AD 值。
5. 目标使用同人、同日、同 PC/boot 分组内，开始时间距目标不超过 1 小时且已完成的最近一次可用 Test5。来源按录制选择，缺指保持无效。目标录制随后才加入来源列表，避免自标自测。
6. 有可用逐指来源时，AD 等于 0 的点输出 0 N。其余超出实测标定 AD 域的点输出 NaN，并将有效掩码置为 false。首条 Test5 没有前次来源，但仍会生成自己的来源曲线供后续动作使用。

默认合并加载和卸载、保留原始同步时间关系。首秒零点锚与路由指别的静息扣除用途不同，后者只用于识别受测指。

硬零点使用原始 AD。设备静息 AD 非零时，仍按现有曲线映射该值。历史导出的曲线及力标签可用原始清单重新生成。

旧版按“默认薄膜力峰值最高的一指”归属动作的统计，可用以下路由复现。此模式会受到预压、残余和高基线干扰，输出中会保存所选模式。

```bash
python -m cmcu_calibration --manifest recordings.json --out work/legacy \
  --routing default_force_peak
```

## 质量过滤

各规则可单独启用。默认启用斜率的整条录制过滤，滑窗和旧基础质检采用显式开关，便于复现此前分别开展的分析。

| 检查 | 定义 | 开关与处理 |
|---|---|---|
| 数据读取与时钟 | 时间戳递增、形状和有限值有效；若提供 `clock_ok` 则检查；MCAP 另检查各流 recv-sync 中位偏移差 ≤50 ms | 始终检查，失败记录为 `error` |
| 斜率 | 每个“录制×手指”用 2≤BSQ≤25 N 拟合 `默认薄膜力 = k × BSQ + b`，保留 0.3≤k≤3 | `--slope-scope recording` 默认任一已测指超界即整条拒绝；`finger` 按指；`off` 仅记录诊断 |
| 滑窗 | 完整按压中存在 1.5 s 窗，BSQ>3 N 累计覆盖≥1.2 s，且十指任一路原始 AD>45000 连续≥135帧 | `--sliding-scope press` 按压过滤，`recording` 整条过滤，`off` 默认仅诊断 |
| 旧基础曲线质检 | 至少20点、8个AD值、AD跨度≥100；Spearman≥0.2；拟合输出跨度≥1 N；残差P90同时>10 N且>拟合力跨度一半则拒绝 | `--basic-qc` 按指启用 |

斜率筛查的是设备默认力，映射表拟合的是原始 AD。两者各自使用对应数据。

滑窗按原生行数统计连续帧，延续历史实现，未在帧间时间缺口处额外切断连续段。`strict_passed` 另外记录连续高 AD 段逐帧同时满足 BSQ>3 N 的结果，用于诊断。原规则判断的是同一合格窗口内的两项条件。

BSQ 覆盖时长不足记为 `BSQ_duration_too_short`，AD 连续帧不足记为 `AD_run_too_short`。按压过滤会从建表样本中排除不通过的按压，也会将被评 Test5 中未通过按压及其余非路由时段标为无效。普通 stage 使用附近曲线输出，不用“BSQ没有受力”判定普通动作无效。

```bash
# 斜率整条过滤，并启用滑窗按压过滤
python -m cmcu_calibration --manifest recordings.json --out work/both \
  --slope-scope recording --sliding-scope press

# 复核历史滑窗分析，不额外叠加斜率过滤
python -m cmcu_calibration --manifest recordings.json --out work/sliding \
  --routing default_force_peak --slope-scope off --sliding-scope press

# 基础标定版本的质量门槛
python -m cmcu_calibration --manifest recordings.json --out work/basic \
  --slope-scope off --basic-qc
```

阈值命中属于采集异常候选，可能来自薄膜状态、佩戴、接触、指别归属或动作。绝对高 AD 也可能包含基线；斜率和滑窗结果需结合原始信号解释。

## 输入 NPZ

| 字段 | 形状和单位 |
|---|---|
| `bsq_t`、`bsq_force` | `(N,)`，整数纳秒、N；Test5 必需，普通 stage 可整体省略 BSQ 流 |
| `left_t`、`right_t` | 各自 `(M,)`，严格递增的整数纳秒，使用同步时钟 |
| `left_ad`、`right_ad` | 各自 `(M,5)`，原始 AD |
| `left_force`、`right_force` | 各自 `(M,5)`，默认薄膜力，单位 N |
| `clock_ok` | 可选标量布尔值，false 时拒绝 |

十指顺序为左拇、左食、左中、左无名、左小、右拇、右食、右中、右无名、右小。MCAP 入口读取 `/bsq/middle/force` 及左右 `/cmcu05b/*/cmcu05b`，`pressures` 从 mN 转成 N。缺 `ad_values` 的旧 schema 会明确报告错误。

部分旧 NPZ 保存了每手9路，入口兼容这些数组并使用前5路，沿用原分析的十指顺序。

## 输出与追溯

- `models.json` 保存每次来源 Test5 的逐指单调表和适用分组。若零节点位于实测域外，`observed_ad_range` 记录实测 AD 上下界。使用 `predict_lookup` 应用该曲线和有效域。
- `force/<id>.npz` 保存左右手原生时间、标定力、逐点 `*_valid`、默认力和 `source_id`。未匹配来源、域外或质量拒绝点为 NaN/false。
- `quality/<id>.json` 和汇总 `quality.json` 保存零点、来源、来源间隔、路由拒绝原因、逐指斜率、滑窗结果、有效覆盖和录制状态。
- `summary.json` 保存配置及汇总。`first_record_rejection` 按首个录制拒绝原因计数；`recording_reasons` 可重叠。逐指拒绝和按压拒绝在各自明细中记录。
- `run.json` 的 `complete` 表示本次批处理已完成。读取输出时以本次 `quality.json` 的 ID 清单为准。输出文件采用临时文件替换，同一输出目录的并发运行由系统锁串行化。

## 原始 AD 平台专项复核

`tools/screen_ad_plateaus.py` 来自远端 `c0ad0d0` 的平台复核脚本。它查找“BSQ上升、AD变化很小”的重复按压，输出复核清单。输入格式和完整口径见源分支 [data_quality/task103](https://github.com/OriginFlow-AI/wild-few-shot/tree/c0ad0d0e0384f7ebe786a6e56eb894c4813e0518/data_quality/task103)。

```bash
python tools/screen_ad_plateaus.py --aligned-dir aligned_test5 \
  --records records.json --out work/plateau
```

## 实现来源与验证

- 标定节点算法沿用本地查看器的 `fit_lookup_curve`，并保持同 AD 中位数和按频数加权的 PAVA。
- 单指路由沿用 `routing.py`。历史默认力峰值路由作为显式兼容选项。
- 斜率算法来自 `compute_channels_1118.py`，阈值为 0.3–3。点数少于2或参考力无跨度时记录 `insufficient`。
- 滑窗算法迁自 `check_gjs_sliding.py`、`sliding_quality_10d.py`。
- 可选基础质检来自提交 `309a98e` 的实验脚本。

```bash
python -m unittest discover -s tests -v
```

原始 MCAP、NPZ、模型权重和访问凭据保留在各自数据存储中。此分支包含实现、入口和测试。

### 2026-11-23 迁移验证

- 13 项自包含测试通过，覆盖加权保序回归、域外掩码、时间与设备边界、命令行入口、过滤作用于被评按压、坏MCAP隔离、重试及旧9路NPZ。
- 两项外部数据回归通过。在兼容路由下，11月18日128条录制、1,267个通道的斜率与旧统计一致；129条录制、2,948次按压的滑窗判定和连续帧数与旧审计一致。
- 一条本地原始MCAP的BSQ时间/力、左右AD/默认力，与原有NPZ提取结果一致。
- 129条真实NPZ通过完整默认批处理，读取错误为0。默认AD增量路由下，90条质量拒绝、18条缺前次来源、21条有前次来源并完成应用，生成39条可供后续使用的来源模型。这些数量反映默认路由与整条过滤的组合，旧峰值路由的96条斜率异常录制属于另一配置。

外部回归使用数据集持有者提供的冻结审计，环境变量分别指向原生NPZ目录、`audit_by_finger.json` 和 `sliding_10d_audit.json`。

```bash
CMCU_NATIVE_DIR=/path/to/native \
CMCU_SLOPE_AUDIT=/path/to/audit_by_finger.json \
CMCU_SLIDING_AUDIT=/path/to/sliding_10d_audit.json \
python -m unittest discover -s tests -v
```

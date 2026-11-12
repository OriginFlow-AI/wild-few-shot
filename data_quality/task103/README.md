# Task103 数据问题清单

本目录记录 Task103 2026-10-19 至 2026-11-10 冻结清单中已排查的数据问题，供复核、回源和重采使用。证据来自 Task103 AP304 实验的原始流提取与标定质检，基准提交为 [309a98e](https://github.com/OriginFlow-AI/wild-few-shot/tree/309a98ed9d9a78ccba48824b3888ac0594231004/experiments/task103_ap304_huber_ablation_20261110)。

| 文件 | 使用方式 |
|---|---|
| [data_quality_findings.csv](data_quality_findings.csv) | 通过 `recording_id` 或 `object_key` 定位原始 MCAP。含 26 条记录级排除和 329 个指通道排除。 |
| [calibration_quality_examples.png](calibration_quality_examples.png) | BSQ 力值异常、BSQ 恒定和正常拟合的对照。数值见同名 JSON。 |
| [full_calibration_quality_summary.json](full_calibration_quality_summary.json) | test5/test7 质检统计、不可用记录、无可用指的 test5 及其事件排除理由。 |
| [accepted_calibration_diagnostics.csv](accepted_calibration_diagnostics.csv) | 16,307 条已通过基础质检的指曲线，其中 4,040 条有人工复核标记。 |
| [accepted_calibration_worst_residuals.png](accepted_calibration_worst_residuals.png) | 已保留曲线中残差较大的例子。对应数值和适用范围见诊断 JSON。 |

先处理 16 条提取失败（时钟逆序或 MCAP 损坏）、3 条参考信号不可用、7 条无可用标定指。指级问题按 `finger` 定位，239 个弱单调、88 个严重拟合失配、2 个拟合力跨度不足，共涉及 267 条 test5。只有指级证据时应保留其它可用指。无有效按压、读档失败和参考力异常分别记录，复核后再确定是否需要重采。

优先检查同一 AD 在重复按压下是否对应一致的 BSQ 力。严重失配的门槛为残差 P90 同时大于 10 N 和拟合力跨度的一半。轻度复核标记保留在诊断表，适合排定回看顺序；其本身不改变数据可用状态。模型 Test 误差没有参与这些筛查。

新增问题时沿用 CSV 字段，至少记录对象路径、发现范围、手指（如适用）、可复查证据和当前处理状态。原始记录和文件保留在数据存储中。

跨人划分发现多条不同 MCAP 共用占位 ETag `00000000000000000000000000000000-1`。在 K2 四组候选记录中有 4,693 条使用该值。它只能作为缺失内容标识记录，去重时应检查 recording ID、非占位 ETag 与大小，以及实际信号内容哈希。

## 2026-11-12 标定源复查和目标覆盖

重新对冻结清单的 **1,977 条 test5** 做时间对齐、BSQ 去零、逐指路由和 PAVA 拟合。1,973 条生成的曲线、按压事件和参考路由与旧版逐文件一致，其中 **1,966 条至少有一指可用并被应用**。4 条来源不可用，7 条准备完成但无可用指，已有规则屏蔽了 329 个拟合失败的指通道。逐条表记录拟合成功指、失败原因、按压排除原因和实际使用次数。

| 问题 | 复核证据 | 筛选时如何使用 |
|---|---|---|
| 同一指在不同按压下标定不一致 | 647 个指通道触发重复按压复核。其中 264 个通道来自 237 条 test5，旧的单调性和整条拟合残差规则均未标记这些通道。 | 优先看留一次按压的误差及覆盖率，按指回看原始 AD、BSQ 和按压路由。 |
| 独立重复按压证据不足 | 3,095 个通道只有一次合格按压，6 个通道虽有重复但留出按压无有效计分点，另有 1,049 个可计分通道覆盖率低于 50%。 | 同时查看按压次数、有效点和 AD 覆盖。低误差与低覆盖同时出现时，仍需回看力值范围。 |
| 目标 recording 找不到有效标定 | 原始跨人数据的 10,528 条中，2,339 条缺同人、同日、同 PC、同启动会话的 test5，70 条的最近来源无可用曲线，18 条标定来源与目标间时钟不连续。 | 保留 recording 和来源 ID，按缺来源、来源无可用指、时钟异常分别排查。 |
| 部分手指或力段被屏蔽 | 成功应用标定的 8,101 条中，856 条保留不足一半的原生有效力点。59 条在有曲线的通道上，超过 20% 的原生有效点超出曲线 AD 域。 | 分开检查缺曲线和 AD 越界的点数。优先复核整手只剩少数手指的来源，和目标按压力度超出标定幅度的 recording。 |
| 老版 CMCU schema 缺 AD | 对 41 条抽取报 `ad_values` 的 recording 回看 MCAP schema。左右手均声明 `pressures`，均缺 `ad_values`，41 条也都缺同会话 test5。全 stage 原始标签抽取另外有 6 条时钟逆序、1 条解压损坏。 | 将原始压力字段存在与可做 AD 标定分开记录，按 MCAP schema 定位批次。 |

重复按压复核使用同一 test5 同一指的留一次按压诊断。留出覆盖率至少 50%，且留出 MAE 同时超过 3 N、拟合力跨度的 20% 和整条拟合 MAE 的 3 倍时标记复核。这些标记保留供后续筛选，实际应用继续按已复验的原质检规则及目标 AD 域生成有效掩码。

- `b994adab2acc0cb51c92` 左拇指整条拟合 MAE **4.68 N**，两次按压交替留出后 MAE **22.35 N**，覆盖率 **58.2%**。其中一次按压的全部 143 个有效点都落在另一条曲线的 AD 范围内，MAE 仍为 **24.12 N**。[按压对照图](calibrated_subject_audit_20261112/test5_repeat_disagreement_example.png)
- `dedab4d14e39aab4069f` 复用来源 `895b434d1bbb2dfd7464`，只保留 **9.6%** 的原生有效力点，主要因为其余指缺少曲线。`6a6af567b7876828da2d` 在有曲线的指上仍有 **38.6%** 的 AD 点超出标定域，整体保留 **55.3%**。[目标覆盖证据](calibrated_subject_audit_20261112/calibrated_target_findings.csv)

| 文件 | 内容 |
|---|---|
| [test5_source_review.csv](calibrated_subject_audit_20261112/test5_source_review.csv) | 1,977 条来源的逐条复核记录，含 MCAP 路径。 |
| [test5_repeat_disagreement_review.csv](calibrated_subject_audit_20261112/test5_repeat_disagreement_review.csv) | 647 个重复按压不一致的通道，含覆盖、误差、旧标记和使用次数。 |
| [calibrated_target_findings.csv](calibrated_subject_audit_20261112/calibrated_target_findings.csv) | 缺标定、时钟问题、低覆盖或明显越界的目标 recording。 |
| [all_stage_raw_extraction_findings.csv](calibrated_subject_audit_20261112/all_stage_raw_extraction_findings.csv) | 全 stage 原始标签抽取的 48 条排除记录。 |
| [raw_no_ad_schema_audit.json](calibrated_subject_audit_20261112/raw_no_ad_schema_audit.json) | 41 条缺 AD recording 的双手 CMCU schema 核验。 |
| [calibrated_subject_data_audit.json](calibrated_subject_audit_20261112/calibrated_subject_data_audit.json) | 复查总数、覆盖统计、按 stage 的可用数据数和典型案例。 |

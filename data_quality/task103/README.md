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

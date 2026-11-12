# Task103 模型的 Benchmark v4.05 Test

在 `.1` 重建 v4.05 的固定 109 段 Test，保留 6 人、12,505 个窗口的冻结掩码和 500,040 个有效原生力时间点。只读取 RRD 的双手 16 通道 EMG、各手前五指 CMCU 和时间戳。按大小和 checker SHA256 核对源对象，用 v4.05 冻结 LUT 从原始 CMCU N 一次换算真值，再核对历史指标保存的逐人、手、指标签统计量。

以随 A 提供的 Online W100 作为所有 EMG 模型的统一推理方式。保持 RRD 的 EMG 数值，用因果保持对齐到 2 kHz，50 Hz 模型输出保持到下一次更新。预测与原生力时间点按时间戳对齐。每个模型核对权重与归一化，先做 4,000 点直接前向和 Online W100 的等价性与重置检查。排序模型保持无量纲输出，报告 Corr 和 Kendall τ-b。

| 脚本 | 用途 |
|---|---|
| `prepare_test.py` | 核对冻结掩码和源对象，暂存 RRD 并提取 EMG、原始力与标定力。 |
| `verify_targets.py` | 对比重建后的真值与历史 v4.05 的逐人、手、指统计。 |
| `evaluate.py` | 单模型 W100 推理和计分。 |
| `launch.py` | 自动选共享 GPU，运行模型清单并汇总。 |
| `registry.json` | Task103 正式实验模型及参考 `A_ep1` 的权重来源。 |

冻结测试来自 `/data/zjl/task46/emg_force_ex45_newcal_v405_20261028`。运行目录为 `/pub_data/zjl/task103_benchmark_v405_20261112`。运行时使用冻结的 `fusion2force` 读档代码和 Rerun SDK 0.36.3。Online W100 来自 `/data/lr_data/benchmark_calibrated_huber_20261105/contact_eval/adapters.py`。

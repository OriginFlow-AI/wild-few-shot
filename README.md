# Task103：AP304 二阶段标定与 loss 对照

分支：`zjl/task103-ap304-huber-calibration-ablation`。

本实验从同一 AP304 排序预训练 best-tau 权重开始，对 task103 的普通 stage 做全参数回归微调；test5 约一半做 val，其余 test5 与 test7 做 test。比较四组：

| 实验 | Train / Val 标签 | Test 标签 | Loss |
|---|---|---|---|
| 1 | BSQ 重标定的 CMCU | 重标定 CMCU | 纯 Huber，δ=1 N |
| 2 | 同 1，共享 checkpoint | BSQ | 纯 Huber，δ=1 N |
| 3 | 设备原有 CMCU force | BSQ | 纯 Huber，δ=1 N |
| 4 | 与 3 相同 | BSQ | `(1−Corr)+0.3×MAPE+0.03×MAE` |

三套训练共用初始权重、EMG、划分和有效点；第 3、4 组连 force 归一化也相同。结果尚未完成，不能把预处理或 smoke 通过视为实验收益。

- [完整协议、数据来源和复现](experiments/task103_ap304_huber_ablation_20261110/README.md)
- [配置](experiments/task103_ap304_huber_ablation_20261110/config.json)
- [运行检查与阶段状态](experiments/task103_ap304_huber_ablation_20261110/results/)
- [一阶段权重与共同二阶段初始化](models/ap304_stage1/)（Git LFS）

`.1` 运行目录为 `/pub_data/zjl/task103_ap304_huber_calibration_ablation_20261110`。GPU 按剩余显存和利用率自动选择，允许与已有任务共用；已按真实 Train 基准冻结三套训练共用microbatch16，最多三张卡并行不同实验，尽量于北京时间11月11日08:00前完成。

原始 MCAP、EMG、视频、访问密钥和运行缓存不入库。

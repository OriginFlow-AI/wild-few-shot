# wild-few-shot
野采标定代码仓库，main分支暂时用来整理进展。单独分支上放实验脚本。

每个分支，创建分支名.md或者分支名.html之类的放到main分支，然后在下面一句话说明内容。
# 实验

- [codex/task103-cmcu-calibration-20261109](codex/task103-cmcu-calibration-20261109.md)：固定 AP304，比较是否先用 BSQ 重标定 CMCU；test7 用 BSQ 真值评价，MAE 6.53→3.26 N（降低 50%）。

- [AP304 原 loss 与 A-loss](codex/ap304-original-a-20261118.md)：全量排序预训练与 best_tau 选模记录、第二阶段原 loss 和 A-loss 训练包及薄膜评估口径。

- [syh/structure/ring-rq：旧 Ring-RQ 架构](ring_rq/README.md)：W304/E9/D4 模型、Train-only RQ拟合、Kendall两阶段训练、本地数据读取与离线推理；代码包不包含私有数据和权重。

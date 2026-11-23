# wild-few-shot
野采标定代码仓库，main分支暂时用来整理进展。单独分支上放实验脚本。

每个分支，创建分支名.md或者分支名.html之类的放到main分支，然后在下面一句话说明内容。
# 实验

- [codex/task103-cmcu-calibration-20261109](codex/task103-cmcu-calibration-20261109.md)：固定 AP304，比较是否先用 BSQ 重标定 CMCU；test7 用 BSQ 真值评价，MAE 6.53→3.26 N（降低 50%）。

- [AP304 原 loss 与 A-loss](codex/ap304-original-a-20261118.md)：全量排序预训练与 best_tau 选模记录、第二阶段原 loss 和 A-loss 训练包及薄膜评估口径。

- [Stage3 EMG记忆标定：A / LoRA / F](syh/task43-emg-memory-calibration.md)：同一A-loss基座上比较固定基线、LoRA适配与本人Stage3标定记忆，提供训练、推理和数据读取代码。

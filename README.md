# wild-few-shot
野采标定代码仓库，main分支暂时用来整理进展。单独分支上放实验脚本。

每个分支，创建分支名.md或者分支名.html之类的放到main分支，然后在下面一句话说明内容。
# 实验

- [codex/task103-cmcu-calibration-20261109](codex/task103-cmcu-calibration-20261109.md)：固定 AP304，比较是否先用 BSQ 重标定 CMCU；test7 用 BSQ 真值评价，MAE 6.53→3.26 N（降低 50%）。

- [AP304 原 loss 与 A-loss](codex/ap304-original-a-20261118.md)：全量排序预训练与 best_tau 选模记录、第二阶段原 loss 和 A-loss 训练包及薄膜评估口径。

- [Stage3 EMG记忆标定：A / LoRA / F](syh/task43-emg-memory-calibration.md)：同一A-loss基座上比较固定基线、LoRA适配与本人Stage3标定记忆，提供训练、推理和数据读取代码。

## Force3 架构消融

- [Ring-RQ 架构](syh/structure/ring-rq.md)（[实验分支](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/ring-rq)）：W304/E9/D4 Encoder–Decoder 与冻结残差量化读出，提供训练、数据读取、离线及 W100 连续推理、配套最佳权重和历史评估。
- [三视图集成架构](syh/structure/threeview.md)（[实验分支](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/threeview)）：共享 Package W304/E9 Encoder，三个独立 W128/D2、各固定屏蔽24维的读出分支，最终取三路预测均值；提供两阶段训练、集成推理及数据读取代码。

两架构按共同55个时刻、相同 Kendall/A-loss 两阶段协议开展受控对照，目前尚无两臂同预算的完整结论。Ring-RQ 分支中的历史3572点/W100评估与新消融的时间支持不同，不直接作为三视图相对收益的依据。

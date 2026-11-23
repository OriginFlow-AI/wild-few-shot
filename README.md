# wild-few-shot

## 本分支：三视图集成架构

`syh/structure/threeview` 提供当前Force3三视图的训练、集成推理与本地数据读取代码：[使用说明](threeview/README.md)、[数据schema](threeview/DATA_SCHEMA.md)、[验证范围](threeview/VALIDATION.json)。Package W304/E9与三路W128/D2 mask24、冻结RQ/frame/Motion；原两阶段Kendall/A-loss和55点口径。此次为代码发布，不含波形、私有split、标定表或训练权重。

Ring-RQ已在[独立分支](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/ring-rq)发布，其旧模型成绩不能当作三视图结果。

野采标定代码仓库，main分支暂时用来整理进展。单独分支上放实验脚本。

每个分支，创建分支名.md或者分支名.html之类的放到main分支，然后在下面一句话说明内容。
# 实验

- [codex/task103-cmcu-calibration-20261109](codex/task103-cmcu-calibration-20261109.md)：固定 AP304，比较是否先用 BSQ 重标定 CMCU；test7 用 BSQ 真值评价，MAE 6.53→3.26 N（降低 50%）。

- [AP304 原 loss 与 A-loss](codex/ap304-original-a-20261118.md)：全量排序预训练与 best_tau 选模记录、第二阶段原 loss 和 A-loss 训练包及薄膜评估口径。

- [Stage3 EMG记忆标定：A / LoRA / F](syh/task43-emg-memory-calibration.md)：同一A-loss基座上比较固定基线、LoRA适配与本人Stage3标定记忆，提供训练、推理和数据读取代码。

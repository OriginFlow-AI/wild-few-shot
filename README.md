# wild-few-shot
野采标定代码仓库，main分支暂时用来整理进展。单独分支上放实验脚本。

本分支记录数据问题及筛选证据。Task103 的原始流、`test5` 标定源和目标 recording 覆盖问题见 [Task103 数据问题清单](data_quality/task103/README.md)。

每个分支，创建分支名.md或者分支名.html之类的放到main分支，然后在下面一句话说明内容。
# 实验

- [codex/task103-cmcu-calibration-20261109](codex/task103-cmcu-calibration-20261109.md)：固定 AP304，比较是否先用 BSQ 重标定 CMCU；test7 用 BSQ 真值评价，MAE 6.53→3.26 N（降低 50%）。

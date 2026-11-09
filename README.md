# wild-few-shot

野采标定实验；各实验放独立分支。

## Task103：CMCU 重标定是否有收益？

**有。** 固定 AP304 排序模型，两组都用 test5 标定模型输出；区别是 CMCU 是否先用 BSQ 标定。11 月 7/9 日共 28 人，test7 独立评价：

| MAE（0–25 N，人员等权） | CMCU 未重标定 | CMCU 重标定 |
|---|---:|---:|
| 11 月 7 日 | 6.44 N | 2.91 N |
| 11 月 9 日 | 6.62 N | 3.61 N |
| 合计 | **6.53 N** | **3.26 N（降低 50%）** |

主方案用 test7 前后两个 test5；仅用前次 test5 也从 6.68→3.40 N。test7 按侧放后稳定空载去皮，前 4 秒去皮版结论一致。共同可评分 trial×指约 69%，高力段仍有明显误差。

- [详细报告 report.html](experiments/task103_cmcu_calibration_20261109/report.html)：统计、逐人结果、预测与 BSQ 标签同图。下载后直接用浏览器打开，图已嵌入。
- [数据获取与复现](experiments/task103_cmcu_calibration_20261109/REPRODUCE.md)
- [模型](models/ap304_rank/)：只保存本次使用的 best（约 70 MiB，Git LFS）。原始 MCAP/EMG/视频不入库；数据清单、统计结果和六组示例轨迹已保存。

```bash
git lfs pull
python -m pip install -r requirements.txt
# 设置有权限的 S3 endpoint / AWS 凭据后运行；也支持本地 MCAP
python experiments/task103_cmcu_calibration_20261109/scripts/run_experiment.py \
  --run-root work/task103 --device cuda:0 --precision bf16
```

注意：已核对 MCAP→训练格式的 EMG 预处理；不能将带巨大直流偏置的原始 MCAP 值直接送入模型。

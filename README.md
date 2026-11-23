# Ring-RQ 旧架构与最佳权重

## Test 与基线对比：同人口、同 W100

固定权重为**标定 Ring-RQ：一期 EP6 → 二期 EP1**，由完整 Val A-loss 选出。基线为旧 **ex45 W304/E9/D4 Affine A-loss EP1**（权重SHA前缀 `e7110ae4`），不是尚无完整Val的新架构消融基线。

| Test范围 | 模型 | 参考A-loss† ↓ | MAE N ↓ | MAPE % ↓ | Corr ↑ | Kendall τ_b ↑ | R²宏平均 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Task43共同Test：1,395条 / 6人 | 旧 Affine A-loss EP1 | 0.20676 | 0.47048 | 46.82 | 0.82128 | 0.56109 | -1106.71 |
| Task43共同Test：1,395条 / 6人 | **本分支 Ring-RQ 二期EP1** | 0.22416 | 0.53775 | 47.64 | 0.80829 | 0.55721 | -2249.25 |
| Task83 Test：3,379条 / 33人 | 旧 Affine A-loss EP1 | 0.39471 | 0.39096 | 80.05 | 0.48797 | 0.29304 | -8809.85 |
| Task83 Test：3,379条 / 33人 | **本分支 Ring-RQ 二期EP1** | 0.34148 | 0.38079 | 65.82 | 0.56292 | 0.37214 | -18154.04 |
| 合计：4,774条 / 39人 | 旧 Affine A-loss EP1 | 0.36579 | 0.40319 | 74.94 | 0.53925 | 0.33428 | -7624.75 |
| 合计：4,774条 / 39人 | **本分支 Ring-RQ 二期EP1** | 0.32343 | 0.40493 | 63.02 | 0.60067 | 0.40061 | -15707.15 |

合计相对旧基线：Corr **+0.06142**、Kendall τ **+0.06633**、MAPE **降低11.916个百分点**；MAE **增加0.432%**，基本持平。Task43共同Test退步，Task83的MAE/MAPE/Corr/τ改善；R²宏平均受极端负值影响，原样保留。该模型没有在所有来源或指标上同时胜出。

†参考A-loss是将本表宏Corr、MAPE比例、MAE代入 `0.5*(1-Corr)+0.1*MAPE_ratio+0.15*MAE_N`，**不是训练history的production L**。聚合为录制逐指 → 同人/stage/手/指内录制等权 → 有效手侧×手指等权 → stage等权 → 人等权；MAPE仅在 |GT|≥0.5N 计算，各指标独立排除null，不做力幅加权。τ使用完整录制逐指tau-b。

Task43仅保留在旧权威split中同为Test的6人；Task83的33人在该旧索引中没有Train/Val登记。此曝光核验限于该冻结索引。两组人员无交集，合计中Task83占33/39人口权重。两模型训练人口及训练协议不同，这不是仅替换解码头的受控架构消融。

以上均复用已经完成的固定权重评估，不重新读取Test，不用Test选权重或调参。[完整精度、分母与证据SHA](ring_rq/EVALUATION.json)。

本分支为 `syh/structure/ring-rq`，包含旧W304/E9/D4 Ring-RQ训练、数据读取、离线reset2s/W100推理，以及一期EP6→二期EP1最佳权重、配套norm和RQ。

[完整加载说明、完整Test结果与评估口径](ring_rq/README.md) · [权重与SHA清单](ring_rq/assets/phase2_ep1/MANIFEST.json) · [结果证据](ring_rq/EVALUATION.json)

```bash
python -m pip install -r ring_rq/requirements.txt
python -m ring_rq.infer_w100 --emg filtered_segment.npy --output force_w100.npz --device cuda:0
```

输入：一条连续、已滤波但未标准化的float32[T,16] EMG（uV）。默认加载已发布的权重/norm/RQ；每条录制或技术缺口reset，2秒块之间保持连续状态。W100需要CUDA。CPU固定2秒推理见详细README，和上述Test口径不同。

## 一阶段结束后的权重

已新增 **一期EP6终态 / Val τ-best**：[下载纯权重](ring_rq/assets/phase1_ep6/best.pt) · [来源与指标清单](ring_rq/assets/phase1_ep6/MANIFEST.json)。它是上述二期EP1的真实父模型，直接复用已发布的norm和RQ。

[一期加载方法与Val成绩](ring_rq/README.md#一阶段结束后的-ep6-权重)。开头的Test表仅对应二期EP1，本次未新增一期Test评估。

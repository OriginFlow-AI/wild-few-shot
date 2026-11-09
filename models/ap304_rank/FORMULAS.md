# 输入、输出、标签与训练公式

## 1. 输入归一化

原始EMG为float32的 `[T,16]`，2000Hz。对每个电极c：

```text
x[t,c] = (raw_emg[t,c] - emg_mean[c]) / emg_std[c]
```

mean/std来自完整Train并固定；float32先减后除，与生产数据读取一致。不做滤波、通道镜像或force校准。

## 2. AP304与输出尺度

模型width304、8 heads、FFN1216、9 encoder blocks、4 decoder blocks，完整参数18,357,945。4000点输入先形成100个token，去除11个上下文token，输出89点。双stream融合后，15个command分为五指position、velocity、gate：

```text
p = 0.01 * position_command
v = 0.01 * velocity_command
g = sigmoid(gate_command)
前12个有效token: g = 0
b = g*v + (1-g)*p
native[t] = g[t]*native[t-1] + b[t], 初值0
ranking_score z[t] = native[t] / 0.01
```

源模型文件保留历史Force/normalized-Fz注释，不表示本次排序训练输出获得了N值标定。`load_model`包装器与`RankPredictor`已经执行最后的 `/0.01`，用户不要重复执行。

## 3. 原生标签与时间位置

窗起点s，第k个输出(k=0..88)比较的标签为：

```text
y[k,f] = raw_force[s + 470 + 40*k, f]
```

force保留raw数值，不标准化、不映射到统一分布。窗内50Hz，整条trial拼接后的间隔不一定均匀。末窗仅保留晚于已输出末时刻的预测。

## 4. 比较对

只比较同一trial、同一手指。用该trial/指的 `raw_force[470::40]` 计算：

```text
D = P95 - P5
eps = 1e-6 * max(abs(reference))
threshold = max(0.05*D, eps)
```

仅当 `D>eps` 且 `abs(y_i-y_j)>threshold` 时规定大小。相同/接近相同值无顺序约束。它决定监督对，不修改或删除raw记录。

同窗每指最多128对：token间隔5–15、16–40、41–88，各38/52/38对，档权重0.3/0.4/0.3。同trial跨窗每指最多128对，原始时间间隔至少200点(0.1s)。均无放回抽取，候选不足时不复制补足。

## 5. 损失

```text
s = sign(y_i-y_j)
L_pair = softplus(-s * (z_i-z_j))
L_group = 0.8*L_within + 0.2*L_cross + 0.001*mean(z^2)
```

各个有效pair先平均，再按时间档、手指、真实窗、组分层平均。缺失分量的剩余权重重新归一化。整组没有有效排序对则该组不提供梯度；dummy不参与。DDP/梯度累积按全局有效组数归一化。精确实现见 `training_reference/ap304_rank_loss.py`。

Pearson Corr、Kendall tau-b与MAPE均不进入梯度；Val选模使用Stage1–15宏平均tau-b。正排序只约束相对大小，不唯一决定分数的偏移与尺度；小幅L2正则控制幅度，但不是物理标定。

## 6. 明确没有的变换

- 没有force mean/std反归一化。
- 没有ex45的正标定、反标定或LUT。
- 没有负预测截零、MA7、2kHz插值或W100连续状态。
- 没有“score直接等于牛顿值”的公式。

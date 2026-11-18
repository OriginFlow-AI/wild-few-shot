# AP304 全量 raw 排序预训练，从随机初始化到 best_tau.pt

这是全量原 loss 与 A-loss 回归实验共用的第一阶段。seed 43，全模型随机初始化，训练 18,357,945 个参数。第一阶段学到同一 trial、同一根手指不同时刻的力大小顺序，输出无量纲排序分数。第二阶段取本实验选出的权重初始化主干，重置输出头，再学习标定后的牛顿值。[第二阶段训练包](../AP304_full_besttau_original_A/)

## 数据与训练

| 划分 | trial | 受试者 | 用途 |
|---|---:|---:|---|
| Train | 188,777 | 681 | 包含 stage 0–15，保留 4、5，使用全部训练 trial，拟合 EMG 归一化 |
| Val | 10,890 | 39 | 定期选模、早停 |
| Test | 11,130 | 18 | 选定权重后最终评估 |

各划分的受试者互斥。输入为 2 kHz、16 通道 EMG，每 4,000 个采样点独立重置。每个窗口在 start + 470 + 40k 处取 raw force，k = 0…88。末端补最后一个完整窗，评估拼接时去重。排序对仅在同 trial、同指内构造；具体间隔、采样配额和阈值见实际配置。

训练目标为同窗与跨窗成对 logistic 损失（权重 0.8、0.2），加 0.001 倍分数平方正则。AdamW，学习率 1e-4，weight decay 1e-4，2,000 步 warmup 后 cosine 衰减。8 GPU，BF16，每卡每微批 4 组双窗，累积 4 次，每次更新 128 组。每轮 45,321 次更新，最多 3 轮。[实际参数](resolved_spec.yaml) · [损失实现](source_snapshot/emg2force/ap304_rank_loss.py)

## best_tau.pt 怎样选出

1. 每轮在约 25%、50%、75%、100% 处评估完整 Val。对每条 trial、每根手指，沿去重后的时间序列计算预测分数与 raw force 的 Kendall τ-b。真值恒定的项排除；真值有变化而预测恒定的项在选模时记 0。
2. 在每个 stage 内，对有效的「trial × 手指」项等权取均值；再对 stage 1–15 中有有效项的 stage 等权平均。这个阶段使用完整 Val 的 stage 宏平均。第二阶段交接包里的薄膜、逐人等权 Test 指标另有用途。
3. 每次验证分数**严格高于历史最高值**就保存 best_tau.pt。另维护早停参考分数，超过参考值 0.001 才重置耐心计数；至少完成一轮后，连续 4 次验证未达到这个改善阈值即停止。
4. 本次训练进行了 10 次定期验证。**第 2 轮中点，第 67,982 步，Val τ-b = 0.4342530491895346**，为最高值。实际在第 113,303 步早停。随后重新加载胜出权重跑最终 Val 和 Test，Test τ-b = 0.554933407689383。Test 为完整评估集的 stage 宏平均。输出为无量纲分数，CSV 中的 MAPE 是对 raw 数值的诊断。
5. 权重 SHA-256 为 `85445c11c7b20764ddb04bbf7e4baedfb6ff269ac0c3ff608b84e6e7be15d22a`。已比对第一阶段 best_tau.pt 与第二阶段 inputs/ranking_best.pt 的文件哈希。仓库交付源码、配置和选模证据，权重文件保存在原运行目录。

核对时从 epoch_metrics.csv 取 split=val 且 label 为 update*_val 的行，对 tau_b 取最大值。final_val 和 final_test 是选定权重后的复测。[全部验证记录](epoch_metrics.csv) · [胜出轮逐 stage 指标](evaluations/update067982_val/summary.json) · [第一阶段选模回执](selected_checkpoint.json) · [第二阶段权重来源](stage2_weight_binding.json) · [评估与保存实现](source_snapshot/emg2force/ap304_rank_train.py)

## 复现入口

使用原始 split 索引及 EMG/force 缓存，环境见 [版本记录](environment_and_architecture.json)。运行还需 PyYAML 和 SciPy。下例在一个新输出目录写入可迁移配置，先完成数据准备，再启动八卡训练。历史流水线允许 Train 准备好后与 Val/Test 扫描重叠。历史参数原件保留在 resolved_spec.yaml。

```bash
python launch.py --index /path/to/data_splits/full/split.json --cache /data/syh_data/emg_force --output /path/to/new_rank_run
```

原训练器按 SHA-256 核验索引。输出的最佳权重位于新目录的 run/checkpoints/best_tau.pt。原始数据、索引与 GPU 运行环境需另行提供。

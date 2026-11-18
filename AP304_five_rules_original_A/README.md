# AP304 五条规则筛选后训练交接
先看《实验索引.csv》。每个 experiments 子目录对应一个真实运行，内含冻结的模型及训练源码、实际 YAML、初始化回执、运行环境、标定及准备统计。所有版本的指间配对损失系数为零。

## 选哪一组

|数据方案|原 loss|A-loss|初始化与选择权重|
|---|---|---|---|
|排除 stage 0、4、5 并做五规则筛选，93,788 条训练 trial|filtered_original|filtered_A|第一阶段最佳 τ 初始化；使用筛选后 Val 的 trial 等权目标选模。|
|同一五规则数据，全模型随机初始化| |filtered_A_scratch|唯一的从头训练实验。用筛选后 Val 的逐指、stage、受试者等权指标合成目标选模，最佳第 4 轮，第 7 轮早停。与 filtered_A 的选模方式也不同。|

## 训练参数和数据

原 loss = (1−r) + 0.3 M + 0.03 E；A-loss = 0.5 (1−r) + 0.1 M + 0.15 E。r 为训练微批内有效五指和时间点汇总后的 Pearson 相关系数；M 为 |真值| ≥ 0.5 N 的 MAPE 比例值；E 为标定 N 域 MAE。训练损失、验证选模目标、薄膜 Test 汇报是三个不同步骤。

使用 K2RingNetwork、16 通道 2 kHz EMG、每窗 4,000 点。seed 43，AdamW，主干/输出头学习率 2e-5/1e-4，weight decay 1e-4，warmup 1,000 步，梯度裁剪 1.0。8 GPU × 每卡 8 窗 × 累积 4，共 256 窗/更新，BF16。以各运行 resolved_config.yaml 和源码为准。

五规则在原始力上按顺序判定，任一规则命中即剔除整条 trial。指定指在 stage 10 为前三指、11–12 为前两指、其余为五指。① 任一指定指全零；② 任一指定指的非零点中，0 < F ≤ 0.5 N 占比 ≥ 80%；③ 任一指同一正值连续 ≥ 2,000 点；④ 任一指有点 > 60 N；⑤ 任一指前 5 秒众数不含零（并列含零保留）。精确定义见 five_rules.json。筛选系列沿用冻结的全训练集归一化。

## 启动

在解压目录中使用 Linux 和历史 PyTorch 环境。原环境的 Python 可执行文件为 /data/zzy_data/emg2force/.venv-py310/bin/python。环境版本见每组 environment.json。以下以 filtered_A 为例，其他实验替换为表中同名目录。

```bash
python train.py filtered_A --check
python train.py filtered_A --output /path/to/new_output
```

--check 校验索引、模型源码、EMG 归一化、筛选清单及所用第一阶段权重的 SHA-256，并加载该版本训练入口。--output 应指向尚未创建的目录。训练会在该目录创建运行版配置和检查点。该包交付训练代码及参数；数据缓存、split 索引、筛选清单和第一阶段 .pt 权重使用原配置指定的共享文件。换机时使用 --index、--cache-root、--source-checkpoint、--val-filtered、--test-filtered 指向同一文件的副本。缓存需保留大小及纳秒修改时间，以通过历史代码的样本身份检查。随机初始化组无需 .pt。

## 只汇报薄膜 Test

历史训练入口可能输出混合布局 Test。统一横向比较时先按每条记录的 layout=1x5x1 筛选，stage 取 1–15；排除 4、5 的比较要对所有参与模型应用相同 stage 子集。五规则模型区分完整 Test 与五规则 Test。在带逐指统计及 Kendall τ-b 的评估目录上运行：

```bash
python membrane_test.py /path/to/new_output/evaluations/final_test_filtered --exclude45
```

先逐 trial、逐指计算指标，对有效指等权；然后同一人同一 stage 内 trial 等权、同一人各有效 stage 等权、最后逐人等权。缺失 stage 跳过，每项指标单独排除无定义值。左右手试次数在 trial 层体现。只比较同一测试子集及同一汇总方式。部分早期原生评估只保留合并统计，需使用对应补测的逐指记录；缺少逐指数据的总分无法换算成本口径。早期另有“同一人同一 stage 内先对每只手每根指的 trial 平均，再对已有手指等权”的历史薄膜报告，引用旧分数时保留其口径标签。

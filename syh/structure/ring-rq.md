# Force3 架构消融：Ring-RQ

实验分支：[syh/structure/ring-rq](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/ring-rq)。

W304、9层 Encoder、4层 Decoder，使用 Ring-RQ 软读出：五指独立两级残差码本与冻结逐指 MLP。分支包含两阶段 Kendall/A-loss 训练、标定与本地数据读取、独立2秒窗口推理和 W100 连续状态推理。

- [代码与运行说明](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/ring-rq/ring_rq/README.md)
- [数据 schema、单位与标定语义](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/ring-rq/ring_rq/DATA_SCHEMA.md)
- [已发布最佳权重、norm与RQ资产](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/ring-rq/ring_rq/assets/phase2_ep1)

该分支发布的是旧架构训练包及已有固定权重结果，保留原3572点训练/评估和W100连续推理合同。新的受控架构消融将两臂统一为55个时刻，不能把历史模型成绩当成这一轮同协议基线。与[三视图架构](threeview.md)的同预算比较结论尚待完成。

# Force3 架构消融：三视图集成

实验分支：[syh/structure/threeview](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/structure/threeview)。

共享 Package W304/E9 Encoder，三个独立 W128/D2 读出分支分别固定屏蔽24个特征维，屏蔽集合互不重叠。三路共享冻结的 RQ/frame/Motion tokenizer，最终对各分支 fine 预测取均值，无GT历史输入。总参数26.106M，可训练25.986M。

- [架构、训练、推理与运行说明](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/threeview/threeview/README.md)
- [数据 schema、时钟和标定语义](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/threeview/threeview/DATA_SCHEMA.md)
- [训练协议](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/threeview/threeview/PROTOCOL.json)
- [代码验证范围](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/structure/threeview/threeview/VALIDATION.json)

每个2秒输入输出55个50Hz时刻，沿用 Kendall/A-loss 两阶段、辅助/主项梯度L2比7.5%、每卡256/global2048、人员隔离及完整Val选模。训练分别计算三路loss后平均，评估使用集成预测。

当前分支为代码发布，不含私有数据、split、标定表或模型权重。已通过CPU契约检查和有限单卡原执行代码逐位对拍/断点恢复；这些是工程验证，不是模型精度或[Ring-RQ](ring-rq.md)对照收益的结论。新环境训练仍需执行包内八卡恢复资格。

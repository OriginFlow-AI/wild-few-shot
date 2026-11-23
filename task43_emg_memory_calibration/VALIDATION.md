# 发布验证

2026-11-23完成。本次验证只使用合成EMG/Force与外部提供、SHA绑定的历史模型资产，没有读取真实查询或Test信号，没有启动正式训练或恢复已停止实验。

|检查|结果|
|---|---|
|CPU数据与准入测试|11项通过：本地mmap、C/Fortran布局S3 Range、单位、路径边界、split隔离和最近支持选择|
|冻结数值源码|10份模型/训练/采样/统计模块与来源逐字节一致；Calibration类AST一致|
|F完整恢复|真实global1024，连续2步与跨进程1＋1步完整状态逐位一致；21项Adam|
|B_L完整恢复|真实global1024，同样逐位一致；40项Adam|
|原生推理对照冻结原实现|4个合成查询窗，A/B_L/F最大绝对差均为0 N|
|共享Encoder CUDA Graph64对照原实现|三臂最大绝对差均为3.5762786865234375e-7 N，通过atol=rtol=1e-5|
|无支持回退|F与A逐位一致|
|支持重绑定|本人→无支持→本人，恢复后输出逐位一致|
|支持构造|推理记忆与训练静态支持库逐位一致|

恢复比较包含模型、Adam、Python/NumPy/CPU/CUDA RNG、采样计划与游标、scheduler、history和累计best，并覆盖smoke Train→Val。仅资格比较将墙钟字段置0，正式训练仍记录实际耗时。资格使用清单前两条查询、少量Val窗，不是完整Val。

GPU环境为RTX5090、Linux/Python3.12、Torch2.12.1+cu130、NumPy1.26.4、FP32/TF32关闭。有限资格分块32、Val分块8、显存分配上限6GiB；最终推理集成的CUDA allocated峰值2339935232字节，不能当作正式训练或其他设备的显存上限。CPU测试另在Windows/Python3.11.15、NumPy2.4.6通过。

恢复资格后仅增加推理端只读mmap数组的可写副本暂存，以及CLI登记Val-best功能；最终推理集成与冻结原实现对照重新通过，训练数值模块未变。结构化摘要见[VALIDATION.json](VALIDATION.json)，发布文件绑定见[RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)。运行新数据/配置时仍需执行自己的`qualify`，发布资格不替代消费者的输入审查。

这是工程发布验证，不是新的完整Val/Test结果、准确率结论或硬件无关性能保证。权重、norm/LUT和真实清单由资产所有者另行提供。

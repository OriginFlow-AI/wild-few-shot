# syh/task43-emg-memory-calibration

三臂为固定A-loss A_ep1基线、无个人标定的B_L LoRA＋完整Affine输出头、读取本人近期Stage3前60秒的F EMG记忆。

[代码与运行说明](../task43_emg_memory_calibration/README.md) · [数据接口](../task43_emg_memory_calibration/docs/DATA.md) · [来源与资产标识](../task43_emg_memory_calibration/PROVENANCE.json) · [验证](../task43_emg_memory_calibration/VALIDATION.md)

提供完整训练/恢复、只读mmap＋S3 Range、有序预读、静态支持缓存、完整Val和FP32共享Encoder CUDA Graph推理。权重、norm/LUT、真实清单/数据、缓存和凭据由所有者外部提供，不在此公开分支中。

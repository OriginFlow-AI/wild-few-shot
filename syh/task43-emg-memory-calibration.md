# Stage3 EMG记忆标定：A / LoRA / F

实验分支：[syh/task43-emg-memory-calibration](https://github.com/OriginFlow-AI/wild-few-shot/tree/syh/task43-emg-memory-calibration)。

同一A-loss A_ep1基座上，比较固定基线A、无个人标定的B_L LoRA＋完整Affine输出头，以及读取本人近期Stage3前60秒的F EMG记忆。

- [代码与运行说明](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/task43-emg-memory-calibration/task43_emg_memory_calibration/README.md)：训练、完整断点恢复、Val与原生/共享Encoder CUDA Graph推理。
- [数据接口](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/task43-emg-memory-calibration/task43_emg_memory_calibration/docs/DATA.md)：只读mmap/S3 Range、静态支持缓存与共同采样。
- [发布验证](https://github.com/OriginFlow-AI/wild-few-shot/blob/syh/task43-emg-memory-calibration/task43_emg_memory_calibration/VALIDATION.md)：CPU测试、合成GPU smoke、恢复及原实现推理对照。

权重、norm/LUT、真实数据清单、波形与凭据由所有者另行提供，不在该代码分支中。

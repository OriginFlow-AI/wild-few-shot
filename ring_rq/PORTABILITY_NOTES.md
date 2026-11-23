# 独立代码包的兼容性记录

## 2026-11-23：NumPy 2.4 删除私有 npy header 接口

原读取器使用 `np.lib.format._read_array_header`；本地 NumPy 2.4.6 上的 ZIP_STORED 合成读取测试报 AttributeError。代码包改用公开的 v1/v2 header API，其他版本走标准 `np.load`，保留 dtype/shape 和原滤波标记检查。数据布局、偏移定义、只读memmap和数值不变。已补充真实 ZIP_STORED 的合成回归验证。

这是独立发布副本的兼容性修正；没有修改服务器旧实验或活动训练源码。

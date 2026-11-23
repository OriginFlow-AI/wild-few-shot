# 数据接口与读取边界

## 清单

训练PLAN只有`train`和`val`角色；不含真实清单示例，个人标识由数据所有者私下提供。顶层schema为`stage3-prefix60-crossstage-v1`，每角色包含`records`、`supports`、`queries`；`summary[role].windows`必须闭合。

```text
records[record_key]:
  key:          安全的单段record_key，不含路径分隔符
  subject:      已解析受试者身份；同一人的不同别名需先统一
  stage:        权威旧benchmark整数stage，1..15
  hand:         left / right
  layout:       1x5x1（本训练人口为CMCU）
  n:            2kHz样本数
  duration:     n/2000，秒
  start:        Unix秒
  relative:     只读cache/S3对象的相对路径，不含.npy/.npz后缀
  content_hash: 上游冻结内容身份
supports[support_record_key]:
  record:       对应records键
  hand:         同记录手侧
  stop:         120000（固定前60秒）
queries[]:
  query:        查询records键
  support:      supports键
  start:        0
  windows:      floor(n/4000)
  cross_hand:   false
```

校验器检查人员/别名及内容跨split隔离、重复查询内容、shape/时长、Stage3支持与查询分离、Train排4/5、同人同手/同日/已完成支持与两小时限制。它不重新读取全部原始文件做SHA，也不能从裁剪清单证明未列出的其他Stage3不存在；最近支持和源版本需由完整上游元数据冻结。

## 本地优先

`DATA.json.local_roots`为一个或多个授权只读缓存根。默认字段：

```text
<root>/<relative>.emg.npy   float32 [n,16]
<root>/<relative>.fz.npy    float32 [n,5]，原始N，尚未LUT标定
```

也可在记录中提供`emg_path/fz_path`，但必须位于配置的local_roots中。使用read-only mmap，仅拷贝所需窗；不修改源数组、不搬移缓存或别名。`local_force_unit`默认N，显式mN等其他域会报错，避免重复除1000。

`materialize`从同一reader分块复制到当前运行目录的`waveform_cache`，只处理显式指定的Train/Val角色及EMG/Fz字段。此命令不做Force LUT，不读取视频；随后`targets`和支持库各消费原始N并恰好标定一次。不要把已经标定的N冒充原始N缓存。

## S3 API

本地缺失时，通过配置的bucket/prefix读取`<prefix>/<relative>.npz`：

- `emg.npy`：float32、2维、16通道；
- `cmcu05b_<hand>_force_mn.npy`：float32、至少5通道、原始mN，读取前5通道并乘float32 `.001`。

仅支持未加密、未压缩的`ZIP_STORED`成员及NPY1.0/2.0头。读取器解析ZIP与NPY偏移、C/Fortran布局；Fortran按通道列取Range，不能按C行顺序误读。核对206、Content-Range、长度、ETag和Last-Modified条件；冻结记录可额外绑定`etag`。拒绝shape错位、NaN/Inf、压缩成员和单位不符，不补0、不跳样、不自动SSH到其他机器。

对象头定位信息写入运行目录的`source_headers`；这不等于已缓存整个波形。默认Range读取不落盘整录制，只有显式`materialize`才写入波形副本。大数据与凭据不进入Git。

凭据二选一：环境`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`及可选`AWS_SESSION_TOKEN`；或外置`S3_CREDENTIALS_FILE`，JSON字段为`access_key_id`、`secret_access_key`及可选`session_token`。Linux上文件需本人所有且0600。endpoint通过`S3_ENDPOINT_URL`或私有DATA绑定设置，region默认us-east-1。客户端只提供HEAD/GET，日志不输出签名头、响应体或密钥。

## 标签与静态支持库

`targets/<role>/<query>.npz`包含`target[windows,89,5]`、`high[windows]`、`rest[windows]`。high/rest按原训练定义生成，供原75/12.5/12.5 sampler使用，不用Test统计拟合。

每份`.raw`支持库为2140880字节，按以下顺序无损存储：

|字段|shape|dtype|
|---|---|---|
|EMG融合特征＋RMS|5×128×320|little-endian float32|
|Force历史＋有效mask|5×128×2×256|float32|
|幅值摘要|5×128×4|float32|
|条目有效性|5×128|bool|
|覆盖摘要|5×4|float32|

库绑定冻结A特征、原norm/LUT、支持及清单。F训练每次重新编码可训练K/V；推理在冻结F后才允许缓存编码K/V。换上游权重、归一化、标定或支持选择必须重新物化/资格，不能沿用旧目录名假装兼容。

原实验使用16通道EMG与五指Fz；本包没有视频、IMU或BSQ归指解析器。直接实测BSQ或Task103局部CMCU标定是独立的数据合同，不能静默替代此原始CMCU标签接口。

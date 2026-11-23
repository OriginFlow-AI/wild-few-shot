# 读取接口和来源语义

## 清单层

`FULL_SPLIT.json.gz` 为 `{"revision":2,"splits":{"train":[...],"val":[...],"test":[...]}}`。保留原冻结schema；不在本仓库提供真实人员、路径或区间。

`INDEX.json.gz` 为 `{"train":[[[start,end],...],...],"val":[...]}`，每项与相应split记录严格同序，区间为2kHz网格半开整数区间，长度至少4000。窗口必须完整包含于某一个合格区间，不能跨断流。没有INDEX时，`prepare_assets.py --build-index` 从记录中已审计interval及时钟计算。

共同记录字段：`id`、`subject`、`session`、`split`、权威`stage`、`component`、`source_uri`、`local_files`、`hands`；`hands[0]`含`side/intervals`。统计生成还使用`effective_hand_ns/source`。记录与`LABEL_BINDINGS.rows[id]`的`source_uri`必须相同。

## 三类已物化数据

| 类型 | `local_files` | 输入与区间 |
|---|---|---|
| 旧Force NPY对 | `emg`, `fz` | EMG float32[N,16]、Force float32[N,5]，均为既有2kHz对齐processed缓存；interval为采样索引 |
| Task43 processed MCAP | `mcap` | 只接受原schema、单位和processed标记；16路EMG uV、CMCU9原mN取前五通道转N；同2kHz网格；需要`samples_2khz`和原准入回执 |
| Task83 owner NPZ | `npz` | `emg.npy` float32[N,16]、`force.npy` float32[M,5]；ZIP comment=`old_sos_2khz_v1`；interval为绝对ns |

Task43记录的`local_inventory_index`索引`TASK43_LOCAL_INVENTORY.rows`，每个`files`项含`path/dev/ino/bytes/mtime_ns`，并核对`source_uri`。Task83记录直接提供`local_stat={dev,ino,bytes,mtime_ns}`。读取先检查缓存身份；不会修改文件。

Task83的ZIP_STORED成员按npy header直接只读memmap，不拆包；压缩成员回退为内存读取。原生Force仍按其独立时钟存储，不能假定和EMG同长。`CLOCK_BINDINGS.rows[id]={path,sha256}`引用的npz包含`emg_t0_ns/emg_dt_ns/force_t0_ns/force_dt_ns`；时间由整数原点和累积差分恢复，先减整数原点再转换float64插值，禁止外推和跨缺口取邻居。

## 力标签路由

`LABEL_BINDINGS.json.gz`含`rows[id]`，每项包含匹配的`source_uri`、`input_unit="N"`、`layout`、`label_domain`。

- `layout="1x5x1", label_domain="raw_extracted_N"`：膜式原N提取值，经旧ex45 LUT映射一次；超过覆盖范围按最后实际LUT段斜率线性延伸。负值/非有限值拒绝。
- `layout="1x5x3", label_domain="raw_extracted_N"`：gen3原生带符号Fz，恒等复制。
- `label_domain="ex45_calibrated_N"`：仅允许提供精确匹配的`policy_sha256`，恒等复制，避免再次标定。
- 未知域、单位或布局拒绝，不能按录制日期、task、stage或数值大小猜测。

映射发生于只读源加载后的独立内存数组，先于插值、norm、RQ拟合和损失计算。缓存命中复用已映射内存；缓存淘汰后重读原件再映射一次，不写回NPY/NPZ/MCAP。

`LocalReader.load`和训练批读取器拒绝模型Test信号。split元数据中的Test人员仅用于隔离检查；本代码包没有打开Test的启动器。

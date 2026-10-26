# 单模型共享权重实验：复现实测与证据

此文档记录本次 PR #2 新增共享权重策略的架构、复现方法及 2026-10-26 的真实模型结果。
CPU、GPU 主对照与额外 18 请求 stream 对照已完成实测；脚本语法检查、调度单测和合成数组检查与实机测量分开记录。
`shared` 仍需显式启用，**默认保持 `process`，没有游戏内验证**。

## 架构与可变状态

原 `process` 策略让每个并行槽在独立进程中加载完整模型。实验 `shared` 策略只有一个权重 owner，
为每个调用线程构造独立请求 view，并通过 `SharedInferencePool` 限制同时进入推理的请求数量。
CPU 使用线程并行；GPU 可为每个 view 分配独立 CUDA stream，也可用默认 stream 做对照。

view 递归复制 Python `nn.Module` 外壳、`__dict__`、子模块映射和预处理对象，
注册的 parameter / buffer 继续引用 owner 的同一 Tensor 与 storage；经检查的只读 TorchScript MHR 也共享。
这会隔离上游运行期写入的 `batch`、`image_embeddings`、`output`、`prev_prompt`、`is_crop`，
以及模型的 `_person_valid`、`_batch_size`、`_max_num_person`、`hand_batch_idx`、`body_batch_idx` 等请求字段。
它并非让多个线程直接调用同一个有状态 estimator。

设备适配只编译当前 view 使用的私有函数副本：将上游写死的 `.cuda()` / `recursive_to(..., "cuda")`
改为实际设备，并去掉每张图的 `torch.cuda.empty_cache()`。不会在线程中改写全局 `torch.Tensor.cuda`
或上游模块的函数对象。原串行封装仍作为单独参考，因此要用 shared1 区分这些适配与并发本身的影响。

加载时核对已审阅上游文件的规范化 AST 指纹，变化后拒绝静默启用；检查 MHR 内联图中的持久状态写入，
拒绝模块 hooks、训练态和未冻结配置。推理时核对注册 Tensor 是否仍为同一对象，并检查版本计数。
这些检查有明确审计范围，不能代替所有原生算子的完整别名分析或任意上游版本验证。

加载、卸载与配置切换和活动推理互斥；`close()` 停止新请求并等待现有请求结束。
`release_after=0` 要等同组活动请求全部完成后才卸载，避免清掉其他请求正在使用的权重。
CUDA 请求在成功和异常路径都等待自己的 stream 完成；若 stream 同步失败，owner 会拒绝后续请求，直到卸载重载。

## 可选启用与回退

**默认仍为 `process`**。默认并行数 1 沿用原串行封装；`process` 并行数大于 1 时使用原多进程池。
`shared` 即使并行数为 1 也走实验 owner。插件 UI 不会自动替用户切到实验策略。

对包含本次共享策略实现的后端，可显式启用：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:42430/model/config `
    -ContentType application/json `
    -Body '{"recognition_strategy":"shared","recognition_parallelism":3,"release_after":-1}'

# 回到原策略；切换会等待活动推理结束并卸载旧模型。
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:42430/model/config `
    -ContentType application/json `
    -Body '{"recognition_strategy":"process","recognition_parallelism":1,"release_after":0}'
```

也可在启动后端前设置环境变量：

```powershell
$env:POSEGRAPH_RECOGNITION_STRATEGY = 'shared'
$env:POSEGRAPH_RECOGNITION_PARALLELISM = '3'
$env:POSEGRAPH_SHARED_CUDA_STREAMS = '1'
$env:POSEGRAPH_RELEASE_AFTER = '-1'
& backend/.venv-shared-cuda/Scripts/python.exe -m posegraph.app
```

命令需使用已安装本次源码的 Python 环境，并配置已有本地源码 / 权重路径。
`POSEGRAPH_SHARED_CUDA_STREAMS=0` 是共享默认 stream 对照；此选项从启动环境读取，改变后需重启后端，
不是 `POST /model/config` 的字段。HTTP 响应或 `/health` 中的 `model.config.recognitionStrategy`
与 `recognitionParallelism` 可用于确认所选策略。实测统一常驻模型（`release_after=-1`），不把重新加载时间混进正式吞吐。

## 对照设计

每一行必须在**独立 Python 进程**中顺序运行，避免上一种策略的 CUDA 上下文、缓存和模型干扰下一种策略。

| 策略 | workers | 用途 |
|---|---:|---|
| serial | 1 | 原 `Sam3dBodyEstimator`，原封装参考 |
| shared | 1 | 单权重、请求状态隔离的实验适配；隔离适配本身带来的性能变化 |
| shared，独立 CUDA streams | 2、3 | 多请求同时推理，观察增加并发槽后的变化 |
| shared，默认 CUDA stream | 2、3 | 用 `--no-cuda-streams` 分离多线程与独立 stream 的影响 |
| process | 2、3 | 原隔离进程池，各进程加载独立模型；内存不足也是应记录的实验结果 |

并发收益首先比较 `shared N / shared 1`；`shared 1 / serial 1` 是适配收益。
不能把 `shared 3 / serial 1` 的全部差异归因于并发。
CUDA stream 对照需保持 workers、设备、精度、MHR 位置、输入和线程数一致。

默认 MHR 留在 CPU，CPU intraop 线程数固定为 4，interop 为 1，OpenCV 为 1。
进程策略中每个模型进程都采用相同 intraop 线程数，因此其总 CPU 线程预算会随进程数增加。
若改用 `--no-mhr-on-cpu`，必须为所有对照重新运行，不能混用结果。

## 本地资产与输入

脚本只加载已存在的 SAM 源码、`model.ckpt`、`assets/mhr_model.pt` 与模型配置。
不会构造外部 detector、SAM2 segmentor 或 FOV estimator；默认按整图单人范围处理。
脚本启用 Hugging Face / Transformers 离线环境变量，缺失资产会直接失败，不主动下载新权重。

默认输入为指定源码目录中的 `notebook/images/dancing.jpg`。
`--variants` 额外构造水平翻转和四边各去掉约 1/12 的确定性裁剪，保存到结果旁的 `inputs/`。
每个输入都有独立索引、原文件 SHA-256、解码后 RGB SHA-256 与尺寸。
默认原图 / 翻转 / 裁剪各运行两遍，共 6 次正式请求，用于检查不同请求之间是否串扰。
也可用 `--images a.jpg b.jpg c.jpg` 指定多张本地图。

## 运行命令

在实验工作树根目录执行。Python 环境需包含现有后端依赖、与设备匹配的 Torch / torchvision；
`psutil` 用于采集 RSS、USS 与 Windows 私有提交量，缺少时报告明确标记不可用。

```powershell
$python = 'backend/.venv-shared-cuda/Scripts/python.exe'
$common = @(
    '--repo', 'D:/path/to/sam-3d-body',
    '--checkpoint', 'D:/path/to/sam-3d-body-vith',
    '--device', 'cuda', '--precision', 'auto',
    '--variants', '--repeats', '2', '--warmup', '1',
    '--cpu-threads', '4', '--mhr-on-cpu', '--nvidia-smi'
)

& $python backend/scripts/bench_shared_model.py @common --strategy serial --workers 1 --output backend/outputs/shared-model/serial-cuda1.json
& $python backend/scripts/bench_shared_model.py @common --strategy shared --workers 1 --output backend/outputs/shared-model/shared-cuda1.json
& $python backend/scripts/bench_shared_model.py @common --strategy shared --workers 2 --output backend/outputs/shared-model/shared-cuda2.json
& $python backend/scripts/bench_shared_model.py @common --strategy shared --workers 3 --output backend/outputs/shared-model/shared-cuda3.json
& $python backend/scripts/bench_shared_model.py @common --strategy shared --workers 2 --no-cuda-streams --output backend/outputs/shared-model/shared-cuda2-default-stream.json
& $python backend/scripts/bench_shared_model.py @common --strategy shared --workers 3 --no-cuda-streams --output backend/outputs/shared-model/shared-cuda3-default-stream.json
```

逐条检查退出码后再执行下一条；不要把不同策略同时放到后台运行。
`process` 对照使用同一 `$common`，单独指定 `--strategy process --workers 2` 或 `3` 与不同结果文件。
CPU 对照换用相同依赖的环境、`--device cpu --precision fp32`，其余配置保持一致；CPU 不需要 `--nvidia-smi`。

输出强制位于当前实验工作树忽略的 `backend/outputs/`，不把原图变体、权重、NPZ、运行日志或主机信息提交到 Git。
覆盖同名结果会丢失旧样本，复测请使用新的结果文件名。

## 指标与边界

JSON 与同名 NPZ 同时保存。JSON 中记录：

- 输入准备时间、模型准备 / 加载时间、预热时间、正式请求总墙钟时间。
- 正式请求吞吐（images/s），以及从实际开始调用 estimator / pool 到完整结果返回的每图延迟。
  `queue_wait_ms` 独立记录执行器排队时间；它未混入服务耗时。
- 请求的图片索引、重复索引、人数、实际设备和模型内部计时；完成顺序不用于匹配输入。
- 模型准备后内存、进程树 RSS / USS / Windows 私有提交量采样峰值与完整采样序列。
  RSS 相加可能重复统计共享页；`uss_bytes` 是唯一驻留页（Windows 为私有工作集），
  `private_bytes` 是 Windows PrivateUsage 私有提交量，不能把两者当作相同口径。
  采样峰值不是操作系统保证捕获到的瞬时最大值；正式阶段保留的输出数组也计入内存。
- CUDA allocator 的 peak allocated / reserved：单进程在预热后重置；进程策略从实际模型子进程收集，
  **不能把父进程的 CUDA allocator 数字当成全部子进程显存**。
- 可选 `nvidia-smi` 采样为整个 GPU 的占用与利用率，包含其他进程；与 Torch allocator 指标分开解释。
- serial / process 模型参数和 buffer 的 `data_ptr` 样例、指针摘要与唯一 storage 字节数；
  对包装器隐藏的 CPU MHR 也统计 storage。shared 使用实验 owner 的 diagnostics，
  包括线程 view 是否共享参数 / buffer storage、MHR identity、stream handle。
  **指针相同仅在同一进程内有意义，不能拿不同进程的地址数值直接证明共享。**

加载 / 冷启动高峰和正式推理高峰应分开看。默认采样区间 0.2 秒，可用 `--sample-interval` 调整。
小样本 p95 只反映这 6 次请求，不代表稳定生产分位数。

## 完整数值比较

NPZ 按 `r重复索引_i图片索引_h人物索引_字段` 保存完整关节坐标、关节全局旋转、关键点、
边界框、相机平移、焦距和静止参考数组，使用普通数值数组，读取时禁用 pickle。
JSON / NPZ 内的 manifest 保留索引与输入 SHA 映射。

```powershell
& $python backend/scripts/bench_shared_model.py `
    --compare backend/outputs/shared-model/serial-cuda1.json backend/outputs/shared-model/shared-cuda3.json `
    --atol 0.0001 --rtol 0.0001 --output backend/outputs/shared-model/compare-serial-shared3.json
```

也接受 `.npz` 路径。比较器检查输入映射、重复 / 图片键、人数、字段、shape、有限数值，
对每一数组计算 `allclose` 与最大绝对差；不只比较输出 hash 或少量关节。
失败时退出码为 1。应分别比较 shared 1、shared 2、shared 3、默认 stream 对照和 process 对照。
CPU 与 GPU 默认精度不同，不应用跨设备差异直接判断线程串扰。

`atol` 的含义随字段单位而异：坐标是米，旋转是无量纲矩阵元素，像素关键点与焦距是像素。
报告需列出这些字段的实际最大差；若调整阈值，应给出原因，不能仅为获得绿色通过而放宽。

## CPU 真实模型结果

环境：Windows，Intel Core i7-13700KF，Torch `2.11.0+cpu`，fp32；每个模型使用 4 个 intraop 线程，
interop 为 1。无外部 detector。相同 dancing 原图 / flip / crop，各重复 2 次；主表统一 warmup 1 轮。
所有配置均识别出每图 1 人，输入索引与 RGB SHA 映射一致。

原始 JSON / NPZ 在本地 `backend/outputs/shared-model/`，未提交其中的路径、进程号或主机信息。
主表依次来自 `serial-cpu1-t4.json`、`shared-cpu1-t4.json`、`shared-cpu2-t4.json`、
`shared-cpu3-t4.json`、`process-cpu3-t4.json`；完整比较来自相应 `compare-*.json`。

| 配置，warmup1 | 模型准备秒 | 正式墙钟秒 | images/s | 平均 / p95 延迟秒 |
|---|---:|---:|---:|---:|
| serial 1 | 22.830 | 66.984 | 0.089574 | 11.164 / 12.966 |
| shared 1 | 7.522 | 71.979 | 0.083358 | 11.996 / 12.352 |
| shared 2 | 7.832 | 45.385 | 0.132201 | 15.099 / 15.787 |
| shared 3 | 7.028 | 40.697 | 0.147429 | 20.305 / 20.461 |
| process 3 | 10.279 | 39.293 | 0.152698 | 19.541 / 19.697 |

shared3 吞吐是 shared1 的 **1.769 倍**、原串行的 **1.646 倍**。
相对 process3，shared3 吞吐低 **3.45%**；并发提高了批量吞吐，但单次请求延迟也从约 11–12 秒增加至约 20 秒。
这些是该六次请求样本的观察，不是长期服务容量或 SLA。
首次 serial 加载为 22.830 秒，后续运行多为 7–10 秒；没有控制操作系统文件缓存冷热状态，
不能据此宣称 shared 让模型加载快了三倍。

### CPU 内存与共享证据

单位 GiB（2³⁰ 字节）。主表采用结束阶段的 `memory_before_close` 快照；
此时模型尚未卸载，执行器已结束，仍保留待存档的输出和 allocator 缓存。
“加载后 RSS”为 `memory_after_load`；“全程 RSS 峰”为离散采样最大值，包含加载阶段。

| 配置 | 加载后 RSS | 结束 RSS | 结束 USS | 结束私有提交 | 全程 RSS 采样峰 |
|---|---:|---:|---:|---:|---:|
| serial 1 | 4.337 | 4.522 | 未采集 | 5.790 | 5.057 |
| shared 1 | 4.369 | 4.541 | 4.543 | 5.818 | 4.573 |
| shared 2 | 4.318 | 4.629 | 4.631 | 5.957 | 4.746 |
| shared 3 | 4.370 | 4.740 | 4.742 | 6.141 | 5.012 |
| process 3 | 13.073 | 12.812 | 12.562 | 16.640 | 14.038 |

相对 process3，shared3 结束时 USS 低 **62.25%**，私有提交从 16.640 GiB 降至 6.141 GiB。
shared1/2/3 的 owner 均只加载 **1 次**，唯一参数 / buffer storage 均为 **3,671,687,535 字节**（约 3.420 GiB），
并未随线程数复制权重。每个 view 都验证 **1079** 个 Tensor 引用与 owner 相同；shared3 的三个 view 使用同一个 MHR 对象。
process3 的三个模型子进程各报告同等唯一 storage，总权重 storage 约三倍。
Tensor 计数含别名枚举口径差异，判断权重副本应看唯一 storage 与每个 view 的共享验证，不比较跨进程指针数值。

采样有以下边界：

- RSS、USS 和私有提交在同一函数中也由不同 API 依次读取，进程树还逐进程遍历；快照不是原子的。
  各指标的峰值又独立取最大值，因此 process3 的峰 USS 大于峰 RSS 不构成额外内存泄漏证据。
- 名义采样周期为 0.2 秒，正式阶段最大实际间隔约 0.249 / 0.283 / 0.307 / 0.295 / 0.724 秒，
  对应上表五种配置；USS 查询与监控本身有开销，可能漏掉瞬时峰值。
- process3 正式推理阶段 RSS / USS 采样峰为 **13.002 / 12.756 GiB**，其全程最高值来自加载阶段。
  shared3 正式阶段对应 **4.956 / 4.958 GiB**。不要把加载峰与另一配置的 warm 峰直接比较。
- serial 最先运行时脚本尚未加入 USS 采集，因此该项是未采集，不能从 RSS 估算填补。
  serial 与后续配置的监控开销也略有不同。
- process3 每个模型各有 4 个 intraop 线程；shared 的模型算子也可能由多个请求同时占用线程资源。
  线程数参数相同不等于整个进程树 CPU 预算或实际调度完全相同。

CPU shared3 的六次正式请求与原串行参考的九个存档字段逐元素一致，最大绝对差为 0。
这一观察仅覆盖本次输入、配置和运行轮次。

## GPU 真实模型结果

GPU 为 NVIDIA GeForce RTX 4070 Ti，12 GB；Torch `2.11.0+cu128`、torchvision `0.26.0+cu128`。
实际设备为 CUDA，骨干 bf16，MHR 在 CPU；CPU intraop 仍为 4。与 CPU 对照使用相同三种输入、各重复 2 次、warmup 1 轮。
原始证据为 `serial-cuda1-t4.json`、`shared-cuda{1,2,3}-t4.json`、
`shared-cuda{2,3}-default-t4.json`、`process-cuda3-t4.json` 及对应 NPZ / compare JSON。

| 配置，warmup1 | 模型准备秒 | 正式墙钟秒 | images/s | 平均 / p95 延迟秒 | 与 serial 数值比较 |
|---|---:|---:|---:|---:|---|
| serial 1 | 32.065 | 10.035 | 0.597924 | 1.672 / 1.778 | 参考 |
| shared 1 | 17.794 | 9.606 | 0.624578 | 1.601 / 1.675 | 九字段逐元素一致 |
| shared 2 streams | 8.519 | 7.775 | 0.771693 | 2.591 / 2.611 | 九字段逐元素一致 |
| shared 3 streams | 8.480 | 6.769 | 0.886410 | 3.329 / 3.618 | 九字段逐元素一致 |
| shared 2 default stream | 9.021 | 7.900 | 0.759520 | 2.632 / 2.956 | 九字段逐元素一致 |
| shared 3 default stream | 9.509 | 7.295 | 0.822429 | 3.548 / 3.671 | 九字段逐元素一致 |
| process 3 | 12.764 | 6.012 | 0.998031 | 3.004 / 3.109 | 九字段逐元素一致 |

shared1 相比原串行吞吐高 **4.46%**，这是适配路径与本轮测量的综合差异。
shared2 / shared3 相比 shared1 分别为 **1.236 / 1.419 倍**；shared3 相比原串行为 **1.482 倍**。
process3 仍更快：shared3 的吞吐约为其 **88.82%**，即低 **11.18%**。
单次请求平均延迟也从约 1.6 秒变为约 3.3 秒，因此该取舍是批量吞吐与内存，而非每图延迟降低。

首轮独立 streams 对默认 stream 的吞吐差为 workers2 **+1.60%**、workers3 **+7.78%**。
这只有六次请求和一个运行顺序，尚不足以称为稳定的 stream 收益；下文以更多请求并先运行默认 stream、再运行独立 streams 进行复查。
GPU 的首次加载 / 模型准备同样受缓存和 CUDA 初始化顺序影响，不能拿首两行与后续行直接归因为加载优化。

### GPU 显存与主机内存

以下 allocator 数字为 GiB，含已加载权重与 warm 推理激活；nvidia-smi 为正式推理阶段整卡采样最大值，单位 MiB。
它包含桌面及其他进程，不能当成本程序独占显存。
主机内存为结束阶段快照，单位 GiB；私有提交不等于物理驻留 RAM。

| 配置 | CUDA peak allocated / reserved GiB | 整卡 warm 采样峰 MiB | 结束 RSS / USS GiB | 结束私有提交 GiB |
|---|---:|---:|---:|---:|
| serial 1 | 1.693 / 1.824 | 5080 | 2.947 / 2.931 | 5.848 |
| shared 1 | 1.692 / 1.846 | 5157 | 2.944 / 2.928 | 5.857 |
| shared 2 streams | 1.778 / 1.955 | 5291 | 2.948 / 2.932 | 5.983 |
| shared 3 streams | 1.823 / 2.064 | 5423 | 2.969 / 2.953 | 6.121 |
| shared 2 default stream | 1.778 / 1.928 | 5245 | 2.958 / 2.942 | 5.958 |
| shared 3 default stream | 1.837 / 1.998 | 5334 | 2.948 / 2.932 | 6.017 |
| process 3 | **各 worker 峰值之和** 5.079 / 5.521 | 9335 | 9.259 / 8.427 | 18.423 |

process3 的三份 allocator allocated 峰各约 1.693 GiB；reserved 分别约 1.842 / 1.840 / 1.840 GiB。
表中求和仅用于说明多模型各自的资源量，**不是同一时刻的总峰值**，不能拿该和与单进程峰值直接声称测得了某个同时峰值降幅。
整卡正式阶段采样峰则从 process3 的 9335 MiB 变为 shared3 的 5423 MiB；背景用量与离散采样限制仍适用。
主机结束 USS 从 8.427 GiB 变为 2.953 GiB，减少约 **64.96%**。

CUDA shared1/2/3 与默认 stream 对照的 owner 都只加载一次，唯一 storage 均为 **2,408,721,775 字节**：
其中 CUDA **1,714,760,400 字节**（约 1.597 GiB），CPU **693,961,375 字节**（约 0.646 GiB）。
权重 storage 不随 view 数增加。三个独立-stream view 各验证 **1024** 个注册 Tensor 共享，使用三个不同的 stream；
默认-stream view 的 handle 均为 0。GPU 模式下各 view 有独立 CPU-MHR Python 包装器，但内部 TorchScript MHR identity 相同，
没有为三个请求额外加载三份 MHR。与 CPU 的 storage 大小不同，主要因 GPU 对照骨干采用 bf16，不能跨精度直接归因共享。

所有六组候选 GPU 对照的六次正式请求，人数、输入映射与九个存档字段都逐元素一致，最大绝对差为 0；
原始比较器的 `atol=rtol=1e-4` 检查也全部通过。这是当前输入与运行轮次的真实模型证据，不代表所有图片和长期线程交错都已覆盖。

### 额外 18 请求 stream 对照

复查配置为 workers3、warmup2、三种输入各重复 6 次，先默认 stream、再独立 streams。
结果文件为 `shared-cuda3-default-long-t4.json` / `shared-cuda3-streams-long-t4.json`。

| 18 请求，warmup2 | 正式墙钟秒 | images/s | 吞吐增幅 | 批量耗时减幅 | 两配置完整数组比较 |
|---|---:|---:|---:|---:|---|
| default stream，先运行 | 21.152 | 0.850999 | 基准 | 基准 | 参考 |
| 独立 streams，后运行 | 17.949 | 1.002845 | **17.84%** | **15.14%** | 18 请求九字段逐元素一致 |

`compare-cuda3-long.json` 显示所有请求、所有存档字段最大差为 0。
本轮独立 streams 吞吐高 **17.84%**；与首轮六请求的 +7.78% 分开报告，不混算均值或把后一轮 shared3
与前一轮 process3 直接当作同条件胜负。两轮独立 streams 都更快，但样本仍短、运行顺序固定，
无法由此给出稳定生产增益或统计显著性结论。

## 相对共享单任务的并行收益

以下统一以**相同设备的 shared1** 为基准，使用主表的 6 次正式请求、warmup1 数据，
不含模型加载和预热。末 USS 使用 `memory_before_close`，资源增幅从原始字节计算后再舍入。
这里不混入原串行封装与 shared1 本身的适配差异，也不把不同设备、精度或额外 18 请求对照混算。

| 设备 / 并行数 | 6 请求耗时秒 | 吞吐增幅 | 批量耗时减幅 | 末 USS GiB | USS 增加 GiB / 百分比 |
|---|---:|---:|---:|---:|---:|
| CPU shared1 | 71.979 | 基准 | 基准 | 4.543 | 基准 |
| CPU shared2 | 45.385 | **58.59%** | **36.95%** | 4.631 | **+0.089 / +1.95%** |
| CPU shared3 | 40.697 | **76.86%** | **43.46%** | 4.742 | **+0.199 / +4.39%** |
| GPU shared1 | 9.606 | 基准 | 基准 | 2.928 | 基准 |
| GPU shared2 | 7.775 | **23.55%** | **19.06%** | 2.932 | **+0.004 / +0.14%** |
| GPU shared3 | 6.769 | **41.92%** | **29.54%** | 2.953 | **+0.025 / +0.86%** |

GPU shared 策略在同一进程内统计 warm allocator 峰值，因此可用相同口径计算增幅：

| 并行数 | peak allocated GiB | allocated 增加 GiB / 百分比 | peak reserved GiB | reserved 增加 GiB / 百分比 |
|---|---:|---:|---:|---:|
| shared1 | 1.692 | 基准 | 1.846 | 基准 |
| shared2 | 1.778 | **+0.085 / +5.03%** | 1.955 | **+0.109 / +5.93%** |
| shared3 | 1.823 | **+0.131 / +7.74%** | 2.064 | **+0.219 / +11.85%** |

GPU shared1 allocated 的原始值为 **1,817,278,976 字节 / 1.692472935 GiB**，按三位小数显示为 **1.692 GiB**。
百分比以原始数值计算，不用表格的舍入值反推。
process3 的性能和内存保留在前文独立对照；其 GPU allocator 是三个 worker 各自峰值之和、并非同时峰值，
不在本增幅表中与单进程 shared1 混算。

## 真实 HTTP 集成

使用隔离 CUDA 环境启动单个临时后端，绑定仅本机可访问的随机端口，不占用已有后端端口。
启用 shared / workers3 / 独立 streams / MHR CPU / 常驻模型，使用与基准 RGB SHA 完全相同的三种输入，
以无损 PNG 并发调用 `POST /sessions`。本次检查通过：

- 健康状态采样观察到共享 owner 同时处理 **3** 个请求；加载模型副本为 **1**，返回三个独立会话。
- 下载、解包三个会话归档，当前归档提供的 **8** 项数值字段与 GPU 基准逐元素一致，最大绝对差 **0**。
  基准的第九项 `keypoints_3d` 未由现有 HTTP 归档保存，因此未将它计入 HTTP 对照；基准本身已检查九项。
- 删除会话后读取返回 404；卸载后 `loaded=false`、`modelCopies=0`、活动推理为 0。
- 临时进程已结束、监听已清理，没有调用游戏 bridge。

本地证据为 `backend/outputs/shared-model/http-integration-*.json` 及同名前缀日志，未提交主机路径、进程信息或归档文件。
该结果验证实际 HTTP、线程池、会话与卸载链路，仍不代表游戏内操作已验证。

## 验证层级与未覆盖范围

- 脚本语法 / CLI、合成数组比较与假模型生命周期测试只验证工具和调度边界，不当作真实模型性能。
- 上述 CPU / GPU 性能表来自真实本地 SAM 权重；CPU shared3 与 GPU 对照的数值一致性来自完整关节 / 旋转等存档数组。
- 45 项后端测试在隔离 Python 环境通过，覆盖隔离、生命周期、局部配置更新和 API 等基础边界；它们与上述真实权重运行分开解释。
- GPU 主对照、18 请求 stream 复查与后端 HTTP 真实集成均已完成。
- 未测试游戏内按钮、图库并行交互、Brio、与游戏争用 GPU 时的体验或长时间服务。

本次提交增加可显式启用的 shared 策略，默认仍为 process。即使当前三张图的结果一致，也不能由此推出所有输入、
TorchScript 内部行为、图形驱动和长期并发都安全；新增上游版本还需重新审计可变请求状态及共享 buffer。

## 4～8 并发补测：固定 24 次识别

CPU / GPU 分别新增共享 N1、N3 对照及 N4～8 实测。每档独立进程一次，原图 / flip / crop 各重复 8 次，
固定 warmup2、intraop4 / interop1、MHR CPU，GPU 独立 streams；保持模型常驻，正式耗时不含加载和预热。
以下增幅统一相对同设备本轮 N1，不混用前文 6 / 18 请求。USS 为私有驻留快照，资源单位 GiB。

### CPU

| 并发 N | 24 请求秒 | images/s | 吞吐增幅 | 批量耗时减幅 | 末 USS GiB | USS 增加 GiB / % | 数值比较 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 294.805 | 0.081410 | +0.00% | 0.00% | 4.488 | +0.000 / +0.00% | EXACT 216/216 |
| 3 | 177.197 | 0.135443 | +66.37% | 39.89% | 4.800 | +0.311 / +6.94% | EXACT 216/216 |
| 4 | 154.145 | 0.155697 | +91.25% | 47.71% | 4.875 | +0.386 / +8.60% | **FAIL**（已比216） |
| 5 | 146.263 | 0.164088 | +101.56% | 50.39% | 4.948 | +0.460 / +10.24% | EXACT 216/216 |
| 6 | 142.676 | 0.168214 | +106.63% | 51.60% | 5.052 | +0.564 / +12.56% | EXACT 216/216 |
| 7 | 167.604 | 0.143194 | +75.89% | 43.15% | 5.134 | +0.645 / +14.38% | EXACT 216/216 |
| 8 | 164.471 | 0.145922 | +79.24% | 44.21% | 5.225 | +0.737 / +16.42% | EXACT 216/216 |

### GPU

| 并发 N | 24 请求秒 | images/s | 吞吐增幅 | 批量耗时减幅 | 末 USS GiB | USS 增加 GiB / % | 数值比较 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 34.941 | 0.686868 | +0.00% | 0.00% | 2.932 | +0.000 / +0.00% | EXACT 216/216 |
| 3 | 25.303 | 0.948507 | +38.09% | 27.58% | 2.949 | +0.017 / +0.58% | EXACT 216/216 |
| 4 | 24.665 | 0.973037 | +41.66% | 29.41% | 2.972 | +0.041 / +1.39% | EXACT 216/216 |
| 5 | 24.576 | 0.976545 | +42.17% | 29.66% | 2.993 | +0.061 / +2.07% | EXACT 216/216 |
| 6 | 26.199 | 0.916069 | +33.37% | 25.02% | 2.996 | +0.064 / +2.20% | EXACT 216/216 |
| 7 | 26.713 | 0.898428 | +30.80% | 23.55% | 3.007 | +0.076 / +2.58% | EXACT 216/216 |
| 8 | 25.774 | 0.931172 | +35.57% | 26.24% | 3.001 | +0.070 / +2.38% | EXACT 216/216 |

### GPU warm allocator 峰值

allocated 含已加载权重与推理激活；reserved 还包含缓存预留，两者均不是整卡独占显存。

| N | allocated GiB| 增加 GiB / % | reserved GiB | 增加 GiB / % |
|---|---:|---:|---:|---:|
| 1 | 1.692 | +0.000 / +0.00% | 1.848 | +0.000 / +0.00% |
| 3 | 1.845 | +0.153 / +9.02% | 2.070 | +0.223 / +12.05% |
| 4 | 1.895 | +0.202 / +11.96% | 2.182 | +0.334 / +18.08% |
| 5 | 1.962 | +0.269 / +15.90% | 2.293 | +0.445 / +24.10% |
| 6 | 2.024 | +0.331 / +19.58% | 2.404 | +0.557 / +30.13% |
| 7 | 2.097 | +0.404 / +23.89% | 2.512 | +0.664 / +35.94% |
| 8 | 2.179 | +0.486 / +28.73% | 2.617 | +0.770 / +41.65% |

14 档均正常完成 24 次请求，实际调用并发峰达到所设 N，加载次数与模型副本均为 1；每个线程 view 的注册 Tensor 均共享。
同设备的唯一权重 storage 不随 N 增长（CPU 约 3.420 GiB；GPU 约 GPU 1.597 + CPU 0.646 GiB）。
数值比较覆盖每档 24×9=216 数组：GPU 七档和 CPU N1/3/5/6/7/8 均逐元素一致。
**CPU N4 未满足本轮 atol=rtol=1e-4 判据**：208/216 数组通过，8 个旋转矩阵数组的最大元素差为 7.338e-4，原因待定位；未改写为全部通过。

本轮 CPU N6 吞吐最高（比 N1 +106.63%），GPU N4 / N5 接近（N5 比 N4 仅 +0.36%）；更高并发未继续改善整批速度。
单次延迟会增加：CPU N1→N8 平均 / p95 从 12.28 / 15.46 秒变为 51.85 / 63.88 秒；
GPU 对应从 1.46 / 1.53 秒变为 8.35 / 9.59 秒。

每档仅一个批次，顺序未随机化；N5 / N7 有尾任务，不能当作持续满载吞吐。warmup2 在 N1/3/4/5/6/7/8
分别调用 6/6/8/10/12/14/16 次，不代表每线程每输入均预热两遍。峰在途 N 不等于 GPU kernel 全程重叠；
USS 与采样峰不是原子快照，结果只适用于当前硬件、输入和线程设置，未做游戏争用测试。

### 真实 HTTP 八并发

另起临时后端连续处理 24 个独立会话，实际活动推理峰 8、模型副本 1。创建后 RAM 保留 8 份、另外 16 份临时溢出；
全部归档读取及重读的八项既有字段与 GPU 基准逐元素一致，删除后会话 / 临时存档均为 0，卸载后模型副本为 0，临时进程与监听均已清理。
原始 JSON / NPZ / HTTP 日志保留在忽略的 `backend/outputs/shared-model/scale-b24-w2/`，不提交本机路径、会话、输出或运行环境。

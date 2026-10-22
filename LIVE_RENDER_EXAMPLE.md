# 可运行的直播渲染循环示例

[live_render_example.py](live_render_example.py) 展示如何把已有帧源、业务 owner 和输出 sink 接到统一 `render_frame` 入口：worker 内初始化一次，持续消费当前帧与素材组，在暂停、恢复、换流和退出时管理资源。[live_render_demo.py](live_render_demo.py) 提供演示输入和检查型 sink，供命令行直接运行。

这是 GPU 渲染与生命周期接线示例，不包含视频解码、编码、RTMP 或真实推流。演示的相机、运动和 Gaussian 回调明确是 Synthetic；生产接入应替换帧源/sink，并沿用业务自己的真实回调。渲染库版本与接口说明见 [README](README.md) 和 [RENDER_API_REVIEW.md](RENDER_API_REVIEW.md)。

## 运行演示

先按 README 安装项目环境。在仓库根目录运行：

```powershell
.\scripts\run.ps1 live-example --device cuda:0 --frames-per-phase 3 --output results/live_render_demo.json
```

默认走 dense 模糊。下面显式开启可分离模式，并保存可选的 PPM 画面快照；可分离核是已知演示核，不能据此假定生产回调生成的核也可分离。

```powershell
.\scripts\run.ps1 live-example --device cuda:0 --separable --frames-per-phase 3 --output results/live_render_demo_separable.json --snapshot-dir results/live_render_frames
```

`--frames-per-phase` 默认 3、最小 2，以覆盖连续暂停和重用；报告目标必须不存在，快照目录必须新建或为空，CLI 拒绝覆盖旧结果。演示实际调用 `initialize_renderer` 和 Triton 渲染，需要可用 CUDA 设备；`--device cpu` 会提前报错，不是实际渲染的替代模式。CPU 单元测试通过 mock 检查 worker 的生命周期与异常清理，不代表 CPU 可以运行该公开 GPU 初始化流程。首次运行还包含内核编译，不能用演示总耗时当作稳定性能。

## 动态路数和素材变化

B 表示同一画面、几何与广告位布局下的素材组数量，每组产生一张结果；它不是独立摄像机数量。单个广告位在 B 组中的源高宽必须相同，不同广告位可以使用不同尺寸。输入尺寸或素材布局与业务选项不一致时，应修正输入或显式配置 resize，不能用缓存隐藏差异。

演示按以下八个阶段运行，每个阶段默认三帧：

| 阶段 | B | 目的 |
|---|---:|---|
| `initial_B1` | 1 | 首次初始化后渲染，返回 HWC |
| `add_B3` | 3 | 复用 worker，返回 BHWC |
| `reorder_replace_B2` | 2 | 换素材对象、改变成员顺序 |
| `inplace_revision_B2` | 2 | 修改像素并推进素材 revision |
| `paused_B0` | 0 | 跳过渲染与输出；进入暂停时释放渲染资源 |
| `resume_B4` | 4 | 使用原 worker 实例恢复，不每帧重新初始化 |
| `no_ads_B4` | 4 | 保留四组背景输出，跳过共享相机/运动核准备 |
| `camera_reset_B1` | 1 | 重置相机历史，frame ID 从 0 开始；输出 PTS 继续递增 |

**B=0 与无广告不同。** 空素材组是调度层暂停，不应传给 `render_frame`；暂停期间仍可接收真实相机记录。无广告则保留 B，令选中 way 的广告位列表为空，仍返回 B 张背景图。非 None 的 `ad_params_{way_id}` 优先于通用 `ad_params`，因此专用空列表不会回退到通用广告。

持续暂停不应每帧反复释放；只有从活动进入暂停时清理库持有资源，恢复时复用初始化后的对象。B、组顺序和源对象变化按现有素材 key 检测；原地修改像素必须推进 `owner.imgs_raw_version` 对应的业务 revision。帧号更新不能代替素材 revision，增加 B 也不能代替正确的素材版本。

## 接到真实业务

生产端复用 worker 函数，提供自己的 owner、已有配置、帧源和 sink：

```python
from live_render_example import run_live_worker

counts = run_live_worker(
    owner,
    existing_options,
    frame_source,
    sink,
    device=device,
    use_separable=False,
    history_capacity=9,
)
```

`owner` 必须专用于本 worker，保留真实 `compute_camera_motion_rate`、`motion_rate_to_sigma`、`create_gaussian_kernel`。`existing_options` 保留原图像布局、通道顺序、alpha、padding、对齐和缩放约定；此 helper 显式选择 `batch_chunk_size="auto"` 与 legacy 几何/相机规则，尺寸缓存容量沿用 `existing_options.max_cached_sizes`。底层统一初始化的其它设置见接口报告。一个 owner 服务一个串行 worker/计算流，不要在多个并发消费任务之间共享同一实例。

### 每帧 packet

`frame_source` 是可迭代对象，并实现 `close()`。每次迭代返回下列字典；为便于生产者实现，可以所有帧都带齐字段。

| 字段 | 读取时机 | 内容 |
|---|---|---|
| `frame_id` | 所有帧 | 当前相机历史的整数帧号；同一历史段中单调，重置时可从 0 开始 |
| `camera` | 所有帧 | 含 `K`、`w2c`、一维 `dist_coeffs` 的 CPU 数组字典；worker 分别复制成独立记录 |
| `material_groups` | 所有帧 | list/tuple；B 等于长度，空列表表示暂停 |
| `route_ids` | 所有帧 | 长度 B 的 tuple，元素为互不重复的非空字符串，顺序与素材组严格对应；暂停时为 `()` |
| `frame` | B>0 | 当前背景 Tensor 或数组，按 `existing_options.frame_layout` 解读 |
| `data` | B>0 | 本帧广告位参数；含 `ad_params` 或对应 `ad_params_{way_id}`。worker 浅复制字典，并用当前独立相机记录设置 K/畸变系数 |
| `mask` | B>0 | 当前 mask，遵循原广播、布局和缩放约定 |
| `material_revision` | B>0 | 当前素材内容版本，赋给 `owner.imgs_raw_version`；静态素材可保持不变，原地修改后必须推进 |
| `pts` | B>0 | 交给 sink 的输出时间戳，原样传递；单位和时间基由业务 source/sink 约定 |
| `way_id` | B>0，可选 | 默认 1；用于选择对应广告位列表，不代表 B |
| `reset_camera_history` | 可选 | 默认 false；新历史段首包置 true，清历史与库内资源，允许 frame ID 重新开始 |
| `drop_asset_cache` | B>0，可选 | 默认 false；显式要求本帧调用 `executor.invalidate_assets()` |

一个广告位仍提供 `ad_placement_id`、`ad_id`、边界内的 `bbox`、ROI 局部 `m_inv` 和可选透明度。每个素材组用 `(ad_placement_id, ad_id)` 或旧兼容 placement key 定位 `{"img": RGBA_image}`。新对象替换、顺序和 B 改变会被原缓存 key 识别；对于同一对象原地修改内容，仍需 revision 或显式失效。

worker 不修改生产者的原 `data` 字典。浅复制不会隔离嵌套广告位数组；并发生产者不能在当前同步渲染读取期间覆写这些输入，跨 CUDA 流还需要先建立依赖。演示生成器按顺序交接，不代表库自动解决并发 producer 的同步。

相机历史在 worker 开始时创建一次。每帧应保存独立的 CPU 相机记录；生产者会复用的 K、w2c 和畸变系数缓冲应在入队或交接时复制。相机滤波可能替换当前 `data_dict['dist_coeffs']`，所以当前帧字典不能直接与保留历史记录共用。默认半径 4 下，最新帧至少需要 5 条记录；完整四帧前瞻窗口至少需要 9 条，更长延迟需要更大容量。

frame ID 用于相机邻域；输出 PTS 用于下游时间线。seek/换流时应显式重置历史和渲染资源，再允许新的 frame ID 序列，不要把旧相机记录带入新流。若下游要求连续 PTS，应按业务时钟继续推进，不因 frame ID 归零而回退。`reset_camera_history` 本身不重写 PTS，也不替业务重置编码器。

## 输出和关闭责任

标准初始化产生独立、可立即读取的 NumPy `uint8`。B1 为 HWC，多组为 BHWC；组下标跟随本帧素材组的顺序，不能把动态数组下标当作永久业务流身份。业务应随帧传递稳定的流标识和时间戳，让 sink 按当前映射分发。

sink 实现三个方法：

| 方法 | 契约 |
|---|---|
| `set_routes(route_ids)` | 仅在有序 tuple 改变时调用；按稳定 ID 协调新增、移除和重排，未变化的路不必重建编码器 |
| `write_batch(route_ids, output, *, pts)` | 接收本帧完整批次，按本帧 route 顺序分发；返回前完成本次同步消费，显式保留的引用须有界并由 close 释放 |
| `close()` | 返回前排空或取消全部待消费工作并完成关闭 |

worker 不负责启动网络发送线程。若改成异步 sink，必须自行设置队列上限及背压/丢帧策略，明确持有输出直到消费完成，并在退出时等待在途消费结束。将 `output[index]` 视图入队会保留整个 BHWC 底层存储；如果只需要单路独立所有权，应明确复制该路，不能假定视图只占一张图的内存。

`run_live_worker` 接管本次 source/sink 的关闭责任。无论正常结束、初始化失败或循环异常，`ExitStack` 都按 **source → sink → renderer → worker 输入引用** 的顺序尝试清理；即使某个 close 抛错，也继续尝试后续清理。source 的 `close()` 必须停止生产者并释放其输入引用，sink 的 `close()` 必须停止使用输出，之后才释放渲染器。

最后清空本 worker 的历史，将 `owner.paste_data_stack=None`、`owner.imgs_raw_groups=[]`；空组可避免误用旧 `owner.imgs_raw` 回退。其它 owner 字段和业务外部素材 bank 不归这个 helper 管理，仍由业务释放。`release_renderer(owner)` 释放库缓存、复制流和上下文引用，不覆盖业务仍持有的输出，也不替业务清空发送队列。仅看到 CUDA reserved 未下降不能判断泄漏。示例不在逐帧循环调用 `gc.collect()` 或全局 `empty_cache()`。

worker 在 sink 调用的 `finally` 中删除自己的 output 引用，但异常 traceback 仍可能通过 sink 参数或源迭代器的栈帧保留输入/输出。上层不要无限积累带 traceback 的异常对象；日志和重试队列也要有明确上限。异常单测验证的是清理动作和 owner 状态，不证明所有外部引用、异常栈或分配器保留存储均已释放。

正常结束返回计数字典：`received_frames`、`rendered_frames`、`paused_frames`、`idle_releases`、`camera_resets`、`asset_invalidations`、`route_changes`、`output_images`。渲染帧数包含无广告帧；输出图像数累加每个活动帧的 B；暂停帧不渲染、不写 sink。

## 验证方法

生命周期单元测试入口为项目解释器的 `-B -m unittest test_live_render_example`。实际 GPU 演示分别检查 dense 与显式 separable 路径。画布为 96×160、两个广告位，素材源高宽分别为 24×44 和 20×40；Synthetic owner 默认生成 5×5 Gaussian 核。小画布不属于已有 1080p auto 准入，不能据此声称触发了已测分块优化。

`CheckingSink` 在当前调用内检查 HWC/BHWC、uint8、连续性、route 顺序和递增 PTS。它最多保留首份及上一份输出，加对应的普通 NumPy 快照，用于之后检查旧数组未被覆写；`close()` 再检查并清空这些引用。这是明确有上限的诊断保留，不是异步发送队列。

背景、相机和几何在演示中保持不变，所以逐路 SHA256 用来检查替换/revision 只改变对应素材输出；恢复与重置阶段还会对照此前同素材结果。无广告阶段要求每一路逐字节等于背景。暂停阶段检查素材、Gaussian 缓存和 owner 当前素材引用释放，持续检查 builder/执行器身份不变。结束后断言 source/sink 已关闭、渲染缓存及上下文引用释放、worker 历史和素材引用已清空、sink 保留输出已释放。

JSON 主要字段为 `phases`、`worker`、`sink`、`route_changes`、`observations`、`after_close` 和 `snapshots`。启用快照时，每个活动阶段第一批的第 0 路保存一张 P6 RGB PPM，共七张，暂停阶段没有画面。它们帮助直观看图，不包含音频或时间戳容器。演示是小画布功能检查，不是生产视频验收或性能基准；逐阶段断言也不是 dense 与 separable 两种数学路径之间的逐字节比较。

## 本次验收记录

2026-10-22，在本机 CUDA 环境完成最终版本检查；运行时版本仍为 `2026-10-21-perf4.6-render-api`。两份最终 GPU JSON 均为 `status="complete"`，八个阶段的合成 hash、背景、输出所有权及生命周期断言全部通过。

| 指标 | dense | 显式 separable |
|---|---:|---:|
| 每阶段输入帧数 | 3 | 25 |
| 总输入帧数 | 24 | 200 |
| 渲染 / sink 批次数 | 21 | 175 |
| 输出图像数 | 51 | 425 |
| B=0 暂停帧数 | 3 | 25 |
| 进入暂停时释放次数 | 1 | 1 |
| 相机历史重置次数 | 1 | 1 |
| route tuple 变化次数 | 6 | 6 |
| builder/执行器身份检查次数 | 24 | 200 |
| 最终运行保存的 PPM | 7 | 0，未启用快照 |

两份报告退出时的 `retained_stack_bytes`、素材 source 引用数、Gaussian 条目数、核上下文引用数均为 0；source/sink 关闭、history/owner 当前引用清理、sink 保留输出清理标记均为 true。这些是库与示例对象状态的检查，没有把分配器 reserved 或整个进程存储解释为已归零。

新增 13 项 worker CPU 生命周期/异常测试与既有 59 项测试共 **72 项 unittest 全部通过**，其中公开 API GPU 项显式启用，无 GPU 跳过。33 个 Python 文件通过 Ruff 和 Python 3.9 AST 检查，`git diff --check` 通过；本轮 12 个运行时、内核和策略文件相对本轮开始时未改动。

CLI 也验证了 `--help` 成功；已有报告、CPU 设备、非空快照目录三类拒绝路径均以退出码 2 返回。已有报告的 SHA 保持不变，CPU/非空快照拒绝没有生成新报告。

最终运行命令如下；重跑时应替换为未占用的结果路径：

```powershell
.\scripts\run.ps1 live-example --device cuda:0 --frames-per-phase 3 --output results/live_demo_dense_final.json --snapshot-dir results/live_demo_dense_final_frames
.\scripts\run.ps1 live-example --device cuda:0 --separable --frames-per-phase 25 --output results/live_demo_separable_final_200.json
```

本机 artifact 为 `results/live_demo_dense_final.json`、`results/live_demo_separable_final_200.json`、`results/live_demo_dense_final_frames/`、`results/live_example_regression_final.log`。CLI 拒绝路径日志为 `results/live_example_existing_output.log`、`results/live_example_invalid_cpu.log`、`results/live_example_existing_snapshot.log`。

本次没有运行独立图像 oracle、真实视频编码/网络推流或生产长期负载；200 帧有限演示不能替代这些验收，也不产生吞吐或延迟结论。

原始 JSON、日志和 PPM 位于忽略的 `results/`，不随 Git 分发。API 数值与长期运行的既有证据仍以 [RENDER_API_REVIEW.md](RENDER_API_REVIEW.md) 为对应版本记录；不要把先前的 10000 帧结论当作此次演示已执行的证明。

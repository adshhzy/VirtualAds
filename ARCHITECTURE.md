# 代码阅读与修改指南

当前版本 `2026-10-21-perf4.3-backends`。这是 GPU 贴图库和基准工具，不含视频解码、编码、推流或常驻服务。
这里的 **B（路数）是同一画面/布局下的素材组数**；每组产生一张结果，不等同于任意独立摄像机的并发数。
最新后端测量、策略准入及验收见 [BACKEND_COMPARISON_REVIEW.md](BACKEND_COMPARISON_REVIEW.md)。

## 先从哪里读

1. [shared_paste_example.py](shared_paste_example.py)：worker 初始化、每帧共享准备和业务调用位置。
2. [paste_gpu_batch_adapter.py](paste_gpu_batch_adapter.py) 的 `PasteBatchAdapter.__call__`：从业务输入到 NumPy 输出的主流程。
3. [paste_gpu_batch.py](paste_gpu_batch.py) 的 `paste_gpu_batch` / `_finish_canvas`：纯 Tensor 运算与后端选择。
4. 再按需求查看共享几何、缓存、执行策略或具体 Triton 内核。

动态业务接入可直接参考 [dynamic_paste_example.py](dynamic_paste_example.py)：`initialize_dynamic_worker` 调用一次，随后复用对象调用 `render_dynamic_frame`，B 由当前素材组列表决定。示例不提供合成业务回调；B=0由调度方跳过，B1继续返回HWC。已用合成测试数据验证1→3→8→5→12→1与整批逐字节一致，以及旧输出存活、way专用无广告优先和空B拒绝，记录见 `results/dynamic_business_example.json`。

```mermaid
flowchart TD
    A[业务帧、相机参数、素材组] --> B[SharedParameterBuilder.prepare]
    B --> C[六项 paste_args]
    A --> D[PasteBatchAdapter.__call__]
    C --> D
    D --> E[整理素材、ROI 网格、mask 和模糊核]
    E --> F[choose_execution：整批或分块]
    F --> G[paste_gpu_batch：逐广告位处理 B 组]
    F --> K[paste_gpu_bucketed：逐桶处理 B × M 个素材]
    G --> H[_finish_canvas：畸变、模糊、融合]
    K --> H
    H --> I[GPU BHWC uint8]
    I --> J[同步导出独立 NumPy]
```

## 文件职责

| 文件 | 负责什么 | 适合在这里修改什么 |
|---|---|---|
| `shared_paste_example.py` | 业务对象接入，初始化 builder/adapter | worker 接入方式 |
| `paste_camera_filter.py` | CPU 相机历史选择、滤波、调用业务运动估计 | 相机数据校验和滤波；行为变更须单独验收 |
| `paste_shared_params.py` | 共享畸变网格、ROI 网格、精确 sigma 核缓存 | 共享准备和缓存实现 |
| `paste_shared_types.py` | `MotionKernelSpec`：二维核与已验证的一维因子 | 共享数据契约 |
| `paste_gpu_batch_adapter.py` | 输入布局、素材缓存、执行编排、NumPy 所有权 | 业务接口适配和输出策略 |
| `paste_execution_policy.py` | 离线实测配置的匹配与回退 | 新硬件/布局/B 的实测策略 |
| `paste_gpu_batch.py` | 纯 Tensor 贴图流程，batch/bucketed，后端路由 | GPU 阶段组织和参考路径 |
| `paste_gpu_kernels.py` | Triton ROI 采样、融合、dense 模糊 | 对应数值内核 |
| `paste_gaussian_kernels.py` | 横纵可分离模糊与最终融合 | 两阶段启动参数和内核 |
| `paste_shared_kernels.py` | Triton 畸变和 ROI 网格生成 | 几何网格内核 |

运行时代码不依赖 benchmark 或测试文件；共享准备和 adapter 共同使用 `MotionKernelSpec`。
保留平铺文件和原导入路径，已有业务代码不需要迁移包名。

## 一帧如何走完

`setup_shared_paste` 在 worker 内执行一次；builder 和 adapter 后续复用。
`prepare_shared_paste_args` 每帧读取实时 K/畸变/运动参数，输出以下原有六项元组：

```python
(device, distortion_grid, filtered_coefficients,
 motion_kernel, mask_canvas, warp_grid_dict)
```

`motion_kernel` 可以是原 Tensor，或带已验证因子的 `MotionKernelSpec`。元组顺序保持兼容。
实际素材尺寸明确后，adapter 才生成每个广告位的 ROI 网格。

`PasteBatchAdapter.__call__` 现在主要编排以下步骤：

| 阶段 | 方法 | 边界 |
|---|---|---|
| 选择输入 | `_groups`、`_ad_parameters` | 确定 B 和当前 way 的广告位 |
| 无广告返回 | `_render_background_only` | 不要求 `paste_args` 有效 |
| 背景与素材 | `_prepare_frame`、`_collect_ads` | 规范为 float32 Tensor；复用素材缓存 |
| ROI 输入 | `_prepare_warp_inputs` | 根据实际素材尺寸准备 grid 与透明度 |
| 全图输入 | `_prepare_canvas_inputs`、`_prepare_motion_kernel` | 畸变、mask、共享/逐组核 |
| 选策略 | `_select_execution` | 记录 `last_execution`；未知配置回退整批 |
| 执行与输出 | `_render_prepared` | batch/bucketed/分块；NumPy 或显式 Tensor 输出 |

GPU 核心先按广告位顺序覆盖 RGBA 画布，再做全图畸变、模糊、背景/mask 融合和 uint8 转换。
`_finish_canvas` 明确区分 `torch`、dense `triton` 和 `separable` 三条模糊路径。
不可交换 ROI 写入顺序、alpha 运算顺序或模糊位置，它们会改变画面。

### batch、bucketed 与分块分别合并什么

`batch` 和 `bucketed` 都先把同一广告位的 B 组素材整理为 `[B,4,src_h,src_w]`。
这里的堆叠发生在**素材组维度 B**；单个广告位在 B 组中的源尺寸必须相同。

| 路径 | 核心 Python 循环 | 一次采样处理的维度 | 如何得到整张画布 |
|---|---|---|---|
| `batch` | 按原顺序逐广告位循环 | 当前广告位的 B 组素材 | 每位 ROI 覆盖写入 `[B,4,H,W]` 画布 |
| `bucketed` | 逐桶循环 | 当前桶的 B × M 个素材，M 是桶内广告位数 | 桶内结果拼成 palette，再按 `canvas_lut` 取回每个画布像素的最终所有者 |

`bucketed` 默认按源高宽分桶，`area_caps` 还可按 ROI 面积细分。每桶素材为
`[B,M,4,src_h,src_w]`，grid 为 `[B,M,L,2]`，L 是补齐后的 ROI 像素容量；
采样时分别展平为 `[B*M,4,src_h,src_w]` 和 `[B*M,1,L,2]`。
因此矩形 ROI 先展平并补齐到 L，而非补齐成统一的最大高宽矩形。
LUT 按原广告位顺序保留最后覆盖者；后位透明像素也会覆盖前位，而不是跳过覆盖。

这并未消除外围的逐广告位工作：素材检查/转换、布局规划，以及 grid/透明度的
pad 和 stack 仍包含 slot 循环。当前 adapter 每帧重新打包 grid 和透明度；素材桶可复用缓存。
两条核心路径最终都把 `[B,4,H,W]` 画布交给 `_finish_canvas`，完成相同的全图阶段。

`batch_chunk_size` 切的是 **B 维度**，不会把广告位变成桶。共享相机/核/畸变准备，以及
adapter 的素材收集、ROI grid 和透明度准备，都在分块前执行一次；随后按块切分非广播的
B 维输入，广播到 B 的单份输入继续共享。当前分块执行用于 `batch` 后端，各块包含全部广告位，
块内仍按广告位循环，最后按原组顺序导出 NumPy。

## 形状与坐标约定

`H/W` 是背景高宽，`h/w` 是单个素材或 ROI 高宽，`N` 是可广播的 1 或 B。
一个广告位在不同素材组中必须具有相同源尺寸；不同广告位可以尺寸不同。

| 数据 | 核心形状 | 含义 |
|---|---|---|
| 背景 | `[N,3,H,W]` | float32，RGB/BGR 顺序原样保留 |
| 每位素材 | `[B,4,h,w]` | RGB 和原始 alpha 均为 0～255 |
| 每位 grid | `[N,roi_h,roi_w,2]` | 反向采样坐标 |
| 畸变 grid | `[N,H,W,2]` | 全图反向采样坐标 |
| mask | 可广播到 `[B,3,H,W]` | 1 保留背景，0 允许广告 |
| 透明度 | 可广播到 ROI 的 `[B,1,h,w]` | 0～1 |
| dense 核 | `[4,1,kh,kw]` 或 `[B,4,1,kh,kw]` | 共享或逐组二维卷积核 |
| 可分离因子 | `ky[kh]`、`kx[kw]` | 两个 FP32 一维向量，共享到组/通道 |
| 核心输出 | `[B,H,W,3]` | GPU uint8 |
| 默认业务输出 | B1 为 `[H,W,3]`，多路为 `[B,H,W,3]` | 同步 NumPy uint8；可显式关闭单组压缩 |

`bbox=(x1,y1,x2,y2)` 右下不包含，必须在画布内。共享 ROI 建网格使用 ROI 局部坐标和 `z + 1e-8`。
`normalization="legacy"`、`camera_index_policy="legacy"` 和 `legacy_alpha=True` 保持历史画面；其它值会改变语义。
Tensor 的 stride 和广播视图有实际性能意义，不要为了整洁随意加入 `.contiguous()`。

## 缓存与输出生命周期

- 素材缓存由尺寸、素材身份、版本、设备、B 等区分；原地修改素材后递增 `imgs_raw_version` 或显式失效。
- 尺寸缓存保存各广告位已整理的 `[B,4,h,w]` 素材，`batch` 同样使用它；缓存命中不表示启用了 `bucketed` 计算分桶。
- `bucketed` 另有当前布局的素材打包缓存。bbox 平移或改变 ROI 尺寸都会改变 adapter 的布局键，重建 `canvas_lut` 并重新打包当前素材桶；底层素材尺寸缓存仍可命中。不能把动态布局按静态 LUT 复用来估算耗时。
- Gaussian 核按精确 sigma 和回调/配置版本缓存；没有 sigma 量化。相同 `frame_id` 的 K、畸变、矩阵仍可能变化，因此不缓存整帧几何值。
- 每次 NumPy 输出拥有独立存储。固定页路径直接返回 host Tensor 的 NumPy 视图，数组持有其底层 Tensor。
- 分块路径仍在调用方计算流串行计算；内部复制流等对应事件后复制。GPU 源和 host 结果一直存活到复制完成，正常返回与异常退出都同步复制流。
- 每个 worker 使用自己的实例，调用方按同一计算流串行调用。跨流生产输入需先建立依赖。

`batch_chunk_size="auto"` 只匹配离线表；`0` 整批；正整数手动分块。默认后端仍为 `batch`，也不强行启用可分离模糊。
本轮新增准入限定在 B8/B16、4/16 广告位、已测源尺寸循环、108×320 ROI 等条件内，继续检查既有硬件、软件和 Tensor 布局。
精确准入表及回退条件以 [paste_execution_policy.py](paste_execution_policy.py) 和 [最新报告](BACKEND_COMPARISON_REVIEW.md) 为准，不能从广告位数或 B 单独推断；32 广告位不在此承诺内。
[动态路数报告](DYNAMIC_BATCH_REVIEW.md) 保留早期策略的历史测量。

## 基准与回归工具

| 命令 | 文件 | 验证范围 |
|---|---|---|
| `scripts/run.ps1 shared` | `benchmark_shared_paste.py` | 含共享准备与 NumPy 的完整调用 |
| `scripts/run.ps1 backends` | `benchmark_backend_comparison.py` | 同输入、独立实例的 batch 整批/auto/手动分块与 bucketed 完整调用对比；含动态 bbox、打包和布局更新 |
| `scripts/run.ps1 check-backends` | `check_backend_contracts.py` | 布局归属、透明覆盖、广播输入、动态几何/素材缓存及旧输出所有权对照 |
| `scripts/run.ps1 check-policy` | `test_backend_policy.py` | CPU 元数据测试；验证自动策略完整结构匹配、未知输入回退及原有档位优先 |
| `scripts/run.ps1 batch` | `benchmark_paste_gpu_batch.py` | 核心、adapter、dense 内核及诊断 |
| `scripts/run.ps1 check` | `test_dynamic_batch.py` | 广播、尾块、动态 B、缓存和输出生命周期 |
| `scripts/run.ps1 compare` | `benchmark_dynamic_comparison.py` | 隔离加载旧源码，对照当前完整调用；`--slots` 指定广告位数，默认 16 |
| `scripts/run.ps1 dynamic` | `benchmark_dynamic_batch.py` | 不同分块/启动配置筛选 |
| `scripts/run.ps1 tune` | `benchmark_blur_tuning.py` | 单独模糊内核调参；不是完整耗时 |

`benchmark_support.py` 只共享报告、轮换顺序和同步计时；不放业务计算。
基准使用合成运动/sigma/Gaussian 回调，不代表真实业务。不要把独立阶段计时相加，也不要把跨轮数字当收益。

本轮完整筛查覆盖 B=1/4/8/16、4/16/32 广告位、1～3 种素材尺寸与静态/平移/缩放布局，共 108 个场景。
其中 bucketed 有 104 个场景最大差 1 级 uint8，另外 4 个逐字节一致场景更慢，未满足零差异和性能准入，故默认继续采用 batch。
详细数据、确认测量和限制见 [BACKEND_COMPARISON_REVIEW.md](BACKEND_COMPARISON_REVIEW.md)。

`backends` 的计时从共享准备开始，到独立 NumPy 返回且 GPU 完成为止；只把合成源输入更新放在计时外。
各候选持有独立 builder/adapter 和可变相机状态，并在下一次返回前保留上一次输出。
报告包含轮换顺序、逐样本/P95、布局重建和缓存计数，以及包含其他候选常驻缓存的进程显存峰值。
`--diagnose` 另做带插桩的分阶段检查，不混入主计时；默认的 `cold_assets_ms` 是核预热后失效素材缓存的一次完整调用，
会记录布局是否保留，不能当作稳定 P95 或自动策略依据，可用 `--skip-cold-assets` 跳过。

正确性默认检查第 0、1、7 帧，`--check-timed-steps` 扩展到所有实际计时帧；报告的 exact gate 只覆盖列出的 `checked_steps`。
`backends` 会记录 uint8 差 1 并继续测量，但任何非零差异都不具备自动策略的画面一致性资格，差值超过 1 则明确失败。
`check-backends` 的 CPU 对照要求逐字节一致，CUDA 默认允许差 1 并报告实际 exactness；`--max-uint8-diff 0` 可要求严格一致。
这些工具不自动改写执行策略。确认测量时，应让轮数是候选数的整数倍，以完整轮换候选顺序。

```powershell
.\scripts\run.ps1 backends --batches 8 --slots 16 --profiles three --layouts static --check-timed-steps
.\scripts\run.ps1 check-backends --device cpu
.\scripts\run.ps1 check-backends --device cuda:0 --output results/backend_contracts.json
.\scripts\run.ps1 check-policy
```

## 后续迭代习惯

格式统一为 Ruff 0.14.1、100 列、UTF-8/LF、Python 3.9 语法目标；实际 GPU 环境仍是 Python 3.12。
格式与 lint 工具使用独立 uv 工具环境，不改变 GPU 运行依赖：

```powershell
.\scripts\lint.ps1          # 只检查
.\scripts\lint.ps1 -Fix     # 格式化后检查；不自动更改语义
.\scripts\run.ps1 check --device cuda:0 --profile-checks
```

新增处理优先放在负责该阶段的现有函数附近。内核保留数学坐标和 stride 命名；host 代码使用能说明职责的名称。
更改执行策略必须补完整 NumPy 对照；更改数值核必须补原算法/旧源码对照，不依靠逐轮放宽容差。
本轮验收见 [BACKEND_COMPARISON_REVIEW.md](BACKEND_COMPARISON_REVIEW.md)；此前可读性重构记录保留在 [READABILITY_REVIEW.md](READABILITY_REVIEW.md)。

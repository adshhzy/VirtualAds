# Codex 后续迭代

## 项目与运行

- VirtualAds 是 Python GPU 库和命令行基准，没有 Web 服务。当前版本 `2026-10-23-perf4.9-prepare`，准备阶段优化基线 `6cebfce` 和验收见 `PREPARATION_OPTIMIZATION_REVIEW.md`。三参数接口说明见 `RENDER_IMG.md`；上传优化基线 `b82f35b` 和验收见 `UPLOAD_OPTIMIZATION_REVIEW.md`。旧统一入口迁移基线 `4b2b311` 和验收见 `RENDER_API_REVIEW.md`。原包 perf4.1 基线为 `0b97beb`；最初 perf4 为 `480fad5`。保持 batch 默认与 auto 表；perf4.7 新增 mask 羽化会改变遮挡边缘过渡。
- 先尊重调用方的 `PASTE_GPU_PYTHON`；未设置时使用 `scripts/run.ps1` 的默认解释器，即仓库父目录下 `env\paste_gpu_performance\Scripts\python.exe`。不要在新机器上硬编码旧 E 盘路径；不要直接用系统 Python 运行项目，也不要改动其他项目环境。若环境不存在，按 README 创建。
- 当前 Windows 锁文件要求现有 Python 3.12 与 uv。`scripts/setup.ps1 -PythonPath ... -UvPath ...` 创建项目环境并安装 `requirements-lock.txt`，不会下载/安装 Python 或 uv；torch 独立使用官方 cu128 源。自定义 `-EnvironmentPath` 后相应设置 `PASTE_GPU_PYTHON`。依赖调整应同步锁文件并验证 CUDA / Triton 兼容。
- 本机已测参考：Windows、RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128；当时解释器为 `E:\dev\virtualAds\env\paste_gpu_performance\Scripts\python.exe`。此路径是历史记录，不是所有克隆的固定入口。
- `scripts/run.ps1 shared ...` 对应 `benchmark_shared_paste.py`；`scripts/run.ps1 batch ...` 对应 `benchmark_paste_gpu_batch.py`。
- `scripts/run.ps1 backends ...` 对应 `benchmark_backend_comparison.py`；`before_bucketed` 候选配合 `--baseline` 在同进程隔离比较旧桶、当前桶和 batch，`--require-exact` 指定必须零差异的候选。`check-backends` 对应 `check_backend_contracts.py`；`check-policy` 对应 CPU 元数据测试 `test_backend_policy.py`。旧 batch 源码对照仍用 `compare` / `benchmark_dynamic_comparison.py`，`--slots` 为单个正整数，默认 16。
- `scripts/run.ps1 render-img-modes ...` 对应 `benchmark_render_img_modes.py`，读取本机 NPY 与解码后的 24-bit BMP，复制为 B8/S16，比较完整三参数入口的 eager、dense/separable 整批与分块。基准中的 eager 同时切换共享几何与贴图/模糊，公共 mask 保持 Triton；不要只传 eager options 后误测被初始化覆盖的 Triton 路径。每次计时输出在计时外逐像素检查；任何非零差异不准自动替换，真实样例结论见 `RENDER_IMG_MODE_REVIEW.md`。
- `scripts/run.ps1 check-streaming` 对应 `check_streaming_stability.py`，需要 `--baseline` 指向保留源码；`--frames` 是固定/动态两阶段合计测量帧数，不含预热。新克隆先按 README 的 `git archive`/`Expand-Archive` 步骤恢复 `6edf109`；旧桶基线为 `9c6e930`。禁止为恢复基线覆盖或删除已有 results。运行示例与内存指标边界见 `STREAMING_REVIEW.md`。
- 底层兼容入口为 `paste_renderer.initialize_renderer` / `render_frame` / `release_renderer`；`paste_render_executor.py` 的 `RenderOptions` / `RenderExecutor` 是规范实现。`paste_gpu_batch_adapter.py` 只保留旧导入/函数兼容，勿再复制一份执行逻辑。共享参数在 `paste_shared_params.py` / `paste_camera_filter.py`，纯 Tensor 路由在 `paste_gpu_batch.py`，GPU 内核在 `paste_*_kernels.py`。
- 三参数业务入口是 `ImageRenderer.render_img(data_dict, frame, imgs_raw)`，初始化一次的对象显式持有状态，无模块全局 worker。`paste_route_plan.py` 按整数路号扫描 ad_params_N，精确pair查imgs_raw，按有序bbox/素材尺寸等兼容条件分组；不读ad_paste_data/way_ad_map/brightness/image_path，不用透明占位slot伪造相同布局。旧render_frame继续兼容。
- 启动与线上接线总览为 `STARTUP_AND_API.md`；`scripts/run.ps1 render-img-example` 启动 `render_img_example.py` 的有限合成三参数演示。它不提供 HTTP/解码/推流服务，不依赖测试或基准 helper；生产继续使用原有 owner 回调。当前共享初始化固定 B1 squeeze，不能在新入口文档声称 `RenderOptions(squeeze_single=False)` 可覆盖；下游可把 HWC 视为 1HWC。
- `paste_shared_mask.py` / `paste_mask_kernels.py` 先maximum合并局部mask再做全帧Gaussian羽化，mask=1保留背景；默认sigma1.5/radius3，replicate边界，radius0关闭。借用工作区必须在下一帧前消费，不能当独立输出保存。相机/mask/背景上传每帧共用，不按路重复；最终NumPy仍独立。
- 可运行业务循环在 `live_render_example.py`，演示源/合成回调/检查 sink 在 `live_render_demo.py`；`scripts/run.ps1 live-example --device cuda:0` 为小画布演示，默认 dense，可显式 `--separable` 检查已知演示核。接入字段、source/sink 生命周期见 `LIVE_RENDER_EXAMPLE.md`；不把演示 Synthetic 回调用于生产，也不把本地 sink 称为实际编码/推流。
- `paste_asset_cache.py` 负责历史素材 stack 的预算/LRU/revision；当前素材与源身份引用由执行器持有。优先阅读 `README.md`、`ARCHITECTURE.md` 和 `RENDER_API_REVIEW.md`；`STREAMING_REVIEW.md`、`BUCKETED_OPTIMIZATION_REVIEW.md` 等为历史记录，不能重写旧结果或据旧文档扩大当前策略范围。results/logs/env/缓存不随 Git 分发；报告中的原始测试 artifact 不在新克隆内，恢复源码不能恢复旧测量数据。
- 新克隆用已有 uv 执行 `uv.exe tool run --from ruff==0.14.1 ruff format --check .` 和 `uv.exe tool run --from ruff==0.14.1 ruff check .`。`scripts/lint.ps1` 仅适用于已有 `..\env\tools\bin\uv.exe` 的工作区，不读取 setup 的 `-UvPath`；`-Fix` 仅统一格式再检查。Ruff 0.14.1/100列/LF/Python3.9语法目标，配置在 `pyproject.toml`，工具隔离于GPU环境。保留平铺导入路径；优先按阶段职责放置 helper，不引入只为包装参数的层层对象。

## 必须保持的行为

- `AdapterOptions is RenderOptions`、`PasteBatchAdapter is RenderExecutor`：旧名称必须保持同一个 class，不能改成子类或额外包装实例。旧函数、`__call__`、共享准备和动态示例函数签名/返回值继续兼容；三参数业务用 `ImageRenderer.render_img`，已有 `render_frame` 和手工准备流程仍可运行。
- 统一入口应调用 `executor(...)`，标准类再转调 `render_prepared`；不得直接绕过旧业务子类的 `__call__` 覆盖。
- 旧 adapter 模块是兼容 shim，helper 属性读取可代理，但给旧模块私有全局赋值不会转发。需注入测试依赖时定位实际实现模块 `paste_render_executor`，不要另复制执行代码或把私有 monkeypatch 当作业务 API。
- owner 的 `_paste_batch_adapter` / `_shared_paste_builder` 是唯一执行器/builder 存储位，不能另增一套缓存。每帧共享准备一次，ROI 网格仍待实际素材尺寸明确后通过 builder 回调产生。无广告绕过共享准备；`ad_params_{way_id}` 的非 None 值优先。
- 新 `render_frame` 的 `material_groups=None` / `camera_history=None` 意味着沿用 owner 已有字段；显式空组拒绝。旧动态 wrapper 的必要参数和显式 None 行为继续兼容，不可顺手改成新 API 的默认语义。
- 初始化在 worker 内调用一次；ImageRenderer退出用with/close，可复用释放用其release_resources，包含mask工作区和owner当前素材引用。兼容render_frame才直接用release_renderer(owner)，它只清执行器/shared builder、不清新类mask/历史。真实业务回调、独立NumPy及有界输出队列所有权保持。
- 直播示例中 B=0 是调度暂停，无广告是 B 张背景返回；连续暂停只在进入暂停时释放，恢复复用实例。换流/seek 显式清历史并重置帧号，PTS 按下游时间线管理；同步 sink 在 close 返回前完成消费。异步 sink 必须限制队列并在退出时排空/取消，单路 NumPy view 仍可能保留整个批次存储。
- `run_live_worker` 的 source→sink→renderer→worker 输入清理顺序、初始化失败/close异常时的后续清理须保持。sink 调用异常时用 finally 删除 worker 的 output 引用；异常 traceback 仍可持有 sink 参数或源帧，不能把清理单测解释为所有外部引用释放，上层也不能无界保存异常对象。
- 默认保持 `normalization="legacy"`、`camera_index_policy="legacy"`；性能修改不要同时混入 `matched` / `corrected` 的画面行为变更。
- 默认 NumPy `uint8`：B1 为 HWC、多组为 BHWC；显式 `squeeze_single=False` 保留 B1 维。固定页输出必须独立，不得被下一帧覆盖。
- K、畸变系数、单应矩阵可能原地更新；不能仅按 frame_id 缓存几何结果。尺寸按实际素材 / 显式 resize 后的高宽处理。
- 保留 ROI 局部坐标、`z + 1e-8`、现有采样 / alpha / 通道顺序语义。不默认引入 FP16、TF32 或低秩近似。
- `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 缺少生产定义。基准回调只用于合成 fixture，不得将它们冒充生产定义或在接入代码中替代业务方法。
- Gaussian 缓存使用精确 sigma，默认最多 16 项且只保留当前 owner/回调/版本/device 上下文；配置变更需更新版本或清缓存。上下文变化应释放旧 owner 强引用。可分离路径必须验证实际二维核，不靠函数名推定可分离。
- builder / adapter 在 worker 内初始化后复用，调用方在同一计算流上串行使用；不每帧重建或清缓存。分块路径内部使用专属复制流，事件连接计算与复制；返回及异常退出前等待复制完成，禁止覆写在途GPU源或调用方持有的CPU结果。
- `paste_upload_buffer.PinnedUploadBuffer` 只复用有界存储，所有参数逐帧重写。固定页 host 重写前须等待其上次 H2D 事件；借用 GPU 视图只在同计算流排队消费，不能暴露为持久结果。矩阵/透明度各最多 1 MiB host + 1 MiB device；背景最多 32 MiB host + 32 MiB device uint8 和 128 MiB device float32，均按实际需求延迟分配。超预算是临时工作集，不得称其仍在缓存预算内。release 必须清所有上传持有；Gaussian clear_cache 单独不清矩阵工作区。
- `scripts/run.ps1 uploads` 对比隔离旧源码的完整 `render_img`；`profile-uploads` 的 CPU/CUDA 及父子阶段不可相加。`check-uploads` 用真实输入做固定与动态 B/S 长测，记录旧输出、H2D 工作区增长/复用和释放；不可把进程 reserved 或有界业务队列当作库泄漏。
- `paste_route_preparation.py` 的普通 CPU 工作区最多保留 1 MiB；只在标准三参数内部路径借用，公共 route plan 独立。素材描述只在本次调用传递，不跨帧缓存；shape/dtype/device 快照与 layout 必须同时获取。源状态或 options 变化回退原扫描；异构声明尺寸保守回退；共享业务回调安装自定义入口后须将已借用参数复制成独立快照，不能重读源重新规划。保留自定义 warp 回调后的输入保护。
- mask 集中上传工作区默认最多保留 32 MiB host + 32 MiB device，按可见 patch 总面积延迟分配，另列 `cache_info()['patch_upload']`。mask 数值逐帧验证/重写；GPU/混合来源与 negative/conjugate Tensor 逻辑视图沿用旧路径，不能因改复制顺序改变旧结果。超预算临时 host 在后续空帧/fallback/释放时也必须结束持有。
- `scripts/run.ps1 prepare` 分开比较 baseline/route_only/mask_only/both，私有开关仅供诊断；正式轮次须平衡且验证实际工作区计数。`check-prepare` 在真实输入长测中额外更新 mask 并记录普通 CPU 参数空间；逐像素对照范围与持续帧数要分开报告。`test_prepare_cuda` 需显式 `PASTE_TEST_DEVICE=cuda:0`，对照目录默认 `results/baseline_prepare_6cebfce`，可用 `PASTE_PREPARE_BASELINE` 指定。
- 更新 `RENDERER_VERSION` 时同步 `benchmark_shared_paste.BENCHMARK_VERSION` 和 `benchmark_paste_gpu_batch` 的两个版本常量；旧脚本会因 bundle 版本不匹配拒绝运行，不得把 CLI 拒绝算作测试通过。
- `batch_chunk_size="auto"` 只查询 `paste_execution_policy.py` 的离线实测表，未知硬件/形状/布局/B回退整批；0强制原整批，正整数手动分块。修改策略必须补对应完整NumPy对照，不能以微基准推断整个流程收益。
- perf4.4 的 108 个合成场景已检查帧中，bucketed 与 batch 全部零差异；动态 bbox 且素材与有序分组不变时不重新打包素材。但该轮完整调用性能不足以支持自动切换，默认 `batch` 和既有分块策略表保持不变，证据见 [BUCKETED_OPTIMIZATION_REVIEW.md](BUCKETED_OPTIMIZATION_REVIEW.md)。任何非零画面差异都不能因性能收益而获准切换后端。
- 既有新增 auto 档位仅在 B8/B16、4/16 广告位、已测源尺寸循环及 108×320 ROI 等条件内精确匹配，并沿用现有硬软件限制；不外推到 32 广告位。
- 桶素材缓存 key 为素材 key 加有序桶成员，只保留当前一份。bbox/ROI 容量变化且分组不变时重建 LUT 但复用打包素材；`area_caps` 改变成员分组时必须重打包。身份/版本/B/顺序/尺寸变化、素材缓存 miss、显式失效及禁用缓存仍按接口失效；实时 grid/透明度不缓存。
- Triton zeros/border 桶路径复用 batch 的同一 warp/alpha 内核，以 B×M 和 1×L 写出 patch，再走 palette/LUT；eager/reflection 保留原 PyTorch 采样。修改时仍须验证数值顺序及完整输出。
- 历史素材 stack 受 `max_cached_bytes`（默认 512 MiB）、尺寸数和条目数共同约束；当前完整素材工作集另计，可大于预算。revision 精确变化必须清旧代；历史条目和当前工作集须保留 source 强引用，不能用裸 `id` 替代所有权。预算不含 source storage、桶打包、LUT、输出或分配器保留量。
- 图像/map 计划缓存只允许有限 CPU 元数据，不保留动态像素/透明度/mask。每帧仍校验实时输入形状、成员与布局，并应用当前 Tensor；不能以相同 frame_id、shape 或对象 id 跳过值变化。输入准备阶段的局部校验与直接 core 的公开校验都须保留完整契约。
- `_warp_paste_into_supported` 只供已经完成支持/padding 检查的内部路径使用；公开 `warp_paste_into` 仍须拒绝不支持的输入。删除重复 host 检查不能改变 launch 或 GPU 数学。
- 兼容render_frame的`BoundedCameraHistory`由业务显式接入；ImageRenderer自行创建有界历史并复制每帧CPU相机状态。按单调frame ID更新，seek用renderer.reset_history清历史和工作集。只有此容器走直接窗口lookup，普通Mapping保留原行为、不裁剪外部历史。
- builder/adapter 的 `release_resources()` 在调用之间或退出时释放库持有资源；返回的独立 NumPy 仍由调用方拥有，不能复用覆盖其存储。业务输入、历史和下游队列也须有明确上限/释放时机。不得用逐帧 `gc.collect()`、`empty_cache()` 或全局同步掩盖资源问题，也不能把 `reserved` 直接解释为泄漏。

## 回归与性能证据

从项目根目录运行与改动有关的检查。两种素材数的共享流程检查：

```powershell
.\scripts\run.ps1 check --device cuda:0 --profile-checks
.\scripts\run.ps1 check-policy
.\scripts\run.ps1 check-backends --device cpu
.\scripts\run.ps1 check-backends --device cuda:0 --output results/backend_contracts.json
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --check-only
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 300x600 --check-only
.\scripts\run.ps1 batch --device cuda:0 --batch 8 --slots 16 --blur-backend triton --ad-sizes 200x200 500x500 300x600 --check-only
```

直播缓存、局部输入准备或公共 kernel guard 改动先运行 CPU 测试，再做 GPU 完整输出与持续调用检查：

```powershell
$workspaceRoot = Split-Path -Parent (Get-Location).Path
$pythonCandidate = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else { Join-Path $workspaceRoot 'env\paste_gpu_performance\Scripts\python.exe' }
$projectPython = (Resolve-Path -LiteralPath $pythonCandidate -ErrorAction Stop).Path
& $projectPython -m unittest test_streaming_history test_streaming_lifecycle test_adapter_preparation test_core_guards test_backend_policy
& $projectPython -B -m unittest test_render_api
& $projectPython -B -m unittest test_live_render_example
& $projectPython -B -m unittest test_route_plan test_shared_mask test_render_img
.\scripts\run.ps1 check-render-img --device cuda:0 --frames 1000 --output results/render_img_check.json
.\scripts\run.ps1 check-streaming --device cuda:0 --entry-point render_frame --baseline results/baseline_render_api_4b2b311 --baseline-label 4b2b311 --frames 10000 --warmup 200 --sample-every 100 --check-every 250 --revision-every 1 --retain-outputs 3 --output results/render_api_streaming_final_10000.json
```

上述命令在仓库根目录执行；基线须先恢复，`--output` 应改为未占用的新结果文件名。不能把 README 的示例路径或旧报告中的本机路径当作当前环境事实。

接口组织改动还须针对 `4b2b311` 的隔离旧源码对照完整调用：新 `render_frame` 与旧共享准备+adapter 输出逐字节一致，`last_execution` 决策一致；覆盖动态 B、同帧原地更新、revision/成员变化、无广告与 way 优先、释放重用和旧输出存活。`compare` / `check-streaming --entry-point render_frame` 只切换当前路径，旧 baseline/oracle 必须继续使用隔离旧实现；默认 `legacy` 路线不变。CPU mock/别名测试不能代替 GPU 图像对照，`test_render_api` 的额外 GPU 项需显式设 `PASTE_TEST_DEVICE=cuda:0`。具体执行记录见 `RENDER_API_REVIEW.md`。本轮不以类/函数迁移推断性能收益。

持续运行报告要区分库 retained stack、当前工作集、源输入、业务输出、CUDA allocated/reserved 以及固定页分配器保留量；记录预热后趋势、释放前后和重用后的状态。只对实际检查帧声称逐字节一致，有限帧合成测试不能证明生产直播永不泄漏。共享 GPU 的基准与长测必须串行，避免相互污染。

布局 / 单帧相关改动再检查 B1；缓存改动检查 `--sigma-pattern repeated` 及三尺寸容量。完整计时将共享基准的 `--check-only` 换为 `--diagnose`，保持相同 GPU、分辨率、B、广告位、核尺寸和预热 / 迭代数。报告完整调用结果与正确性差异，不能把独立微基准相加，或与旧纯 adapter 的基线直接相减。真实生产性能还需要实际回调、视频和编码 / 发送流程验证。

后端或准入策略改动还须运行对应 `check-backends` / `check-policy`，并用 `backends --check-timed-steps --require-exact bucketed` 验证当前桶实际计时帧逐字节一致；策略门槛测试不能替代 GPU 图像对照。基准需计入共享准备、素材/每帧 grid/透明度打包、动态布局和可立即读取的独立 NumPy 输出；确认轮数应是候选数的整数倍。`cold_assets_ms` 为核已预热后的单次素材重建，不能当作稳定 P95 或自动策略依据。桶改动使用 `backends --baseline results/baseline_bucketed_9c6e930 --candidates before_bucketed bucketed batch_whole --require-exact bucketed` 对照旧桶、新桶和 batch；batch 回归仍可用保留源码与 `compare --slots ...` 同进程对照。

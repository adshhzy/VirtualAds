# Codex 后续迭代

## 项目与运行

- VirtualAds 是 Python GPU 库和命令行基准，没有 Web 服务。当前版本 `2026-10-21-perf4.5-streaming`。原包 perf4.1 基线为 `0b97beb`；最初 perf4 为 `480fad5`。本轮整理直播缓存上限、重复准备和资源生命周期，未改 batch 默认或 auto 表；实现与验收状态见 `STREAMING_REVIEW.md`。
- 先尊重调用方的 `PASTE_GPU_PYTHON`；未设置时使用 `scripts/run.ps1` 的默认解释器，即仓库父目录下 `env\paste_gpu_performance\Scripts\python.exe`。不要在新机器上硬编码旧 E 盘路径；不要直接用系统 Python 运行项目，也不要改动其他项目环境。若环境不存在，按 README 创建。
- 当前 Windows 锁文件要求现有 Python 3.12 与 uv。`scripts/setup.ps1 -PythonPath ... -UvPath ...` 创建项目环境并安装 `requirements-lock.txt`，不会下载/安装 Python 或 uv；torch 独立使用官方 cu128 源。自定义 `-EnvironmentPath` 后相应设置 `PASTE_GPU_PYTHON`。依赖调整应同步锁文件并验证 CUDA / Triton 兼容。
- 本机已测参考：Windows、RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128；当时解释器为 `E:\dev\virtualAds\env\paste_gpu_performance\Scripts\python.exe`。此路径是历史记录，不是所有克隆的固定入口。
- `scripts/run.ps1 shared ...` 对应 `benchmark_shared_paste.py`；`scripts/run.ps1 batch ...` 对应 `benchmark_paste_gpu_batch.py`。
- `scripts/run.ps1 backends ...` 对应 `benchmark_backend_comparison.py`；`before_bucketed` 候选配合 `--baseline` 在同进程隔离比较旧桶、当前桶和 batch，`--require-exact` 指定必须零差异的候选。`check-backends` 对应 `check_backend_contracts.py`；`check-policy` 对应 CPU 元数据测试 `test_backend_policy.py`。旧 batch 源码对照仍用 `compare` / `benchmark_dynamic_comparison.py`，`--slots` 为单个正整数，默认 16。
- `scripts/run.ps1 check-streaming` 对应 `check_streaming_stability.py`，需要 `--baseline` 指向保留源码；`--frames` 是固定/动态两阶段合计测量帧数，不含预热。新克隆先按 README 的 `git archive`/`Expand-Archive` 步骤恢复 `6edf109`；旧桶基线为 `9c6e930`。禁止为恢复基线覆盖或删除已有 results。运行示例与内存指标边界见 `STREAMING_REVIEW.md`。
- `shared_paste_example.py` 展示业务接入；`paste_gpu_batch_adapter.py` 负责接口、素材缓存和输出，`paste_gpu_batch.py` 负责核心路由。共享参数逻辑位于 `paste_shared_params.py` / `paste_camera_filter.py`，GPU 内核在 `paste_*_kernels.py`。
- `paste_asset_cache.py` 负责历史素材 stack 的预算/LRU/revision；当前素材与源身份引用由 adapter 持有。优先阅读 `README.md`、`ARCHITECTURE.md` 和 `STREAMING_REVIEW.md`；`BUCKETED_OPTIMIZATION_REVIEW.md`、`BACKEND_COMPARISON_REVIEW.md` 等为历史记录，不能据旧文档扩大当前策略范围。results/logs/env/缓存不随 Git 分发；报告中的原始测试 artifact 不在新克隆内，恢复源码不能恢复旧测量数据。
- 新克隆用已有 uv 执行 `uv.exe tool run --from ruff==0.14.1 ruff format --check .` 和 `uv.exe tool run --from ruff==0.14.1 ruff check .`。`scripts/lint.ps1` 仅适用于已有 `..\env\tools\bin\uv.exe` 的工作区，不读取 setup 的 `-UvPath`；`-Fix` 仅统一格式再检查。Ruff 0.14.1/100列/LF/Python3.9语法目标，配置在 `pyproject.toml`，工具隔离于GPU环境。保留平铺导入路径；优先按阶段职责放置 helper，不引入只为包装参数的层层对象。

## 必须保持的行为

- 默认保持 `normalization="legacy"`、`camera_index_policy="legacy"`；性能修改不要同时混入 `matched` / `corrected` 的画面行为变更。
- 默认 NumPy `uint8`：B1 为 HWC、多组为 BHWC；显式 `squeeze_single=False` 保留 B1 维。固定页输出必须独立，不得被下一帧覆盖。
- K、畸变系数、单应矩阵可能原地更新；不能仅按 frame_id 缓存几何结果。尺寸按实际素材 / 显式 resize 后的高宽处理。
- 保留 ROI 局部坐标、`z + 1e-8`、现有采样 / alpha / 通道顺序语义。不默认引入 FP16、TF32 或低秩近似。
- `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 缺少生产定义。基准回调只用于合成 fixture，不得将它们冒充生产定义或在接入代码中替代业务方法。
- Gaussian 缓存使用精确 sigma，默认最多 16 项且只保留当前 owner/回调/版本/device 上下文；配置变更需更新版本或清缓存。上下文变化应释放旧 owner 强引用。可分离路径必须验证实际二维核，不靠函数名推定可分离。
- builder / adapter 在 worker 内初始化后复用，调用方在同一计算流上串行使用；不每帧重建或清缓存。分块路径内部使用专属复制流，事件连接计算与复制；返回及异常退出前等待复制完成，禁止覆写在途GPU源或调用方持有的CPU结果。
- `batch_chunk_size="auto"` 只查询 `paste_execution_policy.py` 的离线实测表，未知硬件/形状/布局/B回退整批；0强制原整批，正整数手动分块。修改策略必须补对应完整NumPy对照，不能以微基准推断整个流程收益。
- perf4.4 的 108 个合成场景已检查帧中，bucketed 与 batch 全部零差异；动态 bbox 且素材与有序分组不变时不重新打包素材。但该轮完整调用性能不足以支持自动切换，默认 `batch` 和既有分块策略表保持不变，证据见 [BUCKETED_OPTIMIZATION_REVIEW.md](BUCKETED_OPTIMIZATION_REVIEW.md)。任何非零画面差异都不能因性能收益而获准切换后端。
- 既有新增 auto 档位仅在 B8/B16、4/16 广告位、已测源尺寸循环及 108×320 ROI 等条件内精确匹配，并沿用现有硬软件限制；不外推到 32 广告位。
- 桶素材缓存 key 为素材 key 加有序桶成员，只保留当前一份。bbox/ROI 容量变化且分组不变时重建 LUT 但复用打包素材；`area_caps` 改变成员分组时必须重打包。身份/版本/B/顺序/尺寸变化、素材缓存 miss、显式失效及禁用缓存仍按接口失效；实时 grid/透明度不缓存。
- Triton zeros/border 桶路径复用 batch 的同一 warp/alpha 内核，以 B×M 和 1×L 写出 patch，再走 palette/LUT；eager/reflection 保留原 PyTorch 采样。修改时仍须验证数值顺序及完整输出。
- 历史素材 stack 受 `max_cached_bytes`（默认 512 MiB）、尺寸数和条目数共同约束；当前完整素材工作集另计，可大于预算。revision 精确变化必须清旧代；历史条目和当前工作集须保留 source 强引用，不能用裸 `id` 替代所有权。预算不含 source storage、桶打包、LUT、输出或分配器保留量。
- 图像/map 计划缓存只允许有限 CPU 元数据，不保留动态像素/透明度/mask。每帧仍校验实时输入形状、成员与布局，并应用当前 Tensor；不能以相同 frame_id、shape 或对象 id 跳过值变化。输入准备阶段的局部校验与直接 core 的公开校验都须保留完整契约。
- `_warp_paste_into_supported` 只供已经完成支持/padding 检查的内部路径使用；公开 `warp_paste_into` 仍须拒绝不支持的输入。删除重复 host 检查不能改变 launch 或 GPU 数学。
- `BoundedCameraHistory` 必须由业务显式接入，按单调 frame ID 添加真实独立状态，seek 时 reset。只有该类型走直接窗口 lookup；普通 Mapping 保留全部 key 校验与原行为，不自动裁剪外部历史。
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
.\scripts\run.ps1 check-streaming --device cuda:0 --baseline results/baseline_streaming_6edf109 --frames 10000 --warmup 200 --sample-every 100 --check-every 250 --revision-every 1 --retain-outputs 3 --output results/streaming_10000.json
```

上述命令在仓库根目录执行；基线须先恢复，`--output` 应改为未占用的新结果文件名。不能把 README 的示例路径或旧报告中的本机路径当作当前环境事实。

持续运行报告要区分库 retained stack、当前工作集、源输入、业务输出、CUDA allocated/reserved 以及固定页分配器保留量；记录预热后趋势、释放前后和重用后的状态。只对实际检查帧声称逐字节一致，有限帧合成测试不能证明生产直播永不泄漏。共享 GPU 的基准与长测必须串行，避免相互污染。

布局 / 单帧相关改动再检查 B1；缓存改动检查 `--sigma-pattern repeated` 及三尺寸容量。完整计时将共享基准的 `--check-only` 换为 `--diagnose`，保持相同 GPU、分辨率、B、广告位、核尺寸和预热 / 迭代数。报告完整调用结果与正确性差异，不能把独立微基准相加，或与旧纯 adapter 的基线直接相减。真实生产性能还需要实际回调、视频和编码 / 发送流程验证。

后端或准入策略改动还须运行对应 `check-backends` / `check-policy`，并用 `backends --check-timed-steps --require-exact bucketed` 验证当前桶实际计时帧逐字节一致；策略门槛测试不能替代 GPU 图像对照。基准需计入共享准备、素材/每帧 grid/透明度打包、动态布局和可立即读取的独立 NumPy 输出；确认轮数应是候选数的整数倍。`cold_assets_ms` 为核已预热后的单次素材重建，不能当作稳定 P95 或自动策略依据。桶改动使用 `backends --baseline results/baseline_bucketed_9c6e930 --candidates before_bucketed bucketed batch_whole --require-exact bucketed` 对照旧桶、新桶和 batch；batch 回归仍可用保留源码与 `compare --slots ...` 同进程对照。

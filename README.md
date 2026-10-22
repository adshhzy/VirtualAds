# VirtualAds

VirtualAds 是 PyTorch / Triton GPU 广告贴图库和命令行基准程序，没有 Web 服务或浏览器界面。当前版本为 `2026-10-23-perf4.9-prepare`。

新业务入口为 `ImageRenderer.render_img(data_dict, frame, imgs_raw)`：worker 初始化一次，从 `ad_params_N` 读取各路配置，合成并羽化共用 mask，相机/畸变准备一次后按兼容布局批量渲染。详见 [三参数接口说明](RENDER_IMG.md)。`render_frame` 和旧 adapter 继续兼容；既有贴图数学、默认 `batch` 与 `auto` 策略保持，新增 mask 羽化按要求改变遮挡边缘过渡。

## 开始使用

先读 [启动与在线业务接入](STARTUP_AND_API.md)：安装环境、各模块启动方式、生产 owner 回调、三参数调用、动态路数和退出清理都在其中。在线业务使用进程内 Python API；renderer 在直播 worker 内创建一次，再逐帧调用。

已安装项目环境后，在仓库根目录运行新的三参数演示：

```powershell
.\scripts\run.ps1 render-img-example --device cuda:0 --frames-per-phase 2 --output results/render_img_demo.json
```

[render_img_example.py](render_img_example.py) 自带合成输入，演示动态路数、素材替换/原地更新、B0 暂停恢复和关闭，执行完退出。报告路径须未占用。需要旧 `render_frame` 的 packet/sink 示例时，继续使用 [live_render_example.py](live_render_example.py)。生产接入使用自己的相机回调、帧源和编码器。

## 版本与验证记录

perf4.7 三参数入口验收通过 117 项单元回归、1027 次合成调用及 3336 样例配置的 1080p 接线检查；逐路旧入口对照零像素差，mask 独立 Gaussian 参考误差小于 `1e-6`。该轮输入图像/回调为合成数据，验收边界与命令见 [RENDER_IMG.md](RENDER_IMG.md)。
perf4.8 集中上传矩阵/透明度，并将连续 CPU uint8 背景的 float32 转换移到 GPU。真实 3336 输入的 B8/S16 固定/动态 bbox 完整调用平均耗时分别下降 17.75% / 8.94%，全部计时帧与旧源码逐像素一致；另完成 10,000 帧工作区复用检查，测量边界和内存成本见 [上传优化报告](UPLOAD_OPTIMIZATION_REVIEW.md)。

perf4.9 复用本帧素材描述和 CPU 参数工作区，并集中上传局部 CPU mask。同一 3336 样例下，相对 perf4.8 的固定/动态 bbox 完整调用均值降至 8.745 / 8.907 ms，分别下降 26.98% / 26.96%。四候选共 7,680 次计时调用全部逐像素一致，另完成 10,000 帧固定/动态工作区复用检查；默认 auto 不变。分项收益、分阶段测量及内存成本见 [准备阶段优化报告](PREPARATION_OPTIMIZATION_REVIEW.md)。

先读 [代码阅读与修改指南](ARCHITECTURE.md) 和 [统一渲染接口说明](RENDER_API_REVIEW.md)。perf4.5 的性能/长期运行证据保留在 [直播运行报告](STREAMING_REVIEW.md)，不作为本轮新增收益。

统一入口迁移验收时，59 项单元测试及既有 GPU 回归通过；公开入口完成 10000 帧合成长测，42 次旧源码抽查画面与策略一致。完整调用均值在四个 B 档位有升有降，不据此宣称提速；采样范围、内存观察及限制见 [接口验收记录](RENDER_API_REVIEW.md)。

[dynamic_paste_example.py](dynamic_paste_example.py) 与 [shared_paste_example.py](shared_paste_example.py) 保留旧业务辅助函数并委托统一入口；新调用方式见下方“接入与迭代约束”。所有方式均沿用真实业务回调。

旧接口的 [直播 worker 示例](LIVE_RENDER_EXAMPLE.md) 通过 `live-example` 命令演示动态 B、素材 revision、暂停/恢复、无广告及换流清理；演示源与 sink 使用合成数据和本地检查，不包含编码或真实推流。

`ImageRenderer` 自带默认容量 9 的有界相机历史，退出使用 `with` 或 `renderer.close()`，同时释放 mask 工作区；旧 `render_frame` 路径由业务显式维护 `BoundedCameraHistory` 并调用 `release_renderer(owner)`。两种方式均需限制下游输出队列。素材历史 stack 默认预算为 512 MiB，当前素材、mask 和输出另计，这不是进程总内存上限。此前 10000 帧观察及未定根因编译异常保留在 [直播运行报告](STREAMING_REVIEW.md)。

格式与静态检查使用 Ruff 0.14.1，运行于独立 uv 工具环境；新克隆的通用命令见下文，不改 GPU 依赖。

代码来源于[原分享会话](https://chatgpt.com/share/6ad7de37-4b64-83ec-9ac2-1ad7013e17b5)及[perf4.1 分享会话](https://chatgpt.com/share/6ad82a9c-6d28-83ec-a454-ff17371706f4)。归档与原包校验见 [PERF4_1_REVIEW.md](PERF4_1_REVIEW.md)，初次部署记录见 [VALIDATION.md](VALIDATION.md)；当前代码包含后续本地优化。原包说明保留在 [SHARED_PIPELINE_README.md](SHARED_PIPELINE_README.md) 和 [paste_gpu_performance_README.txt](paste_gpu_performance_README.txt)。

## 文档导航

| 阅读目的 | 入口 |
|---|---|
| 从启动到线上三参数调用、一次性/逐帧参数、退出清理 | [STARTUP_AND_API.md](STARTUP_AND_API.md)、[render_img_example.py](render_img_example.py) |
| 安装、运行与基线恢复 | 本页下面的步骤 |
| 三参数入口、逐路配置、共用 mask 羽化 | [RENDER_IMG.md](RENDER_IMG.md)、[paste_renderer.py](paste_renderer.py) |
| 一帧执行顺序、模块职责、形状与缓存 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 矩阵/透明度集中上传、uint8 背景上传与性能证据 | [UPLOAD_OPTIMIZATION_REVIEW.md](UPLOAD_OPTIMIZATION_REVIEW.md) |
| 本帧素材元数据复用、CPU 参数工作区和 mask 集中上传 | [PREPARATION_OPTIMIZATION_REVIEW.md](PREPARATION_OPTIMIZATION_REVIEW.md) |
| 统一入口、业务循环与旧接口迁移 | [RENDER_API_REVIEW.md](RENDER_API_REVIEW.md)、[paste_renderer.py](paste_renderer.py) |
| 旧 render_frame 的直播 worker、帧源/sink 接入与动态阶段 | [LIVE_RENDER_EXAMPLE.md](LIVE_RENDER_EXAMPLE.md)、[live_render_example.py](live_render_example.py) |
| perf4.5 性能、长期运行与已知限制 | [STREAMING_REVIEW.md](STREAMING_REVIEW.md) |
| 兼容旧业务辅助函数 | [dynamic_paste_example.py](dynamic_paste_example.py)、[shared_paste_example.py](shared_paste_example.py) |
| Codex / 开发者修改约束与回归 | [AGENTS.md](AGENTS.md) |
| 较早版本的验证记录 | [桶优化](BUCKETED_OPTIMIZATION_REVIEW.md)、[后端对比](BACKEND_COMPARISON_REVIEW.md)、[初次部署](VALIDATION.md) |

报告是相应版本、机器与合成输入的测量记录，不能直接当作新机器上的性能或生产验收结论。仓库包含代码与文档；`results/`、`logs/`、环境目录和编译缓存不随 Git 分发。报告中的原始 JSON、日志及诊断快照路径是测试机器的本地记录，新克隆不含这些文件；旧版本源码可按下文从 Git 历史恢复，原测量结果则不会由此恢复。

## 新克隆：环境与启动

当前锁文件与 PowerShell 安装入口针对 Windows / Python 3.12，包含 `triton-windows`；它不是可直接跨平台使用的通用锁文件。先准备 Git、**已安装的 Python 3.12 和 uv**，再创建项目环境。`setup.ps1` 使用传入的可执行文件安装锁定依赖，不负责下载或安装 Python / uv。GPU 渲染还需要可用的 NVIDIA CUDA 设备与驱动；报告中的已测 GPU 为 RTX 4090。

在希望存放项目的父目录打开 PowerShell，完整克隆仓库：

```powershell
git clone https://github.com/adshhzy/VirtualAds.git
Set-Location VirtualAds
```

下面通过已安装的 Windows `py` launcher 选择 Python 3.12，并从 PATH 查找 uv；如果没有 `py` launcher，可直接把现有 Python 3.12 的绝对路径传给 `-PythonPath`。

```powershell
$pythonLauncher = (Get-Command py.exe -CommandType Application -ErrorAction Stop).Source
$python312Path = & $pythonLauncher -3.12 -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw 'An existing Python 3.12 installation is required.' }
$uvExecutable = (Get-Command uv.exe -CommandType Application -ErrorAction Stop).Source
.\scripts\setup.ps1 -PythonPath $python312Path -UvPath $uvExecutable
.\scripts\run.ps1 shared --version
.\scripts\run.ps1 batch --version
```

默认项目环境在**仓库父目录**的 `env\paste_gpu_performance`，缓存在同一父目录的 `env\cache`；环境名不会随克隆目录名改变。例如仓库为 `C:\Work\VirtualAds` 时，解释器为 `C:\Work\env\paste_gpu_performance\Scripts\python.exe`。无需激活环境。多个克隆若共用同一父目录，默认也会共用这个环境；需要隔离时使用下述覆盖参数。

`requirements-lock.txt` 锁定依赖，包括 PyTorch `2.9.1+cu128`、NumPy `2.5.2`、SciPy `1.18.1` 和 `triton-windows 3.5.1.post24`。安装脚本分别从 PyTorch 官方 cu128 源下载 torch、从 PyPI 下载其他依赖，并执行依赖和导入检查。依赖安装完成后，`--help` / `--version` 不需要 GPU。

也可直接指定现有工具与独立环境路径。以下路径是示例，按本机安装位置替换：

```powershell
.\scripts\setup.ps1 -PythonPath C:\Python312\python.exe -UvPath C:\Tools\uv.exe -EnvironmentPath D:\venvs\paste_gpu -CachePath D:\cache\paste_gpu
$env:PASTE_GPU_PYTHON = 'D:\venvs\paste_gpu\Scripts\python.exe'
.\scripts\run.ps1 shared --version
```

路径参数中的相对路径按调用者当前目录解析。`run.ps1` 优先使用 `PASTE_GPU_PYTHON`，未设置时才使用仓库父目录下的默认环境；已有的 `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` 也会保留使用。改变 `setup.ps1 -EnvironmentPath` 后应相应设置 `PASTE_GPU_PYTHON`。

`setup.ps1 -CachePath` 指定 uv 安装缓存；编译缓存由 `run.ps1` 根据仓库父目录设置，或使用上述两个已有的环境变量。

本机历史记录：测试仓库为 `E:\dev\virtualAds\paste_gpu_performance`，解释器为 `E:\dev\virtualAds\env\paste_gpu_performance\Scripts\python.exe`。不传参数的 setup 默认寻找 `..\env\python312\python.exe` 和 `..\env\tools\bin\uv.exe`，这是当时工作区的工具布局；新克隆不会自动获得这两个可执行文件。

## 格式与静态检查

在仓库根目录使用 PATH 中已安装的 uv：

```powershell
uv.exe tool run --from ruff==0.14.1 ruff format --check .
uv.exe tool run --from ruff==0.14.1 ruff check .
```

需要自动格式化时，将第一条中的 `--check` 去掉，再执行第二条。未加入 PATH 的 uv 可用 `& '实际路径\uv.exe'` 调用。原工作区也可运行 `.\scripts\lint.ps1` 或 `.\scripts\lint.ps1 -Fix`，但该脚本固定寻找 `..\env\tools\bin\uv.exe`；它不读取 setup 的 `-UvPath` 或 `PASTE_GPU_PYTHON`。

## 运行旧 render_frame 直播循环示例

在仓库根目录运行小画布 GPU 演示，默认 dense 模糊：

```powershell
.\scripts\run.ps1 live-example --device cuda:0 --frames-per-phase 3 --output results/live_render_demo.json
```

追加 `--separable` 使用已验证的合成核；追加 `--snapshot-dir results/live_render_frames` 保存可选 PPM。报告文件须不存在，快照目录须新建或为空。帧源、packet、sink 和暂停/重置规则见 [LIVE_RENDER_EXAMPLE.md](LIVE_RENDER_EXAMPLE.md)。演示需要 CUDA/Triton，CPU 单测只检查生命周期；它不包含视频编码或实际网络推流。

本次 dense 24 帧、separable 200 帧演示及含既有回归的 72 项单元测试通过；检查范围是合成 hash、无广告背景、路由及生命周期，具体计数和限制见示例文档。

## 运行分享页要求的基准

共享参数准备、贴图、模糊和独立 NumPy 输出一起计时。下面两条命令均为 1080p、B8、16 个广告位、9×9 核，默认预热 20 次、计时 100 次，sigma 连续变化：

```powershell
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --diagnose
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 300x600 --diagnose
```

需要同轮比较完整 eager 基线时，追加 `--include-eager`。不要把不同运行、不同机器或不同计时范围的节省量相加。

把 `--diagnose` 换为 `--check-only` 可只检查正确性。`--sigma-pattern repeated` 可另外检查精确 sigma 缓存复用。原核心 / adapter 基准不包含共享参数构造成本：

```powershell
.\scripts\run.ps1 batch --device cuda:0 --batch 8 --slots 16 --blur-backend triton --ad-sizes 200x200 500x500 300x600 --check-only
```

脚本转发剩余参数并保留 Python 退出码。日志由调用者保存，例如：

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --diagnose *> results\shared-two-sizes.log
```

## 从 Git 历史恢复基线源码

`compare`、`check-streaming` 和 `backends --candidates before_bucketed ...` 需要独立的旧源码目录。目录被忽略，但相应源码在仓库 Git 历史中；请在仓库根目录执行以下 PowerShell。此例恢复 perf4.5 直播对照用的 `6edf109`，不改变当前 checkout，也不覆盖已有目录或 ZIP：

```powershell
$baselineCommit = '6edf109'
$baselineFolder = 'baseline_streaming_6edf109'
$resultsRoot = Join-Path (Get-Location).Path 'results'
$baselinePath = Join-Path $resultsRoot $baselineFolder
$archivePath = Join-Path $resultsRoot ($baselineFolder + '.zip')
if ((Test-Path -LiteralPath $baselinePath) -or (Test-Path -LiteralPath $archivePath)) {
    throw 'Baseline destination or ZIP already exists; inspect it or choose a new folder name.'
}
git cat-file -e ($baselineCommit + '^{commit}')
if ($LASTEXITCODE -ne 0) { throw 'Baseline commit is missing; use a complete Git clone.' }
New-Item -ItemType Directory -Path $resultsRoot -Force | Out-Null
git archive --format=zip -o $archivePath $baselineCommit
if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
Expand-Archive -LiteralPath $archivePath -DestinationPath $baselinePath -ErrorAction Stop
```

按所需对照替换上面两个变量，其他步骤不变：

| `$baselineCommit` | `$baselineFolder` | 对照入口 |
|---|---|---|
| `4b2b311` | `baseline_render_api_4b2b311` | 本轮接口迁移前的源码；用于统一入口与旧完整流程逐字节对照 |
| `6edf109` | `baseline_streaming_6edf109` | `check-streaming --baseline results/baseline_streaming_6edf109`；`compare --baseline results/baseline_streaming_6edf109 --baseline-label before` |
| `9c6e930` | `baseline_bucketed_9c6e930` | `backends --baseline results/baseline_bucketed_9c6e930 --candidates before_bucketed bucketed batch_whole` |
| `3ecc3b7` | `baseline_backend` | 较早 auto 扩张对照：`compare --baseline results/baseline_backend --baseline-label before_auto` |

普通 `git clone` 会包含这些历史；GitHub 的源码 ZIP 下载不包含 Git 历史。若使用浅克隆，先检查 `git rev-parse --is-shallow-repository`，结果为 `true` 时可执行 `git fetch --unshallow origin` 后重试。不要为恢复基线删除现有 `results`；如果目标已存在，先核对内容或改用新的目录名，并把相同路径传给 `--baseline`。基准的 `--output` 也应选择未占用的新文件名，原始历史报告中的结果路径不代表待覆盖目标。

## 后端对比与回归

perf4.4 筛查覆盖 B=1/4/8/16、4/16/32 个广告位、1～3 种素材尺寸、静态/平移/缩放 bbox，共 108 个合成场景；该轮所有场景已检查帧中的新 bucketed 与 batch 输出均逐字节一致。
完整计时包含共享准备、素材缓存与打包、每帧 grid/透明度打包、布局/LUT 更新，以及完成 GPU 工作后的独立 NumPy 返回。
动态 bbox/ROI 变化仍重建布局和 LUT；素材与有序桶成员不变时不再重新打包素材，`area_caps` 导致分组成员变化时重新打包。
该轮完整调用性能不足以支持自动切换，本轮也未调整默认 `batch` 或既有分块表。桶优化证据见 [perf4.4 报告](BUCKETED_OPTIMIZATION_REVIEW.md)；更早的旧桶观察保留在 [历史报告](BACKEND_COMPARISON_REVIEW.md)。

| 命令 | 工具 | 用途 |
|---|---|---|
| `scripts/run.ps1 backends` | `benchmark_backend_comparison.py` | 比较 batch、当前桶和隔离加载的旧桶；记录完整延迟、正确性和缓存/布局计数 |
| `scripts/run.ps1 render-img-modes` | `benchmark_render_img_modes.py` | 用真实 NPY/BMP 输入比较 eager、dense/可分离模糊的整批与分块；完整 render_img 计时及逐像素检查，见 [模式复测](RENDER_IMG_MODE_REVIEW.md) |
| `scripts/run.ps1 uploads` | `benchmark_upload_comparison.py` | 隔离旧源码，对比完整 B8/S16 调用，静态/动态 bbox，每计时帧逐像素检查 |
| `scripts/run.ps1 prepare` | `benchmark_prepare_comparison.py` | 对照旧版、仅路由准备优化、仅 mask 上传优化和两项合用；平衡轮换、完整调用和逐像素检查 |
| `scripts/run.ps1 profile-uploads` | `profile_upload_stages.py` | 分开记录 CPU 阶段、CUDA 流区间及实际 GPU kernel/传输活动；用于定位而非吞吐结论 |
| `scripts/run.ps1 check-uploads` | `check_upload_streaming.py` | 真实样例的工作区复用、动态 B/slot、B0 恢复和旧源码周期对照 |
| `scripts/run.ps1 check-prepare` | `check_prepare_streaming.py` | 动态路数/slot、逐帧 mask 更新、参数工作区复用、素材切换、暂停恢复与退出清理 |
| `scripts/run.ps1 render-img-example` | `render_img_example.py` | 新业务三参数入口的有限合成演示：动态路数、素材变化、暂停恢复和退出 |
| `scripts/run.ps1 check-backends` | `check_backend_contracts.py` | 检查后端覆盖顺序、广播、布局/素材更新和输出所有权 |
| `scripts/run.ps1 check-policy` | `test_backend_policy.py` | CPU 元数据测试，检查策略精确准入与未知输入回退 |
| `scripts/run.ps1 compare` | `benchmark_dynamic_comparison.py` | 隔离加载旧源码对照当前完整调用；`--slots` 指定广告位数，默认 16 |
| `scripts/run.ps1 check-streaming` | `check_streaming_stability.py` | 检查持续输入变化、输出生命周期和资源保留；使用方法见直播报告 |
| 项目 Python `-B -m unittest test_render_api` | `test_render_api.py` | 公共入口与旧接口兼容；默认 CPU，GPU 验收方式见统一接口报告 |

```powershell
.\scripts\run.ps1 backends --batches 8 --slots 16 --profiles three --layouts static --check-timed-steps
.\scripts\run.ps1 check-backends --device cpu
.\scripts\run.ps1 check-policy
```

同进程对照旧桶、当前桶与 batch，并要求当前桶对 batch 逐字节一致：

```powershell
.\scripts\run.ps1 backends --baseline results/baseline_bucketed_9c6e930 --candidates before_bucketed bucketed batch_whole --require-exact bucketed --batches 8 --slots 16 --profiles three --layouts static translate resize --rounds 3 --check-timed-steps
```

`--baseline` 指向按上节恢复的源码目录，`before_bucketed` 使用其中独立加载的运行时；恢复只需 Git 历史，不需要原测试机器的 `results/`。
`compare` / `check-streaming` 的 `--entry-point render_frame` 可检查当前公共入口，隔离 baseline 仍使用旧调用路线；默认 `legacy` 保留原工具行为。本轮 `4b2b311` 对照命令见 [统一接口报告](RENDER_API_REVIEW.md)。
`backends` 默认检查第 0、1、7 帧；确认策略时使用 `--check-timed-steps` 覆盖实际计时帧，零差异门槛只对报告中的 `checked_steps` 成立。
`--require-exact` 指定的候选出现任何非零像素差即失败；旧桶可保留差异记录。冷素材单次耗时、独立诊断与确认测量的使用方式见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [桶优化报告](BUCKETED_OPTIMIZATION_REVIEW.md)。

## 接入与迭代约束

新的三参数接入和暂停/关闭规则见 [RENDER_IMG.md](RENDER_IMG.md)。以下为兼容 `render_frame` 的底层接入示例：保留业务原有的布局、通道、alpha、缩放等选项，在 worker 内调用一次 `initialize_renderer`；每帧调用 `render_frame`，由它选择当前 way 的广告位、完成一次共享准备并交给执行器。

```python
from paste_renderer import initialize_renderer, render_frame, release_renderer

executor = initialize_renderer(owner, existing_options)
try:
    output = render_frame(
        owner, frame_input, data_dict, frame_id,
        mask_canvas=mask_canvas, material_groups=current_groups,
        camera_history=camera_history, device=device,
    )
finally:
    release_renderer(owner)
```

示例变量来自业务现有数据；连续直播应将初始化放到循环外、`render_frame` 放到循环内，完整有界历史/输出示例见 [RENDER_API_REVIEW.md](RENDER_API_REVIEW.md)。`material_groups` / `camera_history` 省略或为 `None` 时使用 owner 已有字段；显式空素材组拒绝，B=0 由调度方跳过。无广告时跳过共享相机/核准备。

`RenderOptions` 是规范配置类，`RenderExecutor` 承接输入准备、缓存、后端执行和 NumPy 导出。旧 `AdapterOptions` / `PasteBatchAdapter` 分别是同一个类的兼容别名；旧函数、`setup_shared_paste`、`prepare_shared_paste_args` 和动态示例函数继续可用。新旧入口使用 owner 上同一组 builder/执行器对象，不产生第二套缓存。

原始会话未提供 `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 的生产实现；基准中的实现是合成回调。接入时沿用业务原方法，再用真实视频、参数和负载验收。

`RenderOptions.batch_chunk_size` 默认 `"auto"`：仅匹配已测硬件和输入结构时应用离线策略，未知情况回退整批。`0` 强制原整批，正整数强制指定分块上限。执行器的 `last_execution` 可查看实际选择和回退原因。NumPy保持独立存储，返回即可读取；内部复制流不改变调用方串行调用约定。自动策略不强行启用可分离模糊，仍需真实核验证。

此前扩展的分块档位仍限定在 B8/B16、4 或 16 个广告位、已测素材尺寸循环和 108×320 ROI 等完整条件内，并保留现有硬件/软件门槛及更早的已测档位。精确准入表见 [paste_execution_policy.py](paste_execution_policy.py)；本轮未改动该表，也不自动启用 bucketed。不要据此推广到 32 个广告位或任意输入结构。

默认 `normalization="legacy"`、`camera_index_policy="legacy"` 保留原行为；`matched` / `corrected` 是需要单独验收的语义修正。保持 B1 输出 `[H,W,3]`、多组输出 `[B,H,W,3]` 的 NumPy `uint8` 协议；K、畸变系数和单应矩阵须实时读取。GPU worker 初始化及预热后复用 builder / adapter，素材变更按既有接口使缓存失效。

独立 NumPy 输出由其底层 PyTorch Tensor 持有存储；调用方不再引用后，存储才可由分配器回收复用。释放 builder/adapter 不会覆盖调用方仍持有的输出，也不会清空全局分配器。CUDA `reserved` 或进程工作集暂时不下降，不能单独作为泄漏证据；同样不能仅据一个稳定快照宣称没有泄漏。

原包文档中的“本地没有 CUDA GPU”描述的是打包时的环境。初次部署的 GPU 校验见 [VALIDATION.md](VALIDATION.md)，本轮接口迁移验收状态见 [RENDER_API_REVIEW.md](RENDER_API_REVIEW.md)；合成基准通过不等于完成生产视频验收。

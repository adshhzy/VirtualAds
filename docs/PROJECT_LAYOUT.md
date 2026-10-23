# 目录结构与迁移指南

运行时代码集中在 `virtualads/`，演示、性能基准、检查工具、单元测试和说明文档分别存放。目录整理保持渲染数学、动态路数、独立 NumPy 输出和既有 batch auto 策略；运行时版本仍为 `2026-10-23-perf4.9-prepare`。

首次安装和在线调用见 [启动与在线接入](STARTUP_AND_API.md)，一帧处理顺序见 [模块协作](ARCHITECTURE.md)，数据字段见 [render_img 接口](RENDER_IMG.md)。下文所有路径均相对仓库根目录。

## 目录职责

```text
VirtualAds/
├── virtualads/              核心运行库；生产调用只依赖此包及运行依赖
│   ├── __init__.py          公共 API：ImageRenderer、RenderOptions、render_frame 等
│   ├── paste_renderer.py    三参数入口、兼容入口、生命周期管理
│   ├── paste_route_*.py     动态路数规划、本帧元数据与 CPU 参数空间
│   ├── paste_shared_*.py    共用相机/几何、mask、数据类型和网格内核
│   ├── paste_camera_filter.py
│   ├── paste_render_executor.py
│   ├── paste_gpu_batch.py   batch/bucketed 计算流程
│   ├── paste_*_kernels.py   GPU 采样、模糊、融合、mask 内核
│   ├── paste_asset_cache.py
│   ├── paste_execution_policy.py
│   ├── paste_*upload*.py    背景、矩阵、透明度和 mask 上传空间
│   ├── paste_gpu_batch_adapter.py
│   ├── shared_paste_example.py
│   └── dynamic_paste_example.py
├── compat/                  旧平铺模块名的安装别名，无第二份计算实现
├── examples/                三参数与旧直播 worker 的可运行演示
├── benchmarks/              完整调用对比、模式比较和内核微基准
├── tools/                   正确性、长期运行检查和分阶段 profiling
├── tests/                   单元回归及显式 CUDA 合同测试
├── scripts/                 PowerShell 安装、运行、测试、格式检查入口
├── docs/                    当前使用说明、接口与架构
│   ├── reports/             各版本历史验收和性能报告
│   └── legacy/              原分享包的历史说明
├── README.md                项目入口与常用命令
├── AGENTS.md                后续代码修改约束
├── pyproject.toml           包安装/构建与 Ruff 配置
└── requirements-lock.txt    已验证 Windows GPU 环境依赖
```

`results/`、日志、环境、缓存、构建目录由本机生成并被 Git 忽略。原有结果和基线保留在各自位置，不因整理目录重写测量记录。根目录不再堆放 Python 源文件。

两个名称含 `example` 的旧 helper 仍放在 `virtualads/`：它们是旧业务实际导入的兼容函数，委托同一渲染入口，不是需要独立启动的演示程序。真正可运行的直播演示在 `examples/`。核心库不导入 `examples`、`benchmarks`、`tools` 或 `tests`。

## 安装和导入

新环境使用 `scripts/setup.ps1`。脚本安装锁定依赖后，以可编辑模式安装本项目；开发时修改源码即生效，不需每次重装。安装注册规范包和兼容模块，不把整个仓库根目录当作历史平铺模块搜索路径。

已有旧版环境升级目录后，先在仓库根目录执行一次安装。`$projectPython` 必须指向原项目环境；下面沿用启动指南的解释器选择：

```powershell
$workspaceRoot = Split-Path -Parent (Get-Location).Path
$pythonCandidate = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else {
    Join-Path $workspaceRoot 'env\paste_gpu_performance\Scripts\python.exe'
}
$projectPython = (Resolve-Path -LiteralPath $pythonCandidate -ErrorAction Stop).Path
$uvExecutable = (Get-Command uv.exe -CommandType Application -ErrorAction Stop).Source
& $uvExecutable pip install --python $projectPython --no-deps --no-build-isolation --editable .
if ($LASTEXITCODE -ne 0) { throw 'VirtualAds installation failed.' }
& $projectPython -c 'from virtualads import ImageRenderer, RenderOptions; print(ImageRenderer.__module__)'
```

`--no-deps` 保留已验证的 GPU 依赖；`--no-build-isolation` 使用锁文件已安装的 setuptools。若当前环境不完整，应运行 setup，而不是用此短命令替代依赖安装。一个环境只能指向一个本项目可编辑 checkout；多个克隆请使用独立环境。

新业务使用：

```python
from virtualads import ImageRenderer, RenderOptions
from virtualads import initialize_renderer, render_frame, release_renderer
```

需要具体类型或底层实现时使用 `virtualads.paste_renderer` 等包路径。旧写法在安装后仍可用：

```python
import paste_renderer
import virtualads.paste_renderer
from paste_gpu_batch_adapter import AdapterOptions
from virtualads import RenderOptions

assert paste_renderer is virtualads.paste_renderer
assert AdapterOptions is RenderOptions
```

`compat/` 将旧模块名映射到规范包中的同一模块，类、全局状态和缓存不会复制。已有业务不必一次修改所有导入。新代码不继续增加平铺别名；旧 adapter 自身的私有属性代理边界仍按原契约处理。

生产可以构建和安装 wheel。分发内容是 `virtualads` 和旧名兼容模块；源码中的演示、基准、测试、报告不属于运行包。生产环境仍需按部署平台准备并验证 PyTorch/Triton/CUDA，包安装本身不验证 GPU。可编辑安装依赖原 checkout 存在；迁移或删除 checkout 前，应重新安装到新路径或改用 wheel。

在已安装锁定依赖的项目环境中构建：

```powershell
& $uvExecutable build --wheel --no-build-isolation --python $projectPython --out-dir dist
if ($LASTEXITCODE -ne 0) { throw 'VirtualAds wheel build failed.' }
```

将产物传到已准备好 GPU 依赖的生产环境，再安装。路径按部署位置替换；升级后 wheel 文件名中的版本也应相应更新：

```powershell
$productionPython = 'D:\venvs\virtualads-live\Scripts\python.exe'
& $uvExecutable pip install --python $productionPython --no-deps .\dist\virtualads-0.4.9-py3-none-any.whl
if ($LASTEXITCODE -ne 0) { throw 'VirtualAds wheel installation failed.' }
```

## 启动和测试

模块文件不需要逐个启动。业务 worker 创建一次 `ImageRenderer`，逐帧调用 `render_img`，退出时关闭。已有 `scripts/run.ps1` 子命令保持不变，内部改为 Python `-m` 模块启动；从仓库根目录直接运行时，使用完整模块名：

| 用途 | 保留的脚本入口 | 等效 Python 模块 |
|---|---|---|
| 三参数动态直播演示 | `render-img-example` | `-m examples.render_img_example` |
| 旧入口直播演示 | `live-example` | `-m examples.live_render_example` |
| 共享流程基准 | `shared` | `-m benchmarks.benchmark_shared_paste` |
| 核心/执行器基准 | `batch` | `-m benchmarks.benchmark_paste_gpu_batch` |
| 执行策略与动态 B 对比 | `compare` / `dynamic` | `-m benchmarks.benchmark_dynamic_comparison` / `-m benchmarks.benchmark_dynamic_batch` |
| batch/bucketed 对比 | `backends` | `-m benchmarks.benchmark_backend_comparison` |
| 真实样例模式/上传/准备对比 | `render-img-modes` / `uploads` / `prepare` | `-m benchmarks.benchmark_render_img_modes` / `-m benchmarks.benchmark_upload_comparison` / `-m benchmarks.benchmark_prepare_comparison` |
| 单独模糊调参 | `tune` | `-m benchmarks.benchmark_blur_tuning` |
| 三参数/后端正确性 | `check-render-img` / `check-backends` | `-m tools.check_render_img` / `-m tools.check_backend_contracts` |
| 持续运行 | `check-streaming` / `check-uploads` / `check-prepare` | `-m tools.check_streaming_stability` / `-m tools.check_upload_streaming` / `-m tools.check_prepare_streaming` |
| 分阶段诊断 | `profile-uploads` | `-m tools.profile_upload_stages` |
| 动态批次检查 | `check` | `-m tests.test_dynamic_batch` |
| 策略合同 | `check-policy` | `-m tests.test_backend_policy` |

例如：

```powershell
.\scripts\run.ps1 render-img-example --device cuda:0
& $projectPython -m examples.render_img_example --device cuda:0
.\scripts\test.ps1
.\scripts\test.ps1 tests.test_render_api tests.test_route_plan
& $projectPython -B -m unittest discover -s tests -t .
```

`scripts/test.ps1` 未给测试名时自动发现测试；给定测试模块时，只执行所列模块。默认测试中的显式 CUDA 项会跳过；需要 GPU 合同检查时设置 `PASTE_TEST_DEVICE=cuda:0` 并选择相应测试。某些旧源码对照还要求先恢复基线目录，见各报告。不要在多个进程同时跑 GPU 性能或长测。

不能用 `python examples/render_img_example.py` 或先切进子目录再直接执行脚本来替代上述模块启动方式。包内相对导入依赖模块上下文；`python virtualads/paste_renderer.py` 也不是业务服务启动命令。

## 旧路径和历史记录

| 原来根目录的文件 | 现在的位置/调用 |
|---|---|
| `paste_*.py`，两个旧 helper | `virtualads/`；安装后的旧名来自 `compat/` |
| `render_img_example.py`、`live_render_*.py` | `examples/` |
| `benchmark_*.py` | `benchmarks/` |
| `check_*.py`、`profile_upload_stages.py` | `tools/` |
| `test_*.py` | `tests/`；unittest 名称增加 `tests.` 前缀 |
| 使用与架构说明 | `docs/` |
| `*_REVIEW.md`、`VALIDATION.md` | `docs/reports/` |
| 两份原包说明 | `docs/legacy/` |

`git archive` 恢复的历史提交仍保留当时平铺布局。比较工具会隔离加载指定旧源码，不能把旧目录改成新结构后再声称是原基线；源文件 SHA 和历史数字保持原样。新报告记录实际新路径。历史报告中的直接 `python file.py` 命令用于对应快照，复跑当前源码应使用上表模块入口或不变的 PowerShell 子命令。

`benchmarks/source_layout.py` 统一定位包目录和旧平铺快照中的实现文件，并生成仓库相对路径的来源记录；只查路径，不导入指定快照或修改模块搜索路径。基线依赖缺失时应报错，不能悄悄读取当前 checkout。

本次整理只改变文件组织、导入和启动/安装方式；不据此宣称新增性能收益。迁移前对照提交为 `b65c3eb`，验收记录应与既有性能报告分开保存。

## 目录迁移验收（2026-10-23）

本机项目环境已完成可编辑安装，未重新解析或升级 GPU 运行依赖。wheel 已构建，并安装到独立目标目录；从仓库外路径验证新旧模块导入指向同一实现。产物包含 20 个核心/兼容 helper 实现、包入口和 20 个旧名别名，不包含示例、基准、工具或测试。

`tests.test_package_layout` 的四项检查通过，覆盖新旧模块/class 身份、旧模块 patch 转发、公共包惰性导出和仓库外可编辑安装导入。文档检查覆盖本地文件链接、Python 3.9 代码块语法及 PowerShell 语法；历史报告除迁移提示与相对链接外，文本和数字保持原样。

20 个核心实现文件相对迁移前 `b65c3eb` 逐字节不变。Windows / RTX 4090 项目环境完成以下结构、启动和像素回归：

| 检查 | 本轮结果 |
|---|---|
| `scripts/test.ps1` 默认发现 | 共 234 项：219 通过，15 项显式 CUDA 测试按默认设置跳过 |
| 显式 CUDA 合同测试 | `tests.test_prepare_cuda`、`tests.test_upload_cuda`、`tests.test_opacity_upload`、`tests.test_render_api`，共 35 项全部通过 |
| CLI 模块启动 | `scripts/run.ps1` 的 19 个子命令 `--help` 全部通过 |
| 三参数演示 | `render-img-example`，8 个阶段、16 次调用通过 |
| 旧直播演示 | `live-example`，接收 24 帧、渲染 21 帧、暂停 3 帧、输出 51 张图像 |
| 三参数完整对照 | `check-render-img --frames 100`，实际 127 次调用、424 路图像，最大 uint8 像素差为 0 |
| 3336 样例 B8 / S16 迁移前后对照 | 固定/动态 bbox × 两候选 × 两轮 × 每轮 12 步，共 96 次计时调用、768 路图像全部逐像素一致，auto 决策一致 |
| 旧 `render_frame` 接口 | 隔离 `4b2b311`，B=1/3/8/16、16 slot，auto 和强制整批均通过精确快照检查 |

3336 对照记录为 `results/structure_20261023/layout_pixels.json`，旧接口对照为 `results/structure_20261023/legacy_api_check.json`。本轮只验证目录迁移后的调用、输入/输出及兼容性；短测轮数不用于声明性能收益，也未重做 10,000 帧长测。构建与回归 artifact 位于本机忽略目录 `results/structure_20261023/`；既有性能和长期运行数字继续以各版本历史报告为准。

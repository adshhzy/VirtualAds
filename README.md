# GPU 广告贴图库

这是 PyTorch / Triton GPU 贴图库和命令行基准程序，没有 Web 服务或浏览器界面。当前本地版本为 `2026-10-21-perf4.5-streaming`，位于 `E:\dev\virtualAds\paste_gpu_performance`。

本轮整理直播长期运行的缓存上限、输入准备和资源释放。默认后端仍为 `batch`，既有 `auto` 分块策略表保持不变。
先读 [代码阅读与修改指南](ARCHITECTURE.md)；本机 12 场景完整调用平均耗时下降 0.64%～6.72%，具体条件及持续运行验收状态见 [直播运行报告](STREAMING_REVIEW.md)。上一轮桶优化结果保留在 [桶优化报告](BUCKETED_OPTIMIZATION_REVIEW.md)，不据此自动切换到 bucketed。

[dynamic_paste_example.py](dynamic_paste_example.py) 展示业务 worker 初始化一次、每帧更新素材组/相机参数并返回原 NumPy 接口；沿用真实业务回调。

直播接入可显式使用 `BoundedCameraHistory(capacity=9)`，按单调帧号写入独立相机记录，并给下游输出队列设置容量。素材历史 stack 默认预算为 512 MiB，当前素材工作集与桶打包存储另计，这不是进程总内存上限。worker 退出时调用 builder 和 adapter 的 `release_resources()`；具体循环示例、10000 帧观察及仍未确定根因的编译异常见 [直播运行报告](STREAMING_REVIEW.md)。

格式与静态检查：`.\scripts\lint.ps1`；自动格式化后检查：`.\scripts\lint.ps1 -Fix`。工具使用独立 uv 环境，不改 GPU 依赖。

代码来源于[原分享会话](https://chatgpt.com/share/6ad7de37-4b64-83ec-9ac2-1ad7013e17b5)及[perf4.1 分享会话](https://chatgpt.com/share/6ad82a9c-6d28-83ec-a454-ff17371706f4)。归档与原包校验见 [PERF4_1_REVIEW.md](PERF4_1_REVIEW.md)，初次部署记录见 [VALIDATION.md](VALIDATION.md)；当前代码包含后续本地优化。原包说明保留在 [SHARED_PIPELINE_README.md](SHARED_PIPELINE_README.md) 和 [paste_gpu_performance_README.txt](paste_gpu_performance_README.txt)。

## 环境与启动

在 PowerShell 中执行：

```powershell
Set-Location E:\dev\virtualAds\paste_gpu_performance
.\scripts\setup.ps1
.\scripts\run.ps1 shared --version
.\scripts\run.ps1 batch --version
```

独立 Python 环境位于 `E:\dev\virtualAds\env\paste_gpu_performance`。默认基础 Python 为 `..\env\python312\python.exe`，uv 为 `..\env\tools\bin\uv.exe`，下载与编译缓存位于 `..\env\cache`。无需激活环境，也不依赖系统 `python` 的选择。

`requirements-lock.txt` 锁定依赖，包括 PyTorch `2.9.1+cu128`、NumPy `2.5.2`、SciPy `1.18.1` 和 `triton-windows 3.5.1.post24`。安装脚本分别从 PyTorch 官方 cu128 源下载 torch、从 PyPI 下载其他依赖，并执行依赖和导入检查。正式 GPU 基准需要可用的 NVIDIA CUDA 设备；`--help` / `--version` 不需要 GPU。

迁移目录时可明确覆盖路径：

```powershell
.\scripts\setup.ps1 -PythonPath C:\Python312\python.exe -UvPath C:\Tools\uv.exe -EnvironmentPath D:\venvs\paste_gpu -CachePath D:\cache\paste_gpu
$env:PASTE_GPU_PYTHON = 'D:\venvs\paste_gpu\Scripts\python.exe'
.\scripts\run.ps1 shared --version
```

路径参数中的相对路径按调用者当前目录解析。`run.ps1` 默认使用上述独立环境；可通过 `PASTE_GPU_PYTHON` 指定另一个解释器。已有的 `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` 环境变量也会保留使用。

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

## 后端对比与回归

perf4.4 筛查覆盖 B=1/4/8/16、4/16/32 个广告位、1～3 种素材尺寸、静态/平移/缩放 bbox，共 108 个合成场景；该轮所有场景已检查帧中的新 bucketed 与 batch 输出均逐字节一致。
完整计时包含共享准备、素材缓存与打包、每帧 grid/透明度打包、布局/LUT 更新，以及完成 GPU 工作后的独立 NumPy 返回。
动态 bbox/ROI 变化仍重建布局和 LUT；素材与有序桶成员不变时不再重新打包素材，`area_caps` 导致分组成员变化时重新打包。
该轮完整调用性能不足以支持自动切换，本轮也未调整默认 `batch` 或既有分块表。桶优化证据见 [perf4.4 报告](BUCKETED_OPTIMIZATION_REVIEW.md)；更早的旧桶观察保留在 [历史报告](BACKEND_COMPARISON_REVIEW.md)。

| 命令 | 工具 | 用途 |
|---|---|---|
| `scripts/run.ps1 backends` | `benchmark_backend_comparison.py` | 比较 batch、当前桶和隔离加载的旧桶；记录完整延迟、正确性和缓存/布局计数 |
| `scripts/run.ps1 check-backends` | `check_backend_contracts.py` | 检查后端覆盖顺序、广播、布局/素材更新和输出所有权 |
| `scripts/run.ps1 check-policy` | `test_backend_policy.py` | CPU 元数据测试，检查策略精确准入与未知输入回退 |
| `scripts/run.ps1 compare` | `benchmark_dynamic_comparison.py` | 隔离加载旧源码对照当前完整调用；`--slots` 指定广告位数，默认 16 |
| `scripts/run.ps1 check-streaming` | `check_streaming_stability.py` | 检查持续输入变化、输出生命周期和资源保留；使用方法见直播报告 |

```powershell
.\scripts\run.ps1 backends --batches 8 --slots 16 --profiles three --layouts static --check-timed-steps
.\scripts\run.ps1 check-backends --device cpu
.\scripts\run.ps1 check-policy
```

同进程对照旧桶、当前桶与 batch，并要求当前桶对 batch 逐字节一致：

```powershell
.\scripts\run.ps1 backends --baseline results/baseline_bucketed_9c6e930 --candidates before_bucketed bucketed batch_whole --require-exact bucketed --batches 8 --slots 16 --profiles three --layouts static translate resize --rounds 3 --check-timed-steps
```

`--baseline` 指向保留源码目录，`before_bucketed` 使用其中独立加载的运行时；该目录不随 Git 提交。
`backends` 默认检查第 0、1、7 帧；确认策略时使用 `--check-timed-steps` 覆盖实际计时帧，零差异门槛只对报告中的 `checked_steps` 成立。
`--require-exact` 指定的候选出现任何非零像素差即失败；旧桶可保留差异记录。冷素材单次耗时、独立诊断与确认测量的使用方式见 [ARCHITECTURE.md](ARCHITECTURE.md) 和 [桶优化报告](BUCKETED_OPTIMIZATION_REVIEW.md)。

## 接入与迭代约束

生产接入入口是 `shared_paste_example.py`，调用 `setup_shared_paste` 初始化 worker，再每帧调用 `prepare_shared_paste_args` 与 adapter。需要接入现有业务对象和真实数据，不能把基准 fixture 当成生产应用。

原始会话未提供 `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 的生产实现；基准中的实现是合成回调。接入时沿用业务原方法，再用真实视频、参数和负载验收。

`AdapterOptions.batch_chunk_size` 默认 `"auto"`：仅匹配已测硬件和输入结构时应用离线策略，未知情况回退整批。`0` 强制原整批，正整数强制指定分块上限。`last_execution` 可查看实际选择和回退原因。NumPy保持独立存储，返回即可读取；内部复制流不改变调用方串行调用约定。自动策略不强行启用可分离模糊，仍需真实核验证。

既有分块准入仍限定在 B8/B16、4 或 16 个广告位、已测素材尺寸循环和 108×320 ROI 等完整条件内，并保留现有硬件/软件门槛。精确准入表见 [paste_execution_policy.py](paste_execution_policy.py)；本轮未改动该表，也不自动启用 bucketed。不要据此推广到 32 个广告位或任意输入结构。

默认 `normalization="legacy"`、`camera_index_policy="legacy"` 保留原行为；`matched` / `corrected` 是需要单独验收的语义修正。保持 B1 输出 `[H,W,3]`、多组输出 `[B,H,W,3]` 的 NumPy `uint8` 协议；K、畸变系数和单应矩阵须实时读取。GPU worker 初始化及预热后复用 builder / adapter，素材变更按既有接口使缓存失效。

独立 NumPy 输出由其底层 PyTorch Tensor 持有存储；调用方不再引用后，存储才可由分配器回收复用。释放 builder/adapter 不会覆盖调用方仍持有的输出，也不会清空全局分配器。CUDA `reserved` 或进程工作集暂时不下降，不能单独作为泄漏证据；同样不能仅据一个稳定快照宣称没有泄漏。

原包文档中的“本地没有 CUDA GPU”描述的是打包时的环境。初次部署的 GPU 校验见 [VALIDATION.md](VALIDATION.md)，本轮结果状态见 [直播运行报告](STREAMING_REVIEW.md)；合成基准通过不等于完成生产视频验收。

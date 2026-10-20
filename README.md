# GPU 广告贴图 perf4

这是 PyTorch / Triton GPU 贴图库和命令行基准程序，没有 Web 服务或浏览器界面。版本为 `2026-10-20-perf4`，已部署到 `E:\dev\virtualAds\paste_gpu_performance`，后续可直接在 Codex 中打开此目录迭代。

来源：[原分享会话](https://chatgpt.com/share/6ad7de37-4b64-83ec-9ac2-1ad7013e17b5) 的 `paste_gpu_performance_perf4.zip`。压缩包 SHA256 为 `17c6354f8c478a74daa52cc6f0294a21fb71fc1eb22b2e811b8e6d1130b6a26b`；Git 原始代码基线为 `480fad5`。原包的说明保留在 [SHARED_PIPELINE_README.md](SHARED_PIPELINE_README.md) 和 [paste_gpu_performance_README.txt](paste_gpu_performance_README.txt)。

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

把 `--diagnose` 换为 `--check-only` 可只检查正确性。`--sigma-pattern repeated` 可另外检查精确 sigma 缓存复用。原核心 / adapter 基准不包含共享参数构造成本：

```powershell
.\scripts\run.ps1 batch --device cuda:0 --batch 8 --slots 16 --blur-backend triton --ad-sizes 200x200 500x500 300x600 --check-only
```

脚本转发剩余参数并保留 Python 退出码。日志由调用者保存，例如：

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --diagnose *> results\shared-two-sizes.log
```

## 接入与迭代约束

生产接入入口是 `shared_paste_example.py`，调用 `setup_shared_paste` 初始化 worker，再每帧调用 `prepare_shared_paste_args` 与 adapter。需要接入现有业务对象和真实数据，不能把基准 fixture 当成生产应用。

原始会话未提供 `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 的生产实现；基准中的实现是合成回调。接入时沿用业务原方法，再用真实视频、参数和负载验收。

默认 `normalization="legacy"`、`camera_index_policy="legacy"` 保留原行为；`matched` / `corrected` 是需要单独验收的语义修正。保持 B1 输出 `[H,W,3]`、多组输出 `[B,H,W,3]` 的 NumPy `uint8` 协议；K、畸变系数和单应矩阵须实时读取。GPU worker 初始化及预热后复用 builder / adapter，素材变更按既有接口使缓存失效。

原包文档中的“本地没有 CUDA GPU”描述的是打包时的环境。本次部署的实际 GPU 校验和计时结果见 [VALIDATION.md](VALIDATION.md)；合成基准通过不等于完成生产视频验收。

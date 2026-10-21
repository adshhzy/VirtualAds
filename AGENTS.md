# Codex 后续迭代

## 项目与运行

- 这是 Python GPU 库和命令行基准，没有 Web 服务。当前本地版本 `2026-10-21-perf4.2-readable`。原包 perf4.1 基线为 `0b97beb`；最初 perf4 为 `480fad5`。动态路数优化基线为 `fe07b03`，本轮仅可读性重构，验收见 `READABILITY_REVIEW.md`。
- 默认解释器：`E:\dev\virtualAds\env\paste_gpu_performance\Scripts\python.exe`。不要直接用系统 Python，也不要修改其他项目的环境。
- 使用 `scripts/setup.ps1` 从 `requirements-lock.txt` 安装依赖；torch 独立使用官方 cu128 源。依赖调整应同步锁文件并验证 CUDA / Triton 兼容。
- `scripts/run.ps1 shared ...` 对应 `benchmark_shared_paste.py`；`scripts/run.ps1 batch ...` 对应 `benchmark_paste_gpu_batch.py`。
- `shared_paste_example.py` 展示业务接入；`paste_gpu_batch_adapter.py` 负责接口、素材缓存和输出，`paste_gpu_batch.py` 负责核心路由。共享参数逻辑位于 `paste_shared_params.py` / `paste_camera_filter.py`，GPU 内核在 `paste_*_kernels.py`。
- 优先阅读 `README.md`、`ARCHITECTURE.md` 和 `READABILITY_REVIEW.md`；`DYNAMIC_BATCH_REVIEW.md` 记录现行调度策略的历史测量，其余 review/validation 文档保留历史说明。日志、输出和缓存不提交 Git。
- 用 `.\scripts\lint.ps1` 检查格式与静态问题；`-Fix` 仅统一格式再检查。Ruff 0.14.1/100列/LF/Python3.9语法目标，配置在 `pyproject.toml`，工具隔离于GPU环境。保留平铺导入路径；优先按已有阶段职责放置helper，不引入只为包装参数的层层对象。

## 必须保持的行为

- 默认保持 `normalization="legacy"`、`camera_index_policy="legacy"`；性能修改不要同时混入 `matched` / `corrected` 的画面行为变更。
- 默认 NumPy `uint8`：B1 为 HWC、多组为 BHWC；显式 `squeeze_single=False` 保留 B1 维。固定页输出必须独立，不得被下一帧覆盖。
- K、畸变系数、单应矩阵可能原地更新；不能仅按 frame_id 缓存几何结果。尺寸按实际素材 / 显式 resize 后的高宽处理。
- 保留 ROI 局部坐标、`z + 1e-8`、现有采样 / alpha / 通道顺序语义。不默认引入 FP16、TF32 或低秩近似。
- `motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 缺少生产定义。基准回调只用于合成 fixture，不得将它们冒充生产定义或在接入代码中替代业务方法。
- Gaussian 缓存使用精确 sigma；配置变更需更新版本或清缓存。可分离路径必须验证实际二维核，不靠函数名推定可分离。
- builder / adapter 在 worker 内初始化后复用，调用方在同一计算流上串行使用；不每帧重建或清缓存。分块路径内部使用专属复制流，事件连接计算与复制；返回及异常退出前等待复制完成，禁止覆写在途GPU源或调用方持有的CPU结果。
- `batch_chunk_size="auto"` 只查询 `paste_execution_policy.py` 的离线实测表，未知硬件/形状/布局/B回退整批；0强制原整批，正整数手动分块。修改策略必须补对应完整NumPy对照，不能以微基准推断整个流程收益。

## 回归与性能证据

从项目根目录运行与改动有关的检查。两种素材数的共享流程检查：

```powershell
.\scripts\run.ps1 check --device cuda:0 --profile-checks
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --check-only
.\scripts\run.ps1 shared --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 300x600 --check-only
.\scripts\run.ps1 batch --device cuda:0 --batch 8 --slots 16 --blur-backend triton --ad-sizes 200x200 500x500 300x600 --check-only
```

布局 / 单帧相关改动再检查 B1；缓存改动检查 `--sigma-pattern repeated` 及三尺寸容量。完整计时将共享基准的 `--check-only` 换为 `--diagnose`，保持相同 GPU、分辨率、B、广告位、核尺寸和预热 / 迭代数。报告完整调用结果与正确性差异，不能把独立微基准相加，或与旧纯 adapter 的基线直接相减。真实生产性能还需要实际回调、视频和编码 / 发送流程验证。

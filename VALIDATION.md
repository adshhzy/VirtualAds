# 本机部署验证（2026-10-21）

当前版本已同步到 perf4.1，包内 13 文件与最新版压缩包逐字节一致；
新版本回归、同轮 eager 对照和进一步分析见 [PERF4_1_REVIEW.md](PERF4_1_REVIEW.md)。
以下保留首次 perf4 部署的历史结果，不能把跨轮耗时差解释为 perf4.1 算法收益。

## 首次 perf4 部署记录

原始 perf4 压缩包中的 13 个文件保持不变，Git 初始提交为 `480fad5`。
本次增加独立环境、依赖锁定、PowerShell 运行脚本和迭代说明。

## 环境

- Windows 11，NVIDIA GeForce RTX 4090 24GB，驱动 610.88。
- Python 3.12.14；PyTorch 2.9.1+cu128；CUDA 12.8。
- triton-windows 3.5.1.post24；NumPy 2.5.2；SciPy 1.18.1。
- 解释器：`E:\dev\virtualAds\env\paste_gpu_performance\Scripts\python.exe`。
- 完整依赖见 `requirements-lock.txt`；`uv pip check` 通过。

## 共享参数完整调用

两组均为 1920×1080、batch=8、16 个广告位、9×9 模糊核，预热 20 次后测量 100 次。
沿用 `normalization=legacy`、`camera_index_policy=legacy`、逆畸变 10 次、连续变化 sigma。
所有路径包含共享准备、适配器和固定页 NumPy 输出；不含视频解码、编码和发送。

| 源广告尺寸 | 原共享准备 + dense | 新共享准备 + dense | 新共享准备 + separable | 总节省 |
|---|---:|---:|---:|---:|
| 200×200、500×500 | 11.676 ms | 6.931 ms | 6.376 ms | 5.300 ms（45.4%） |
| 200×200、500×500、300×600 | 11.566 ms | 6.465 ms | 6.149 ms | 5.417 ms（46.8%） |

表中为平均耗时；三尺寸组对应 P95 分别为 12.435、6.659、6.359 ms。
这是本机同轮合成基准结果，不能与分享页另一台 4090 D 的历史数字直接相减。

两组动态输入正确性检查均通过：每组 3 帧 K/畸变/单应矩阵更新，最大网格误差
`3.5762786865234375e-7`（容差 `2e-6`），最大 uint8 像素差为 1。
连续 sigma 下，优化路径计时区间内核缓存均为 0 hits / 100 misses。

原始输出保存在 `results/shared_2sizes.log` 和 `results/shared_3sizes.log`。

## 其他已通过检查

- 小图 320×180、B2、三尺寸共享流水线检查：最大 uint8 差 0。
- 原基准的 24 个 core、48 个 adapter、8 个动态矩阵用例及透明覆盖、无广告 HWC 输出检查通过；核心用例最大 uint8 差 0。
- 40 个密集融合模糊 kernel 用例通过，最大 uint8 差 0。
- 实际 1080p/B8/16 位三尺寸 fixture 通过；core 和 adapter 最大 uint8 差均为 1。
- 三尺寸缓存重复调用 16 hits / 0 misses，尺寸分布为 6/5/5 个广告位。
- B1、小图、repeated sigma、含 eager 对照通过，最大 uint8 差 0；优化路径各 100 次内核缓存命中、0 次未命中。
- 两个 benchmark 的版本均为 `2026-10-20-perf4`。
- 安装脚本重复执行成功；运行脚本跨目录版本查询、三尺寸参数转发及真实 GPU 小图检查通过。

详细日志见 `results/batch_checks.log`、`results/shared_smoke.log`、
`results/shared_repeated_sigma.log`；环境记录见 `results/environment.json`。
`results/` 不纳入 Git，本文保留验收摘要。

## 业务接入边界

本次已部署并运行算法库及命令行基准，没有常驻服务。
测试使用包内明确标注的合成素材与运动/sigma/Gaussian 回调。
接入直播业务时仍需提供原来的 `compute_camera_motion_rate`、`motion_rate_to_sigma`、
`create_gaussian_kernel` 和真实帧/相机参数，并沿用原采样和 alpha 配置。
接入方式见 `SHARED_PIPELINE_README.md` 及 `shared_paste_example.py`；
默认先用 dense，separable 在真实素材验证后启用。

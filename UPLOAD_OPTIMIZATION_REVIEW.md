# 上传优化与验证（2026-10-22）

当前版本 `2026-10-22-perf4.8-uploads`，对照基线 `b82f35bf634c89862efdc98a0b563996304095ae`。
本轮只调整 CPU→GPU 输入准备及存储复用。贴图、畸变、模糊和融合的数学实现、NumPy 输出协议、batch auto 表均保持原样。

## 改了什么

| 位置 | 修改 |
|---|---|
| `paste_shared_params.py` | 各 slot 的 CPU `[3,3]` / `[1|B,3,3]` 矩阵集中传一次，保留各自 rank 和广播维度；GPU/混合输入沿用原转换 |
| `paste_render_executor.py` | CPU float32 标量/向量透明度集中传一次，GPU 除 scale 后还原各 slot 的广播视图；其他 dtype、空间 map、GPU Tensor 及特殊逻辑视图沿用原路径 |
| `paste_frame_upload.py` | 连续 CPU uint8 背景按原始字节上传，在 GPU 转 float32；各布局组共用一次上传；其他输入沿用原转换 |
| `paste_upload_buffer.py` | 共用的有界固定页 staging / GPU 空间；容量增长、H2D 事件保护和释放集中在一处 |

每帧仍读取当前像素、矩阵和透明度，复用的是存储空间。没有按 frame_id 或对象身份缓存这些值。未调整 ROI 网格公式、归一化、slot 覆盖顺序、FP32 精度或 blur backend。

固定页内存只有在前次 H2D 完成后才可由 CPU 重写；借用的 GPU 输入按同一计算流顺序消费。上传 helper 不持有业务源数组。旧 `render_frame` 的背景转换保留原实现；共享 builder 和执行器继续支持旧接口。

## 输入与计时

- RTX 4090、Windows、Python 3.12.14、PyTorch 2.9.1+cu128、Triton 3.5.1，CPU 线程设置保持原值 24。
- `3336.npy`、对应 1920×1080 原始帧、500×500 测试广告。复用上轮相同的 24-bit BMP 解码结果。
- 循环复制原路配置和广告顺序为 **8 路、每路 16 slot**，共 128 个配置。所有素材 key 指向同一张测试图；仍有 16 份 B8 GPU 素材 stack，不消除重复 slot。
- `ad_h_w=(200,200)` 的显式 resize、相机 legacy 行为、mask sigma1.5/radius3 保持原样。
- 延续样例演示回调 `SyntheticOwner(kernel_size=5)`：静止相机、sigma0.65、dense5×5；没有提供的生产运动回调不能由此替代。
- auto 在此输入下仍为整批 `chunk_size=0`，回退原因 `unsupported_backend_or_output`；没有扩大策略准入范围。
- 完整调用包含路由、素材缓存检查、动态布局/ROI 准备、相机与 mask、上传、GPU 渲染和可立即读取的独立 NumPy；不含输入解码、外部输入更新、编码或网络发送。

基线全部 `paste*.py` 从 Git 快照作为独立 package 导入，并核对来源和源码 SHA。每候选每场景 **4 轮 × 240 帧 = 960 帧**，每轮预热 50 帧并轮换执行顺序。动态 bbox 使用 12 步平移周期，ROI 尺寸、相机和矩阵不变；每阶段完整覆盖 20 个周期。

每次计时输出都在计时外检查全部 8 路像素和执行决策。校验本身约耗时 15～17 ms，会影响调用间隔和运行状态；这些完整调用延迟不能直接当作直播持续吞吐量。

## 完整调用结果

单位 ms；降幅指相同指标的耗时下降。

| 场景 | 优化前平均 | 优化后平均 | 平均降幅 | 优化前 P95 | 优化后 P95 | P95 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| 固定 bbox | 14.815 | **12.185** | **17.75%** | 17.970 | **15.618** | **13.09%** |
| 动态 bbox | 14.820 | **13.495** | **8.94%** | 17.683 | **16.070** | **9.12%** |

最终共 **3,840 次计时调用、30,720 路图像**逐像素一致，最大 uint8 差为 0；其中新代码 1,920 次调用、15,360 路图像。16 个测量阶段均通过独立输出、下一帧及关闭后旧输出不变检查。

各场景四轮的平均及 P95 都有改善，但存在波动：动态 bbox 第 2 轮仅改善 0.80% / 0.42%。较早的 4×120 帧初轮测得固定/动态平均降幅 8.09% / 13.21%；它在特殊 negative-bit Tensor 透明度回退修复前执行，保留为初轮记录，不与最终样本混合或替换最终结果。

## 分阶段结果

独立诊断各记录 200 帧 CPU 阶段；另用 CUDA profiler 记录 10 帧，排除首帧初始化影响后统计 9 帧真实设备活动。CPU 阶段墙钟含提交和可能等待，CUDA event 区间还含 CPU 供给空隙；父子阶段以及 CPU/GPU 时间均不可相加。

| CPU 调用阶段，平均 ms | 优化前 | 优化后 |
|---|---:|---:|
| 背景准备/上传 | 1.718 | **0.402** |
| ROI 参数与网格准备 | 2.606 | **1.532** |
| 配置解析/分组 | 1.507 | 1.281 |
| 素材缓存检查 | 0.837 | 0.693 |
| 共用 mask | 1.035 | 0.852 |
| 共用相机/畸变参数 | 0.442 | 0.359 |
| 核心渲染提交 | 2.103 | 1.761 |
| NumPy 导出（含等待前序 GPU） | 3.045 | 3.065 |

未改动阶段也受主机调度、插桩和运行状态影响；不能把上述差值逐项相加作为完整调用收益。吞吐相关结论以交替多轮的完整调用为准。

| GPU 实际活动，平均 ms/帧 | 优化前 | 优化后 |
|---|---:|---:|
| 背景 H2D | 1.530 | **0.260** |
| uint8→float32 转换 | — | 0.017 |
| 全部 H2D | 1.561 | **0.283** |
| 全图畸变采样 | 0.684 | 0.684 |
| dense 模糊 + alpha/mask 融合 + uint8 | 0.599 | 0.627 |
| 16 slot warp 贴图核合计 | 0.216 | 0.216 |
| NumPy 输出 D2H | 1.890 | 1.890 |

真实 GPU 活动仅取 trace 中 kernel/memcpy 类别，没有把重复的 CPU 父项或 GPU annotation 算入。背景 H2D 是全部 H2D 的子项；同样不能相加。

每帧 H2D 从 **41 次降至 11 次**：背景 1、mask 6、相机参数 2 保持；矩阵 16→1，透明度 16→1。矩阵与透明度共 5,120 字节，小数据准备收益主要来自调用/提交/同步减少。

| 传输字节/帧 | 优化前 | 优化后 |
|---|---:|---:|
| 背景 H2D | 24,883,200 | **6,220,800** |
| 全部 H2D | 25,215,144 | **6,552,744** |
| 完整 B8 NumPy D2H | 49,766,400 | 49,766,400 |

背景传输量严格降低 75%；全量 H2D 还包含未变的 mask 和参数。输出协议保持不变，所以约 47.46 MiB 的回传仍保留。

## 内存复用与长期检查

默认矩阵、透明度各最多保留 1 MiB host + 1 MiB GPU；背景最多保留 32 MiB host uint8 + 32 MiB GPU uint8 + 128 MiB GPU float32。这些是延迟分配的容量上限；只持有一个当前上下文，不按历史 B/shape 累积多份。超过上限仍可使用临时工作集，不能把超额输入称为缓存预算内存。

本样例新增工作区实际占用 **5.9375 MiB 固定页内存、37.9375 MiB GPU 内存**。其中背景浮点 payload 为 23.73 MiB，几何增长后的保留容量为 32 MiB；其余是 uint8 背景及参数。固定 B8/S16 下四类工作区各只分配一次。

完整性能基准中，进程 CUDA allocated 稳态从 **111.82→149.29 MiB**，峰值从 **756.97→771.78 MiB**。这是用有界存储复用换取较少分配和传输的成本；素材 stack 两版都为 78.125 MiB。每个测量阶段关闭后，库持有的新增工作区及 CUDA allocated 均为 0，allocator reserved 不要求归零。

另完成 **10,000 帧**检查：

- 前 5,000 帧固定 B8/S16；后 5,000 帧在 B1/2/3/4/8/16、S1/3/7/16 间变化，同时改变 bbox、逐路矩阵及透明度。
- 每帧原地修改背景 8×8 区域；每阶段每 500 帧修改局部素材并推进 revision。
- 固定阶段上传工作区累计分配次数恒为 `(1,1,1,1)`；动态容量预热后恒为 `(1,1,2,2)`，顺序是背景 uint8、背景 float32、矩阵、透明度。
- 动态阶段上传工作区约 5.9424 MiB pinned / 37.9424 MiB GPU，没有随帧数继续增加；素材缓存随 B、slot、revision 的变化独立计量。
- 72 个旧源码对照帧、442 路输出全部零差，同时检查路号及策略；149 次旧输出快照检查通过。长测没有逐帧执行旧源码 oracle。
- B=0 暂停恢复、同帧号原地更新、素材切换、上传后 mask 校验异常、显式释放重用、退出清理通过。释放后的重用可以重新分配，不把其计数增长误判为泄漏。
- 退出并释放测试持有的结果/快照后，上传空间、素材、Gaussian、历史均归零，CUDA allocated/active 为 0。reserved 仍约 2,082 MiB，属于进程分配器保留。

进程分配器在阶段起始仍有少量扩容，两阶段各自第 250 帧采样后，device/host allocation 次数才稳定；不能写成所有内存块在预热后都从未分配。固定阶段第 250→4,999 帧 private 增加约 1.00 MiB。动态阶段同时保留两份返回结果及等量 CPU 快照，必须把队列大小变化与库持有量分开。

本机 PyTorch 的 pinned live 字节计数存在不自洽：46 个采样点 allocated 均大于 reserved，其中 24 个还大于 peak。报告保留这些原始值并标为无效，未用其判断泄漏；检查结合自有工作区、分配次数、CUDA live 和进程内存趋势。有限帧、固定分辨率、单设备及演示回调不能证明生产直播永不泄漏。

## 回归

- CPU：176 项执行，其中 166 通过、10 项需要显式 CUDA 的检查跳过；另一次显式 CUDA 回归 30 项全部通过，含 8 个新异步上传测试、透明度缩放精度及旧公共入口。
- GPU `check_render_img --frames 1000`：共 1,027 帧 / 3,574 路图像检查，最大 uint8 差 0；mask 与 SciPy 参考最大误差约 2.38e-7。
- 两尺寸/三尺寸共享几何回归通过；这些历史脚本比较 eager、dense、separable，沿用最大 1 级 uint8 容差，实测为 1。它们不是本轮旧版→新版的零差依据；本轮严格零差依据是完整快照对照与上述真实输入检查。
- 底层 batch 回归：24 core + 48 adapter、8 个矩阵更新、40 个融合模糊核案例通过；三尺寸 fixture 和缓存命中检查通过。
- 旧 `render_frame` 对照 `b82f35b`：B1/3/8/16、每路 16 slot、auto/整批的快照输出全部严格一致。该检查迭代数很少，仅用于兼容验收，不作为性能结论。
- Python 3.9 AST、Ruff format/check、差异空白检查通过。

补修两个旧基准脚本过期的版本检查常量，使 shared/batch CLI 接受 perf4.8。完整性能采集后这两文件仅版本字符串变化；17 个运行时文件与正式报告的 SHA 保持一致，测量函数及运行时没有后续改动。

## 复跑

输入不随 Git 分发。先按 [模式复测报告](RENDER_IMG_MODE_REVIEW.md) 的 System.Drawing 步骤解码 JPEG，设置 `$samplePath`、`$frameBmpPath`、`$adBmpPath`。

```powershell
$snapshotDirectory = Join-Path (Get-Location).Path 'results\baseline_upload_b82f35b'
$snapshotArchive = Join-Path (Get-Location).Path 'results\baseline_upload_b82f35b.zip'
if ((Test-Path -LiteralPath $snapshotDirectory) -or (Test-Path -LiteralPath $snapshotArchive)) {
    throw '基线目标已存在，请复用经核对的目录或选择新路径，不覆盖旧记录。'
}
git archive --format=zip --output=$snapshotArchive b82f35b
if ($LASTEXITCODE -ne 0) { throw '基线导出失败' }
Expand-Archive -LiteralPath $snapshotArchive -DestinationPath $snapshotDirectory

$inputArguments = @('--device', 'cuda:0', '--sample', $samplePath,
    '--frame-bmp', $frameBmpPath, '--ad-bmp', $adBmpPath)

.\scripts\run.ps1 uploads @inputArguments --baseline $snapshotDirectory --rounds 4 --steps 240 --warmup 50 --output results/uploads_new.json
.\scripts\run.ps1 profile-uploads @inputArguments --baseline $snapshotDirectory --steps 200 --output results/uploads_profile_before_new
.\scripts\run.ps1 profile-uploads @inputArguments --steps 200 --output results/uploads_profile_after_new
.\scripts\run.ps1 check-uploads @inputArguments --baseline $snapshotDirectory --frames 10000 --check-every 250 --output results/uploads_streaming_new.json
```

GPU 测试必须串行。已有输出会被拒绝；不要用删除旧报告的方式绕过保护。阶段 profiler 使用实际 CUDA 活动，不支持相关 profiler 的环境应先处理工具能力，不能把 CPU 时间冒充 GPU 时间。

本机原始记录位于忽略的 `results/3336_uploads_20261022/`：`comparison_final.json`、`stages_before/`、`stages_after/`、`streaming_10000.json`、`render_img_1000.json`、`legacy_snapshot_check.json`。初轮 `comparison.json` 和短测 `smoke_streaming.json` 也保留，未替换原始测量。源码和输入 SHA 在对应正式报告中。

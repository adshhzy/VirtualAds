# 准备阶段优化与验证（2026-10-23）

> 目录迁移说明（2026-10-23）：本文的版本、数值、源码文件名和直接 Python 命令保留当时记录；旧快照仍是平铺目录。当前源码按功能分目录，源码名及启动命令映射见 [目录与迁移指南](../PROJECT_LAYOUT.md)。现有 `scripts/run.ps1` 子命令不变；复跑当前源码时从仓库根目录运行。原始测量 artifact 不随仓库分发。

版本 `2026-10-23-perf4.9-prepare`，对照基线 `6cebfceec9e25c572bfdefa20e2844a2fb6cc962`。
本轮不修改广告采样、ROI 网格、畸变、Gaussian、slot 覆盖顺序、NumPy 输出或 auto 表。

## 实现

1. `paste_route_plan.py` 在本次调用内复用形状的 layout 描述，并将已经读取的素材描述交给标准执行器。`_collect_ads` 继续实时处理版本和素材缓存；来源/形状/dtype/device/options 不匹配时重新扫描。
2. `paste_route_preparation.py` 用一份有界普通 CPU 字节空间承接同 dtype float32/float64 矩阵和 float32 标量透明度。仍调用 `np.stack(..., out=...)`，保留原 dtype 与成员顺序；减少目标数组分配，**仍保留到 pinned staging 的最终复制**。混合 dtype、GPU 矩阵及空间透明度沿用原转换。
3. `paste_shared_mask.py` 将可见 CPU mask 转成 float32 后集中上传，再按原顺序 clamp、maximum 合并和整图羽化。GPU/混合来源、negative/conjugate Tensor 逻辑视图保留逐块上传；这些特殊视图的旧跨设备转换与提前 CPU 转换存在差异，不能自动替换。

每帧仍读取动态路数、slot 顺序、bbox、素材身份和 revision、矩阵、透明度及 mask 数值。没有按帧号或对象 ID 缓存动态结果。复用的是本次解析结果和存储空间；输入配置字典、素材与返回的 NumPy 不被改写。

公共 `build_route_plan` 仍返回独立的参数数组。只有标准三参数内部流程借用 CPU 工作区，跨组使用不同切片。自定义执行器/warp callback 沿用原入口；共享业务回调中途安装自定义 hook 时，先复制已经生成的借用参数，再调用 hook，保持回调前的快照。B1 原矩阵引用保持不变。

source metadata 与 layout 同时取快照，避免自定义 Mapping 在后续查找中修改先前素材形状时绕过旧校验。声明尺寸不一致的兼容组保守回退原素材检查。自定义 warp 回调后的保护检查保留。

## 对照方法

- 同一 RTX 4090 / Windows / PyTorch 2.9.1+cu128 / Triton 3.5.1，沿用 CPU 线程设置 24。
- 原始 `3336.npy`、1920×1080 帧及 500×500 广告，使用上轮相同解码 BMP，广告补 alpha=255，依样例显式 resize 到 200×200。
- 循环复制成 B8/S16；仍是三个原始 ROI 和一份实际图片。各重复 slot 均执行，不合并相同输出路、不删除重复广告。
- 演示 `SyntheticOwner(kernel_size=5)`，静止相机 sigma0.65；共用 mask sigma1.5/radius3。未提供的生产运动回调不在此次验收内。
- 当前 auto 在此 fixture 仍选择整批 `chunk_size=0`，reason=`unsupported_backend_or_output`。旧 k9 策略表不变。
- 从 `render_img` 调用前到独立、可立即读取的 NumPy 返回后计时，包含路由、素材检查、相机/mask/ROI 准备、上传、渲染与回传。输入解码、输入字段更新、逐像素校验、编码与网络发送不计时。
- 每次计时输出都在计时外对全部 8 路逐像素检查，并验证策略。该校验影响帧间 CPU 负载；不能用延迟倒数宣称真实推流 FPS。

`benchmark_prepare_comparison.py` 复用上轮原计时循环，分别建立独立 worker，构造时设置两个私有诊断开关。四个候选为旧源码、仅 route 准备优化、仅 mask 集中上传、两项合用。执行顺序逐轮轮换，正式轮次是候选数的整数倍；报告核对实际优化计数及源码 SHA，不能只凭开关标签认定启用。

## 完整调用结果

每候选每场景 4 轮 × 240 次，轮前预热 50 次。动态 bbox 是 12 步平移，ROI 尺寸和矩阵不变，每个阶段完整覆盖 20 个周期。单位 ms：

| 候选 | 固定 bbox Mean / P95 | 动态 bbox Mean / P95 | 平均降幅：固定 / 动态 |
|---|---:|---:|---:|
| 旧版 6cebfce | 11.975 / 14.877 | 12.194 / 14.882 | — |
| 仅 CPU 路由准备优化 | 11.086 / 13.409 | 11.035 / 13.710 | 7.43% / 9.51% |
| 仅 mask 集中上传 | 9.705 / 10.926 | 9.578 / 11.289 | 18.96% / 21.45% |
| **两项合用** | **8.745 / 10.136** | **8.907 / 10.874** | **26.98% / 26.96%** |

两项合用的 P95 分别下降 **31.87% / 26.93%**，各场景四轮的均值和 P95 均较旧版改善超过 5%。保留两项优化为标准路径默认行为，auto 表无需改变。

分项收益不能相加。在 mask 已启用时，CPU 路由优化带来的额外均值下降为 9.89% / 7.01%，P95 下降为 7.22% / 3.68%；各轮增量均为正，但并非每轮或每项 P95 都达到 5%。mask 对本样例收益更大。

四候选共 **7,680 次计时调用、61,440 路图像**全部逐像素一致，最大 uint8 差为 0，auto 决策相同。其中两项合用为 1,920 次调用、15,360 路图像。32 个阶段均通过连续输出独立、下一帧及 close 后旧输出不变检查。

所有启用 route 的阶段均记录 292 次本帧元数据命中、0 次回退，矩阵/透明度各 4,672 次工作区写入，CPU 工作区只分配一次。启用 mask 的阶段均记录 292 次集中上传、一次分配、无 oversize；未启用项计数为 0。292 包含 50 次预热、240 次计时和 2 次所有权检查。

报告中的 26 个运行时/基准依赖 SHA 与最终文件一致；基线 17 个运行时文件逐一与 Git 提交核对。未用前一天的延迟数字代替本轮同进程对照。

## 分阶段结果

阶段诊断各 200 帧，CUDA profiler 另采 10 帧，排除第一帧后统计真实 kernel/memcpy 活动。CPU wall、CUDA event 与父子阶段存在重叠，不相加。吞吐判断以平衡轮换的完整调用为准。

| CPU 阶段，平均 ms | 优化前 | 两项合用 |
|---|---:|---:|
| 路由准备 | 1.564 | 0.820 |
| 素材检查 | 0.848 | 0.020 |
| 共用 mask | 1.056 | 0.502 |
| ROI 参数与网格准备 | 1.884 | 0.986 |
| 背景准备/上传 | 0.608 | 0.397 |
| 共用相机参数 | 0.436 | 0.315 |
| 核心渲染提交 | 2.220 | 1.080 |
| NumPy 导出，含前序 GPU 等待 | 3.559 | 3.232 |

素材描述的创建移入路由准备，因此不能把素材检查降幅单独理解为完整收益。新增 mask 上传子阶段约 0.104 ms，已包含在共用 mask 中。前后是独立诊断运行，未修改的背景、核心提交和输出也有时序波动，不能把全部下降归因于代码修改。

| GPU 实际活动/帧 | 优化前 | 两项合用 |
|---|---:|---:|
| 全部 H2D 次数 | 11 | 6 |
| 全部 H2D 字节 | 6,552,744 B | 6,552,744 B |
| 全部 H2D 时间 | 0.282864 ms | 0.280006 ms |
| mask H2D 次数 | 6 | 1 |
| mask H2D 字节 | 326,784 B | 326,784 B |
| mask H2D 时间 | 0.019342 ms | 0.016387 ms |
| ROI 网格 kernel | 16 次 / 0.062414 ms | 16 次 / 0.062752 ms |
| 广告贴图 kernel | 16 次 / 0.217073 ms | 16 次 / 0.216262 ms |
| NumPy D2H | 1 次 / 2.292873 ms | 1 次 / 1.921195 ms |

全部 kernel 均为 52 次，D2H 字节均为 49,766,400 B；D2H 代码与传输量没有变化，表中耗时波动不代表新增输出优化。mask 原输入虽为 uint8，旧 `copy_` 已转换成 float32 再传输，所以本轮没有减少或增加其 H2D 字节。

每帧 `cudaStreamSynchronize` 从 8 次降至 2 次，减少的 6 次来自旧 mask 上传；新增一次 mask H2D 事件等待保护固定页复用，总 `cudaEventSynchronize` 从 4 次变为 5 次。主要改善是 CPU 准备及上传提交方式，不是 DMA 本身节省了数毫秒。profiler 收尾的一次全局同步不属于逐帧业务调用。

阶段诊断前后输入和完整 B8 reference SHA 相同。优化后 844 次路由准备全部命中，mask 集中上传也是 844 次，均只分配一次，确认 profiler 包装没有让快路径失效。阶段端点零差不能替代前述每个计时帧的全量像素检查。

## 内存与持续运行

普通 CPU 参数工作区默认最多保留 1 MiB；mask upload 默认分别最多保留 32 MiB pinned host 和 32 MiB GPU。这些都是延迟分配预算，不在初始化时预分配上限。超过预算使用临时空间，不保留历史形状列表；后续空 mask/fallback 也结束对超额上传 staging 的持有。

CPU 参数数组是内部借用视图，不交给外部保存；mask 上传的固定页源在前次 H2D 完成后才允许重写。释放清理库持有量，不逐帧 `gc.collect()` / `empty_cache()`。调用方持有的独立 NumPy 和分配器 reserved 内存另计。

本样例新增工作区实际持有量如下，未包含上一轮已有背景/矩阵/透明度上传空间：

| 项目 | 固定 B8/S16 | 动态容量预热后 |
|---|---:|---:|
| 普通 CPU 参数空间 | 11,264 B（11 KiB） | 22,528 B（22 KiB） |
| mask 固定页空间 | 326,784 B | 326,784 B |
| mask GPU 空间 | 326,784 B | 326,784 B |
| **本轮新增总持有** | **649.25 KiB** | **660.25 KiB** |

完整性能基准中 CUDA allocated 从 156,353,536 B 变为 156,680,704 B，增加 327,168 B（含分配器对齐），各阶段 close 后均为 0。全部上传工作区合计固定时 pinned/GPU 为 6.2491 / 38.2491 MiB，动态时为 6.2540 / 38.2540 MiB；普通 CPU 参数空间另计。

已完成 5,000 帧固定 B8/S16 和 5,000 帧动态 B1/2/3/4/8/16、S1/3/7/16；逐路矩阵/透明度变化，逐帧更新背景局部像素和 mask，定期切换素材 revision。mask 更新周期为 7，避免默认每 250 帧的抽检总落在同一数值。恢复检查另外覆盖 float32 mask 0.25/0.75 的原地更新。

- 72 次旧源码对照、442 路图像全部零差，149 次旧输出快照检查通过。不是全部 10,000 帧都运行旧源码 oracle。
- 按背景 uint8、背景 float32、矩阵、透明度、mask、route 的顺序，固定阶段分配计数恒为 `(1,1,1,1,1,1)`；动态容量预热后恒为 `(1,1,2,2,1,2)`。没有 oversize 或普通 CPU 临时分配。
- 暂停恢复、同帧原地修改、素材 revision、无效 bbox/上传后 mask 异常恢复、显式释放重用和退出通过。释放后重建使累计计数增加，属于预期行为。
- 退出并释放测试持有的输出/快照后，库工作区、素材、历史持有量归零，CUDA allocated/active/inactive-split 为 0；CUDA reserved 仍为 2,082 MiB，固定页 reserved 约 672.56 MiB。

进程分配器在阶段起始仍会扩容，两阶段分别从第 250 帧采样起 device/host allocation 次数稳定为 18/11、31/33，retry/OOM 全为 0。固定第 250 帧到结尾 RSS 增加约 1.16 MiB、private 减少约 3.14 MiB；动态扣除显式 CPU 快照后的 private 仍波动约 1.80 MiB，不能写成进程内存完全不变。

本机 pinned live 计数仍不自洽：46 个采样点 allocated 都大于 reserved，其中 24 个还大于 peak，报告标记为无效。未据其声称固定页 live 内存归零；判断结合库自有空间、分配次数、CUDA live 和进程趋势。

## 回归与复跑

- CPU 契约回归：227 项，其中 212 通过、15 项显式 CUDA 检查跳过。
- 显式 CUDA 回归：35 项全部通过，含 mask 上传对旧源码 float32 严格一致、连续异步消费、不同 dtype/stride、GPU/mixed fallback、超预算、释放重用和异常恢复。
- `check-render-img --frames 1000 --sample ...`：1,027 次合成调用、3,574 路图像最大 uint8 差 0；mask 对独立参考最大误差约 2.38e-7；另通过原 NPY 的接线检查。
- 旧 `render_frame` 与 6cebfce 在 B1/3/8/16、每路 16 slot、auto/整批下快照严格一致。其短计时仅用于兼容检查，不作为本轮性能证据。
- 旧接口与自定义回调兼容、公共 plan 所有权、同帧原地修改、旧输出存活纳入检查。
- 58 个 Python 文件的 Ruff format/check、Python 3.9 AST 和 Git 差异空白检查通过。

输入文件不随 Git 分发，解码步骤见 [模式对照报告](RENDER_IMG_MODE_REVIEW.md)。在仓库根目录先设置 `$samplePath`、`$frameBmpPath`、`$adBmpPath`，恢复基线时不要覆盖已有结果：

```powershell
$snapshotDirectory = Join-Path (Get-Location).Path 'results\baseline_prepare_6cebfce'
$snapshotArchive = Join-Path (Get-Location).Path 'results\baseline_prepare_6cebfce.zip'
if ((Test-Path -LiteralPath $snapshotDirectory) -or (Test-Path -LiteralPath $snapshotArchive)) {
    throw '目标已存在，请复用核对后的目录或另选路径，不覆盖旧记录。'
}
git archive --format=zip --output=$snapshotArchive 6cebfce
if ($LASTEXITCODE -ne 0) { throw '导出基线失败' }
Expand-Archive -LiteralPath $snapshotArchive -DestinationPath $snapshotDirectory
$inputArguments = @('--device', 'cuda:0', '--sample', $samplePath,
    '--frame-bmp', $frameBmpPath, '--ad-bmp', $adBmpPath)

.\scripts\run.ps1 prepare @inputArguments --baseline $snapshotDirectory --rounds 4 --steps 240 --warmup 50 --output results/prepare_ablation_new.json
.\scripts\run.ps1 profile-uploads @inputArguments --baseline $snapshotDirectory --steps 200 --output results/prepare_profile_before_new
.\scripts\run.ps1 profile-uploads @inputArguments --steps 200 --output results/prepare_profile_after_new
.\scripts\run.ps1 check-prepare @inputArguments --baseline $snapshotDirectory --frames 10000 --check-every 250 --output results/prepare_streaming_new.json
```

GPU 检查必须串行，已有输出会被拒绝。原始报告位于忽略的本机目录 `results/3336_prepare_20261023/`：`ablation.json`、`stages_before/`、`stages_after/`、`streaming_10000.json`、`render_img_1000.json`、`legacy_snapshot_check.json`。源码和输入哈希在各报告中。有限帧、固定分辨率、固定 mask 布局、单 GPU 与演示回调不能证明生产直播的所有情况。

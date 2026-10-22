# render_img 模式对照（2026-10-22）

本轮比较真实单帧输入下的完整 `ImageRenderer.render_img`，检验 dense、分块、separable 和 eager 是否满足保持原画面的自动准入要求。当前默认仍是 dense + `batch_chunk_size="auto"`；auto 只选择已有离线分块配置，不自动切换 blur 算法。该样例不匹配现有 k9 profile 时继续整批回退，不为此新增无收益的 chunk=0 表项。

**结论：正式静态、平移与默认 auto 对照均已完成，本轮不新增自动配置。** dense 分块未同时达到静态/平移的均值与 P95 性能门槛，eager/separable 未达到零像素差门槛。本例继续执行 dense 整批，既有 k9 auto 表原样保留；这是依据实测结果作出的策略决定。初步 probe 和被中断的旧静态计时不用于正式结论或策略准入。

## 输入与比较范围

- 使用提供的 `3336.npy`、真实 1920×1080 背景帧和一张 500×500 测试广告。JPEG 解码结果先保存为无压缩 RGB BMP，文件读取/解码不计入渲染时间；广告补 alpha=255，按样例 `ad_h_w=(200,200)` 显式缩放，保持原 `m_inv` 的素材坐标尺度。
- 原样例两路各 3 个广告。基准按原数值路号和 slot 顺序循环扩成 **8 路 × 每路 16 slot**。它仍然只有 3 个原始 ROI 和 1 份实际素材像素数组，不代表 16 个不同位置、不同素材的真实业务场景；重复 ROI 仍按原顺序覆盖。
- 所有候选使用相同采样、alpha、normalization 和相机策略。运动回调使用明确标注的演示 `SyntheticOwner(kernel_size=5)`；静态相机对应 sigma=0.65 的 5×5 运动模糊核，不能替代尚未提供的生产回调。
- 共用 mask 在所有候选中保持 Triton：局部 mask 先 maximum 合并到整帧，再执行 sigma=1.5、radius=3 的 7×7 高斯羽化。这里的 mask 羽化与 5×5 广告运动模糊是两项不同处理。

默认 9 个候选为 `dense_whole`、`dense_c1/c2/c4`、`separable_whole`、`separable_c1/c2/c4` 和 `eager`。可额外选 `dense_auto` 检查当前策略，但不能据此省略整批基准。

`eager` 同时替换共享参数 builder 和执行器：共享几何与广告 warp 使用 eager，blur 使用 PyTorch；最终 NumPy 导出方式不变，mask 仍使用 Triton。因此该行不是“全流程禁用 Triton”。畸变图上的画布采样在现有各模式中本来就使用 PyTorch `grid_sample`。

## 方法与准入标准

实现见 [benchmark_render_img_modes.py](benchmark_render_img_modes.py)。每个候选/轮次创建独立 worker、配置字典和有界相机历史，共用只读背景/素材输入；先预热填满历史。本轮静态正式矩阵为 9 个候选 × 9 轮，每轮预热 30 次、测量 100 次，即每候选 900 次完整调用，逐次检查输出。候选顺序逐轮轮换，轮数为候选数的整数倍，使各执行位置出现次数相同。

计时从调用 `render_img` 前开始，至独立 NumPy 输出可读取且 GPU 同步结束为止，包含路由元数据、共享 mask/相机/几何、素材准备、ROI 网格、核心计算、H2D/D2H 和输出导出。NPY/BMP 读取、fixture 构造、帧号/bbox 更新、JIT/预热、CPU 图像校验、内存/所有权检查和编码/网络不计时。每次计时输出在校验后释放，下一次计时不持有上一份结果；另用两次调用检查旧 NumPy 未覆写且两份结果不共享存储。

每个候选/轮次在正式预热前，先针对每个固定输入 pattern 完整计算候选对 `dense_whole` 的最大 uint8 差异、变化通道数和变化像素数。零差 pattern 直接使用 dense reference；非零 pattern 额外保存独立、普通 CPU 存储的候选 reference。随后重新预热，再开始计时。

每次实际计时调用之后，使用 `np.array_equal` 对全部 8 路、全部像素进行严格比较。只有输出完全等于对应 reference，才由相等关系的传递性复用已知的 dense 差异计数；这不是抽查、容差比较或假设输出不变。任何变化都现场重新完整计算对 dense 的差异，并记为 `unexpected_output_changes`，拒绝自动准入。非零差异仍记录而不终止整个矩阵。

dense reference 是 GPU dense whole 的普通 CPU 副本，用于检验模式切换是否改画面，不是独立 CPU 数学真值。非零候选 reference 的数量受 pattern 数限制，在该候选/轮次后释放；静态最多额外一份 B8 副本，平移的零差 dense 候选不新增候选副本。

报告将 `validation_samples_ms` 与渲染时间分开。全像素相等检查仍在帧间产生约 16–20 ms 的 CPU 工作；完成的静态矩阵中，各候选校验均值为 15.96–17.61 ms。这段工作虽不计入 timer，仍可能影响后续 GPU 调用的时序、CPU 调度与频率。因此本轮只能比较相同方法下的候选，不能直接与上一轮约 12.52 ms 的数字相减并宣称优化收益，也不代表包含校验后的整条视频链路吞吐。

除静态布局外，单独执行 12 步平移周期：所有路/slot 同步改变 bbox 的画布位置，保持 ROI 尺寸、原 `m_inv`、相机、素材和 mask 不变。实际偏移写入报告；这验证实时 bbox 放置与输出，不等同于任意形变、素材更新或动态 mask。平移参考仅在逐步确认 dense whole 的 8 路完全相同后保存每步首路，实际候选仍逐一检查全部 8 路。

平移确认只比较静态像素门槛通过的 `dense_whole` 和 `dense_c1/c2/c4`：4 候选 × 4 轮，每轮预热 30 次、测量 120 次，即每候选 480 次；每轮覆盖 10 个完整平移周期。eager/separable 静态已出现非零差异，因此不再重复动态性能确认。最后额外比较 `dense_whole` 与 `dense_auto`，采用 2 个候选 × 2 轮、每轮预热 30 次和测量 100 次，检查保留默认设置的实际决策与输出。

自动准入须同时满足：

1. 所有已检查调用对 dense whole **逐字节一致，最大差异为 0**，且 `unexpected_output_changes=0`；视觉上难察觉的 ±1 也不豁免。
2. 静态和平移确认轮的完整调用均值、P95 均稳定改善至少 5%，并检查每轮结果，不能以单次最低值或微内核计时替代。
3. 独立 NumPy、旧输出存活、关闭清理与原 API 合同保持；硬软件、B、kernel、实际有序素材/ROI 布局和广播方式的范围清楚。未知组合继续回退。

本轮没有候选同时满足两项性能指标和像素门槛，因此保留默认与现有表，不添加新自动 profile。dense 与 separable 的浮点计算顺序不同，不能仅因原二维核可分离就认定 uint8 结果相同。

内存记录区分库缓存与进程级 CUDA allocated/reserved。各候选串行运行，但分配器可能保留前一候选释放的空间；reserved 高于零不能直接解释为泄漏。本轮有限单帧/平移实验也不能证明长期生产直播的内存趋势。

## 排除的旧测量

`results/3336_modes_20261022/static.json` / `static.log` 在第一轮中断，**不属于完整正式矩阵**。旧实现每帧都计算详细差异计数；`validation_diagnostics.json` 中这项 CPU 校验平均耗时 154.13 ms，随后 dense 调用均值为 64.26 ms。改为完整 `np.array_equal` 后，校验均值为 16.39 ms，渲染均值为 12.40 ms；随后无校验复测为 12.50 ms。计时区外的校验工作仍干扰了原测量序列，不能将旧结果归因于渲染算法性能。

排除说明见同目录 `EXCLUDED_RUN.md`，旧代码保存在 `benchmark_full_diff_snapshot.py`。原始 JSON、日志、测量值和诊断记录均保留；修正后的正式矩阵另存 `static_v2.json`，不覆盖旧记录。本次基准修正未修改运行时数学、Torch 线程数、默认数值 flags、核大小或策略。

## 已完成的组件像素诊断

本机记录为 `results/3336_modes_20261022/pixel_diagnostics.json`。以下数字均为**每路**结果对 dense reference 的差异，诊断同时确认 8 路相同。

| 单独切换的部分 | 最大 uint8 差异 | 变化通道数/路 | 变化像素数/路 |
|---|---:|---:|---:|
| dense reference | 0 | 0 | 0 |
| 共享几何改为 eager | 1 | 273 | 268 |
| 广告 warp 改为 eager | 1 | 181 | 175 |
| blur 改为 PyTorch，其他加速设置保留 | 0 | 0 | 0 |
| blur 改为 separable | 1 | 544 | 531 |
| eager 共享几何、warp、PyTorch blur，mask 仍为 Triton | 1 | 323 | 315 |

这些组件切换实验支持“几何/warp/separable 路径切换可能改变最终量化结果”的判断。通道集合会重叠，后续浮点运算与 uint8 截断还会改变差异，**不能将这些数量相加分摊总体误差，也不能外推所有输入的误差上界**。PyTorch blur 在这一帧零差异不等于其他帧、核或配置也零差异。

诊断保留当时的数值设置：`cudnn.allow_tf32=True`、`matmul.allow_tf32=False`、float32 matmul precision 为 `highest`，未借改变这些设置追求不同结果。该记录不含正式性能结论。eager 与 separable 的当前诊断已出现非零差异，尚不满足保持 dense 原画面的自动替代要求。

## 正式结果与复现

静态正式报告 `results/3336_modes_20261022/static_v2.json` 的 `status=complete`，共 81 个 phase、8100 次计时调用、64800 张逐路输出检查。全部 `unexpected_output_changes=0`；各候选每轮的原始时序、校验耗时和输出差异均保留在报告中。下表只取完整报告的 aggregate，不使用 partial 或初步 probe 数值。

实测环境为 Windows 11、RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128、triton-windows 3.5.1.post24、NumPy 2.5.2。运行时版本为 `2026-10-22-perf4.7-render-img`，采样时 Git HEAD 为 `5da4cce`；基准源码 SHA-256 为 `062dab469096b76482f9bf5b5d47b2c867d89e8c6b695f8cdf54cfed9048f85b`，其他源码与输入哈希见报告 `provenance`。

平移报告 `results/3336_modes_20261022/translated.json` 同样为 `status=complete`，共 16 个 phase、1920 次计时调用、15360 张逐路输出检查。4 个 dense 候选均为最大差异 0，且无 unexpected output change。

| 候选 | 静态 mean / P95（ms） | 平移 mean / P95（ms） | 所有计时帧最大差异 | 准入结果 |
|---|---|---|---|---|
| dense whole | 13.10 / 16.09 | 14.35 / 17.82 | 0 | 保留为本例执行路径 |
| dense C1 | 18.91 / 23.87 | 17.83 / 23.85 | 0 | 不准入：性能未达门槛 |
| dense C2 | 14.66 / 17.98 | 12.80 / 17.39 | 0 | 不准入：未同时满足两种布局和两项指标 |
| dense C4 | 13.76 / 17.19 | 14.28 / 17.89 | 0 | 不准入：无稳定性能收益 |
| separable whole | 15.05 / 17.21 | 未测：静态像素门槛未过 | 1 | 不准入 |
| separable C1 | 18.24 / 24.35 | 未测：静态像素门槛未过 | 1 | 不准入 |
| separable C2 | 15.29 / 18.52 | 未测：静态像素门槛未过 | 1 | 不准入 |
| separable C4 | 13.84 / 17.09 | 未测：静态像素门槛未过 | 1 | 不准入 |
| eager，mask 保持 Triton | 26.08 / 32.69 | 未测：静态像素门槛未过 | 1 | 不准入 |

未纳入的具体原因：

- dense C1 在静态和平移中均更慢。
- dense C2 的平移 mean 改善 10.79%，但 P95 仅改善 2.40%，低于 5% 门槛；静态 mean/P95 反而慢 11.94%/11.78%。单项动态均值收益不足以准入。
- dense C4 的静态 mean/P95 慢 5.07%/6.86%；9 轮中只有 4 轮 mean 更快、1 轮 P95 更快，没有任何一轮两项同时改善至少 5%。平移 mean 仅改善 0.48%，P95 慢 0.40%，没有稳定收益。
- eager 和所有 separable 候选最大 uint8 差异为 1，不满足保持 dense 原画面的要求；不以差异较小为理由自动切换算法。

因此不新增 profile，也不为已自然回退的 dense 整批路径添加 chunk=0 条目。手动 C1/C2/C4 与显式 separable 入口继续保留，使用者仍需按自己的画面与延迟目标验收。

### 默认 auto 复核

`results/3336_modes_20261022/auto.json` 为 `status=complete`：2 候选 × 2 轮，每候选 200 次调用，共 3200 张逐路输出检查，全部最大差异 0、`unexpected_output_changes=0`。

| 设置 | mean / P95（ms） | 实际 chunk | 决策 reason |
|---|---:|---:|---|
| 显式 dense whole | 13.49 / 16.84 | 0 | `explicit` |
| 默认 dense auto | 13.87 / 16.25 | 0 | `unsupported_backend_or_output` |

两者 `batch=8`、`blur_launch_config=None`，均运行 dense 整批核心。auto 的 reason 表示该 dense5 配置不在现有 separable9 实测范围内，不代表渲染失败或改用 eager。两行时序只是此次对照的观测，不能解释为自动切换了算法或获得新的优化收益。

### 输出与内存验证

三份完整报告合计 10420 次计时调用、83360 张逐路输出检查；101 个 phase 的独立 NumPy/旧输出未覆写检查均通过。关闭后，报告中的库素材 stack、mask 工作区、Gaussian 条目、素材源引用、owner 当前素材组和相机历史均为 0。

这不等于进程 CUDA allocated 全部归零：静态参考关闭时 allocated 为 0，首次 eager phase 后，静态 81 个 phase 的关闭后 allocated 均固定为 **8.125 MiB**。本轮未定位这部分进程级存活分配的来源，不将它猜测为某个库泄漏或用 reserved 解释。平移和 auto 两个独立运行中，各 phase 关闭后的 allocated 为 0。有限运行和这些清理字段不能替代长期生产内存验证。

平移矩阵的计时阶段 CUDA peak allocated 如下，是进程级峰值总量，不是增量，也不含 CPU NumPy/reference 存储：

| 候选 | CUDA peak allocated（MiB） |
|---|---:|
| dense whole | 756.966 |
| dense C1 | 311.708 |
| dense C2 | 374.989 |
| dense C4 | 502.966 |

分块在此测量中降低了峰值显存，但没有满足本轮以完整调用 mean/P95 为依据的自动延迟准入标准。

收尾验证通过：已有报告拒绝覆盖且 SHA-256 未改变；CPU 设备请求被拒绝且未创建输出；41 个 Python 文件通过 Ruff 和 Python 3.9 AST 检查，9 项策略测试通过。运行时算法与 auto 策略表没有因本轮基准发生变更。

### 复现命令

复现时从仓库根目录运行。仓库不包含用户 JPEG 或中间 BMP，先在 Windows PowerShell 中替换下面三个原始输入路径，使用系统 `System.Drawing` 解码 JPEG 并保存 24 位 BMP；不需要安装 Pillow。输出只放到 `results/`，原始 JPEG/NPY 保持不变。函数拒绝覆盖已有输出，并在成功或异常时释放图像句柄。

```powershell
$samplePath = '请替换为3336.npy的绝对路径'
$frameJpegPath = '请替换为3336.jpg的绝对路径'
$adJpegPath = '请替换为ads_sample.jpg的绝对路径'
$inputDirectory = Join-Path (Get-Location).Path 'results\3336_modes_inputs'
$frameBmpPath = Join-Path $inputDirectory 'frame_decoded.bmp'
$adBmpPath = Join-Path $inputDirectory 'ad_decoded.bmp'

foreach ($destination in @($frameBmpPath, $adBmpPath)) {
    if (Test-Path -LiteralPath $destination) {
        throw "拒绝覆盖已有文件：$destination；请改用新的 inputDirectory。"
    }
}
New-Item -ItemType Directory -Path $inputDirectory -Force | Out-Null
Add-Type -AssemblyName System.Drawing

function Export-JpegAsBmp {
    param([string]$Source, [string]$Destination)
    if (Test-Path -LiteralPath $Destination) {
        throw "拒绝覆盖已有文件：$Destination"
    }
    $sourceFile = (Resolve-Path -LiteralPath $Source -ErrorAction Stop).Path
    $decodedImage = $null
    try {
        $decodedImage = [System.Drawing.Image]::FromFile($sourceFile)
        if ($decodedImage.PixelFormat -ne [System.Drawing.Imaging.PixelFormat]::Format24bppRgb) {
            throw '此预处理要求24位RGB JPEG；请核对是否使用了相同的原始样例。'
        }
        $decodedImage.Save($Destination, [System.Drawing.Imaging.ImageFormat]::Bmp)
    }
    finally {
        if ($null -ne $decodedImage) { $decodedImage.Dispose() }
    }
}

Export-JpegAsBmp -Source $frameJpegPath -Destination $frameBmpPath
Export-JpegAsBmp -Source $adJpegPath -Destination $adBmpPath
```

在同一个 PowerShell 会话中执行正式对照。以下命令给出本轮实际参数；JSON 输出必须尚不存在，旧结果已存在时改用新的结果文件名。

```powershell
$inputArgs = @('--device', 'cuda:0', '--sample', $samplePath, '--frame-bmp', $frameBmpPath, '--ad-bmp', $adBmpPath)

.\scripts\run.ps1 render-img-modes @inputArgs --layout static --rounds 9 --warmup 30 --iterations 100 --output results/3336_modes_static_new.json
.\scripts\run.ps1 render-img-modes @inputArgs --layout translated --candidates dense_whole dense_c1 dense_c2 dense_c4 --rounds 4 --warmup 30 --iterations 120 --output results/3336_modes_translated_new.json
.\scripts\run.ps1 render-img-modes @inputArgs --layout static --candidates dense_whole dense_auto --rounds 2 --warmup 30 --iterations 100 --output results/3336_modes_auto_new.json
```

原始用户输入、解码中间文件与 `results/` 不随 Git 分发。复跑须自行提供相同输入；报告记录文件及源码 SHA-256，不能只凭文件名认为两轮工作量相同。接口和上轮功能验收见 [RENDER_IMG.md](RENDER_IMG.md)，既有自动策略历史见 [DYNAMIC_BATCH_REVIEW.md](DYNAMIC_BATCH_REVIEW.md) 与 [BACKEND_COMPARISON_REVIEW.md](BACKEND_COMPARISON_REVIEW.md)。

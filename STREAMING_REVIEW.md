# perf4.5：直播准备与资源生命周期

版本：`2026-10-21-perf4.5-streaming`。本轮以 `6edf109` 为前版对照，处理长期复用 worker 时的缓存保留、逐帧重复准备和退出释放。默认后端继续为 `batch`，`batch_chunk_size="auto"` 的准入表未改，GPU 数学与独立 NumPy 输出协议保持原样。

当前已完成 CPU/GPU 回归、12 场景完整性能对照与 10000 帧合成持续调用。平均耗时下降 0.64%～6.72%；长测中同结构 GPU 活动内存未见增长，CPU RSS 仍有小幅增长，固定页统计存在不自洽字段，不能宣称全链路零增长或生产永不泄漏。上一轮桶优化结果保留在 [BUCKETED_OPTIMIZATION_REVIEW.md](BUCKETED_OPTIMIZATION_REVIEW.md)，不在本报告重写历史结论。

## 实现范围

### 素材缓存有界，但当前工作集仍完整

`paste_asset_cache.py` 的 `SizeAssetCache` 从 adapter 中独立出来。历史 `[B,4,h,w]` stack 同时受尺寸数、每尺寸条目数和字节预算约束；新增 `AdapterOptions.max_cached_bytes` 默认 `512*1024*1024`，要求非负 Python `int`。设为 0 不保留历史 stack，仍允许重复帧复用当前完整素材工作集。

当前 `_ads` 工作集与当前桶打包分别保留，可超过历史预算。这保证素材较多时仍能完整渲染，也避免单次工作集超过预算后每帧全部重建。同一个 stack 可同时被当前工作集和 LRU 引用，不意味着复制两份。`cache_info()` 的 `retained_stack_bytes` 对这些 Tensor 引用去重；`bytes`、`current_bytes`、`bucket_bytes` 不能直接相加当作真实总量。

预算只统计缓存拥有的 stack 逻辑字节数。它不包含源对象的底层存储、当前桶打包、LUT、临时工作区、输出或分配器保留量。特别是源图为大数组的切片时，保留一个 source 引用可能保留更大的底层 storage。因此 512 MiB 不是整个 worker 的内存上限。

每次有广告调用仍读取有序成员身份、shape/dtype/device、B 和 resize 等结构。历史条目与当前工作集保留对应 source 强引用，避免 Python 回收旧对象并复用 `id` 后产生错误缓存命中。原地修改像素须递增 `owner.imgs_raw_version` 或调用 `invalidate_assets()`；仅保持相同对象身份不会自动检测像素变化。

revision 精确变化时清旧代历史及当前素材/桶引用，累计计数继续记录；回到旧 revision 也重新构造。`cache_ads=False` 或 revision 为 `None` 时，在下一次有广告调用清旧缓存并逐帧重读。无广告路径不承担素材清理；worker 停用时用显式 `release_resources()`。`invalidate_assets()` 清素材与图像布局计划，并重置相应缓存计数。

桶素材和 bbox 布局解耦的 perf4.4 行为保留：bbox/ROI 变化时重建 LUT，素材与有序桶成员相同则复用桶打包；`area_caps` 改变成员分组仍需重打包。尺寸缓存本身不等于启用 bucketed 计算。

### 仅复用元数据，不缓存实时几何值

图像布局计划与 mask/透明度 map 计划缓存只存 shape、布局、排列/广播操作等 CPU 元数据，并各设最多 128 项。计划应用于当前 Tensor；不保存旧 map 值，也不跳过实时成员、形状或设备变化。

adapter 的各准备阶段负责自身的输入局部校验和 float32/device 规范化，移除进入 core 前对同一准备结果的重复全树扫描。独立调用纯 Tensor core 时，公开 `validate_inputs()` 仍是调用方的校验入口。

batch core 逐 slot 判断 Triton 支持条件及 padding 后，调用私有 `_warp_paste_into_supported`，避免公开包装器再次扫描相同元数据。公开 `warp_paste_into` 仍执行保护检查；展开、stride、启动参数和 GPU 算术共用。不能把这个私有入口作为业务绕过检查的接口。

K、dist_coeffs、m_inv 可能在同一帧号或相同对象中原地更新，仍每帧构造当前几何。sigma 仍由真实业务回调逐帧产生，精确缓存不做量化；连续 sigma 通常会持续 miss，这不能靠改变数值规律来“提升命中”。

### 有界相机历史与 Gaussian 上下文

业务可以显式选择 `BoundedCameraHistory(capacity=9)`。写入帧号须单调递增，允许更新最新帧；回看、seek 或换流须先 `reset()`。记录按引用保留，生产者复用数组存储时应在写入前复制。滤波器只对这个专用类型做 O(radius) 的直接窗口查询；普通 Mapping 保留原来的全表遍历和全部 key 校验，库不自动删除调用者历史。

容量按记录数计，半径默认 4：处理最新帧至少保留 5 条；中心帧允许四帧前瞻时至少保留 9 条。更长延迟或前瞻需更大容量，稀疏帧号可能仍不足三个邻居并触发原 fallback。它不承诺恢复已经驱逐的旧帧，也不改变 legacy/corrected 的选择规则或业务 motion 回调次数。

Gaussian LRU 默认最多 16 项，只保留当前 owner、回调身份、配置版本、device 和缓存配置上下文。上下文变化时释放旧项及旧 owner 强引用；当前上下文保留身份引用，继续防止 `id` 复用。`builder.clear_cache()` 和 `builder.release_resources()` 都清缓存、身份引用与统计。它们不清外部相机历史，也不会使调用方已经持有的 Tensor 失效。

## 直播循环接入示意

下面只展示准备与生命周期，不实现解码、编码或发送。假定每个 packet 的 `camera` 字段提供 CPU NumPy 的 K、w2c、dist_coeffs，`data` 提供真实 ad_params；其他业务数据类型须按原回调契约处理。owner 的三个真实回调保持不变。

```python
from collections import deque

from dynamic_paste_example import initialize_dynamic_worker, render_dynamic_frame
from paste_camera_filter import BoundedCameraHistory


def run_worker(owner, base_options, incoming_packets, device):
    builder = initialize_dynamic_worker(owner, base_options)
    history = BoundedCameraHistory(capacity=9)  # 初始化一次
    outputs = deque(maxlen=3)  # 演示只保留最近三帧；实际满队列策略由业务决定
    try:
        for packet in incoming_packets:
            frame_id = packet["frame_id"]
            # 独立记录，不引用下一次采集会覆盖的相机数组。
            record = {name: packet["camera"][name].copy()
                      for name in ("K", "w2c", "dist_coeffs")}
            history[frame_id] = record
            groups = packet["material_groups"]
            if not groups:  # B=0 跳过渲染，历史仍正常推进
                continue
            data = dict(packet["data"])
            data["K"] = record["K"]
            data["dist_coeffs"] = record["dist_coeffs"]
            owner.imgs_raw_version = packet["asset_revision"]
            outputs.append(render_dynamic_frame(
                owner, packet["frame"], data, frame_id,
                material_groups=groups, camera_history=history,
                mask_canvas=packet["mask"], device=device,
            ))
    finally:
        outputs.clear()
        try:
            owner._paste_batch_adapter.release_resources()
        finally:
            builder.release_resources()
            history.clear()
```

`data` 与历史 record 是两个字典，滤波替换 `data['dist_coeffs']` 不会替换保留的原始记录。`asset_revision` 应在源像素原地改变时更新，静态素材不必每帧递增。seek/换相机时应在下个写入前显式重置 history 与 builder 缓存；素材/布局也发生切换或 worker 停用时释放 adapter。业务自己持有的素材、packet 和队列引用仍须按其生命周期管理。

每次输出仍是独立的 NumPy `uint8`：B1 为 HWC，多组为 BHWC。固定页路径由 NumPy 持有底层 host Tensor；最后一个数组/Tensor 引用释放后，存储才可由 PyTorch 分配器回收复用。`release_resources()` 不覆盖仍被消费者持有的输出，也不清空全局分配器。

## 测量方法

先完成数值与输入契约回归，再测性能。性能比较使用同进程隔离加载的 `results/baseline_streaming_6edf109`，新旧候选各有独立 builder/adapter 和可变输入状态，轮换候选顺序。完整调用计时含共享相机/几何/核准备、素材处理、实时 map、GPU 运算及可立即读取的独立 NumPy 返回；输入 fixture 更新放计时外，不能预构造渲染中间结果。

持续运行检查命令：

```powershell
.\scripts\run.ps1 check-streaming --device cuda:0 --baseline results/baseline_streaming_6edf109 --frames 10000 --warmup 200 --sample-every 100 --check-every 250 --revision-every 1 --retain-outputs 3 --output results/streaming_10000.json
```

`--frames` 是固定与动态两个阶段合计测量调用数，不含预热，默认 5000；上述命令是 10000，分为固定 5000 帧与动态 5000 帧。固定阶段 B8、静态素材；动态阶段轮换 B1/4/8/16，使用 `--revision-every 1` 每帧改变实际 RGB 与 revision，同时改变素材尺寸顺序、平移/缩放 bbox，并持续改变 sigma。动态预热覆盖有限 B/源顺序组合。CPU smoke 用较小图和帧数，只验证检查工具流程，不能代替 GPU 长测。

工具定期对照旧源码图像与 auto 决策，要求已检查样本逐字节一致；这些检查不意味着未采样帧都经过对照。输出保留包括有限队列及首帧锚点/快照，用于检查后续调用不覆写旧数组；weakref 检查队列驱逐和释放后的对象生命周期。周期性基线输出和缓存清理后再采样，减少参考实例常驻资源对当前实例趋势的干扰。

报告同时记录以下指标，不能只看任务管理器中的一个数字：

| 指标 | 解释与边界 |
|---|---|
| 进程 RSS/working set、private bytes | 含输入 bank、有限输出、参考快照、Python/NumPy/Torch 和运行时资源；工作集可受操作系统回收策略影响 |
| CUDA allocated/active | 当前由分配器管理的活动资源；仍含业务输入与存活输出，需要结合释放边界解释 |
| CUDA reserved、inactive split、分配重试/OOM | 用于观察缓存分配器保留和碎片；reserved 不是泄漏字节数 |
| host/pinned 分配统计 | API 可用时记录原始计数；缺值不能当作零，字段自洽性不足时不能据此推算存活字节或释放量 |
| cache_info 与输出保留量 | 历史 stack 预算、当前/bucket/去重 retained 字节、版本/驱逐计数、Gaussian 项数及消费者保留字节数 |

该长测复用合成 fixture 的有限相机历史，未单独统计 `BoundedCameraHistory` 容量；专用历史容器的数值和引用释放由 CPU 测试覆盖。脚本记录每阶段消费者输出释放后的状态，不能将其表述为 builder/adapter 全部资源释放后的 GPU 采样；后者已另做独立 release/reuse 测试，见下文。

重点比较预热后固定场景是否继续增长、动态组合重复后是否形成稳定工作集，以及消费者释放后活动引用是否减少。完整退出验收同时检查库资源释放前后与实例重用。不得逐帧 `gc.collect()`、`empty_cache()` 或额外全局同步改变正常直播行为来掩盖增长。PyTorch 可以保留已释放的 GPU/固定页内存供后续复用，`reserved` 或 RSS 不立即下降不等于泄漏；另一方面，有界单项缓存和一次平稳采样也不足以证明全链路无泄漏。

## 完整调用性能结果

环境为 Windows、RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1。1080p、9×9 可分离核，素材尺寸循环为 200×200 / 500×500 / 300×600。前版 `6edf109` 与本版均使用既有 batch auto 策略和默认 blur 启动配置。

每个场景、每个候选共 4 轮；每轮先预热 30 次，再测量 100 次，两候选交替先后顺序。下表均值和 P95 由该候选合计 400 个样本计算，P95 是合并样本的第 95 百分位，不是四个分位数的平均。素材按固定组合复用，因此素材阶段主要走缓存命中；共享参数和 sigma 随帧更新，计时到独立 NumPy 完成返回。正确性在第 0、7 帧逐字节对照通过，同时检查旧输出独立性，未逐一对照全部计时帧。

单位：毫秒；下降比例为 `(前版均值 - 本版均值) / 前版均值`。

| 广告位 | B | 前版均值 | 本版均值 | 均值下降 | 前版 P95 | 本版 P95 |
|---|---|---|---|---|---|---|
| 4 | 1 | 1.608 | 1.597 | 0.64% | 1.737 | 1.684 |
| 4 | 4 | 3.083 | 3.000 | 2.69% | 3.261 | 3.183 |
| 4 | 8 | 4.379 | 4.270 | 2.48% | 4.470 | 4.368 |
| 4 | 16 | 6.689 | 6.570 | 1.77% | 6.830 | 6.675 |
| 16 | 1 | 2.574 | 2.420 | 5.99% | 2.719 | 2.548 |
| 16 | 4 | 4.188 | 3.960 | 5.44% | 4.345 | 4.082 |
| 16 | 8 | 5.672 | 5.387 | 5.04% | 5.759 | 5.482 |
| 16 | 16 | 8.610 | 8.185 | 4.93% | 8.715 | 8.344 |
| 32 | 1 | 4.369 | 4.145 | 5.13% | 5.133 | 4.919 |
| 32 | 4 | 6.598 | 6.275 | 4.90% | 6.981 | 6.589 |
| 32 | 8 | 8.745 | 8.157 | 6.72% | 9.131 | 8.545 |
| 32 | 16 | 13.059 | 12.318 | 5.67% | 13.397 | 12.751 |

三份原始报告为 `results/streaming_compare_s4_final.json`、`results/streaming_compare_s16_final.json`、`results/streaming_compare_s32_final.json`；合并表为 `results/streaming_performance_summary.json`。仅使用这轮最终确认数据，较早 S16 preflight 不混入统计。复现命令对 slots=4、16、32 分别执行：

```powershell
.\scripts\run.ps1 compare --baseline results/baseline_streaming_6edf109 --baseline-label before --batches 1 4 8 16 --slots 16 --chunks auto --configs default --height 1080 --width 1920 --kernel-size 9 --ad-sizes 200x200 500x500 300x600 --rounds 4 --warmup 30 --iterations 100 --device cuda:0 --output results/streaming_compare_s16_final.json
```

本机这轮观察为小幅完整调用改善，未给出统计置信区间，也不能外推到生产视频、任意素材或更高负载。32 广告位的测量不扩张 auto 准入表。

## 10000 帧持续调用结果

`results/streaming_10000.json` 已完成，独立只读审计为 `results/streaming_10000_audit.json`。固定和动态阶段各 5000 帧、各 50 个窗口采样；动态阶段每帧实际更新 RGB/revision，两个阶段都持续产生精确 sigma。测量阶段记录的图像与 auto 决策对照各 21 次、合计 42 次，全部一致；不是每一帧都执行旧源码图像对照。

| 观察项 | 固定 5000 帧 | 动态 5000 帧 |
|---|---|---|
| 同 B/源顺序/布局/输出保留量的 GPU allocated | 50 个采样点一致 | 按相同结构分组后的采样值一致 |
| allocated 减去去重素材 stack 的残差 | 同结构首末差 0 | 12 种结构各自首末差 0，每种仅 4～5 个采样点 |
| CUDA reserved 窗口采样 | 1746 MiB，保持不变 | 4058 MiB，保持不变 |
| 预热后新增 device alloc/free、重试、OOM | 均为 0 | 均为 0 |
| 第一个至最后一个窗口 RSS 变化 | +1.188 MiB | +2.117 MiB |
| 同范围 private bytes 变化 | +1.551 MiB | +2.266 MiB |
| 旧输出快照、独立性及队列驱逐 weakref | 通过 | 通过 |

动态 B 与源尺寸顺序会改变必要工作集，不能直接用两个不同结构端点的显存差得出趋势。上述残差只扣除已知 stack，并非完整 GPU 资源归因；每种动态结构仅有 4～5 个点，也不足以估计可靠的长期增长斜率。相机历史使用合成 fixture 的有限字典；`BoundedCameraHistory` 的独立容量/数值/引用生命周期由 15 项 CPU 测试覆盖。

历史素材缓存始终满足预算与条目约束，Gaussian 最多保留 16 项；动态阶段记录了 5000 次 revision 切换，没有累计保留旧版本。每阶段输出队列驱逐时记录 4996 次 weakref 释放，末尾清空剩余消费者输出也通过检查。旧数组保留期间未被后续帧覆写。

固定页分配器统计需要单独说明：104 个含预热后/窗口/消费者释放后的采样点中，`allocated_bytes.current` 超过 `reserved_bytes.current`，动态阶段还出现 current 大于 peak。该组字段不能用作实时存活字节数或泄漏增长的证据，也没有据此给出 pinned live-byte 残差。50 个窗口里的 reserved 原始报告值固定阶段为 320 MiB、动态阶段为 1000 MiB，并保持不变；这仍只作为诊断记录。没有确定这些异常计数的根因。

长测显示有限合成负载下 GPU 活动工作集重复稳定、消费者所有权正常；同时保留 CPU 小幅增长和统计质量限制。它未包含真实解码、编码、网络或业务消费者队列，不能替代更长时间的生产全链路验收。

## 退出释放与重用

独立结果 `results/streaming_release_gpu.json` 覆盖 B16 和 B8，均选择既有 auto chunk4。保留旧 NumPy 输出时执行 builder/adapter 的 `release_resources()`，素材/cache plan、Gaussian 上下文、source 引用、内部复制流和事件引用均清空，旧输出内容保持不变。随后同实例同帧重用重新构造资源，输出与释放前以及保留旧源码均逐字节一致，两个输出存储独立。

两组共 4 次旧源码对照和 4 个输出 weakref 释放检查通过。GPU allocated 在释放时分别从 1261.868 MiB（B16）和 964.139 MiB（B8）降至 672.389 MiB；剩余资源包括测试持有的输入 bank，不能期待进程 allocated 归零。分配器 reserved 可继续保留 2442 MiB，这不影响引用清理和实例重用。该脚本的部分 pinned 计数也不自洽，未用于解释实时存活字节。

## 保留的编译异常

最终 B1 repeated-sigma 共享回归的首次运行，在 dense blur `_blur_blend_u8` 编译的 `TritonGPUCoalesce` 阶段出现 Windows stack overflow / `PassManager::run failed`。同一命令使用正常缓存重试后通过。另一次本版新缓存尝试在畸变 kernel 的 launcher 初始化处被 Windows 应用控制策略拒绝，未到 dense 编译，因此不能把它算作首次异常的复现，也不能由它判定本版 dense 冷编译是否正常。保留基线 `6edf109` 的单次新缓存运行成功。

相关 dense wrapper、kernel 和 helper 的 Python AST 与基线相同，未因本事件修改内核、系统安全策略或全局栈设置。根因仍未确定，重试成功不等于已修复；不能把长测、性能和重用通过外推为所有未缓存 Triton 特化都能可靠编译。

详细四次运行与证据路径见 `results/streaming_compile_incident.txt` 和 `results/streaming_dense_ast_audit.json`。重试共享基准的最大 uint8 差 1 是原有 dense/separable/参考路径对照的容差范围；本轮新旧 batch 版本对照、长测采样及重用对照仍均为差 0，未为本轮放宽容差。

## 证据与源码版本

报告中 `results/` 路径均为此工作区保留的本地 artifact，被 Git 忽略，不随仓库提交。性能原始样本、10k 原始 JSON、独立内存审计、release/reuse 结果、编译异常四份日志与 AST 审计各自保留，不能把不同测试的通过范围合并扩大。

收尾核对的 10 个 `paste_*.py` 运行时文件和 7 个 `benchmark*.py` 文件 SHA-256 均与 `streaming_10000.json` 的 `current_sources` 相同。长测之后只为诊断工具增加 `pinned_counter_issues` / `pinned_live_bytes_valid` 字段，显式标记原始统计自洽性；未修改生产 runtime、性能基准或已有 10k 结果，也未重跑覆盖原长测。

长测时的完整诊断脚本保存在 `results/check_streaming_stability_10000_snapshot.py`，SHA-256 为 `9576bab4cfab74febcb25520402b87e97d6d539808db8ffc4cd2994477a01b44`，与原始 10k 报告记录的 harness hash 一致。复核原结果应读取该快照，不能假定收尾后的诊断脚本与测试时逐字节相同。

只读证据清单 `results/streaming_evidence_manifest.json` 验证通过：运行时/基准及 harness 快照哈希一致，12 个性能场景每候选 400 个样本的 mean/P50/P95 重算一致，各轮新旧 auto 决策一致，27 个 Python 文件可按 Python 3.9 语法解析。增加诊断标记后的工具另做 32 帧 GPU smoke，`results/streaming_diagnostics_final.json` 显示完成并确认 validity 标记生效；该短测只验证工具变更，不替代原 10000 帧证据。

## 验收状态

| 检查 | 当前结果 |
|---|---|
| CPU 主回归 | 42 项通过：4 项 adapter 准备、11 项素材生命周期、15 项相机/核生命周期、3 项公开 core guard、9 项策略 |
| 动态 B GPU 回归 | 265 项通过 |
| 后端契约 | GPU 115 项与另计 8 项布局检查通过，最大输出差 0；CPU 77 项通过 |
| 共享与核心回归 | shared_two、shared_three_repeated、shared_b1_repeated_retry、batch_three 最终通过；B1 首次 JIT 异常另行保留记录 |
| S4/16/32 × B1/4/8/16 新旧完整性能对照 | 12 场景完成，每候选 400 样本；均值下降 0.64%～6.72%，本轮 P95 均下降；正确性仅对照第 0、7 帧 |
| 固定/动态持续 GPU 调用与消费者释放检查 | 10000 帧完成，42 次记录图像与 auto 决策对照均一致；资源趋势及限制见上节 |
| 独立 GPU release/reuse | B16/B8、chunk4 通过且经独立审计；4 次旧源码对照差 0、4 个输出 weakref 释放，资源引用清空且可重用 |
| 最终诊断工具与证据复核 | 新增 pinned validity 标记后 32 帧 GPU smoke 通过；证据 manifest、42 项 CPU、27 文件 Ruff/Python 3.9 AST 通过 |

CPU 回归可用项目 Python 运行 `-m unittest test_streaming_history test_streaming_lifecycle test_adapter_preparation test_core_guards test_backend_policy`；公开 warp guard 测试覆盖不支持输入与 padding 的拒绝路径。收尾再次通过 42 项 CPU 回归及 27 个 Python 文件的 Ruff 检查。保留测试时的源码/依赖版本与上述适用范围，不能把有限帧合成验收表述成生产直播或推流全链路验收。

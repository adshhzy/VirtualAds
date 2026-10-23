# 动态路数优化与实测（2026-10-21）

> 目录迁移说明（2026-10-23）：本文的版本、数值、源码文件名和直接 Python 命令保留当时记录；旧快照仍是平铺目录。当前源码按功能分目录，源码名及启动命令映射见 [目录与迁移指南](../PROJECT_LAYOUT.md)。现有 `scripts/run.ps1` 子命令不变；复跑当前源码时从仓库根目录运行。原始测量 artifact 不随仓库分发。

当前本地版本：`2026-10-21-perf4.2-dynamic`。修改前基线是 Git `0b97beb7b641af5a27baaa65a2f594dae4835b2d` 的 perf4.1。
本次完成代码实现、不同路数基线/候选筛选、严格像素回归及优化前后对比。没有使用真实业务运动估计、sigma 和 Gaussian 回调；这些生产定义仍需由接入方提供。

## 最终结果与测量方法

环境：Windows / RTX 4090 24 GB / Python 3.12.14 / torch 2.9.1+cu128 / CUDA 12.8 / triton-windows 3.5.1.post24。
输入：1080×1920，16 个广告位，素材尺寸按200×200、500×500、300×600循环，各ROI为108×320；共享背景/单通道mask，9×9已验证可分离核，连续sigma、动态相机参数。
每个路径每轮预热20次、计时100次，共4轮；轮换先后顺序使原版和新版各先执行两轮。表中均值和P95基于每路径合并的400个完整调用样本。
计时从共享参数准备到返回可立即读取、独立存储的NumPy，并同步GPU完成；输入fixture更新在计时外。
两个版本在同一进程的不同包命名空间运行，分别持有相机历史和适配器，使用相同只读GPU输入。对照直接加载修改前源码，不用“新版关闭部分开关”冒充旧版本。

| B（路数） | 原版均值 ms | 新版均值 ms | 均值降低 | 原版P95 ms | 新版P95 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 2.578 | 2.532 | 1.8% | 2.949 | 2.666 |
| 2 | 3.140 | 3.193 | -1.7% | 3.257 | 3.344 |
| 3 | 3.738 | 3.659 | 2.1% | 4.333 | 3.784 |
| 4 | 4.176 | 4.200 | -0.6% | 4.341 | 4.342 |
| 5 | 4.759 | 4.806 | -1.0% | 5.241 | 5.281 |
| 6 | 5.385 | 4.982 | 7.5% | 5.856 | 5.377 |
| 7 | 5.968 | 5.427 | 9.1% | 6.352 | 5.787 |
| 8 | 6.507 | 6.026 | 7.4% | 6.801 | 6.356 |
| 9 | 6.891 | 6.161 | 10.6% | 7.265 | 6.609 |
| 10 | 7.339 | 6.182 | 15.8% | 7.719 | 6.802 |
| 11 | 7.859 | 6.559 | 16.5% | 8.268 | 7.149 |
| 12 | 8.346 | 7.152 | 14.3% | 8.745 | 7.698 |
| 13 | 8.944 | 7.498 | 16.2% | 9.327 | 8.046 |
| 14 | 9.702 | 8.114 | 16.4% | 9.869 | 8.266 |
| 15 | 10.320 | 8.442 | 18.2% | 10.399 | 8.538 |
| 16 | 10.769 | 8.761 | 18.6% | 10.993 | 9.082 |

1～5路保留整批。该组小幅正负变化不作为加速收益，B2存在约1.7%的均值回退，本轮P95也略高；不宣称所有路数均提速。6～16路均应用每块最多4路的内部流水线，所有这些路数的均值和P95在最终对照中均改善。
此前三轮确认也保留在 `results/dynamic_final_comparison.json`，不删除较慢或波动的轮次；最终主表使用先后顺序均衡的四轮结果。
这些数字是本机合成基准，不等同于解码、编码、发送在内的完整生产服务延迟。

主数据：[四轮原始样本](../../results/dynamic_final_balanced.json)、[逐轮日志](../../results/dynamic_final_balanced.log)。

## 实际采用的执行策略

`AdapterOptions.batch_chunk_size`：

- `"auto"`（默认）：仅查询离线实测表，不逐帧搜索或等待凑批。
- `0`：强制原整批执行，模糊默认启动参数也保持原值。
- 正整数：显式指定分块上限，要求batch后端、CUDA、固定页NumPy输出；尾块按真实路数处理，不补齐。

当前自动表覆盖B1～16：1～5整批，6～16每块最多4路。B6/7/9/10/11/13/14/15也分别测过，不是从B8外推。
例如B7为4+3，B13为4+4+4+1。同一adapter可以在调用间改变B；调用中素材组及顺序必须保持稳定。
共享参数、素材收集、网格和校验执行一次；只切分批次维度。单例输入保留广播，逐组输入按B切片。
计算流顺序执行各块，独立复制流等待生产事件后复制到本次专属host切片，后续计算与复制有机会重叠；正常返回和异常退出均等待复制流完成。
GPU源持有至复制完成。调用方继续得到普通NumPy，B1 HWC、多路BHWC、uint8，旧结果不被之后的调用覆盖。

自动策略保守匹配：
- 精确本机Windows、RTX4090（不包括4090D）、CC8.9、上述torch/CUDA/Triton版本。
- batch后端、Triton/auto加速、separable、固定页NumPy、legacy alpha、zeros padding、align_corners=False。
- 上述1080p、16广告位、三尺寸顺序与ROI尺寸、共享HWC存储背景、共享连续单通道mask、共享连续网格与标量透明度。
- 共享9×9核、两个连续FP32一维因子；权重、矩阵、畸变值始终实时读取，不缓存动态数值。

未知硬件、分辨率、布局、B>16、二维dense模糊等情况自动回退原整批。两尺寸素材也仍回退；没有把三尺寸测量当成其他布局的证据。
改变bbox位置允许继续使用当前ROI尺寸配置，但改变尺寸或素材尺寸模式会回退。
`adapter.last_execution` 提供实际B、块大小、模糊配置及选择/回退原因。手动覆盖未知模糊启动参数时，auto回退整批，避免组合未经验证的配置。
自动策略不强行启用separable；接入仍需验证真实业务核的可分离性。

## 模糊核调优结论

新增横向和纵向独立配置：
`blur_launch_config=(h_block_x,h_block_y,h_warps,v_block_x,v_block_y,v_warps)`。
核心和adapter可显式传递；None保留两段128×4、4warps。只改变线程组织，不引入FP16、近似核或改变Alpha表达式。

离线筛选8个B值、每阶段26个配置，共416组stage配置校验；横向FP32中间结果bitexact、纵向uint8输出exact。
随后把候选放回完整调用，五轮复核每块4路下的64×4/8warps配置：

| B | 原模糊配置均值 ms | 候选配置均值 ms | 判断 |
|---|---:|---:|---|
| 8 | 5.65 | 5.59 | 小幅改善 |
| 12 | 7.23 | 7.35 | 回退 |
| 16 | 8.82 | 8.79 | 差异很小 |

因此本版默认保留原模糊启动参数，不把不稳定微小差异作为收益；独立调参入口与可复现工具保留。
数据：[kernel筛选](../../results/blur_tuning_all_batches.json)、[完整组合测试](../../results/dynamic_combined.json)、[五轮复核](../../results/dynamic_blur_confirmation.json)。
未实施改变算法的全图融合、ROI裁剪、sigma量化、CUDA Graph或输出缓冲环。

## 正确性与兼容性

- 新增动态回归265项通过：B1/2/3/5/8、非整除尾块、1|B广播、非连续输入、共享及逐组dense核、separable、素材版本/成员顺序、同frame_id原地几何更新、异常恢复、旧NumPy保留。
- 用同一adapter/builder执行8→1→5→12→3→8；实际auto选择符合策略，所有图与强制原参数整批执行逐字节一致，未知尺寸回退。
- 四轮对照中B1～16在两个动态参数步骤均直接与未修改perf4.1逐字节一致。
- 原24 core + 48 adapter + 8动态矩阵 + 40融合模糊检查通过；两尺寸/三尺寸shared检查通过。相对独立原算法reference仍满足uint8最大差1、grid最大差3.5763e-7（原阈值2e-6），未累积放宽容差。
- 发现并修复原dense及separable内核对合法空间广播mask/background的Triton编译失败：按真实维度加载后广播数值，卷积/融合算术保持不变；新增48组专项回归。
- 所有Python文件通过Python3.9语法解析，本机Python3.12 GPU验证。未额外部署Python3.9 GPU环境。

日志：[动态回归](../../results/dynamic_final_checks.log)、[core回归](../../results/dynamic_batch_core_check.log)、[两尺寸shared](../../results/dynamic_shared_2sizes_check.log)、[三尺寸shared](../../results/dynamic_shared_3sizes_check.log)。

## 复现与接入

继续使用已有独立环境，不更改依赖。项目目录执行：

```powershell
.\scripts\run.ps1 check --device cuda:0 --profile-checks
.\scripts\run.ps1 compare --batches 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --rounds 4 --output results/recheck.json
.\scripts\run.ps1 dynamic --chunks 0 2 4 8 --configs legacy --output results/chunk-candidates.json
.\scripts\run.ps1 tune --rounds 3 --output results/kernel-candidates.json
```

比较工具默认使用已保留的 `results/baseline_perf4_1`。results不进Git；新checkout可从历史生成：

```powershell
New-Item -ItemType Directory -Force results | Out-Null
git archive 0b97beb -o results/baseline_perf4_1.zip
Expand-Archive -LiteralPath results/baseline_perf4_1.zip -DestinationPath results/baseline_perf4_1
```

原业务接入签名、`setup_shared_paste` 和 `prepare_shared_paste_args` 不变。若已经通过真实核验证并启用separable，默认auto会在匹配条件下启用策略。
显式控制示例：

```python
from dataclasses import replace
from paste_gpu_batch_adapter import PasteBatchAdapter

# 初始化worker时选择；不要每帧重建adapter。
owner._paste_batch_adapter = PasteBatchAdapter(
    replace(existing_options, batch_chunk_size="auto")
)
# 回退旧执行：batch_chunk_size=0
# 手动验证其它输入：batch_chunk_size=4，需pinned_output=True、return_numpy=True
# 查看选择：owner._paste_batch_adapter.last_execution
```

后续推广到其它GPU、分辨率、广告布局或核尺寸，先保留原整批对照，执行完整NumPy测量及相同的精确回归，再扩展离线配置。现有素材缓存仍按条目数限额，未加入字节预算；大B和多种素材组合仍需评估实际显存与下游保留输出造成的固定页内存占用。


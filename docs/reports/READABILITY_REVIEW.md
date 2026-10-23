# 可读性重构与验收（2026-10-21）

> 目录迁移说明（2026-10-23）：本文的版本、数值、源码文件名和直接 Python 命令保留当时记录；旧快照仍是平铺目录。当前源码按功能分目录，源码名及启动命令映射见 [目录与迁移指南](../PROJECT_LAYOUT.md)。现有 `scripts/run.ps1` 子命令不变；复跑当前源码时从仓库根目录运行。原始测量 artifact 不随仓库分发。

当前版本 `2026-10-21-perf4.2-readable`；重构前基线为 Git `fe07b03`，即已经完成动态路数优化的 perf4.2。
本轮目标是让代码便于理解和继续修改，不引入新的图像算法或扩大自动优化范围。
业务阅读入口见 [ARCHITECTURE.md](../ARCHITECTURE.md)。

## 已完成的结构整改

| 范围 | 改动 | 保留的约定 |
|---|---|---|
| 适配器 | 入口拆成参数选择、背景早返回、素材收集、frame/ROI/canvas/kernel 准备、策略选择、渲染与导出；素材描述与缓存填充分离 | 原调用签名、选项字段顺序/默认值、早返回和缓存行为 |
| 共享准备 | 相机行选择、系数堆叠、sigma 校验、CPU 核校验/上传、矩阵批量准备分别命名 | 回调传参/次数、精确 sigma、实时几何和数值运算次序 |
| 执行策略 | 后端、frame/kernel、广播布局、广告位布局分开匹配 | 原测量表、门槛顺序和 reason 字符串 |
| GPU 核心 | 提取公共模糊校验、因子校验和 PyTorch dense 模糊；明确 `_finish_canvas` 阶段 | Triton 数学表达式、启动参数、广播和 alpha 语义 |
| 内核调用 | 展开参数列表，说明 shape、stride、ROI 坐标和横纵启动配置 | 原 JIT 签名与 12 个内核 AST |
| 基准工具 | CLI 解析与执行分离，比较状态由 `ComparisonCase` 管理，共享同步计时/统计/轮换/报告函数 | 计时范围、默认参数、数据格式与独立状态 |
| 风格与导航 | 统一 Ruff 格式，补 lint 命令、编辑器配置、模块阅读指南 | 平铺模块名及原导入路径 |

没有重排 `AdapterOptions` 字段来换取表面整齐，避免破坏位置参数调用；仅按职责加说明。
没有把图像元组改成新的外部协议，也没有给每个广告位引入新的对象模型。
Triton 运算中的惯用坐标/stride 缩写保留，主要改善 host 侧编排和文档。

基准检查顺带补齐两类已有防错缺口：

- 对照图先复制冻结，并检查返回数组之间不共享内存，防止错误的输出复用覆盖 reference 后仍被判为通过。
- `--source-dir` 检查必要源码、按版本发现的本地依赖及实际导入来源，缺失快照明确失败，避免静默测到当前代码。

## 画面、接口与工具检查

- 所有原运行时函数/方法的参数和返回注解对照通过；`AdapterOptions` 字段顺序和默认值不变。
- 全部 Python 源文件通过 Python 3.9 语法解析和本机 Python 3.12 compile；实际 GPU 验证在 Python 3.12 完成。
- `scripts/lint.ps1` 格式与静态检查通过，`git diff --check` 通过。
- 独立代理 CPU 新旧对照：40 组 adapter、104 组几何、24 组相机/回调、18 组 sigma 缓存/回调、4 组矩阵准备均一致；另有策略元数据检查。
- [动态 GPU 回归](../../results/readability_dynamic.log)：265 项通过，包含动态路数、尾块、广播/非连续输入、逐组核、旧输出保留和异常恢复。
- [两尺寸](../../results/readability_shared_two_sizes.log)/[三尺寸](../../results/readability_shared_three_sizes.log)共享流程通过；[B1 repeated sigma](../../results/readability_shared_single_repeated.log)通过，优化准备路径计时区间 8 次核缓存命中、0 次 miss。
- [完整核心检查](../../results/readability_core_cold_after.log)：24 core、48 adapter、8 动态矩阵及 40 dense 模糊检查通过。与独立原算法 reference 的既有 uint8 容差仍为 1，没有放宽。
- 新旧源码 1～16 路均在两个动态输入步骤直接逐字节一致；NumPy 仍为 uint8，B1 HWC、多路 BHWC，每次独立，返回即可读取。
- [动态候选工具](../../results/readability_dynamic_tool_smoke.json)和[模糊调优工具](../../results/readability_tune_tool_smoke.json)的小型 GPU 冒烟检查通过；基准防错、计时边界和 CLI 另做 CPU 替身验证。

首次完整核心检查曾在小尺寸 9×9 dense 模糊特例中发生 `TritonGPUCoalesce` 编译错误，日志保存在 [首次尝试](../../results/readability_batch_core_first_attempt.log)。
没有因此改运算或降低检查标准。核 AST 和 host launch 与旧版一致；[旧新版独立核复现](../../results/readability_blur_probe.log)均通过，随后旧/新版本各自使用全新 Triton 缓存完成完整核心检查也均通过（[旧版](../../results/readability_core_cold_before.log)、[新版](../../results/readability_core_cold_after.log)）。
该错误未再次复现，具体根因尚未确认。格式化会改变 Triton 的源码缓存键，部署首次执行仍需预热并检查编译状态。

## 重构前后性能

RTX 4090 / Windows / torch 2.9.1+cu128 / triton-windows 3.5.1.post24；1080p、16 广告位、三种素材尺寸、共享 9×9 可分离核。
两个版本在同进程的隔离包中运行，输入像素相同、相机状态独立；各轮先后顺序轮换。
4 轮，每路径每轮预热 20 次、计时 100 次，汇总 400 个完整调用样本；包括共享准备和同步 NumPy，不含 fixture 更新/解码/编码。
此处的“前”已经包含上一轮动态分块优化，不能与 perf4.1 的历史数字混算。

| B | 前均值 ms | 后均值 ms | 均值变化 | 前 P95 ms | 后 P95 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 2.622 | 2.649 | +1.04% | 3.288 | 3.293 |
| 2 | 3.240 | 3.339 | +3.05% | 3.914 | 3.912 |
| 3 | 3.900 | 3.747 | -3.93% | 4.459 | 4.411 |
| 4 | 4.160 | 4.188 | +0.68% | 4.785 | 4.793 |
| 5 | 4.851 | 4.865 | +0.29% | 5.431 | 5.417 |
| 6 | 4.960 | 5.019 | +1.21% | 5.525 | 5.620 |
| 7 | 5.290 | 5.308 | +0.33% | 5.844 | 5.782 |
| 8 | 6.053 | 6.114 | +1.01% | 6.425 | 6.408 |
| 9 | 6.351 | 6.325 | -0.42% | 6.672 | 6.603 |
| 10 | 6.543 | 6.525 | -0.28% | 6.958 | 6.951 |
| 11 | 6.662 | 6.642 | -0.30% | 6.809 | 6.916 |
| 12 | 7.289 | 7.181 | -1.48% | 7.463 | 7.401 |
| 13 | 7.645 | 7.480 | -2.15% | 7.881 | 7.654 |
| 14 | 8.321 | 8.142 | -2.15% | 8.585 | 8.358 |
| 15 | 8.619 | 8.366 | -2.94% | 8.829 | 8.583 |
| 16 | 9.040 | 9.061 | +0.24% | 9.485 | 9.434 |

表中均值变化为 `(重构后/重构前 - 1)×100%`，正数表示更慢。
主测均值变化约 -3.9%～+3.1%，多数路数接近；本轮不把正负差异宣称为性能优化收益。
B2 首次均值约 +3.1%，但中位数接近，因此追加独立 8 轮、每轮 200 次的确认，结果见下表；两轮均保留，不用复测覆盖主表。

| B2 加长确认 | 重构前 | 重构后 |
|---|---:|---:|
| 均值 ms | 3.479 | 3.451 |
| P95 ms | 3.871 | 3.837 |

这些测量未显示 B2 的稳定退化；函数拆分仍可能引入微小 CPU 编排开销，不能保证所有真实业务输入耗时完全相同。
本轮保持原策略：匹配已测配置时 B1～5 整批，B6～16 每块最多 4 路；原模糊启动参数不变。
真实生产回调和完整视频流程仍未提供，本报告使用原有合成 fixture。

数据：[四轮全部样本](../../results/readability_comparison.json)、[逐轮日志](../../results/readability_comparison.log)、[B2 加长确认](../../results/readability_b2_confirmation.json)。

## 复现

```powershell
.\scripts\lint.ps1
.\scripts\run.ps1 check --device cuda:0 --profile-checks
.\scripts\run.ps1 compare --baseline results/baseline_readability --baseline-label perf4.2-before --batches 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --rounds 4 --output results/readability_recheck.json
```

`results` 不提交 Git；新 checkout 可恢复重构前基线：

```powershell
New-Item -ItemType Directory -Force results | Out-Null
git archive fe07b03 -o results/baseline_readability.zip
Expand-Archive -LiteralPath results/baseline_readability.zip -DestinationPath results/baseline_readability
```

生产依赖、默认图像语义和已有业务调用方式保持不变；Ruff 仅在独立 uv 工具环境中使用。

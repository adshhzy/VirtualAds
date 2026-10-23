# 统一渲染入口与兼容说明

> 目录迁移说明（2026-10-23）：本文的版本、数值、源码文件名和直接 Python 命令保留当时记录；旧快照仍是平铺目录。当前源码按功能分目录，源码名及启动命令映射见 [目录与迁移指南](../PROJECT_LAYOUT.md)。现有 `scripts/run.ps1` 子命令不变；复跑当前源码时从仓库根目录运行。原始测量 artifact 不随仓库分发。

版本：`2026-10-21-perf4.6-render-api`。对照基线：`4b2b311`。

本轮将业务调用收敛到 `paste_renderer.render_frame`，把原 adapter 的执行实现放入 `paste_render_executor.RenderExecutor`。新入口串联原有共享准备和执行阶段，业务无需每帧手工拼接 `paste_args`。数值内核、默认 `batch`、离线 `auto` 表、素材缓存及独立 NumPy 输出规则保持原有实现；本轮不以接口迁移宣称性能提升。

## 业务接口

从 [paste_renderer.py](../../virtualads/paste_renderer.py) 导入 `initialize_renderer`、`render_frame`、`release_renderer`。该模块同时导出 `RenderOptions`、`RenderExecutor`、`BoundedCameraHistory` 和 `RENDERER_VERSION`。

```python
def initialize_renderer(
    owner, base_options, *, use_separable=False,
    normalization="legacy", camera_index_policy="legacy",
    distortion_inverse=True, max_cached_sizes=3,
): ...

def render_frame(
    owner, frame_input, data_dict, frame_id, *, mask_canvas,
    material_groups=None, camera_history=None, device=None,
    way_id=1, camera_result=None,
): ...

def release_renderer(owner): ...
```

`initialize_renderer` 在每个 GPU worker 内调用一次，返回 `RenderExecutor`。`base_options` 使用业务原有的 `RenderOptions` 或兼容别名 `AdapterOptions`，保留通道布局、缩放、alpha、padding、对齐、素材 resize 及 `batch_chunk_size` 等未覆盖设置。它沿用旧 `setup_shared_paste` 的集成选择：

| 设置 | 统一初始化的值 |
|---|---|
| 计算后端 / 加速 | `backend="batch"`、`acceleration="triton"` |
| 模糊 | 默认 dense `"triton"`；显式 `use_separable=True` 时为 `"separable"` |
| 几何来源 | `grid_source="matrix"`、`matrix_coordinates="roi"`，ROI 网格回调绑定当前 builder |
| 输出 | `pinned_output=True`、`return_numpy=True`、`squeeze_single=True` |
| 其它覆盖 | `max_cached_sizes` 使用本函数参数，`profile_every=0` |
| 共享准备 | 默认 legacy 归一化 / 相机索引、逆畸变、10 次迭代；畸变对齐沿用 `base_options` |

这些是统一业务集成的设置，不是 `RenderOptions()` 每个字段的默认值。需要自行选择 bucketed、Tensor 输出或关闭 B1 压缩的已有低层调用，可继续使用执行器及显式共享准备接口。可分离模式必须先验证真实业务二维核，初始化不会替换生产回调。

`render_frame` 必须在初始化之后调用。传入非空 `material_groups` 更新 owner 的当前素材组，B 等于列表长度；省略或传 `None` 时沿用 owner 的 `imgs_raw_groups`，其未设置时保留原单组 `imgs_raw` 兼容行为。显式空列表不是无广告信号，而是错误；B=0 由业务调度跳过。

`camera_history=None` 沿用 owner 的 `paste_data_stack`；非 None 则设置该引用。入口不添加、复制或裁剪历史。`device` 可从 Tensor 背景推断；有广告的 NumPy 背景需要显式设备。`camera_result=(filtered_coefficients, motion_rate)` 可复用之前计算的 CPU 相机结果，此时不重复调用滤波器，也不由 builder 重写 `data_dict['dist_coeffs']`。

默认返回可立即读取、连续且独立的 NumPy `uint8`：B1 为 `[H,W,3]`，多组为 `[B,H,W,3]`。保留上一输出不会阻止下一帧渲染，也不会被下一帧覆盖；调用方需要自行限制输出队列长度。

## 一帧执行顺序与输入责任

1. 入口更新显式提供的素材组和历史引用，按 `ad_params_{way_id}` 的非 None 值优先、否则通用 `ad_params` 的规则选择广告位。专用空列表表示当前 way 无广告，即使通用列表非空。
2. 无广告直接进入执行器的背景返回路径，不运行相机滤波、运动核或共享几何准备。仍按当前 B 返回背景，并记录 `last_execution.reason="no_ads"`。
3. 有广告时共享准备一次，读取当前 K、畸变和运动状态，产生畸变网格和运动核。未提供 `camera_result` 时，相机滤波按原行为可能替换 `data_dict['dist_coeffs']`。
4. 执行器收集当前素材和实际尺寸，再通过 builder 回调生成 ROI 网格；不能在素材收集前按标称尺寸缓存 ROI，也不能仅凭相同 `frame_id` 重用 K/畸变/单应矩阵结果。
5. 原执行实现完成输入校验、素材缓存、策略选择、整批或 B 分块计算，最后等待输出复制完成并返回独立 NumPy。

最小业务输入边界如下：

| 输入 | 约定 |
|---|---|
| `owner` | 沿用真实 `compute_camera_motion_rate(poses)`、`motion_rate_to_sigma(motion_rate)`、`create_gaussian_kernel(sigma)`；保存当前素材/历史及库的 worker 状态 |
| `data_dict` | 当前帧可变字典；有广告时含 `K`、`dist_coeffs` 和选中 way 的广告位列表 |
| 广告位 | 含 `ad_placement_id`、`ad_id`、边界内的 `bbox=(x1,y1,x2,y2)`、ROI 局部反向矩阵 `m_inv`，以及可选 `transparency`；右下边界不包含 |
| 素材组 | 每组按 `(ad_placement_id, ad_id)` 或兼容的 placement key 定位 `{"img": image}`；RGBA 通道范围及布局沿用业务选项。每位在 B 组中的实际源尺寸须一致，不同位可以不同 |
| 素材版本 | B、组顺序及对象变化按现有 key 检测；原地修改像素必须推进 `owner.imgs_raw_version` 或显式失效，不能只改 `frame_id` |
| 相机历史 | 整数帧号映射到含 `K`、`w2c`、一维 `dist_coeffs` 的记录。参与 median 的系数必须是 CPU 数据；`w2c` 原对象交给业务回调处理 |
| `mask_canvas` | 保留原 mask 广播/布局/缩放规则；1 保留背景，0 允许广告 |

相机邻域有至少三条有效记录时要求包含当前帧，滤波会替换当前 `data_dict` 的系数字段。不要直接把历史记录字典作为可变 `data_dict`，也不要让历史长期引用生产者下一帧会原地覆写的数组。普通 Mapping 继续完整校验帧号、遵循旧滤波行为；`BoundedCameraHistory` 才使用直接邻域查询。默认半径 4，最新帧至少保留 5 条记录；四帧前瞻的完整窗口至少保留 9 条，延迟更大的消费者需自行扩大容量。稀疏历史仍可能进入少于三条的原有回退。

## 一次初始化、逐帧调用、退出释放

下面只展示业务接线，不实现解码、编码或发送。`packets` 是业务已有的帧迭代器；示例约定其中相机值已是 CPU 数组，素材与背景遵循既有选项。真实 motion/Gaussian 回调已在 owner 上，不在此重定义。

```python
from collections import deque
import numpy as np

from paste_renderer import (
    BoundedCameraHistory,
    initialize_renderer,
    release_renderer,
    render_frame,
)


def render_worker(owner, existing_options, packets, device):
    history = BoundedCameraHistory(capacity=9)
    recent_outputs = deque(maxlen=3)
    initialize_renderer(owner, existing_options)  # 每个 worker 一次
    try:
        for packet in packets:
            frame_id = packet["frame_id"]
            raw = packet["camera"]
            record = {
                "K": np.asarray(raw["K"]).copy(),
                "w2c": np.asarray(raw["w2c"]).copy(),
                "dist_coeffs": np.asarray(raw["dist_coeffs"]).copy(),
            }
            history[frame_id] = record  # 单调帧号；B=0 时也保留当前相机记录
            groups = packet["material_groups"]
            if not groups:
                continue

            data = dict(packet["data"])
            data["K"] = record["K"]
            data["dist_coeffs"] = record["dist_coeffs"].copy()
            owner.imgs_raw_version = packet["material_revision"]
            output = render_frame(
                owner, packet["frame"], data, frame_id,
                mask_canvas=packet["mask"], material_groups=groups,
                camera_history=history, device=device,
                way_id=packet.get("way_id", 1),
            )
            recent_outputs.append(output)
            # 实际下游也须有队列上限；此示例不发送图像。
    finally:
        recent_outputs.clear()
        try:
            release_renderer(owner)
        finally:
            history.clear()
```

示例的队列只保留最近三份输出，不代表具体背压或丢帧策略。业务在 seek / 换流前显式 `history.reset()`，并在调用之间释放或清理相应 builder 状态；不要为普通逐帧变化重建 worker。一个 owner 对应一个串行 worker/计算流；跨流生产的输入需要先建立依赖。

`release_renderer` 依次释放执行器和 builder 的库内缓存/引用，执行器先完成在途复制；即使执行器释放抛错，也通过 `finally` 尝试释放 builder。它保留这两个实例供之后重用，不删除调用方历史、不覆盖已返回输出，也不调用全局 `empty_cache()`。调用方持有的输入、历史及输出队列仍由调用方管理；CUDA `reserved` 不能直接解释为存活对象或泄漏。

## 旧接口如何兼容

| 原接口 | 本轮关系 |
|---|---|
| `paste_gpu_batch_adapter.AdapterOptions` | 与规范 `RenderOptions` 是同一个 class，字段及位置参数顺序保留 |
| `paste_gpu_batch_adapter.PasteBatchAdapter` | 与规范 `RenderExecutor` 是同一个 class，不是子类或包装实例 |
| 旧 `paste_gpu_batch_adapter(...)` 函数 | 保留签名、owner 上的 lazy 初始化及调用语义 |
| 原 `adapter(...)` / `__call__` | 标准类按原签名转调 `render_prepared`；统一入口仍调用 `executor(...)`，保留子类覆盖 `__call__` 的分派 |
| `setup_shared_paste(...)` | 委托新初始化，仍返回 **builder**；新 `initialize_renderer` 返回 **executor** |
| `prepare_shared_paste_args(...)` | 仍返回原顺序的六项元组，可继续配合旧函数或执行器 |
| `initialize_dynamic_worker(...)` | 保留原签名、强制 auto 的原设置及 builder 返回值 |
| `render_dynamic_frame(...)` | 保留必填素材组/历史/设备参数，验证后委托新入口；显式 `camera_history=None` 仍写入 owner，和新入口的 None 沿用语义不同 |

`owner._paste_batch_adapter` 与 `owner._shared_paste_builder` 继续作为唯一兼容存储位。新旧入口读取同一执行器/builder，不存在并行缓存；业务使用公共函数即可，不必直接改这两个私有字段。

平铺导入和包内导入均保留。旧模块的 helper 属性读取仍兼容，但测试若需要替换私有全局，应针对实际实现模块 `paste_render_executor`；给旧 shim 的私有属性重新赋值不会转发。私有全局 monkeypatch 不属于公共业务接口。

旧示例模块中原本可导入的名字也继续导出：`shared_paste_example.PasteBatchAdapter` / `SharedParameterBuilder`，以及 `dynamic_paste_example.prepare_shared_paste_args` 保持对应对象身份。

## 验证边界与记录

本轮以基线 `4b2b311` 的隔离旧源码作为画面对照，新 `render_frame` 对比旧共享准备加 adapter 的完整调用，要求 NumPy 形状、dtype、连续性、逐字节画面及 `last_execution` 一致。测试须覆盖动态 B、same-frame 原地更新、素材 revision/顺序、way 优先与无广告、旧输出存活及释放重用；不得以别名测试或 AST 相同代替 GPU 输出验证。

接口 CPU 测试入口为项目解释器的 `-B -m unittest test_render_api`，默认执行 16 项 CPU 并跳过 1 项 GPU。最终显式设置 `PASTE_TEST_DEVICE=cuda:0` 后，新 API 的 17 项与原 42 项 CPU 共 **59 项 unittest 全部通过**。GPU 项实际调用 `initialize_renderer(use_separable=True)`，执行 B3→1→2→3，与旧 wrapper 逐字节一致，并验证输出独立及释放重用。子类 `__call__` 分派有独立回归；素材 revision 测试在同一个 frame ID 下逐步修改像素，每步同时对照上一图与新执行器重新构建的结果，负向变更验证漏掉 revision 会使测试失败。

GPU dynamic 265 项、GPU backend 115 次比较加 8 项布局、CPU backend 77 次比较加 8 项布局均通过；backend 的 batch/bucketed 比较为逐字节一致。shared 两/三尺寸 B8、三尺寸 B1 repeated 及 batch B8 三尺寸 dense 均通过原参考对照，最大 uint8 差 1 / grid 差约 `3.576e-7` 属于既有参考容差，不是当前接口对旧源码出现非零差异。本轮 B1 repeated 一次通过，不代表历史编译事件的根因已确定或修复。

公开入口对旧源码的完整对照使用以下命令。先按 README 恢复 `4b2b311`；输出文件须选未占用的新名称。

```powershell
.\scripts\run.ps1 compare --device cuda:0 --entry-point render_frame --baseline results/baseline_render_api_4b2b311 --baseline-label before_render_api --batches 1 4 8 16 --slots 16 --chunks auto --rounds 4 --warmup 20 --iterations 100 --output results/render_api_compare_final.json
.\scripts\run.ps1 check-streaming --device cuda:0 --entry-point render_frame --baseline results/baseline_render_api_4b2b311 --baseline-label 4b2b311 --frames 10000 --warmup 200 --sample-every 100 --check-every 250 --revision-every 1 --retain-outputs 3 --output results/render_api_streaming_final_10000.json
```

两项工具的 `--entry-point` 默认仍为 `legacy`；显式 `render_frame` 只切换当前路径，保留当前 builder/执行器配置，隔离 baseline/oracle 仍走旧共享准备与 adapter。报告记录 `arguments.entry_point`。因此这类持续/性能对照验证公开渲染入口；真实 `initialize_renderer` 设置另由接口 GPU 测试覆盖，不混为同一项证据。

### 完整调用对照

最终 `render_api_compare_final.json` 使用 RTX 4090、1080p、16 个广告位、200×200 / 500×500 / 300×600 源尺寸、9×9 已验证可分离核。在同一进程中保留独立的新旧可变状态，每个 B 交替执行 4 轮，每轮每候选预热 20 次、计时 100 次；每候选共 400 个样本。计时包含共享准备至完成 GPU 工作后的独立 NumPy 返回，输入 fixture 更新在计时外。

下表单位为毫秒；均值变化按 `(新入口 / 基线 - 1) × 100%` 计算，负值表示本次测量较低。

| B | 旧入口 mean | 新入口 mean | 均值变化 | 旧入口 P95 | 新入口 P95 |
|---|---:|---:|---:|---:|---:|
| 1 | 5.593 | 5.748 | +2.77% | 8.098 | 8.056 |
| 4 | 9.617 | 9.500 | -1.22% | 11.409 | 11.717 |
| 8 | 11.668 | 11.331 | -2.89% | 13.491 | 13.620 |
| 16 | 16.084 | 16.344 | +1.62% | 18.922 | 19.311 |

本轮均值约为 -2.9%～+2.8% 的混合变动，不是提速结论，也不支持修改策略。每个 B 的第 0、7 帧分别作新旧逐字节对照，共 8 个检查帧均通过；不是逐计时帧图像对照。每轮记录的执行决策均相同：B1/B4 整批，B8/B16 为 chunk 4。

### 公开入口的 10000 帧持续调用

最终 `render_api_streaming_final_10000.json` 状态为 `complete`，当前路径使用 `render_frame`，隔离 oracle 使用 `4b2b311` 的旧调用路线。同一组当前 builder/执行器连续经历固定 5000 帧与动态 5000 帧，每阶段先预热 200 次。设置为 RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128、1080p、16 广告位、三种源尺寸和 9×9 核。

固定阶段使用 B8 与静态素材/布局；动态阶段在 B1/4/8/16、素材顺序、bbox 平移/缩放之间切换，并在每帧实际更新 RGB 和 revision，累计 5000 次 revision。每 250 帧及末帧对照旧源码，每阶段 21 次，共 **42 次图像逐字节一致且 auto 决策一致**。这不是对每个测量帧都运行旧源码对照。

每帧检查输出协议和存储独立性，在采样/对照及阶段结束时检查首帧快照，并限制为 anchor 加最多 3 个近期输出；每阶段通过 weakref 确认 4996 个已移出队列的输出被回收。该 fixture 使用固定邻域的普通历史 Mapping，不能称为 `BoundedCameraHistory` 的 10000 次业务接入验证；后者的单调性、窗口与长期容量由既有 8 项有界历史 CPU 测试覆盖，所在 15 项 suite 另含 7 项核生命周期测试。

以下数字来自预热后第一个至最后一个采样窗口，单位 MiB；动态 allocated 与 stack 随当前 B/素材签名变化，不能直接用端点相减判断泄漏。

| 指标 | 固定 5000 帧 | 动态 5000 帧 |
|---|---:|---:|
| RSS 增量 | +1.078 | +2.309 |
| Private bytes 增量 | +1.230 | +2.152 |
| CUDA allocated 范围 | 964.305～964.305 | 709.396～1256.881 |
| 库 retained stack 范围 | 291.748～291.748 | 36.469～583.496 |
| Gaussian 缓存条目 | 16 | 16 |
| device alloc/free 次数增量 | 0 / 0 | 0 / 0 |
| host alloc/free 次数增量 | 0 / 0 | 0 / 0 |
| allocation retries / OOM 最大计数 | 0 / 0 | 0 / 0 |

`retained_stack_bytes` 统计历史/当前/桶 stack 的去重存储；当前工作集可以超过历史缓存的 512 MiB 预算。RSS 仍有上述小幅增长，不能写成完全零增长。两阶段 `pinned_live_counters_valid=false`，固定页 allocated/reserved 的原始计数不自洽，因此不据这些字节值归因 live 内存；分配次数与原始 reserved 仅作诊断。计数是含周期性 oracle 工作的进程观测；库未在帧循环加入 `gc.collect()` 或 `empty_cache()`。

这些结果说明本机、该合成变化序列和有界消费者下的观察，不证明生产长时间永不泄漏，也不包含解码、编码或网络队列。初次 `render_api_streaming_10000.json` 曾为修复 `__call__` 兼容分派而主动中断，不计入上述完成证据。

静态检查已通过 30 个 Python 文件的 Ruff 与 Python 3.9 AST，以及 `git diff --check`。长测记录的 20 个源码文件 SHA 均与最终工作区一致；`paste_gpu_batch.py`、三份 kernel 模块、共享参数、执行策略、素材缓存、相机滤波和共享类型共 9 个核心文件 SHA 与 `4b2b311` 完全相同。旧版性能、10000 帧长期运行及未定根因编译事件保留在 [STREAMING_REVIEW.md](STREAMING_REVIEW.md)，不重写为本轮结果。

本机证据文件（均为忽略的本地 artifact）：`results/render_api_unittest_final.log`、`results/render_api_compare_final.json`、`results/render_api_streaming_final_10000.json`、`results/render_api_streaming_final_summary.json`。长测 JSON 同时记录当前与基线源码 hash、入口选择和原始窗口；源码变更后不能继续把旧运行结果当作新实现证据。

`results/` 不随 Git 分发；基线源码可按 [README 的恢复步骤](../../README.md) 从 `4b2b311` 恢复到新的目录。恢复源码不会恢复测试机器上的原始 JSON/日志。合成回调与测试画面不能替代真实业务回调、视频及下游编码/发送链路的生产验收。

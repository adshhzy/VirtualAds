# 启动与在线业务接入

适用于当前 `2026-10-23-perf4.9-prepare`。线上调用入口是 Python 方法 `ImageRenderer.render_img(data_dict, frame, imgs_raw)`：业务进程创建一个 renderer，收到一帧便调用一次。仓库提供渲染库、可运行演示和测试命令；没有 HTTP 监听端口、Web 服务、视频解码器或推流服务器。

首次接入按“安装 → 三参数演示 → 业务 worker”顺序进行。字段细节见 [RENDER_IMG.md](RENDER_IMG.md)，内部调用顺序见 [ARCHITECTURE.md](ARCHITECTURE.md)，本轮性能和内存证据见 [PREPARATION_OPTIMIZATION_REVIEW.md](reports/PREPARATION_OPTIMIZATION_REVIEW.md)。

目录结构与旧环境迁移见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。运行库集中在 `virtualads/`；安装后使用 `from virtualads import ImageRenderer, RenderOptions`。演示和检查工具从仓库根目录通过 PowerShell 入口或 Python `-m` 启动。

## 1. 安装并检查运行环境

当前安装脚本针对 Windows、已安装的 Python 3.12、uv 和 NVIDIA CUDA GPU。锁文件包含 `triton-windows`，Linux 部署需要另行适配依赖并验证，不直接复用此 Windows 锁文件。安装脚本不会安装 Python、uv 或显卡驱动。

首次克隆，在项目父目录的 PowerShell 中运行：

```powershell
git clone https://github.com/adshhzy/VirtualAds.git
Set-Location VirtualAds
$pythonLauncher = (Get-Command py.exe -CommandType Application -ErrorAction Stop).Source
$python312Path = & $pythonLauncher -3.12 -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw '需要已安装的 Python 3.12。' }
$uvExecutable = (Get-Command uv.exe -CommandType Application -ErrorAction Stop).Source
.\scripts\setup.ps1 -PythonPath $python312Path -UvPath $uvExecutable
```

没有 `py.exe` 时，将现有 Python 3.12 的绝对路径传给 `-PythonPath`。setup 在锁定依赖之后自动以可编辑模式安装本项目。已有工作区无需重复安装 GPU 依赖，但从旧平铺目录升级后，需先在仓库目录确认实际解释器，并执行一次项目安装：

```powershell
$workspaceRoot = Split-Path -Parent (Get-Location).Path
$pythonCandidate = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else {
    Join-Path $workspaceRoot 'env\paste_gpu_performance\Scripts\python.exe'
}
$projectPython = (Resolve-Path -LiteralPath $pythonCandidate -ErrorAction Stop).Path
$uvExecutable = (Get-Command uv.exe -CommandType Application -ErrorAction Stop).Source
& $uvExecutable pip install --python $projectPython --no-deps --no-build-isolation --editable .
if ($LASTEXITCODE -ne 0) { throw 'VirtualAds installation failed.' }
& $projectPython -c 'import torch, triton; print(torch.__version__, triton.__version__); print("CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")'
.\scripts\run.ps1 shared --version
.\scripts\run.ps1 render-img-example --help
```

默认环境位于**仓库父目录**的 `env\paste_gpu_performance`，无需手动激活。使用 `setup.ps1 -EnvironmentPath ...` 安装到其他位置后，将 `$env:PASTE_GPU_PYTHON` 设为该环境的 `Scripts\python.exe`；不要让业务进程误用系统 Python。

setup、`--version` 和 `--help` 通过只说明安装/导入可用；setup 即使打印 `CUDA: False` 也可能完成安装。下一节的实际 CUDA 演示成功才确认 GPU 渲染可以启动。

`run.ps1` 自动切换到仓库目录，并设置缺省 Triton/Torch 编译缓存。业务程序直接使用 `$projectPython` 启动时，可在进程启动前设置相同的 `TRITON_CACHE_DIR` 和 `TORCHINDUCTOR_CACHE_DIR`；这些目录应可写，且不逐帧清理。首次实际渲染会触发 JIT，首次耗时不代表稳态性能。

## 2. 各模块如何启动

下面的命令从仓库根目录运行；结果文件必须使用尚不存在的路径。

| 用途 | 命令/入口 | 运行行为 |
|---|---|---|
| 新三参数接口演示 | `.\scripts\run.ps1 render-img-example --device cuda:0 --frames-per-phase 2 --output results/render_img_demo.json` | 合成小画布，动态路数、换素材、B0 和退出清理，有限帧后退出 |
| 新接口完整正确性检查 | `.\scripts\run.ps1 check-render-img --device cuda:0 --frames 100 --output results/render_img_check.json` | 逐路对照及 mask 独立参考检查，完成后退出 |
| 旧 `render_frame` 直播示例 | `.\scripts\run.ps1 live-example --device cuda:0 --frames-per-phase 3 --output results/live_render_demo.json` | 旧 packet/material_groups 接线，见 [旧直播示例](LIVE_RENDER_EXAMPLE.md) |
| 性能测试与诊断 | `prepare`、`profile-uploads`、`check-prepare` 子命令 | 需要本机真实输入及部分命令的旧源码目录，参数见 [优化报告](reports/PREPARATION_OPTIMIZATION_REVIEW.md) |
| 线上业务 worker | 在业务程序中 import `ImageRenderer`，初始化后循环调用 | 生命周期由业务进程管理，见下一节 |
| `virtualads/paste_route_plan.py`、`virtualads/paste_shared_params.py`、`virtualads/paste_shared_mask.py`、执行器与 kernel 模块 | 由 renderer 自动调用 | 无需逐个启动，也不单独运行这些文件 |

新演示代码在 [render_img_example.py](../examples/render_img_example.py)，使用 [live_render_demo.py](../examples/live_render_demo.py) 中明确标记的 `SyntheticOwner`。其回调、图片、相机和本地输出检查只用于跑通环境，不能代替生产相机算法和编码推流。演示不会读取用户的 3336 附件，也不需要额外下载图片。

若直接启动 Python，等效命令为 `& $projectPython -m examples.render_img_example --device cuda:0 --frames-per-phase 2`，仍从仓库根目录运行。该模块默认只打印摘要；`--output` 指定时另存 JSON。不要按文件路径直接执行子目录中的 Python 文件；运行库中的 `virtualads/paste_renderer.py` 本身也不是服务启动入口。

## 3. 线上接口：初始化一次，逐帧调用

公共调用是实例方法，没有隐式的模块全局 worker：

```python
from virtualads import ImageRenderer, RenderOptions

# owner：当前 worker 专用的业务对象，提供下表三个真实回调。
options = RenderOptions(
    frame_layout="HWC",
    ad_layout="HWC",
    batch_chunk_size="auto",
)
renderer = ImageRenderer(
    owner,
    options,
    device="cuda:0",
    use_separable=False,
    mask_sigma=1.5,
    mask_radius=3,
    history_capacity=9,
)
render_img = renderer.render_img

try:
    output = render_img(data_dict, frame, imgs_raw)
    route_ids = renderer.route_ids
    if route_ids:
        batch = output[None, ...] if output.ndim == 3 else output
        for index, route_id in enumerate(route_ids):
            image = batch[index]
            # 交给该 route_id 对应的编码器；image 是 HWC uint8。
finally:
    renderer.close()
```

这里的 `owner`、`data_dict`、`frame`、`imgs_raw` 来自业务；可直接运行的完整输入在新演示中。业务应把上述初始化和 finally 放在整个直播 worker 外层，不能每处理一帧就创建/关闭 renderer。

### owner 必须提供的回调

| 方法 | 输入 | 返回与责任 |
|---|---|---|
| `compute_camera_motion_rate(self, poses)` | 当前历史窗口内、按帧号排序的 CPU `w2c` 序列 | 沿用现有业务运动度定义；交给下一方法。历史不足三条时共享准备使用 0.0，不调用该方法 |
| `motion_rate_to_sigma(self, motion_rate)` | 上一步的运动度或起始阶段的 0.0 | 有限实数 CPU 标量 sigma，沿用业务映射 |
| `create_gaussian_kernel(self, sigma)` | 业务返回的 sigma | CPU float32/float64 二维核，非空、两个边长均为奇数、权重有限；保持原有核及归一化约定 |

renderer 自动管理 `owner.paste_data_stack`、共享 builder 和执行器，业务不要在并行线程覆盖这些字段。`owner.gaussian_kernel_cache_version` 默认 0；同一 sigma 下若核生成规则变化，推进这个可哈希版本值，让缓存失效。

初始化沿用当前 batch/Triton、ROI 局部矩阵、dense 模糊、独立固定页 NumPy 和原 auto 表。共享初始化会覆盖底层 `RenderOptions` 的 backend、acceleration、输出及 profiling 设置；仅传 `acceleration="eager"`、`return_numpy=False` 或 `squeeze_single=False` 不能改变此业务入口。B1 按现有接口返回 HWC；需要批次维时使用上面的零复制 `output[None, ...]` 视图。切换 `use_separable` 要先验证真实核和完整输出逐像素一致。

### 三个逐帧参数

| 参数 | 线上约定 |
|---|---|
| `data_dict` | 当前帧的相机与各路配置。活动广告帧包含 `frame_id`、`K`、`w2c`、`dist_coeffs`，以及各路 `ad_params_N`；可带 `mask_coor_list`、`imgs_raw_version` |
| `frame` | 一张 CPU NumPy 原始帧，常用连续 uint8 `(1080,1920,3)`；该帧被所有路共用。当前入口也支持已定义的 Tensor/布局转换，但业务应固定自己的格式 |
| `imgs_raw` | 长期复用的素材字典，精确键为 `(ad_placement_id, ad_id)`，值为四通道 HWC 图片，或 `{"img": image}` |

相机数据放在 CPU：`K` 为 3×3，`w2c` 按原业务位姿定义通常为 4×4；`dist_coeffs` 为一维向量，样例是 5 项。每路 `ad_params_N` 为广告配置列表，路号按整数升序输出；`ad_params_2` 与 `ad_params_10` 对应 `route_ids == (2, 10)`，不是批次下标 2、10。

单条广告示例结构如下，数值由当前业务提供：

```python
data_dict["ad_params_2"] = [
    {
        "ad_placement_id": "pitch-side",
        "ad_id": 101,
        "bbox": [100, 200, 420, 308],  # x1,y1,x2,y2，右下不包含
        "m_inv": current_inverse_matrix,  # 本广告的 3×3 ROI 局部反向矩阵
        "transparency": 1.0,
    }
]
imgs_raw[("pitch-side", 101)] = current_ad_rgba
```

广告 bbox 必须在画面内；矩阵的素材坐标尺寸必须与实际输入一致。RGB/BGR 不自动互换，背景和素材前三通道使用相同顺序；图片和 alpha 使用 0..255。JPEG 只有三个通道，装载时补 alpha=255，一次性转换为四通道素材。`transparency` 默认 0..1；其他值域须显式匹配 `transparency_scale`。

默认不根据 `ad_h_w` 缩放。3336 实测将 500×500 素材按配置缩至 200×200，使用了 `RenderOptions(resize_ads_to_ad_h_w=True, ...)`；复现该配置时须保持相同预处理，不能直接以 500×500 解释按 200×200 构造的矩阵。业务已经提供正确尺寸时无需开启 resize。

`mask_coor_list` 每项为 `{"bbox": [x1,y1,x2,y2], "mask": mask_hw}`，局部 mask 尺寸须匹配 bbox，可与画面裁剪求交。mask 为 0..1：1 保留背景，0 允许广告；不是 0..255。共享准备先取 maximum 合并，再整图羽化。`brightness`、`ad_paste_data`、`way_ad_map` 和 `image_path` 不参与当前渲染。

### 返回值

| 当前路数 | 返回形状 | 行为 |
|---|---|---|
| B=0 | `(0,H,W,3)` | 暂停，无图像需要发送 |
| B=1 | `(H,W,3)` | 单路 NumPy uint8 |
| B>1 | `(B,H,W,3)` | 按 `renderer.route_ids` 排序的 NumPy uint8 |

返回后可以立即读取，不需业务额外 `torch.cuda.synchronize()`。下一帧或 close 不会覆盖旧结果。用 `renderer.route_ids` 判断是否有输出，比判断 HWC 的第一维更清晰。每次调用后立刻取出这个 tuple，不在异步消费者中延迟读取 renderer 的下一帧状态。

## 4. 持续处理的业务接线

以下函数使用业务已有的 source/sink，并接管它们的关闭：source 为实现 `close()` 的可迭代对象，每项包含 `data_dict`、`frame`、`pts`，换流时可带 `reset_camera=True`；sink 实现 `set_routes(tuple[int,...])`、`write_batch(route_ids, batch, pts=...)` 和 `close()`。`write_batch` 约定接受统一 BHWC，在返回前完成消费或将独立输出转交有界队列。

```python
from virtualads import ImageRenderer, RenderOptions


def run_online(owner, source, sink, imgs_raw, device="cuda:0"):
    renderer = None
    output = batch = None
    try:
        renderer = ImageRenderer(
            owner,
            RenderOptions(frame_layout="HWC", ad_layout="HWC", batch_chunk_size="auto"),
            device=device,
        )
        render_img = renderer.render_img
        active_routes = None
        for packet in source:
            if packet.get("reset_camera", False):
                renderer.reset_history()
            # source 在本次调用前完成配置与素材更新，不并发改写正在使用的输入。
            output = render_img(packet["data_dict"], packet["frame"], imgs_raw)
            routes = renderer.route_ids
            try:
                if routes != active_routes:
                    sink.set_routes(routes)  # 包括空 tuple：暂停各路输出
                    active_routes = routes
                if routes:
                    batch = output[None, ...] if output.ndim == 3 else output
                    sink.write_batch(routes, batch, pts=packet["pts"])
            finally:
                output = batch = None
    finally:
        # source.close 停止生产；sink.close 排空/取消队列并结束编码。
        try:
            source.close()
        finally:
            try:
                sink.close()
            finally:
                if renderer is not None:
                    renderer.close()
```

这是业务接线模板，不包含帧源、编码器或网络协议实现。将它放入业务主程序，用同一个项目解释器启动；进程守护、重连、时间戳、发送失败策略由现有直播服务管理。当前渲染模块也不会按 `pts` 调度。

一个相机流使用一个串行 worker/owner/renderer，内部可生成动态 B 路广告输出。不要让多个请求线程或多个相机并发调用同一实例。多 GPU worker 分别创建自己的对象和设备上下文；同一实例在同一 CUDA 计算流中串行调用。源 Tensor 在其他流产生时，由调用方先建立依赖。

## 5. 什么只初始化一次，什么逐帧更新

| 生命周期 | 对象/字段 | 做法 |
|---|---|---|
| worker 初始化一次 | owner、RenderOptions、ImageRenderer、source/sink、设备上下文 | 循环外创建，不按路数变化重建 renderer |
| 素材加载/切换时 | `imgs_raw` 中的图片及 alpha/颜色转换 | 尽量循环外解码；更换图片对象会被自动识别 |
| 每帧 | `frame`、`frame_id`、当前 `K/w2c/dist_coeffs`、各 `ad_params_N`、bbox、m_inv、透明度、mask | 提交当前值，原地几何/mask 更新可直接生效；相机历史由 renderer 维护 |
| 同一素材对象改像素时 | `imgs_raw_version` | 修改完成后递增，并在后续帧中保持该版本；或调用一次 `invalidate_assets()`。几何/路数变化不需要推进素材版本 |
| 核生成规则改变时 | `owner.gaussian_kernel_cache_version` | 同 sigma 输出规则改变才更新，不按帧号递增 |
| 换流/seek | `reset_history()` | 在重启 frame_id 前调用；PTS 按下游时间线单独处理 |
| 临时释放 | `release_resources()` | 释放活动空间并保留可复用实例和历史；不每帧调用 |
| worker 退出 | `close()` | 释放库的缓存/空间/历史，幂等；close 后重新创建实例才能渲染 |

减少路数时，从**当前帧字典**移除停用的 `ad_params_N`，不要残留上一帧的键。全部移除就是 B0：首次进入暂停自动释放活动空间，连续暂停不会反复释放，恢复时添加各路配置并继续使用同一实例。`ad_params_2=[]` 表示第 2 路仍输出背景，不表示停止该路。

素材原地变化的典型更新：

```python
imgs_raw[("pitch-side", 101)][...] = next_ad_rgba
asset_revision += 1
data_dict["imgs_raw_version"] = asset_revision
# 后续每帧仍填写 asset_revision，不在下一帧省略后意外退回默认 0。
```

缓存预算不是整个进程的内存上限。异步队列必须有容量和背压/丢帧策略；即使只保留 `batch[index]`，这个 NumPy view 也可能持有整个批次。需要长期只保存单路时可有意 `.copy()`，并接受额外复制成本。业务负责释放自己的素材 bank、外部帧、队列及异常 traceback 引用；不要逐帧 `gc.collect()` 或 `empty_cache()`。

## 6. 常见接入问题

| 现象 | 核对点 |
|---|---|
| 找不到 Python 或导入失败 | 核对 `PASTE_GPU_PYTHON` 和项目安装环境；按本页执行可编辑安装，业务导入 `virtualads`，演示从根目录使用 `-m examples.render_img_example`。wheel 部署见 [目录指南](PROJECT_LAYOUT.md) |
| CUDA 不可用 | 先检查项目解释器中的 `torch.cuda.is_available()`；公开 GPU 入口不通过传 `--device cpu` 替代运行 |
| 提示 owner 回调缺失 | 使用本 worker 的真实业务 owner，补齐三个方法，不直接用裸 object |
| 找不到素材 | 键类型和值必须精确匹配 `(ad_placement_id, ad_id)`，不能用图片路径或仅 placement ID |
| 贴图位置/大小错误 | bbox 的 xy 顺序、右下不含、m_inv 的 ROI 局部坐标、素材实际尺寸及显式 resize |
| 素材改了但仍显示旧图 | 原地改图后推进素材版本或显式失效；替换 ndarray 对象则可自动识别 |
| 动态路数与编码器不对应 | 每帧使用本次 `route_ids`，按数值路号排序，不能硬编码 8 路或沿用旧顺序 |
| 第一帧慢、进程 reserved 不立即下降 | 区分首次 JIT、容量预热、库持有量和分配器 reserved；诊断方法见优化报告 |
| frame_id 回退报错 | 换流/seek 前先 `reset_history()`；同一历史段不能任意回退 |

验收先跑三参数演示，再用真实帧、配置和生产回调检查画面与完整延迟。演示和既有 3336 测量都不包含编码/网络发送，不能据其倒数推算线上推流 FPS。

## 7. 独立示例初次验收记录（2026-10-23，目录整理前）

以下为提交 `b65c3eb` 补充说明与独立演示时的记录，没有修改渲染器、GPU 内核或 auto 表；不将历史测试计数当作目录整理的新增验收。在现有 Windows / RTX 4090 项目环境中完成：

- 新三参数演示 8 个阶段、16 次调用通过，覆盖非连续路号排序、素材替换与 revision、连续 B0、恢复、背景输出和历史重置。退出后检查的库工作区归零，旧 NumPy 输出保持不变。
- 本文 `run_online` 模板从文档代码块直接加载执行，动态/暂停/重置流程以及 sink 抛异常后的关闭顺序通过。
- `check-render-img --frames 100` 完成 127 次调用、424 路图像，与逐路兼容入口最大 uint8 差为 0；mask 对独立参考最大误差约 `2.38e-7`。
- 旧 `live-example` 完成 24 帧，其中 21 次渲染、3 次暂停、51 张输出；新 CLI 的帮助、拒绝 CPU、拒绝覆盖报告及 shared/batch 版本入口通过。
- Ruff format/check、59 个 Python 文件的 Python 3.9 语法检查、文档 Python 代码块编译和本地链接检查通过。

原始 JSON 位于本机忽略目录 `results/startup_api_20261023/`，不随仓库分发；新克隆可按本文命令生成自己的记录。本轮没有重装已有环境，也没有重新进行吞吐或生产推流验收。

按实际执行顺序说明及独立 source/sink worker 示例见 [按帧流程与业务 worker](FRAME_PIPELINE_AND_WORKER.md)。

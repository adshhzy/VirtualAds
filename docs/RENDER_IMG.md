# 三参数业务入口 render_img

`ImageRenderer.render_img(data_dict, frame, imgs_raw)` 直接接收每帧业务字典、原始画面和素材字典。初始化一次，逐帧调用三参数方法；原 `render_frame`、`initialize_renderer` 和 adapter 接口继续兼容。

安装、模块启动命令、owner 回调签名及完整业务循环模板见 [启动与在线业务接入](STARTUP_AND_API.md)。可直接运行 `.\scripts\run.ps1 render-img-example --device cuda:0` 检查三参数接口，源码为 [render_img_example.py](../examples/render_img_example.py)。以下片段使用业务已有对象。

```python
from virtualads import ImageRenderer, RenderOptions

# owner 提供业务原有的运动估计、sigma 和 Gaussian 回调。
# 这些回调的生产定义不在本仓库中，不能用演示回调替换。
options = RenderOptions(frame_layout="HWC", ad_layout="HWC", batch_chunk_size="auto")
with ImageRenderer(owner, options, device="cuda:0") as renderer:
    render_img = renderer.render_img  # 可直接绑定成业务要求的三参数函数
    for data_dict, frame, imgs_raw in frame_source:
        output = render_img(data_dict, frame, imgs_raw)
        batch = None
        try:
            route_ids = renderer.route_ids
            if not route_ids:  # B=0 暂停，无输出需要发送
                continue
            # B1: HWC；B>1: BHWC。这里给业务 sink 统一成 BHWC 视图。
            batch = output[None, ...] if output.ndim == 3 else output
            sink.write_batch(route_ids, batch)
        finally:
            del batch, output  # 暂停或 sink 异常时也不保留上一批视图
```

`frame_source`、`sink` 和 `owner` 由业务提供；上述是接线示例，不启动解码、编码或网络发送。一个 renderer/owner 对应一个串行 worker 和一个 GPU 计算流。`owner` 的三个既有方法必须为 callable：`compute_camera_motion_rate`、`motion_rate_to_sigma`、`create_gaussian_kernel`。默认 dense 模糊；只有实际业务核已验证可分离时才设置 `use_separable=True`。

核的可分离检查不保证最终 uint8 与 dense 逐像素一致。若业务要求保留现有画面，切换前还须使用真实输入做完整 NumPy 对照；`3336` 真实样例的模式性能与像素验证见 [模式复测报告](reports/RENDER_IMG_MODE_REVIEW.md)。auto 只选择分块配置，不代替这项验收，也不自动更改 `use_separable`。

## data_dict 到渲染输入

| 数据 | 读取规则 |
|---|---|
| `ad_params_N` | N 为正整数路号，例如 1、2、8；按 N 数值升序确定输出顺序，允许路号不连续。每路值为 list/tuple，每条广告为 mapping |
| `ad_placement_id`、`ad_id` | 精确读取 `imgs_raw.get((ad_placement_id, ad_id))`，缺失素材报错，不按路径加载、不回退 placement key |
| `bbox` | 画面内整数 `[x1, y1, x2, y2]`，右下不包含，保持该路广告列表顺序 |
| `m_inv` | 每广告一个实时 3×3 逆矩阵；使用既有 ROI 局部坐标及 `z + 1e-8` 约定 |
| `transparency` | 沿用既有透明度和 scale 规则，省略为 1；空间 map 保留原语义 |
| `ad_h_w` | 只有显式开启检查/resize 时使用，否则以真实素材尺寸为准 |
| `K`、`w2c`、`dist_coeffs` | 共用相机数据；CPU 数组，进入有界历史时复制，原字典不被滤波回写 |
| `frame_id` | 相机历史帧号；同一历史段递增或重复当前帧，seek/换流前调用 `reset_history()` |
| `mask_coor_list` | 本帧共用局部 mask 列表，缺省为空，见下节 |
| `imgs_raw_version` | 可选素材内容版本，默认 0；原地修改像素时推进，不能用帧号代替 |

`ad_paste_data`、`way_ad_map`、`brightness`、`image_path` 不参与新入口。`channel_logo_mask` 本轮未定义接入规则，不自动参与 mask。通用 `ad_params` 不用于推断路数；拼错或非规范的 `ad_params_` 路号会报错，避免静默漏路。

`imgs_raw` 是一份素材 bank，不是按路分组的字典。值支持单张 ndarray/Tensor，也兼容 `{"img": image}`；素材保持既有四通道格式与值域。RGB/BGR 不自动交换，应让背景和素材颜色通道顺序一致。旧设备图列表只通过固定 `options.image_index`（未指定为 0）选择，不把路号误当设备索引。同一素材 ID 可被不同路重复使用。

通常 `frame` 为 `(1080,1920,3)`，也支持配置中的其它分辨率/布局；每次只能提供一张原始画面。返回值为独立、可立即读取的 NumPy uint8：B1 是 HWC，B>1 是 BHWC。后续渲染或 close 不覆盖调用方仍持有的结果。

当前共享初始化固定 `squeeze_single=True` 和 NumPy 输出；给 `ImageRenderer` 传 `RenderOptions(squeeze_single=False)` 不会改变 B1 形状。下游需要统一批次维时按上面的 `ndim` 转换，不修改公开输出接口。owner/renderer 不跨线程或相机并发共用。

## 共用 mask 与羽化

每项形如 `{"bbox": [x1,y1,x2,y2], "mask": mask_hw}`。mask 的高宽必须等于 bbox 的高宽，不隐式 resize；超出画面的部分按交集裁剪。输入是 bool 或数值 0..1，不按 uint8 类型猜测 0..255。

一次帧调用按以下顺序构造共享数据：

1. 清零整帧单通道 canvas，按 bbox 放置各局部 mask；重叠处取 maximum，避免重复区域相加变亮。
2. 对合成后的完整 mask 做一次可分离 Gaussian 羽化。默认 `mask_sigma=1.5`、`mask_radius=3`，即 7×7；不是逐 patch 模糊再合并。每 pass 按实际累计权重归一，确保全 1 区域精确保留为 1，避免 uint8 截断造成背景差 1。
3. 图像边界采用 replicate，避免全 1 mask 在画面边缘变淡。`mask_radius=0` 关闭羽化。
4. 相机滤波、畸变图和原有运动模糊核准备一次，连同这张 mask 交给所有路由布局组。

mask=1 保留原画面，mask=0 允许广告显示，介于两者之间形成过渡；它按原融合路径应用于输出画面坐标。Gaussian 是对称羽化，会影响边缘内外几像素；并非“内部始终严格为 1”的单侧扩边。原有广告运动模糊核不被这个独立的 mask 核替换。新增羽化会按要求改变遮挡边缘的过渡；关闭羽化时可对照原硬 mask 效果。

`SharedMaskBuilder` 为当前设备/分辨率复用 canvas、中间结果、输出及 patch scratch，只保留一套空间工作区。输出是内部借用 buffer，使用者不要跨帧保存该 mask；公开返回的最终 NumPy 图像仍有独立所有权。CPU mask 检查有限性及 0..1；GPU mask 按该值域契约读取，避免逐帧回读同步。跨计算流输入需先建立依赖。

## 每路配置与 batch

原 batch 内核共用一套有序 bbox，不能把任意独立路配置直接当成 B。

新入口仅把广告数量、按顺序的 bbox、对应实际素材尺寸等执行要求兼容的路合成 batch。每路素材 ID 可不同；每广告的矩阵和标量透明度合成逐 B 数据，而不是复用第一路的值。空间透明度 map 暂时独立执行，保留其原有广播含义。

不兼容布局分组执行，保留各自 slot 数量、bbox 和覆盖顺序；不补透明占位广告，避免改变重叠区域的覆盖结果。所有组复用同一份相机结果、畸变图、运动核、mask 和上传后的背景；完成后按原数值路号顺序组装 NumPy 输出。单组直接返回原执行器输出，多组会多一次 CPU 结果汇集。

底层仍走 batch 与原有 auto 表，不新增 bucketed 自动准入。auto 针对每个兼容组的 B 判断，不能把独立布局拆组后的性能等同于所有路共享布局的原基准。

perf4.8 起，连续 CPU uint8 背景按字节上传后在 GPU 转 float32，各组 CPU 矩阵和 float32 标量/向量透明度分别集中上传。内部空间有界复用，当前像素/矩阵/透明度每帧重新写入；不要求调用方为几何更新增加素材 revision。其他类型、非连续图像和空间透明度仍按原逻辑处理，NumPy 输出所有权和 auto 表保持不变。完整调用、分阶段和长期检查见 [上传优化报告](reports/UPLOAD_OPTIMIZATION_REVIEW.md)。

perf4.9 进一步复用本帧的素材元数据，并用有界 CPU 工作区承接常见矩阵/透明度数组；局部 CPU mask 集中上传。调用方式和素材 revision 规则不变，bbox、矩阵、透明度及 mask 的原地更新仍逐帧生效。标准路径只在内部借用参数；旧公共计划和自定义回调保持原有所有权。验证与内存成本见 [准备阶段优化报告](reports/PREPARATION_OPTIMIZATION_REVIEW.md)。

## 暂停、素材切换和关闭

- 没有任何 `ad_params_N` 键代表 B=0：返回 `(0,H,W,3)` 的 uint8 数组，进入暂停时释放活动缓存，连续暂停不反复释放。恢复复用原对象。
- 有 `ad_params_N=[]` 代表该路没有广告：仍返回该路背景。全部为空时不准备 mask/畸变/运动核。
- 活动广告帧必须提供完整相机数据；暂停或全部无广告时，若有完整相机数据也会写入历史，否则跳过记录。默认历史容量 9。
- 替换素材图片对象会被缓存识别。同对象原地改像素，需要更新 `data_dict["imgs_raw_version"]`，或在变化时调用一次 `renderer.invalidate_assets()`。
- `renderer.reset_history()` 清历史并释放工作集，允许新流的 frame_id 重新开始；它不替业务维护 PTS 或重置编码器。
- `renderer.release_resources()` 释放缓存、mask、背景与参数上传工作区，但保留历史和可复用的执行对象；`close()` 还清理本 worker 历史和 owner 当前素材引用，可重复调用。close 后应重新初始化，不继续渲染。
- `with` 或 `finally: renderer.close()` 负责库资源。业务仍负责关闭帧源、排空/取消有界发送队列以及释放自己的素材 bank；不要逐帧 gc/empty_cache。不要无限保留带 traceback 的异常对象。

## 验证入口

CPU 合同与参考检查为 `tests.test_route_plan`、`tests.test_shared_mask`、`tests.test_render_img`；它们分别验证路由/几何组装、独立 mask 数值参考、公开调用及生命周期。CPU 测试中的 eager 初始化是测试替身，生产初始化保持 CUDA/Triton。

GPU 完整调用对照可通过以下命令复跑，结果路径须未占用：

```powershell
.\scripts\run.ps1 check-render-img --device cuda:0 --frames 100 --output results/render_img_check.json
# 可选 --sample '你的3336.npy绝对路径'，用样例配置和合成1080p背景/素材检查接线。
```

`tools/check_render_img.py` 使用明确标注的合成回调和图像，将每路结果与兼容 `render_frame` 单路调用逐字节对照。mask 另与 SciPy 独立卷积参考比较；真实视频素材和生产回调仍需业务验收。报告只记录检查与有限运行状态，不作为吞吐基准。

## 三参数入口初次验收记录（2026-10-22，历史）

本机 CUDA 环境完成以下检查，运行时版本 `2026-10-22-perf4.7-render-img`：

| 检查 | 实际结果 |
|---|---|
| 单元回归 | 新增 45 项 + 既有 72 项，共 117 项全部通过；旧公开初始化 GPU 项显式启用，无 GPU 跳过 |
| 定向场景与持续动态 B | 27 个定向调用 + 1000 次动态 B 调用，共 1027 次；3574 张逐路输出对兼容 render_frame 最大 uint8 差异为 0 |
| `3336.npy` 接线 | 两路、每路 3 个广告、3 份唯一素材；使用合成 1080p 背景与 200×200 四通道素材，2 张逐路输出最大差异为 0 |
| mask 独立参考 | 合成检查相对 SciPy 最大绝对误差 `2.384185791015625e-7`；3336 配置为 `1.7881393432617188e-7`，门槛 `1e-6` |
| 工作区与所有权 | 固定分辨率 mask 工作区复用检查 1021 次；旧 NumPy 未覆写检查 4108 次，最多保留 2 份；相机历史最多 9 条 |
| 退出 | retained stack、素材源引用、Gaussian 条目、mask 工作区、相机历史和 owner 当前素材组均为 0 |
| mask 边界补充 | GPU 覆盖 1×1、1×5、5×1、17×131、1080×1920；全 1 精确为 1、空 mask 为 0，工作区指针复用；radius 0/1/6 额外通过 |
| 可分离运动核补充 | 显式 use_separable 的 B1/B8 对单路旧入口零差异，使用已知可分离的合成核 |

定向场景包含非连续路号、独立矩阵/透明度、不同 slot 数/bbox/素材尺寸/覆盖顺序、空间透明度、同帧几何更新、换图/revision/显式失效、动态 mask/裁剪/空 mask/全保护、连续 B0、恢复、无广告、历史重置和关闭。完整 GPU 命令为：

```powershell
.\scripts\run.ps1 check-render-img --device cuda:0 --frames 1000 --sample '本机3336.npy绝对路径' --output results/render_img_gpu_final_1000.json
```

本机原始记录位于忽略的 `results/render_img_gpu_final_1000.json`、`results/render_img_regression_final.log`、`results/render_img_mask_gpu_final.json`、`results/render_img_gpu_options_final.json`，不随 Git 分发；用户的 NPY 附件也不提交。首次发现并修复了全 1 mask 归一化舍入导致的背景差 1，最终结果已覆盖此断言。

40 个 Python 文件通过 Ruff 和 Python 3.9 AST 检查，差异检查通过；既有贴图/共享几何/策略内核未改，执行器仅更新版本号。CLI 已验证拒绝覆盖旧报告且原 SHA 不变、拒绝 CPU 设备，以及拒绝 `-O` 跳过断言；拒绝路径均未生成新报告。

这是有限帧的功能、像素和资源所有权验证，未做生产视频/回调验收、编码推流、吞吐基准或进程/CUDA 分配器长期趋势测量；不能据此声称生产直播永不泄漏或比旧接口更快。旧 `docs/reports/RENDER_API_REVIEW.md` / `docs/LIVE_RENDER_EXAMPLE.md` 的历史验收保持原记录。

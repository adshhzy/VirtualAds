# 一帧处理流程与业务 worker

完整可运行文件：[render_img_worker.py](../examples/render_img_worker.py)。生产可导入其中的 `run_live`；演示 source/sink/owner 为合成实现，不包含解码、编码或网络发送。运行环境安装见 [启动指南](STARTUP_AND_API.md)。

从仓库根目录，用项目解释器运行：

```powershell
$projectPython = if ($env:PASTE_GPU_PYTHON) { $env:PASTE_GPU_PYTHON } else { '..\env\paste_gpu_performance\Scripts\python.exe' }
& $projectPython -m examples.render_img_worker --device cuda:0
```

## 一帧如何经过核心模块

下列文件都位于 `virtualads/`。入口是 `ImageRenderer.render_img(data_dict, frame, imgs_raw)`；它与旧 `render_frame` 共用构造器和执行器，不再调用旧入口重复准备。

| 顺序 | 模块 | 工作 |
|---|---|---|
| 1 | paste_renderer / paste_route_plan | 读取并按数字排序 ad_params_N，按精确素材键取图，保留 slot 顺序，划分兼容布局组。paste_route_preparation 复用 CPU 参数空间和本次调用的素材描述。 |
| 2 | paste_renderer / paste_camera_filter | 复制 K/w2c/dist_coeffs 小数据到有界历史。没有路键时返回 B0，首次暂停释放活动工作集。 |
| 3 | paste_frame_upload / paste_upload_buffer | 常见连续 CPU uint8 背景按字节上传，GPU 转 float32，一帧只准备一次。 |
| 4 | paste_shared_mask / paste_mask_kernels | 局部 mask 按 bbox 合入整图，重叠取 maximum，再做一次整图 Gaussian 羽化，默认 sigma=1.5、radius=3。mask=1 保留背景，0 允许广告。 |
| 5 | paste_shared_params / paste_camera_filter / paste_shared_kernels | 相机滤波、运动度计算一次；K 和畸变参数生成共享畸变网格，w2c 历史参与业务运动度计算。业务回调提供运动模糊核。 |
| 6 | paste_render_executor / paste_asset_cache | 每布局组准备或复用各 slot 的 B 路素材 stack；集中上传当前矩阵、常见透明度，调用共享 builder 生成 ROI 网格。m_inv 使用 ROI 局部坐标。 |
| 7 | paste_execution_policy / paste_gpu_batch / paste_gpu_kernels | 按已测策略整批或 B 维分块，slot 顺序采样并覆盖 RGBA 画布，再对广告画布畸变、运动模糊、背景融合及 uint8 转换。 |
| 8 | paste_render_executor / paste_renderer | D2H 后返回独立、可立即读取的 NumPy；多布局组按 route_ids 汇总。后续调用或关闭不覆盖旧输出。 |

全部无广告的路直接输出背景，不准备 mask、畸变图或运动核。mask 羽化与广告运动模糊各有用途和核；原背景不参与广告画布的整图畸变。

```text
一帧
├─ 共用背景、mask、相机、畸变网格、运动核
└─ for 兼容布局组
   ├─ 各 slot 准备/复用 [B,4,h,w] 素材及实时参数
   └─ 整批或 for B 维 chunk
      ├─ for slot：各路 GPU 并行采样、写入 ROI
      └─ 广告画布畸变 → 模糊 → 背景融合 → uint8 NumPy
```

布局分组不是素材尺寸分桶：slot 数、有序 bbox、对应有效素材尺寸等必须兼容；矩阵值和普通标量透明度可逐路不同。空间透明度等保守独立执行。默认 batch 内仍按 slot 循环，后 slot 的 bbox 整块覆盖前 slot，保持原有语义。

auto 查询离线策略，不在直播循环计时搜索，也不自动开启 bucketed 或可分离模糊。条件不匹配回退整批，实际决策看 renderer.last_execution，不能仅按总路数推断分块。

## 完整业务接线

独立文件中的 `run_live` 接收已有 owner/source/sink/imgs_raw，并负责 source、sink、renderer 的退出清理，包括初始化失败和输出异常路径。生产导入如下，业务对象由现有直播系统提供：

```python
from examples.render_img_worker import run_live

run_live(
    business_owner, frame_source, encoder_sink, imgs_raw,
    device="cuda:0", resize_ads_to_ad_h_w=True,
)
```

这个导入从源码 checkout 使用；wheel 只分发运行库。生产也可将文件中的 run_live 函数放入自己的业务模块，依赖 `from virtualads import ImageRenderer, RenderOptions`，不复制演示类。

- owner 提供原有 compute_camera_motion_rate、motion_rate_to_sigma、create_gaussian_kernel；不要用 SyntheticOwner 替换生产定义。
- source 是可迭代对象，提供 close()。每个 packet 包含 data_dict、frame、pts；换流/seek 时设置 reset_camera=True。
- sink 提供 set_routes(route_ids)、write_batch(route_ids, batch, pts=...)、close()。示例采用同步消费，接收 BHWC；真实异步实现需有界队列并在 close 时排空或取消。
- imgs_raw 键是 (ad_placement_id, ad_id)，值为四通道 HWC 图片；与背景 RGB/BGR 顺序一致。JPEG 加载时补 alpha=255。
- 3336 配置使用 500×500 测试图时保持 resize_ads_to_ad_h_w=True，按配置尺寸缩放；素材已正确预处理时可设 False。

本帧输入结构示例（变量来自当前业务检测/跟踪结果）：

```python
data_dict = {
    "frame_id": frame_id,
    "K": K,
    "w2c": w2c,
    "dist_coeffs": dist_coeffs,
    "imgs_raw_version": asset_revision,
    "mask_coor_list": [{"bbox": person_bbox, "mask": person_mask}],
    "ad_params_1": [{
        "ad_placement_id": "sideline", "ad_id": 101,
        "bbox": ad_bbox, "m_inv": inverse_matrix, "transparency": 1.0,
    }],
    "ad_params_3": route_3_ads,
}
# imgs_raw[("sideline", 101)] 是对应四通道素材。
packet = {"data_dict": data_dict, "frame": frame, "pts": pts}
```

bbox 为 x1,y1,x2,y2，右下不含；广告 bbox 在画面内，mask 尺寸匹配其 bbox，值域 0..1。frame 常见为连续 uint8 (1080,1920,3)。K/w2c/dist_coeffs 放 CPU。输出 B1=HWC，B>1=BHWC，B0=(0,H,W,3)，路号读取本次 renderer.route_ids。brightness、ad_paste_data、way_ad_map 不参与新入口。

## 初始化与更新边界

| 时机 | 对象或数据 | 处理 |
|---|---|---|
| worker 初始化一次 | owner、RenderOptions、ImageRenderer、source/sink | 一个相机对应一个串行 worker，同一计算流使用；内部执行器、历史、工作区自动创建。 |
| 加载/切换素材 | imgs_raw 图片 | 解码、alpha、颜色格式尽量提前完成；替换图片对象自动识别。 |
| 每帧 | frame、frame_id、相机、各路配置、bbox/m_inv/透明度/mask | 提交当前值；允许复用数组，不意味着每帧重新分配。调用期间不可并发改写。 |
| 原地改变素材像素 | imgs_raw_version | 修改后递增，后续帧保持；或调用一次 invalidate_assets。几何变化不需要素材版本变化。 |
| 核生成规则变化 | owner.gaussian_kernel_cache_version | 同 sigma 对应核改变时更新，不按帧递增。 |
| 换流/seek | reset_history | 清历史和工作集，再从新帧号开始；PTS 独立管理。 |
| 退出 | close | 先停源、结束下游消费，再关闭 renderer；调用方释放自己持有的素材。 |

减少路数要删除停用的 ad_params_N 键。全部删除为暂停，首次释放活动工作集，连续暂停不反复释放；添加路键即可恢复。ad_params_N=[] 表示该路输出背景。不要逐帧清缓存、gc 或 empty_cache。

返回 NumPy 独立，不应无限保存。单路 view 可能持有整个批次；长期只保存一张时可有意复制并接受开销。内部缓存/工作区有界不代表进程、业务素材或队列的总内存有界。

## 演示验证范围

独立演示覆盖 B1、非连续路号 B3、素材对象替换、原地改图并递增版本、连续 B0、恢复 B2、无广告 B2、换流重置和退出。检查输出形状/路号、无广告背景、旧输出存活、退出缓存与工作区归零，并额外注入 sink 异常检查清理和错误传播。它是有限合成检查，不是生产解码/编码/推流验收或长时间内存证明。

2026-10-23 本机 CUDA 验证：8 阶段、16 次调用、30 张逐路输出通过；关闭后各项库持有工作区/缓存与历史计数为零，sink 异常清理检查通过。核心算法与 auto 策略未修改。

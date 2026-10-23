"""执行已准备的共享参数，管理素材缓存、GPU调度与独立输出的渲染执行器。

与 paste_gpu_batch.py 放在同一目录。源码兼容 Python 3.9+；PyTorch / Triton
须与实际 Python / CUDA 环境匹配，本机 GPU 验证环境为 Python 3.12。
默认返回 numpy uint8；B=1时为[H,W,3]，B>1时为[B,H,W,3]，
每张图可通过results[b]取出。显式squeeze_single=False保留batch。

已准备共享参数的直接调用示例：
    from paste_render_executor import RenderExecutor, RenderOptions

    owner.imgs_raw_groups = [group0, group1]  # 每份与原 imgs_raw 格式相同
    owner.imgs_raw_version = 0              # 原地修改素材时递增；未设置默认0
    executor = RenderExecutor(RenderOptions(
        ad_layout="HWC", frame_layout="HWC", mask_layout="HWC",
        warp_align_corners=False, distortion_align_corners=False,
        matrix_coordinates="roi",  # 必须核对旧warp函数；也可能应为canvas
    ))
    results = executor.render_prepared(
        owner, frame_input, data_dict, frame_id, way_id, paste_args)

执行器在worker内创建并复用。render_prepared接收共享准备阶段返回的paste_args，
完成实时输入转换、素材缓存、逐广告网格构造、策略选择与输出；__call__保留同一接口。
原PasteBatchAdapter/AdapterOptions名称由兼容模块继续导出为同一类的别名。

输入约定：
* 未设置 imgs_raw_groups 时，自动使用 [self.imgs_raw]，B=1。
* 每个 group[(ad_placement_id,ad_id)] 或 group[ad_placement_id] 的值为
  {'img': [device0_image, device1_image, ...]}；也接受 img 直接存单张图。
* frame_input: HWC/BHWC（默认）或配置 CHW/BCHW；单背景可供全部B组共用。
* 源素材: HWC/CHW/1HWC/1CHW，4通道。auto有歧义时明确报错。
* RGB/BGR和源alpha均为[0,255]，不交换通道；只有显式配置才缩放图片。
* mask与transparency为[0,1]；若原数据为0..255，设置对应scale=255。
* transparency支持标量、[B]、HW、HWC/CHW、BHW、BHWC/BCHW。
* bbox=[x1,y1,x2,y2]为整数、右下端点不含，必须完全位于背景内。
* 默认check_ad_h_w=False，按实际源图size存取缓存，不因字段不一致报错。
  如需严格校验可设置check_ad_h_w=True。若几何按ad_h_w尺寸生成，优先传原预处理后的素材；
  也可显式设置resize_ads_to_ad_h_w=True，在堆叠前按ad_h_w缩放。
  显式resize默认bilinear/align_corners=False/antialias=False，不承诺与
  未提供的原预处理逐像素一致；resize后矩阵及网格须使用目标尺寸坐标约定。
* 单应和畸变grid均为反向采样的归一化NHW2；可省略N，补为1。
* 单应grid缺失时，从当前m_inv重建；必须显式配置matrix_coordinates。
* 畸变grid缺失时必须提供distortion_grid_builder，绝不默认为无畸变。
* 未缓存任何逐帧warp grid/matrix/kernel或可变Tensor/ndarray mask/transparency。
  不可变Python数值mask/transparency可按值缓存，默认开启。调用者提供的
  grid也必须已更新到当前帧；无法仅凭shape检测一个grid的数值是否过期。
* resize目标尺寸也进入素材缓存key，ad_h_w变化时不会复用旧尺寸素材。

缓存与性能：
* 默认启用缓存，owner.imgs_raw_version未设置时按版本0处理；显式设置为
  None或cache_ads=False则不读写缓存。原地修改图像/写入视频素材时必须
  递增版本号或调用invalidate_assets；仅切换size不要增加版本号。
* 缓存按有效素材(H,W)分成两个独立位置，500和200可以同时存在；每个位置
  依据素材身份/版本/设备/B值区分条目，不会将同尺寸的不同广告混用。
  默认max_cached_sizes=2；第三种size按LRU淘汰。如有三种size可设置为3。
  每个size最多max_entries_per_size=128个条目；每个条目是一广告位的[B,4,H,W]。
  历史stack默认另受max_cached_bytes=512MiB限制；0不保留历史。版本变化会
  淘汰旧版本但保留累计统计，不保存多代视频帧。同一版本切回仍在LRU中的素材可命中。
  当前完整工作集独立持有stack和原图引用，允许超过历史预算；重复帧不会因LRU
  容量不足而重新转换。源身份/尺寸/顺序/B等元数据仍逐帧读取；仅布局计划可缓存。
* cache_info()报告历史/当前/桶stack字节与命中/淘汰次数，字节不含原图存储和
  allocator保留。invalidate_assets()清空素材及布局计划；禁用缓存也释放旧素材。
  backend='bucketed'的二次打包缓存仅保留当前有序分桶素材一份；bbox变化而桶成员
  不变时继续复用，不影响上述素材尺寸缓存。
* 几何变化直接适配当前ROI，不要求固定size，不在此处自动torch.compile。
  Python字典、输入检查、CPU/GPU搬运均在图外；该执行器本身不适合capture。
* backend='bucketed'可选；布局变化会重建CPU LUT，动态视频未必更快。
* 对象为有状态缓存，非线程安全；每个并发渲染worker使用自己的实例。
  同一实例按同一CUDA流串行使用；跨流生产输入时由调用者建立流依赖。

性能配置（保持原图像/几何/alpha约定）：
* 默认optimize_views=True：背景/mask/grid尽量使用view，避免逐帧全图连续化。
* 默认cache_scalar_maps=True：按设备/值/scale缓存最多128个数值map，避免
  每个广告位反复创建并上传同一个透明度标量；Tensor和ndarray不走此缓存。
* acceleration='eager'保留PyTorch算子；'auto'在CUDA且Triton可用时融合
  单应采样/ROI写入与最终融合；'triton'要求可用，否则明确报错。
  需与新版paste_gpu_batch.py及paste_gpu_kernels.py同时部署；首次JIT不计稳态。
  反射边界的单应采样和全图畸变仍走PyTorch；模糊按blur_backend选择
  PyTorch、dense Triton或separable路径。
* pinned_output=True是可选D2H优化；为每个结果分配独立固定页Tensor，拷回并
  等待完成后返回其numpy视图，不再做第二次CPU复制。numpy持有底层Tensor，
  调用者仍持有结果时不会被下一帧覆写。默认关闭，需目标机A/B实测；若下游
  长时间积压结果，固定页内存占用也会相应增加，释放后的块由PyTorch管理复用。
* batch_chunk_size='auto'默认查询离线实测配置，未知硬件/尺寸/布局/B回退整批。
  0强制原整批；正整数在B更大时按此上限分块，尾块不补齐。
  手动分块仅支持backend='batch'、CUDA、return_numpy=True、pinned_output=True。
  共享准备只执行一次；当前计算流顺序渲染，各块通过独立复制流写入本次专属的
  固定页结果，全部复制完成后返回。无广告早返回不分块，B不大于上限时仍走整批。
  分块不会改变输出所有权，也不保证更快；需按实际B、画面和设备测量。
  last_execution记录本次选择和回退原因；blur_launch_config可显式配置两个
  separable核的启动参数。未知的手动blur覆盖会使auto回退整批。
* profile_every=N每N帧采样一次；last_profile和profile_summary()包含CPU与
  CUDA流事件计时。采样有同步成本，正式测速/生产关闭；先完成JIT/预热。
  CUDA事件包含流空闲/CPU供给间隙；export_cpu_ms还包含等待前面GPU工作，
  不能把CPU与CUDA分项相加。total_ms是采样帧的同步端到端耗时。
  分块时core/export对应重叠工作提交与最终等待，不能视为独立的计算/复制耗时。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from .paste_gpu_batch import (
        homography_roi_grid,
        make_bucket_layout,
        pack_bucket_ads,
        pack_bucket_grids,
        pack_bucket_transparencies,
        paste_gpu_batch as _paste_gpu_batch,
        paste_gpu_bucketed,
    )
    from . import paste_execution_policy as execution_policy
    from .paste_shared_types import MotionKernelSpec
    from .paste_upload_buffer import PinnedUploadBuffer
    from .paste_route_preparation import PreparedRouteData
else:
    from paste_gpu_batch import (
        homography_roi_grid,
        make_bucket_layout,
        pack_bucket_ads,
        pack_bucket_grids,
        pack_bucket_transparencies,
        paste_gpu_batch as _paste_gpu_batch,
        paste_gpu_bucketed,
    )
    import paste_execution_policy as execution_policy
    from paste_shared_types import MotionKernelSpec
    from paste_upload_buffer import PinnedUploadBuffer
    from paste_route_preparation import PreparedRouteData

if __package__:
    from .paste_asset_cache import SizeAssetCache as _SizeAssetCache, tensor_bytes
else:
    from paste_asset_cache import SizeAssetCache as _SizeAssetCache, tensor_bytes


RENDERER_VERSION = "2026-10-23-perf4.9-prepare"


@dataclass(frozen=True)
class RenderOptions:
    """Worker configuration; field order is preserved for positional callers.

    Input layout/scale settings describe existing data. Geometry and alpha
    settings define image semantics. Execution/cache/output settings change
    scheduling and storage without changing those image semantics.
    """

    backend: str = "batch"  # batch 或 bucketed
    acceleration: str = "eager"  # eager / auto / triton；融合GPU核需附带paste_gpu_kernels.py
    blur_backend: str = "torch"  # torch / triton密集融合 / separable经验证的共享高斯核
    optimize_views: bool = True  # 背景/mask/grid使用stride view，避免逐帧全图连续化拷贝
    cache_scalar_maps: bool = True  # 按值缓存Python标量透明度/mask；不缓存可变Tensor
    pinned_output: bool = False  # 独立固定页结果，numpy共享其存储；同步后返回
    profile_every: int = 0  # 0关闭；N表示每N帧采样一次（采样帧有同步）

    # Input layouts, normalization and explicitly requested material resizing.
    frame_layout: str = "HWC"  # HWC同时接受BHWC；CHW同时接受BCHW
    ad_layout: str = "auto"
    mask_layout: str = "auto"
    transparency_layout: str = "auto"
    mask_scale: float = 1.0
    transparency_scale: float = 1.0
    check_ad_h_w: bool = False  # 默认按实际素材size；需要严格核对时显式开启
    resize_ads_to_ad_h_w: bool = False  # 显式按(H,W)缩放到单应矩阵所用的素材尺寸
    resize_align_corners: bool = False  # 需匹配原素材预处理；独立于warp_align_corners
    resize_antialias: bool = False  # 仅显式resize时使用；默认普通bilinear

    # Sampling and compositing semantics; preserve legacy values for parity.
    warp_align_corners: bool = False
    distortion_align_corners: bool = False
    warp_padding_mode: str = "zeros"
    distortion_padding_mode: str = "zeros"
    legacy_alpha: bool = True
    matrix_coordinates: str | None = None  # roi / canvas；只在从矩阵建grid时使用
    grid_source: str = "prefer_grid"  # prefer_grid / matrix
    image_index: int | None = None  # None采用原来的多设备索引逻辑

    # Asset lifetime, output ownership, and optional geometry builders.
    cache_ads: bool = True  # 未设置版本号时按版本0处理；None禁用
    max_cached_sizes: int = 2  # 500/200可共存；第三种尺寸按LRU替换
    max_entries_per_size: int = 128  # 每个size内分别存当前版本的不同广告/设备/B组合
    return_numpy: bool = True
    squeeze_single: bool = True  # 单组兼容原paste_gpu的HWC输出；多组保留BHWC
    area_caps: tuple[int, ...] | None = None
    distortion_grid_builder: Callable | None = None
    warp_grids_builder: Callable | None = None  # 可选批量网格构造，按实际素材尺寸生成

    # Offline-measured scheduling; no per-frame autotuning.
    batch_chunk_size: int | str = "auto"  # auto离线配置；0整批；正整数为手动分块上限
    blur_launch_config: tuple | None = None  # 显式覆盖separable横纵启动参数，默认由auto配置选择

    # Historical stacks are bounded separately from the active frame's working set.
    max_cached_bytes: int = 512 * 1024 * 1024  # 0不保留历史；当前完整素材集仍可复用


def _choose_plan(shape, candidates, name):
    """纯shape函数：排除无效候选，对仅singleton维不同的等价排列去重。"""
    unique = {}
    for perm, output in candidates:
        effective = tuple(axis for axis in perm if shape[axis] != 1)
        unique.setdefault((effective, output), (perm, output))
    if not unique:
        raise ValueError(f"{name}: unsupported shape {shape}; check layout/size/batch")
    if len(unique) > 1:
        raise ValueError(f"{name}: ambiguous shape {shape}; set an explicit layout")
    return next(iter(unique.values()))


def _image_plan(shape, channels, layout, name):
    shape = tuple(shape)
    mode = layout.upper()
    if mode not in ("AUTO", "HWC", "BHWC", "CHW", "BCHW"):
        raise ValueError(f"{name}: invalid image layout {layout}")
    candidates = []
    if len(shape) == 3 and all(s > 0 for s in shape):
        a, b, c = shape
        if mode in ("AUTO", "HWC", "BHWC") and c == channels:
            candidates.append(((2, 0, 1), (1, c, a, b)))
        if mode in ("AUTO", "CHW", "BCHW") and a == channels:
            candidates.append(((0, 1, 2), (1, a, b, c)))
    if len(shape) == 4 and all(s > 0 for s in shape):
        n, a, b, c = shape
        if mode in ("AUTO", "HWC", "BHWC") and c == channels:
            candidates.append(((0, 3, 1, 2), (n, c, a, b)))
        if mode in ("AUTO", "CHW", "BCHW") and a == channels:
            candidates.append(((0, 1, 2, 3), (n, a, b, c)))
    return _choose_plan(shape, candidates, name)


def _map_plan(shape, batch, hw, channels, layout, name):
    """将标量/逐组标量/空间map规划成可广播NCHW，纯CPU元数据计算。"""
    shape = tuple(shape)
    height, width = hw
    mode = layout.upper()
    if mode not in ("AUTO", "HW", "HWC", "CHW", "BHW", "BHWC", "BCHW"):
        raise ValueError(f"{name}: invalid map layout {layout}")
    candidates = []
    if len(shape) == 0:
        candidates.append(((), (1, 1, 1, 1)))
    elif len(shape) == 1 and shape[0] in (1, batch):
        candidates.append(((0,), (shape[0], 1, 1, 1)))
    elif len(shape) == 2:
        h, w = shape
        if h in (1, height) and w in (1, width):
            candidates.append(((0, 1), (1, 1, h, w)))
    elif len(shape) == 3:
        a, b, c = shape
        if (
            mode in ("AUTO", "HWC", "BHWC")
            and a in (1, height)
            and b in (1, width)
            and c in channels
        ):
            candidates.append(((2, 0, 1), (1, c, a, b)))
        if (
            mode in ("AUTO", "CHW", "BCHW")
            and a in channels
            and b in (1, height)
            and c in (1, width)
        ):
            candidates.append(((0, 1, 2), (1, a, b, c)))
        if mode in ("AUTO", "BHW") and a in (1, batch) and b in (1, height) and c in (1, width):
            candidates.append(((0, 1, 2), (a, 1, b, c)))
    elif len(shape) == 4:
        n, a, b, c = shape
        if n in (1, batch):
            if (
                mode in ("AUTO", "HWC", "BHWC")
                and a in (1, height)
                and b in (1, width)
                and c in channels
            ):
                candidates.append(((0, 3, 1, 2), (n, c, a, b)))
            if (
                mode in ("AUTO", "CHW", "BCHW")
                and a in channels
                and b in (1, height)
                and c in (1, width)
            ):
                candidates.append(((0, 1, 2, 3), (n, a, b, c)))
    return _choose_plan(shape, candidates, name)


def _as_tensor(value, device):
    # NumPy的翻转view可能含负stride，PyTorch不接受，必要时在CPU做一次copy。
    if not isinstance(value, torch.Tensor) and hasattr(value, "strides"):
        if any(s < 0 for s in value.strides):
            value = value.copy()
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _apply_plan(tensor, plan, *, contiguous=True):
    perm, output = plan
    result = tensor.permute(perm).reshape(output)
    return result.contiguous() if contiguous else result


def _plain_values(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _integer_tuple(value, count, name):
    values = _plain_values(value)
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise ValueError(f"{name} must contain {count} integers")
    result = []
    for v in values:
        if isinstance(v, bool):
            raise ValueError(f"{name}: boolean is not a coordinate/size")
        number = float(v)
        if not math.isfinite(number) or number != int(number):
            raise ValueError(f"{name}: expected finite integer, got {v!r}")
        result.append(int(number))
    return tuple(result)


def _bbox(value, hw, name):
    x1, y1, x2, y2 = _integer_tuple(value, 4, name)
    height, width = hw
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{name}={(x1, y1, x2, y2)} outside canvas H,W={hw}, or empty ROI")
    return x1, y1, x2, y2


def _effective_ad_hw(actual_hw, declared_hw, check_ad_h_w, resize_ads, name):
    """纯元数据规则；ad_h_w不能在没有授权resize时被当成强制目标尺寸。"""
    if resize_ads and declared_hw is None:
        raise ValueError(f"{name}: resize_ads_to_ad_h_w=True requires ad_h_w=(H,W)")
    if declared_hw is not None and (resize_ads or check_ad_h_w):
        if len(declared_hw) != 2 or min(declared_hw) < 1:
            raise ValueError(f"{name}: ad_h_w must contain two positive sizes")
        if resize_ads:
            return tuple(declared_hw)
        if tuple(declared_hw) != tuple(actual_hw):
            raise ValueError(
                f"{name}: ad_h_w={declared_hw}, actual H,W={actual_hw}. "
                "If geometry uses actual-size images, set check_ad_h_w=False; "
                "if geometry uses ad_h_w-size images, use matching preprocessed "
                "images or explicitly set resize_ads_to_ad_h_w=True."
            )
    return tuple(actual_hw)


def _slice_batch_args(args, start, stop):
    """Slice validated core inputs along B, retaining singleton broadcast views."""
    frames, ads, grids, distortion, kernel, mask, opacity = args

    def take(value):
        return value if value.shape[0] == 1 else value[start:stop]

    return (
        take(frames),
        tuple(ad[start:stop] for ad in ads),
        tuple(take(grid) for grid in grids),
        take(distortion),
        kernel[start:stop] if kernel.ndim == 5 else kernel,
        take(mask),
        tuple(take(value) for value in opacity),
    )


class _FrameTiming:
    """Sample prepare, core submission, and completed export with three marks.

    Events use the caller's current stream, not a global device synchronization.
    Chunked rendering overlaps work, so these intervals are not additive costs
    of isolated compute and transfer operations.
    """

    def __init__(self, started, device):
        self.cpu = [started]
        self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        self.events = []
        if self.stream is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.stream)
            self.events.append(event)

    def mark(self):
        self.cpu.append(perf_counter())
        if self.stream is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.stream)
            self.events.append(event)

    def finish(self):
        if self.events:
            self.events[-1].synchronize()
        gpu_intervals_ms = (
            [self.events[i].elapsed_time(self.events[i + 1]) for i in range(3)]
            if self.events
            else [None] * 3
        )
        return dict(
            prepare_cpu_ms=(self.cpu[1] - self.cpu[0]) * 1000,
            core_submit_cpu_ms=(self.cpu[2] - self.cpu[1]) * 1000,
            export_cpu_ms=(self.cpu[3] - self.cpu[2]) * 1000,
            prepare_cuda_ms=gpu_intervals_ms[0],
            core_cuda_ms=gpu_intervals_ms[1],
            export_cuda_ms=gpu_intervals_ms[2],
            total_ms=(perf_counter() - self.cpu[0]) * 1000,
        )


class RenderExecutor:
    """Reusable per-worker render executor, asset cache, and synchronous output owner.

    One call prepares the full B batch before selecting whole, chunked, or
    bucketed execution. Instances remain serial and are not thread safe.
    """

    def __init__(self, options: RenderOptions | None = None):
        self.options = options or RenderOptions()
        opt = self.options
        if opt.backend not in ("batch", "bucketed"):
            raise ValueError("backend must be batch or bucketed")
        if opt.acceleration not in ("eager", "auto", "triton"):
            raise ValueError("acceleration must be eager, auto or triton")
        if opt.blur_backend not in ("torch", "triton", "separable"):
            raise ValueError("blur_backend must be torch, triton or separable")
        if type(opt.profile_every) is not int or opt.profile_every < 0:
            raise ValueError("profile_every must be a nonnegative integer")
        if opt.batch_chunk_size != "auto" and (
            type(opt.batch_chunk_size) is not int or opt.batch_chunk_size < 0
        ):
            raise ValueError("batch_chunk_size must be 'auto' or a nonnegative Python integer")
        if (
            opt.batch_chunk_size != "auto"
            and opt.batch_chunk_size
            and (opt.backend != "batch" or not opt.return_numpy or not opt.pinned_output)
        ):
            raise ValueError(
                "batch_chunk_size requires backend='batch', return_numpy=True and pinned_output=True"
            )
        if opt.blur_launch_config is not None:
            if __package__:
                from .paste_gaussian_kernels import _resolve_launch_config
            else:
                from paste_gaussian_kernels import _resolve_launch_config
            _resolve_launch_config(opt.blur_launch_config)
        if opt.grid_source not in ("prefer_grid", "matrix"):
            raise ValueError("grid_source must be prefer_grid or matrix")
        if opt.matrix_coordinates not in (None, "roi", "canvas"):
            raise ValueError("matrix_coordinates must be None, roi or canvas")
        for name in ("mask_layout", "transparency_layout"):
            if getattr(opt, name).upper() not in (
                "AUTO",
                "HW",
                "HWC",
                "CHW",
                "BHW",
                "BHWC",
                "BCHW",
            ):
                raise ValueError(f"invalid {name}: {getattr(opt, name)!r}")
        if any(not math.isfinite(s) or s <= 0 for s in (opt.mask_scale, opt.transparency_scale)):
            raise ValueError("mask_scale and transparency_scale must be finite positive numbers")
        if opt.image_index is not None and (
            not isinstance(opt.image_index, int) or opt.image_index < 0
        ):
            raise ValueError("image_index must be a nonnegative integer")
        if opt.warp_padding_mode not in (
            "zeros",
            "border",
            "reflection",
        ) or opt.distortion_padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError("padding_mode must be zeros, border or reflection")
        self.invalidate_assets()
        self._layout_key = None
        self._layout = None
        self._scalar_maps = OrderedDict()
        self._map_plans = OrderedDict()  # Bounded CPU metadata only; never retains map values.
        self._host_events = {}  # 只复用完成事件；不复用仍由调用者持有的输出
        self._chunk_transfers = {}  # 每设备一个内部复制流和可复用的生产完成事件
        self._opacity_upload = PinnedUploadBuffer()
        self._call_count = 0
        self.last_profile = None
        self.last_execution = None
        self._profile_samples = []

    def release_resources(self):
        """Release worker-owned caches between calls, keeping returned arrays valid.

        Use at a stream/session boundary or when a worker becomes idle, not in
        the frame loop. PyTorch may retain freed storage in its allocators; this
        method does not empty global caches or release the caller's input data.
        The same adapter can render again and rebuild its caches lazily.
        """
        for copy_stream, _ in self._chunk_transfers.values():
            copy_stream.synchronize()
        self._opacity_upload.release_resources()
        self.invalidate_assets()
        self._layout_key = self._layout = None
        self._scalar_maps.clear()
        self._map_plans.clear()
        self._host_events.clear()
        self._chunk_transfers.clear()
        self._profile_samples.clear()
        self.last_profile = self.last_execution = None

    def invalidate_assets(self):
        """清空所有size的素材缓存；素材原地修改后调用或增加版本号。"""
        self._asset_cache = _SizeAssetCache(
            self.options.max_cached_sizes,
            self.options.max_entries_per_size,
            self.options.max_cached_bytes,
        )
        self._asset_key = None
        self._ads = None
        self._asset_source_refs = ()
        self._image_plans = OrderedDict()
        self._bucket_asset_key = None
        self._bucket_ads = None

    def cache_info(self):
        """Report retained stack bytes; source storage and allocator reserves are excluded."""
        current = {id(value): value for value in self._ads or ()}
        buckets = {id(value): value for value in self._bucket_ads or ()}
        retained = {id(value): value for value in self._asset_cache.tensors()}
        retained.update(current)
        retained.update(buckets)
        return dict(
            self._asset_cache.info(),
            scalar_maps=len(self._scalar_maps),
            image_plans=len(self._image_plans),
            current_bytes=sum(tensor_bytes(value) for value in current.values()),
            bucket_bytes=sum(tensor_bytes(value) for value in buckets.values()),
            retained_stack_bytes=sum(tensor_bytes(value) for value in retained.values()),
            opacity_upload=self._opacity_upload.cache_info(),
        )

    def profile_summary(self, reset=False):
        """最近最多120个采样帧；CUDA事件是流上经过时间，可能包含CPU供给间隙。"""
        samples = self._profile_samples
        result = {"samples": len(samples), "last": self.last_profile}
        if samples:
            metrics = (
                "prepare_cpu_ms",
                "core_submit_cpu_ms",
                "export_cpu_ms",
                "prepare_cuda_ms",
                "core_cuda_ms",
                "export_cuda_ms",
                "total_ms",
            )
            for metric in metrics:
                values = sorted(s[metric] for s in samples if s[metric] is not None)
                if values:
                    result[metric] = {
                        "mean": sum(values) / len(values),
                        "p50": values[len(values) // 2],
                        "p95": values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)],
                    }
        if reset:
            self._profile_samples = []
        return result

    def _record_profile(self, timing, frame_id, batch, slots, cache_before):
        metrics = timing.finish()
        metrics.update(
            frame_id=int(frame_id),
            batch=batch,
            slots=slots,
            acceleration=self.options.acceleration,
            backend=self.options.backend,
            blur_backend=self.options.blur_backend,
            asset_hits=self._asset_cache.hits - cache_before[0],
            asset_misses=self._asset_cache.misses - cache_before[1],
        )
        self.last_profile = metrics
        self._profile_samples.append(metrics)
        if len(self._profile_samples) > 120:
            del self._profile_samples[0]

    @staticmethod
    def _groups(owner):
        groups = getattr(owner, "imgs_raw_groups", None)
        if groups is None:
            # 无广告分支不读取此dict；即使owner尚无imgs_raw也可以正常返回背景。
            return [getattr(owner, "imgs_raw", {})]
        if not isinstance(groups, (tuple, list)) or not groups:
            raise ValueError("imgs_raw_groups must be a nonempty list/tuple of group dictionaries")
        return groups

    def _image_index(self, owner, way_id):
        if self.options.image_index is not None:
            return self.options.image_index
        devices = getattr(owner, "device", None)
        if devices is None or isinstance(devices, (str, bytes, torch.device)):
            count = 1
        else:
            try:
                count = len(devices)
            except TypeError:
                count = 1
        if count < 1:
            raise ValueError("owner.device cannot be an empty device collection")
        return (int(way_id) - 1) % count

    def _collect_ads(self, owner, groups, params, device, image_index, *, prepared_sources=None):
        """Return one cached [B, 4, H, W] tensor per slot in the current order.

        Cache keys include ordered member identities and the caller's asset
        revision. Geometry is deliberately excluded: it is prepared live later.
        """
        described = (
            prepared_sources.consume(groups, params, self.options, image_index)
            if prepared_sources is not None
            else None
        )
        if described is None:
            described = self._describe_ad_sources(groups, params, image_index)
        sources, signature, slot_signatures = described
        opt = self.options
        revision = getattr(owner, "imgs_raw_version", 0)
        cache_enabled = opt.cache_ads and revision is not None
        if cache_enabled:
            if isinstance(revision, torch.Tensor):
                raise TypeError("imgs_raw_version must be a Python int/string, not Tensor")
            hash(revision)
        # A revision identifies current source contents, not a video-frame history.
        # Disabled caching also releases previously retained revisions.
        revision_changed = self._asset_cache.set_revision(revision if cache_enabled else None)
        if revision_changed or not cache_enabled:
            self._asset_key = None
            self._ads = None
            self._asset_source_refs = ()
            self._bucket_asset_key = None
            self._bucket_ads = None
        slot_order = tuple((param["ad_placement_id"], param["ad_id"]) for param in params)
        device_key = str(device)
        key = (
            revision,
            device_key,
            image_index,
            opt.ad_layout,
            len(groups),
            slot_order,
            tuple(signature),
            opt.resize_ads_to_ad_h_w,
            opt.resize_align_corners,
            opt.resize_antialias,
            opt.legacy_alpha,
        )
        if cache_enabled and key == self._asset_key and self._ads is not None:
            # The current working set may exceed the historical byte/size limits.
            # Its separate source refs keep every identity in key alive even when
            # the LRU no longer owns those sources. No source pixels were cached
            # during metadata inspection: replacement and shape changes remain live.
            self._asset_cache.hit_current(len(self._ads))
            return self._ads

        # Release the previous working set before constructing the new one. LRU
        # entries, where retained, still support switching B/material combinations.
        self._asset_key = None
        self._ads = None
        self._asset_source_refs = ()
        self._bucket_asset_key = None
        self._bucket_ads = None
        ads_list = []
        for members, member_signature in zip(sources, slot_signatures):
            hw = members[0][2]
            slot_key = (
                revision,
                device_key,
                image_index,
                opt.ad_layout,
                len(groups),
                member_signature,
                opt.resize_ads_to_ad_h_w,
                opt.resize_align_corners,
                opt.resize_antialias,
                opt.legacy_alpha,
            )
            stacked = self._asset_cache.get(hw, slot_key) if cache_enabled else None
            if stacked is None:
                stacked = torch.cat(
                    [
                        self._resize_ad(_apply_plan(_as_tensor(image, device), plan), target_hw)
                        for image, plan, target_hw in members
                    ],
                    dim=0,
                )
                if cache_enabled:
                    source_refs = tuple(image for image, _, _ in members)
                    self._asset_cache.put(hw, slot_key, stacked, source_refs)
            ads_list.append(stacked)

        ads = tuple(ads_list)
        self._asset_key = key if cache_enabled else None
        self._ads = ads if cache_enabled else None
        if cache_enabled:
            self._asset_source_refs = tuple(image for members in sources for image, _, _ in members)
        return ads

    def _cached_image_plan(self, shape, channels, layout, name):
        """Reuse validated layout metadata without retaining source tensors."""
        key = (tuple(shape), channels, layout)
        plan = self._image_plans.get(key)
        if plan is None:
            plan = _image_plan(shape, channels, layout, name)
            self._image_plans[key] = plan
            if len(self._image_plans) > 128:
                self._image_plans.popitem(last=False)
        else:
            self._image_plans.move_to_end(key)
        return plan

    def _describe_ad_sources(self, groups, params, image_index):
        """Validate source metadata before touching the asset cache.

        sources[slot][group] holds (image, layout_plan, effective_hw). The
        ordered signatures detect replacement/reordering without reading pixels.
        """
        sources, signature, slot_signatures = [], [], []
        opt = self.options
        for slot_index, param in enumerate(params):
            placement_id, slot_id = param["ad_placement_id"], param["ad_id"]
            pair = (placement_id, slot_id)
            members = []
            member_signature = []
            expected_hw = None
            declared = param.get("ad_h_w")
            declared_hw = None
            if declared is not None and (opt.check_ad_h_w or opt.resize_ads_to_ad_h_w):
                declared_hw = _integer_tuple(declared, 2, f"ad_params[{slot_index}].ad_h_w")
            for group_index, group in enumerate(groups):
                if not isinstance(group, Mapping):
                    raise TypeError(f"imgs_raw_groups[{group_index}] must be a mapping")
                record = group[pair] if pair in group else group.get(placement_id)
                if not isinstance(record, Mapping) or "img" not in record:
                    raise KeyError(
                        f"group {group_index}, slot {slot_index}: missing {pair!r} / {placement_id!r} image"
                    )
                images = record["img"]
                if isinstance(images, (tuple, list)):
                    if image_index >= len(images):
                        raise IndexError(
                            f"group {group_index}, slot {slot_index}: device image index "
                            f"{image_index} >= {len(images)}"
                        )
                    image = images[image_index]
                else:
                    image = images
                if not hasattr(image, "shape"):
                    raise TypeError(
                        f"group {group_index}, slot {slot_index}: image must be Tensor or ndarray"
                    )
                source_name = f"group {group_index}, slot {slot_index}"
                plan = self._cached_image_plan(image.shape, 4, opt.ad_layout, source_name)
                if plan[1][0] != 1:
                    raise ValueError(
                        f"{source_name}: one image per group required, got {plan[1][0]}"
                    )
                hw = _effective_ad_hw(
                    plan[1][-2:],
                    declared_hw,
                    opt.check_ad_h_w,
                    opt.resize_ads_to_ad_h_w,
                    source_name,
                )
                if expected_hw is not None and hw != expected_hw:
                    raise ValueError(
                        f"slot {slot_index} {pair!r}: source H,W differs across groups: {expected_hw} vs {hw}"
                    )
                expected_hw = hw
                members.append((image, plan, hw))
                member_signature.append(
                    (
                        pair,
                        id(image),
                        tuple(image.shape),
                        str(image.dtype),
                        str(getattr(image, "device", "cpu")),
                        hw,
                    )
                )
            sources.append(members)
            slot_signatures.append(tuple(member_signature))
            signature.extend(member_signature)

        return sources, signature, slot_signatures

    def _resize_ad(self, image, target_hw):
        if tuple(image.shape[-2:]) == tuple(target_hw):
            return image

        # 只有显式启用resize，target_hw才可能与真实shape不同。
        def resize(value):
            return F.interpolate(
                value,
                size=target_hw,
                mode="bilinear",
                align_corners=self.options.resize_align_corners,
                antialias=self.options.resize_antialias,
            )

        if self.options.legacy_alpha:
            return resize(image)
        # corrected模式下先预乘再resize，避免透明像素RGB污染边缘。
        alpha = image[:, 3:4] * (1.0 / 255.0)
        resized = resize(torch.cat((image[:, :3] * alpha, alpha), dim=1))
        alpha_out = resized[:, 3:4]
        rgb = torch.where(
            alpha_out > 0,
            resized[:, :3] / alpha_out.clamp_min(1e-12),
            torch.zeros_like(resized[:, :3]),
        )
        # 核心函数接收straight RGBA；还原后由核心在warp前再次预乘。
        return torch.cat((rgb, alpha_out * 255.0), dim=1)

    def _grid(self, value, hw, batch, device, name):
        grid = _as_tensor(value, device)
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
        if grid.ndim != 4 or grid.shape[0] not in (1, batch) or grid.shape[1:] != (*hw, 2):
            raise ValueError(
                f"{name}: expected [1|{batch},{hw[0]},{hw[1]},2], got {tuple(grid.shape)}"
            )
        return grid if self.options.optimize_views else grid.contiguous()

    def _map(self, value, batch, hw, channels, layout, scale, device, name, full_spatial=False):
        if (
            self.options.cache_scalar_maps
            and isinstance(value, Real)
            and math.isfinite(float(value))
        ):
            # Python数值才按值缓存。Tensor/ndarray每帧读取，防止原地更新后误用旧值。
            key = (str(device), float(value), float(scale))
            result = self._scalar_maps.get(key)
            if result is None:
                result = torch.full((1, 1, 1, 1), float(value), device=device, dtype=torch.float32)
                if scale != 1.0:
                    result = result / scale
                self._scalar_maps[key] = result
                while len(self._scalar_maps) > 128:
                    self._scalar_maps.popitem(last=False)
            else:
                self._scalar_maps.move_to_end(key)
        else:
            tensor = _as_tensor(value, device)
            # Only the permutation/broadcast plan is reusable. Tensor and
            # ndarray contents must still be read on every call.
            key = (tuple(tensor.shape), batch, hw, channels, layout)
            plan = self._map_plans.get(key)
            if plan is None:
                plan = _map_plan(tensor.shape, batch, hw, channels, layout, name)
                self._map_plans[key] = plan
                if len(self._map_plans) > 128:
                    self._map_plans.popitem(last=False)
            else:
                self._map_plans.move_to_end(key)
            result = _apply_plan(tensor, plan, contiguous=not self.options.optimize_views)
            if scale != 1.0:
                result = result / scale
        if full_spatial:
            result = result.expand(result.shape[0], result.shape[1], *hw)
            if not self.options.optimize_views:
                result = result.contiguous()
        return result

    def _export(self, output):
        """Apply B1 shape policy and finish any required CPU copy before returning."""
        # 仅移除单组的batch维，避免squeeze()意外删掉H/W为1的空间维。
        if self.options.squeeze_single and output.shape[0] == 1:
            output = output[0]
        if not self.options.return_numpy:
            return output
        if not self.options.pinned_output or output.device.type != "cuda":
            return output.cpu().numpy()
        # 返回值持有自己的存储，不把仍然存活的结果放进可覆写的环形缓冲区。
        # PyTorch的固定页allocator可复用已释放的内存；首次/队列增长时仍可能分配。
        key = str(output.device)
        with torch.cuda.device(output.device):
            host = torch.empty(output.shape, dtype=output.dtype, device="cpu", pin_memory=True)
            ready = self._host_events.get(key)
            if ready is None:
                ready = torch.cuda.Event()
                self._host_events[key] = ready
            host.copy_(output, non_blocking=True)
            ready.record(torch.cuda.current_stream(output.device))
            ready.synchronize()
        # ndarray.base持有Tensor，底层内存直到下游释放结果后才可由allocator复用。
        return host.numpy()

    def _render_chunked_numpy(self, args, boxes, core_options, chunk_size, timing=None):
        """Overlap per-chunk D2H with later rendering; return a completed private host array."""
        frames, ads = args[:2]
        batch = ads[0].shape[0]
        device = frames.device
        with torch.cuda.device(device):
            compute_stream = torch.cuda.current_stream(device)
            key = str(device)
            state = self._chunk_transfers.get(key)
            if state is None:
                state = (torch.cuda.Stream(device=device), [])
                self._chunk_transfers[key] = state
            copy_stream, produced_events = state
            host = torch.empty(
                (batch, *frames.shape[-2:], 3), dtype=torch.uint8, device="cpu", pin_memory=True
            )
            # Keep all source allocations alive through the final copy, including
            # exceptions after an earlier chunk has already started its transfer.
            outputs = []
            try:
                for index, start in enumerate(range(0, batch, chunk_size)):
                    stop = min(start + chunk_size, batch)
                    output = _paste_gpu_batch(
                        *_slice_batch_args(args, start, stop), bboxes=boxes, **core_options
                    )
                    outputs.append(output)
                    if index == len(produced_events):
                        produced_events.append(torch.cuda.Event())
                    produced = produced_events[index]
                    produced.record(compute_stream)
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(produced)
                        host[start:stop].copy_(output, non_blocking=True)
                if timing is not None:
                    # The current-stream event marks the compute tail; D2H has
                    # already overlapped it. The next interval is only the tail
                    # remaining after all chunks were submitted, not total D2H.
                    timing.mark()
            finally:
                # This stream is private to the adapter. Even an exception must
                # finish outstanding reads/writes before host/outputs can die.
                copy_stream.synchronize()
        if self.options.squeeze_single and batch == 1:
            host = host[0]
        return host.numpy()

    @staticmethod
    def _ad_parameters(data_dict, way_id):
        """Use way-specific parameters when present, otherwise the shared list."""
        params = data_dict.get(f"ad_params_{way_id}")
        if params is None:
            params = data_dict.get("ad_params")
        if params is not None and not isinstance(params, (list, tuple)):
            raise TypeError("ad_params must be list/tuple or None")
        return params

    def _render_background_only(self, frame_input, batch, opt, started, sampled):
        """Preserve the no-ad early return without inspecting paste_args."""
        self.last_execution = dict(
            batch=batch, chunk_size=0, blur_launch_config=None, reason="no_ads"
        )
        tensor = _as_tensor(frame_input, None)
        timing = _FrameTiming(started, tensor.device) if sampled else None
        plan = _image_plan(tensor.shape, 3, opt.frame_layout, "frame_input")
        frame = _apply_plan(tensor, plan, contiguous=not opt.optimize_views)
        if frame.shape[0] not in (1, batch):
            raise ValueError("background batch must be 1 or number of image groups")
        # Keep the original clamp -> uint8 -> BHWC operation order.
        output = (
            frame.expand(batch, -1, -1, -1)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        if timing is not None:
            timing.mark()
            timing.mark()
        return self._export(output), timing

    def _prepare_frame(self, frame_input, device, batch, opt) -> torch.Tensor:
        """Convert the background to float32 [1|B, 3, H, W], preserving views."""
        tensor = _as_tensor(frame_input, device)
        plan = _image_plan(tensor.shape, 3, opt.frame_layout, "frame_input")
        frames = _apply_plan(tensor, plan, contiguous=not opt.optimize_views)
        if opt.batch_chunk_size != "auto" and opt.batch_chunk_size and frames.device.type != "cuda":
            raise ValueError("batch_chunk_size requires CUDA for calls with advertisements")
        if frames.shape[0] not in (1, batch):
            raise ValueError(f"background batch={frames.shape[0]}, expected 1 or {batch}")
        return frames

    def _prepare_warp_inputs(self, params, ads, boxes, grid_dict, device, batch, opt):
        """Build live ROI grids and opacity maps once, before any B chunking.

        Each grid is [1|B, roi_h, roi_w, 2]. Each opacity map broadcasts to
        [B, 1, roi_h, roi_w]; singleton dimensions remain views when requested.
        """
        if grid_dict is not None and not isinstance(grid_dict, Mapping):
            raise TypeError("ad_warp_grids_dict must be mapping or None")
        built_grids = None
        if opt.warp_grids_builder is not None and (
            opt.grid_source == "matrix"
            or grid_dict is None
            or any(grid_dict.get(param["ad_id"]) is None for param in params)
        ):
            try:
                built_grids = opt.warp_grids_builder(
                    params=params,
                    ads=ads,
                    boxes=boxes,
                    device=device,
                    batch=batch,
                    warp_align_corners=opt.warp_align_corners,
                    matrix_coordinates=opt.matrix_coordinates,
                )
                if not isinstance(built_grids, (tuple, list)) or not len(built_grids) == len(
                    params
                ) == len(ads) == len(boxes):
                    raise ValueError("warp_grids_builder must return one grid per ad_param")
                # This extension receives cached tensors. Keep the boundary
                # guard in case a custom callback mutates their metadata.
                for ad in ads:
                    if (
                        ad.ndim != 4
                        or ad.shape[:2] != (batch, 4)
                        or min(ad.shape[-2:]) < 1
                        or ad.device != device
                        or ad.dtype != torch.float32
                    ):
                        raise ValueError("warp_grids_builder must preserve prepared source tensors")
            except Exception:
                # A failed callback may have modified a cached tensor before
                # raising. Do not reuse that working set on the next call.
                self.invalidate_assets()
                raise

        grids = []
        for slot_index, (param, box, ad) in enumerate(zip(params, boxes, ads)):
            x1, y1, x2, y2 = box
            roi_hw = (y2 - y1, x2 - x1)
            grid_value = (
                grid_dict.get(param["ad_id"])
                if grid_dict is not None and opt.grid_source == "prefer_grid"
                else None
            )
            if grid_value is None and built_grids is not None:
                grid_value = built_grids[slot_index]
            if grid_value is None:
                if opt.matrix_coordinates is None:
                    raise ValueError(
                        f"slot {slot_index}: warp grid missing. Set matrix_coordinates='roi' "
                        "or 'canvas' after checking original warp function; cannot infer this safely"
                    )
                matrix = _as_tensor(param["m_inv"], device)
                if matrix.ndim == 3 and matrix.shape[0] not in (1, batch):
                    raise ValueError(
                        f"slot {slot_index}: inverse matrix batch must be 1 or {batch}"
                    )
                grid_value = homography_roi_grid(
                    matrix,
                    tuple(ad.shape[-2:]),
                    box,
                    matrix_uses_canvas_xy=opt.matrix_coordinates == "canvas",
                    align_corners=opt.warp_align_corners,
                    acceleration=opt.acceleration,
                )
            grids.append(
                self._grid(grid_value, roi_hw, batch, device, f"warp grid slot {slot_index}")
            )
        opacity = self._prepare_opacity_maps(params, boxes, device, batch, opt)
        return grids, opacity

    def _prepare_opacity_maps(self, params, boxes, device, batch, opt):
        """Upload mutable host float32 scalars/vectors once for the entire group.

        These are live values, not a value cache. The route planner produces one
        float32 [B] vector per slot. Pack those into a bounded pinned workspace;
        GPU views keep the original singleton/B broadcast shapes. Other dtypes,
        spatial maps and device tensors retain the existing conversion path.
        Immutable Python numbers continue to use the small scalar-value cache.
        """
        host_values = []
        total = 0
        if device.type == "cuda":
            for slot, param in enumerate(params):
                value = param.get("transparency", 1.0)
                if isinstance(value, torch.Tensor):
                    if (
                        value.device.type != "cpu"
                        or value.dtype != torch.float32
                        or value.layout != torch.strided
                        or value.is_neg()
                        or value.is_conj()
                        or value.ndim > 1
                    ):
                        continue
                    value = value.detach().numpy()
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype == np.float32
                    and (value.ndim == 0 or (value.ndim == 1 and value.size in (1, batch)))
                ):
                    host_values.append((slot, value, total))
                    total += value.size

        packed_maps = {}
        if host_values:
            host = self._opacity_upload.prepare((total,), dtype=torch.float32, device=device)
            array = host.numpy()
            for _, value, start in host_values:
                np.copyto(array[start : start + value.size], value.reshape(-1))
            packed = self._opacity_upload.upload()
            # Keep scale arithmetic on the GPU, after float32 conversion, exactly
            # as in _map. Scaling the host values could change rounding.
            if opt.transparency_scale != 1.0:
                packed.div_(opt.transparency_scale)
            for slot, value, start in host_values:
                packed_maps[slot] = packed[start : start + value.size].view(-1, 1, 1, 1)

        maps = []
        for slot, (param, (x1, y1, x2, y2)) in enumerate(zip(params, boxes)):
            value = packed_maps.get(slot)
            if value is None:
                value = self._map(
                    param.get("transparency", 1.0),
                    batch,
                    (y2 - y1, x2 - x1),
                    (1,),
                    opt.transparency_layout,
                    opt.transparency_scale,
                    device,
                    f"transparency slot {slot}",
                )
            maps.append(value)
        return maps

    def _prepare_canvas_inputs(
        self,
        distortion,
        dist_coeffs,
        mask_value,
        data_dict,
        frame_id,
        way_id,
        hw,
        device,
        batch,
        opt,
    ):
        """Read the current distortion grid and mask; neither is cached by frame ID."""
        if distortion is None:
            if opt.distortion_grid_builder is None:
                raise ValueError(
                    "distortion_grid is None: provide a current grid or configure "
                    "distortion_grid_builder; no identity fallback"
                )
            distortion = opt.distortion_grid_builder(
                data_dict=data_dict,
                dist_coeffs_filtered=dist_coeffs,
                height=hw[0],
                width=hw[1],
                device=device,
                dtype=torch.float32,
                batch_size=batch,
                frame_id=frame_id,
                way_id=way_id,
            )
        distortion = self._grid(distortion, hw, batch, device, "distortion_grid")
        mask = self._map(
            mask_value,
            batch,
            hw,
            (1, 3),
            opt.mask_layout,
            opt.mask_scale,
            device,
            "mask_canvas",
            full_spatial=True,
        )
        return distortion, mask

    def _prepare_motion_kernel(self, kernel_value, device, batch, opt):
        """Unpack a dense kernel and optional verified factors without recomputing them.

        Shared kernels become [4, 1, kh, kw]; existing [B, 4, 1, kh, kw]
        kernels retain their batch axis. One-dimensional factors stay shared.
        """
        blur_factors = None
        if isinstance(kernel_value, MotionKernelSpec):
            if kernel_value.factors is not None:
                blur_factors = tuple(_as_tensor(factor, device) for factor in kernel_value.factors)
            kernel_value = kernel_value.kernel
        kernel = _as_tensor(kernel_value, device)
        if kernel.ndim == 2:
            kernel = kernel[None, None].expand(4, 1, -1, -1)
        elif kernel.ndim == 4 and kernel.shape[:2] == (1, 1):
            kernel = kernel.expand(4, 1, -1, -1)
        valid_channels = (kernel.ndim == 4 and kernel.shape[:2] == (4, 1)) or (
            kernel.ndim == 5 and kernel.shape[:3] == (batch, 4, 1)
        )
        if not valid_channels or any(size <= 0 or size % 2 == 0 for size in kernel.shape[-2:]):
            raise ValueError("kernel must be [4,1,odd_h,odd_w] or [B,4,1,odd_h,odd_w]")
        if not opt.optimize_views:
            kernel = kernel.contiguous()
        return kernel, blur_factors

    def _select_execution(self, args, boxes, blur_factors, batch, opt):
        """Resolve offline/manual settings and record the exact settings to execute."""
        if opt.batch_chunk_size == "auto":
            chunk_size, blur_config, reason = execution_policy.choose_execution(
                args, boxes, opt, blur_factors
            )
        else:
            chunk_size, blur_config, reason = opt.batch_chunk_size, None, "explicit"

        if opt.blur_launch_config is not None:
            measured_blur_config = blur_config or (128, 4, 4, 128, 4, 4)
            if opt.batch_chunk_size == "auto" and tuple(opt.blur_launch_config) != tuple(
                measured_blur_config
            ):
                chunk_size, reason = 0, "unmeasured_blur_override"
            blur_config = opt.blur_launch_config
        self.last_execution = dict(
            batch=batch,
            chunk_size=chunk_size,
            blur_launch_config=blur_config,
            reason=reason,
        )
        core_options = dict(
            warp_align_corners=opt.warp_align_corners,
            distortion_align_corners=opt.distortion_align_corners,
            warp_padding_mode=opt.warp_padding_mode,
            distortion_padding_mode=opt.distortion_padding_mode,
            legacy_alpha=opt.legacy_alpha,
            acceleration=opt.acceleration,
            blur_backend=opt.blur_backend,
            blur_factors=blur_factors,
            blur_launch_config=blur_config,
        )
        return chunk_size, core_options

    def _prepare_bucket_inputs(self, args, boxes, batch, opt):
        """Cache assets by their ordered bucket members, independently of ROI layout.

        A layout change rebuilds the LUT and capacities. Packed source tensors
        depend only on assets and ordered groups, so unchanged groups can reuse
        them across bbox/ROI changes. Grids and opacity are always packed live.
        """
        frames, ads, grids, distortion, kernel, mask, opacity = args
        hw = tuple(frames.shape[-2:])
        layout_key = (
            boxes,
            hw,
            str(frames.device),
            tuple(tuple(ad.shape[-2:]) for ad in ads),
            opt.area_caps,
        )
        if layout_key != self._layout_key:
            self._layout = make_bucket_layout(ads, boxes, hw, area_caps=opt.area_caps)
            self._layout_key = layout_key
        # _asset_key already covers revision, device, B, source identities,
        # effective sizes and slot/member order. Capacities, offsets and bbox
        # coordinates do not affect pack_bucket_ads; group membership does.
        bucket_key = (self._asset_key, self._layout["groups"])
        if self._asset_key is None or bucket_key != self._bucket_asset_key:
            self._bucket_ads = pack_bucket_ads(ads, self._layout)
            self._bucket_asset_key = bucket_key
        packed_grids = pack_bucket_grids(grids, self._layout, batch)
        packed_opacity = pack_bucket_transparencies(opacity, self._layout, batch)
        return (
            frames,
            self._bucket_ads,
            packed_grids,
            distortion,
            kernel,
            mask,
            packed_opacity,
            self._layout["canvas_lut"],
        )

    def _render_prepared(self, args, boxes, core_options, chunk_size, batch, opt, timing):
        """Run the backend and export; render_prepared records the final profile mark."""
        if opt.backend == "batch":
            if timing is not None:
                timing.mark()
            if chunk_size and batch > chunk_size:
                # The chunk helper records the compute mark, joins the copy stream,
                # and returns its completed, independently owned NumPy result.
                return self._render_chunked_numpy(args, boxes, core_options, chunk_size, timing)
            output = _paste_gpu_batch(*args, bboxes=boxes, **core_options)
        else:
            bucket_inputs = self._prepare_bucket_inputs(args, boxes, batch, opt)
            if timing is not None:
                timing.mark()
            output = paste_gpu_bucketed(*bucket_inputs, **core_options)
        if timing is not None:
            timing.mark()
        return self._export(output)

    @torch.inference_mode()
    def render_prepared(
        self, owner, frame_input, data_dict: dict, frame_id: int, way_id: int, paste_args
    ):
        """Prepare one frame, render all material groups, and return the configured output."""
        opt = self.options
        self._call_count += 1
        sampled = opt.profile_every > 0 and self._call_count % opt.profile_every == 0
        started = perf_counter() if sampled else 0.0
        cache_before = (self._asset_cache.hits, self._asset_cache.misses) if sampled else None
        groups = self._groups(owner)
        batch = len(groups)
        params = self._ad_parameters(data_dict, way_id)

        # No-ad calls must not inspect or validate paste_args.
        if not params:
            result, timing = self._render_background_only(frame_input, batch, opt, started, sampled)
            slots = 0
        else:
            if not isinstance(paste_args, (tuple, list)) or len(paste_args) != 6:
                raise ValueError(
                    "paste_args must be (device, distortion_grid, dist_coeffs_filtered, "
                    "motion_blur_kernel_4d, mask_canvas, ad_warp_grids_dict)"
                )
            device_value, distortion, dist_coeffs, kernel_value, mask_value, grid_dict = paste_args
            device = (
                torch.device("cuda", device_value)
                if isinstance(device_value, int)
                else torch.device(device_value)
            )
            timing = _FrameTiming(started, device) if sampled else None
            frames = self._prepare_frame(frame_input, device, batch, opt)
            # Resolve index-less 'cuda' before building device-specific cache keys.
            device = frames.device
            hw = tuple(frames.shape[-2:])
            boxes = tuple(
                _bbox(param["bbox"], hw, f"ad_params[{i}].bbox") for i, param in enumerate(params)
            )
            image_index = self._image_index(owner, way_id)
            if isinstance(data_dict, PreparedRouteData):
                ads = self._collect_ads(
                    owner, groups, params, device, image_index, prepared_sources=data_dict.sources
                )
            else:
                ads = self._collect_ads(owner, groups, params, device, image_index)
            grids, opacity = self._prepare_warp_inputs(
                params, ads, boxes, grid_dict, device, batch, opt
            )
            distortion, mask = self._prepare_canvas_inputs(
                distortion,
                dist_coeffs,
                mask_value,
                data_dict,
                frame_id,
                way_id,
                hw,
                device,
                batch,
                opt,
            )
            kernel, blur_factors = self._prepare_motion_kernel(kernel_value, device, batch, opt)
            # Core argument order is part of the existing pure-tensor interface.
            # Each prepare stage validates its own shape/broadcast contract and
            # converts to float32 on this device. Do not scan the same tree again
            # here; direct core callers can still use public validate_inputs().
            args = (frames, ads, tuple(grids), distortion, kernel, mask, tuple(opacity))
            chunk_size, core_options = self._select_execution(args, boxes, blur_factors, batch, opt)
            result = self._render_prepared(
                args, boxes, core_options, chunk_size, batch, opt, timing
            )
            slots = len(params)

        if timing is not None:
            timing.mark()
            self._record_profile(timing, frame_id, batch, slots, cache_before)
        return result

    def __call__(self, owner, frame_input, data_dict: dict, frame_id: int, way_id: int, paste_args):
        """Compatibility call syntax for the explicit render_prepared entry point."""
        return self.render_prepared(owner, frame_input, data_dict, frame_id, way_id, paste_args)


# Preserve direct imports of the former class names; these are aliases, not subclasses.
AdapterOptions = RenderOptions
PasteBatchAdapter = RenderExecutor

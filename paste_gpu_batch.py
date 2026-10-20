"""Batch GPU advertisement compositing, PyTorch 2.x.

接口约定
--------
* batch = 多组素材；每组广告位顺序相同，每个广告位在各组的尺寸相同。
* 不同广告位可以尺寸不同。各组共用每个广告位的 bbox。
* RGB/BGR 通道顺序原样保留；RGB 和原始 alpha 范围均为 [0,255]。
* 核心输入统一为同设备 float32，图像为 NCHW，grid 为 NHW2。
* mask=1 保留背景，mask=0 允许广告；mask 和 transparency 范围 [0,1]。
* grid 必须是反向采样网格；align_corners 必须匹配建网格时的设置。
* blur kernel 要求奇数尺寸、非负且各通道权重和为 1，由调用者保证。
* 保留原代码的 ROI 覆盖语义，包括透明像素覆盖前面广告的行为。
* 不包含相机滤波。请在外部每帧计算一次共享几何，再传入。

legacy_alpha=True（默认）兼容原代码的双乘 alpha；False 在源图先预乘
alpha，之后线性采样/模糊预乘颜色与 alpha，并正确 over 到背景。
False 会改变透明边缘的结果，应按需求显式启用。

返回 GPU uint8 [B,H,W,3]；在计算图之外调用 result.cpu().numpy()。
这里的批处理不依赖原 warp_perspective_torch / apply_distortion 实现，
仅复用它们的 grid；未给出的插值/边界/坐标约定需要调用者核对。

实际 GPU 延迟及编译兼容性应在目标 PyTorch/CUDA/GPU 上验证。

进阶接口 paste_gpu_bucketed：源尺寸分桶、展平ROI补齐、LUT gather合成。
原 paste_gpu_batch 保留用于低显存和端到端性能对比。见 BUCKET_EXAMPLE。
"""

from array import array
from functools import lru_cache, partial
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def _optional_kernels():
    # Keep the eager reference importable without Triton, including torch.compile.
    if __package__:
        from . import paste_gpu_kernels
    else:
        import paste_gpu_kernels
    return paste_gpu_kernels


@lru_cache(maxsize=1)
def _optional_gaussian_kernels():
    if __package__:
        from . import paste_gaussian_kernels
    else:
        import paste_gaussian_kernels
    return paste_gaussian_kernels


def acceleration_status(device="cuda"):
    """Initialization-time diagnostics, not a GPU benchmark or kernel self-test."""
    kernels = _optional_kernels()
    device = torch.device(device)
    cuda_ok = device.type == "cuda" and torch.cuda.is_available()
    result = {
        "device": str(device), "triton_installed": kernels.available(),
        "cuda_available": cuda_ok,
        "available": cuda_ok and kernels.available(),
        "warp_padding_modes": ("zeros", "border"),
        "reflection": "PyTorch warp; Triton final blend still available",
        "blur_backends": ("torch", "triton", "separable"),
        "triton_blur_kernel_limit": "odd kh,kw <= 15; float32 CUDA; explicit opt-in",
    }
    if not cuda_ok:
        result["reason"] = "CUDA device/runtime unavailable"
    elif not kernels.available():
        result["reason"] = "Triton is not installed"
    else:
        result["reason"] = "Available; target-GPU correctness/performance validation still required"
    return result


def _select_accelerator(acceleration, *tensors):
    if acceleration == "eager":
        return None
    if acceleration not in ("auto", "triton"):
        raise ValueError("acceleration must be 'eager', 'auto' or 'triton'")
    kernels = _optional_kernels()
    if kernels.supported(*tensors):
        return kernels
    if acceleration == "triton":
        raise RuntimeError("acceleration='triton' requires Triton and inference float32 CUDA inputs")
    return None


def paste_gpu_batch(
    frames: torch.Tensor,
    ads_by_slot: tuple[torch.Tensor, ...],
    warp_grids: tuple[torch.Tensor, ...],
    distortion_grid: torch.Tensor,
    motion_kernel: torch.Tensor,
    mask_canvas: torch.Tensor,
    transparencies: tuple[torch.Tensor, ...],
    *,
    bboxes: tuple[tuple[int, int, int, int], ...],
    warp_align_corners: bool = False,
    distortion_align_corners: bool = False,
    warp_padding_mode: str = "zeros",
    distortion_padding_mode: str = "zeros",
    legacy_alpha: bool = True,
    acceleration: str = "eager",
    blur_backend: str = "torch",
    blur_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """无 batch 循环的计算核心；外层使用 torch.inference_mode()。

    frames:          [1或B,3,H,W]；1 表示各组共用背景。
    ads_by_slot[i]:  [B,4,h_i,w_i]，不同 i 可以不同尺寸。
    warp_grids[i]:   [1或B,roi_h_i,roi_w_i,2]。
    distortion_grid:[1或B,H,W,2]。
    motion_kernel:   [4,1,kh,kw]（共享）或 [B,4,1,kh,kw]（逐组）。
    mask_canvas:     [1或B,1或3,H,W]。
    transparencies[i]: 可广播到 [B,1,roi_h_i,roi_w_i] 的四维 Tensor；
                       常量 [1,1,1,1]，逐组标量 [B,1,1,1]。
    bboxes[i]:       (x1,y1,x2,y2)，位于画布内，所有 batch 元素共用。

    该函数不做输入拷贝/CPU 转换/字典查询/动态建图。validate_inputs 应在
    初始调用或输入结构变化时在图外运行。固定长度广告位循环可被编译展开。

    acceleration='eager' 保留原始 PyTorch 路径；'auto' 在可用时使用 Triton
    融合 warp/alpha/ROI 写入与最终融合/uint8/BHWC 输出，'triton' 强制要求
    CUDA/Triton。reflection 的 warp 仍使用 PyTorch；畸变不改动。
    blur_backend='torch' 默认保留原模糊；'triton' 显式启用模糊+最终融合，
    仅支持奇数 kh/kw <= 15。核权重每次读取，仅权重值变化不触发重新编译。
    'separable' 要求 blur_factors=(ky,kx) 与共享四通道核对应；奇数长度<=63。
    acceleration可用时使用两阶段Triton，否则用PyTorch两次一维卷积。
    快速路径要求有限 grid/图像/透明度/mask 数值；不会逐帧扫描数值。
    Triton 运行/编译异常不会被静默捕获。先在目标卡验证再启用。
    """
    if blur_backend not in ("torch", "triton", "separable"):
        raise ValueError("blur_backend must be 'torch', 'triton' or 'separable'")
    batch = ads_by_slot[0].shape[0] if ads_by_slot else frames.shape[0]
    height, width = frames.shape[-2:]
    canvas = frames.new_zeros((batch, 4, height, width))
    accelerator = _select_accelerator(acceleration, frames, canvas)

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        source = ads_by_slot[i]
        grid = warp_grids[i].expand(batch, -1, -1, -1)
        opacity = transparencies[i]

        if accelerator is not None and warp_padding_mode in ("zeros", "border"):
            slot_accelerator = _select_accelerator(acceleration, source, grid, opacity, canvas)
            if slot_accelerator is not None:
                slot_accelerator.warp_paste_into(
                    source, grid, opacity, canvas, (x1, y1, x2, y2),
                    align_corners=warp_align_corners,
                    padding_mode=warp_padding_mode, legacy_alpha=legacy_alpha,
                )
                continue

        if legacy_alpha:
            warped = F.grid_sample(
                source, grid, mode="bilinear",
                padding_mode=warp_padding_mode,
                align_corners=warp_align_corners,
            )
            alpha = warped[:, 3:4] * (1.0 / 255.0) * opacity
            patch = torch.cat((warped[:, :3] * alpha, alpha), dim=1)
        else:
            source_alpha = source[:, 3:4] * (1.0 / 255.0)
            premultiplied = torch.cat(
                (source[:, :3] * source_alpha, source_alpha), dim=1
            )
            patch = F.grid_sample(
                premultiplied, grid, mode="bilinear",
                padding_mode=warp_padding_mode,
                align_corners=warp_align_corners,
            ) * opacity

        # 与原实现一致：后一个 bbox 整块覆盖前一个 bbox，不做广告间 over。
        canvas[:, :, y1:y2, x1:x2] = patch

    return _finish_canvas(
        canvas, frames, distortion_grid, motion_kernel, mask_canvas,
        distortion_align_corners, distortion_padding_mode, legacy_alpha,
        acceleration=acceleration, blur_backend=blur_backend, blur_factors=blur_factors,
    )


def _finish_canvas(canvas, frames, distortion_grid, motion_kernel, mask_canvas,
                   distortion_align_corners, distortion_padding_mode, legacy_alpha,
                   *, acceleration="eager", blur_backend="torch", blur_factors=None):
    if blur_backend not in ("torch", "triton", "separable"):
        raise ValueError("blur_backend must be 'torch', 'triton' or 'separable'")
    if blur_backend == "separable":
        if blur_factors is None or len(blur_factors) != 2:
            raise ValueError("separable blur requires verified (ky,kx) factors; use dense blur for other kernels")
        ky, kx = blur_factors
        if (motion_kernel.ndim != 4 or ky.ndim != 1 or kx.ndim != 1
                or (ky.numel(), kx.numel()) != tuple(motion_kernel.shape[-2:])
                or any(k.device != canvas.device or k.dtype != canvas.dtype for k in (ky, kx))
                or any(n < 1 or n > 63 or n % 2 != 1 for n in (ky.numel(), kx.numel()))):
            raise ValueError("separable factors must match shared dense kernel, dtype/device; odd lengths <=63")
    batch, _, height, width = canvas.shape
    background = frames.expand(batch, -1, -1, -1)
    canvas = F.grid_sample(
        canvas, distortion_grid.expand(batch, -1, -1, -1),
        mode="bilinear", padding_mode=distortion_padding_mode,
        align_corners=distortion_align_corners,
    )

    if blur_backend == "separable":
        accelerator = _select_accelerator(acceleration, canvas, background, ky, kx, mask_canvas)
        if accelerator is not None:
            return _optional_gaussian_kernels().separable_blur_blend_to_uint8(
                canvas, background, ky, kx, mask_canvas, legacy_alpha=legacy_alpha,
            )
        # Explicit eager/CPU reference: same zero-padded separable cross-correlation.
        canvas = F.conv2d(canvas, kx.view(1, 1, 1, -1).expand(4, 1, 1, -1),
                          padding=(0, kx.numel() // 2), groups=4)
        canvas = F.conv2d(canvas, ky.view(1, 1, -1, 1).expand(4, 1, -1, 1),
                          padding=(ky.numel() // 2, 0), groups=4)
        return _blend_canvas_eager(canvas, background, mask_canvas, legacy_alpha)

    if blur_backend == "triton":
        # Explicit opt-in: never silently benchmark a PyTorch fallback as Triton.
        accelerator = _select_accelerator("triton", canvas, background, motion_kernel, mask_canvas)
        return accelerator.blur_blend_to_uint8(
            canvas, background, motion_kernel, mask_canvas, legacy_alpha=legacy_alpha,
        )

    kh, kw = motion_kernel.shape[-2:]
    padding = (kh // 2, kw // 2)
    if motion_kernel.ndim == 4:
        # batch 的各组互不混合，4 个 RGBA 通道独立卷积。
        canvas = F.conv2d(canvas, motion_kernel, padding=padding, groups=4)
    else:
        # 将 B 放入通道：一次 grouped conv 实现 B 组不同的模糊核。
        canvas = F.conv2d(
            canvas.reshape(1, batch * 4, height, width),
            motion_kernel.reshape(batch * 4, 1, kh, kw),
            padding=padding, groups=batch * 4,
        ).reshape(batch, 4, height, width)

    accelerator = _select_accelerator(acceleration, canvas, background, mask_canvas)
    if accelerator is not None:
        return accelerator.blend_to_uint8(
            canvas, background, mask_canvas, legacy_alpha=legacy_alpha,
        )

    return _blend_canvas_eager(canvas, background, mask_canvas, legacy_alpha)


def _blend_canvas_eager(canvas, background, mask_canvas, legacy_alpha):
    alpha = canvas[:, 3:4]
    color = canvas[:, :3]
    if legacy_alpha:
        color = color * alpha
    blended = color + background * (1.0 - alpha)
    blended = background * mask_canvas + blended * (1.0 - mask_canvas)
    return blended.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def make_paste_callable(bboxes: Sequence[Sequence[int]], **options) -> Callable:
    """初始化一次：固定 bbox 与模式，动态 tensor 仍作为函数参数传入。"""
    boxes = tuple(tuple(int(v) for v in box) for box in bboxes)
    return partial(paste_gpu_batch, bboxes=boxes, **options)


# -------------------- 按素材尺寸分桶的进阶版本 --------------------

def _plan_buckets(source_hws, bboxes, canvas_hw, area_caps=None):
    """纯 CPU 布局规划；顺序覆盖生成 LUT，无 PyTorch 依赖。"""
    height, width = map(int, canvas_hw)
    boxes = tuple(tuple(int(v) for v in box) for box in bboxes)
    shapes = tuple(tuple(int(v) for v in shape) for shape in source_hws)
    if height < 1 or width < 1 or not boxes or len(boxes) != len(shapes):
        raise ValueError("need positive canvas and equal, nonempty source_hws/bboxes")
    caps = tuple(sorted(set(int(c) for c in area_caps))) if area_caps is not None else None
    if caps is not None and (not caps or caps[0] < 1):
        raise ValueError("area_caps must contain positive capacities")
    groups_by_key = {}
    areas = []
    for i, ((src_h, src_w), (x1, y1, x2, y2)) in enumerate(zip(shapes, boxes)):
        if src_h < 1 or src_w < 1 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("source sizes must be positive; bboxes must be inside canvas")
        area = (y2-y1)*(x2-x1)
        cap = next((c for c in caps if c >= area), None) if caps else None
        if caps and cap is None:
            raise ValueError("ROI exceeds largest area cap")
        groups_by_key.setdefault((src_h, src_w, cap), []).append(i)
        areas.append(area)
    groups, lengths = [], []
    offsets = [0]*len(boxes)
    cursor = 1  # palette 第 0 个像素为全零哨兵，供画布空白处读取。
    for key, members in groups_by_key.items():
        capacity = key[2] if key[2] is not None else max(areas[i] for i in members)
        groups.append(tuple(members))
        lengths.append(capacity)
        for local_id, slot_id in enumerate(members):
            offsets[slot_id] = cursor + local_id*capacity
        cursor += len(members)*capacity

    lookup = array("q", [0])*(height*width)
    # 始终按原广告位顺序写，而非按分桶后顺序写，保留重叠覆盖语义。
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        roi_w = x2-x1
        for local_y in range(y2-y1):
            dst = (y1+local_y)*width+x1
            src = offsets[i]+local_y*roi_w
            lookup[dst:dst+roi_w] = array("q", range(src, src+roi_w))
    return dict(groups=tuple(groups), lengths=tuple(lengths), offsets=tuple(offsets),
                bboxes=boxes, source_hws=shapes, canvas_hw=(height, width),
                lookup=lookup, padding_ratio=(cursor-1)/sum(areas))


def make_bucket_layout(ads_by_slot, bboxes, canvas_hw, *, area_caps=None):
    """初始化一次；返回布局字典和 CUDA/CPU int64 canvas_lut。

    默认按源(h,w)分桶，桶容量为桶内最大 ROI 像素数。area_caps 可用如
    (16384, 65536, 262144) 增加面积子桶，减少补齐浪费，但增加采样次数。
    ROI 网格展平为1×L，因此不存在 max_h×max_w 的交叉膨胀。
    每组共用 bbox；LUT只依赖布局和覆盖顺序，与透明度/素材颜色无关。

    padding_ratio=实际分配采样像素/未补齐ROI像素，不含其他内存成本。
    这是初始化工具，不能放到 torch.compile / CUDA Graph 捕获范围内。
    """
    if not ads_by_slot:
        raise ValueError("bucketed path requires at least one ad slot")
    plan = _plan_buckets([ad.shape[-2:] for ad in ads_by_slot], bboxes, canvas_hw, area_caps)
    lookup = plan.pop("lookup")
    if lookup.itemsize != 8:
        raise RuntimeError("int64 lookup requires 8-byte array('q')")
    plan["canvas_lut"] = torch.frombuffer(lookup, dtype=torch.int64).to(
        device=ads_by_slot[0].device
    ).reshape(*plan["canvas_hw"])
    return plan


@torch.inference_mode()
def pack_bucket_ads(ads_by_slot, layout):
    """仅素材变化时调用；输出每桶[B,M,4,src_h,src_w] float32。"""
    if len(ads_by_slot) != len(layout["bboxes"]):
        raise ValueError("slot count differs from layout")
    batch, device = ads_by_slot[0].shape[0], ads_by_slot[0].device
    for ad, shape in zip(ads_by_slot, layout["source_hws"]):
        if ad.shape != (batch, 4, *shape) or ad.dtype != torch.float32 or ad.device != device:
            raise ValueError("ads must match layout and share batch/device/float32")
    return tuple(torch.stack([ads_by_slot[i] for i in ids], dim=1)
                 for ids in layout["groups"])


@torch.inference_mode()
def pack_bucket_grids(warp_grids, layout, batch_size):
    """图外适配旧网格：每桶[B,M,L,2]。几何不变时缓存。

    视频高频更新时，最好直接批量生成/填充此格式的网格，避免每帧逐slot
    pad/stack带来的开销。这里只提供兼容旧网格的方便入口。
    """
    if len(warp_grids) != len(layout["bboxes"]):
        raise ValueError("grid count differs from layout")
    packed = []
    for ids, length in zip(layout["groups"], layout["lengths"]):
        grids = []
        for i in ids:
            x1, y1, x2, y2 = layout["bboxes"][i]
            if not 0 < (y2-y1)*(x2-x1) <= length:
                raise ValueError("ROI exceeds bucket capacity or is empty; rebuild layout")
            grid = warp_grids[i]
            if grid.ndim != 4 or grid.shape[0] not in (1, batch_size) or grid.shape[1:] != (y2-y1, x2-x1, 2):
                raise ValueError("warp grid must match bbox with batch 1 or B")
            if grid.dtype != torch.float32 or grid.device != layout["canvas_lut"].device:
                raise ValueError("warp grids must be float32 on layout device")
            flat = grid.expand(batch_size, -1, -1, -1).reshape(batch_size, -1, 2)
            grids.append(F.pad(flat, (0, 0, 0, length-flat.shape[1]), value=2.0))
        packed.append(torch.stack(grids, dim=1))
    return tuple(packed)


@torch.inference_mode()
def pack_bucket_transparencies(transparencies, layout, batch_size):
    """透明度变化时调用；每桶[B,M,1,L]，补齐部分为0。"""
    if len(transparencies) != len(layout["bboxes"]):
        raise ValueError("transparency count differs from layout")
    packed = []
    for ids, length in zip(layout["groups"], layout["lengths"]):
        values = []
        for i in ids:
            x1, y1, x2, y2 = layout["bboxes"][i]
            if not 0 < (y2-y1)*(x2-x1) <= length:
                raise ValueError("ROI exceeds bucket capacity or is empty; rebuild layout")
            value = transparencies[i]
            if value.dtype != torch.float32 or value.device != layout["canvas_lut"].device:
                raise ValueError("transparencies must be float32 on layout device")
            flat = value.expand(batch_size, 1, y2-y1, x2-x1).reshape(batch_size, 1, -1)
            values.append(F.pad(flat, (0, length-flat.shape[-1]), value=0.0))
        packed.append(torch.stack(values, dim=1))
    return tuple(packed)


def paste_gpu_bucketed(
    frames, ads_buckets, warp_grids_buckets, distortion_grid,
    motion_kernel, mask_canvas, transparencies_buckets, canvas_lut,
    *, warp_align_corners=False, distortion_align_corners=False,
    warp_padding_mode="zeros", distortion_padding_mode="zeros", legacy_alpha=True,
    acceleration="eager", blur_backend="torch", blur_factors=None,
):
    """仅有分桶循环的纯Tensor核心：K次透视采样+一次索引取值合成画布。

    ads_buckets[g]: [B,M_g,4,src_h,src_w]，建议预先缓存。
    warp_grids_buckets[g]: [B,M_g,L_g,2]。
    transparencies_buckets[g]: [B,M_g,1,L_g]；也可用可广播的标量布局。
    canvas_lut: [H,W] int64，包含0哨兵和对应palette像素索引。
    其他输入与 paste_gpu_batch 一致；外层使用 torch.inference_mode()。

    本函数不修改输入，支持单独 torch.compile 或 StaticPasteGraph。
    bbox不作为Python常量参与核心计算：若桶归属/容量/输入shape不变，
    位置与ROI形状变化可通过更新grid/透明度/LUT数值表示。但适配器的
    layout仍须正确维护；不要每帧按新面积重新分桶并声称无需重编译。

    性能代价：各桶patch并存、维度重排/拼接、全画布gather和padding。
    不保证优于逐slot版本；把图外网格packing也纳入端到端基准。
    acceleration 对最终融合/uint8/BHWC 输出启用 Triton；分桶采样不变。
    blur_backend='triton' 额外将模糊与最终融合合并，限制同 paste_gpu_batch。
    """
    if blur_backend not in ("torch", "triton", "separable"):
        raise ValueError("blur_backend must be 'torch', 'triton' or 'separable'")
    if not ads_buckets or not len(ads_buckets) == len(warp_grids_buckets) == len(transparencies_buckets):
        raise ValueError("need equal, nonempty bucket tuples")
    if canvas_lut.shape != frames.shape[-2:] or canvas_lut.dtype != torch.int64 or canvas_lut.device != frames.device:
        raise ValueError("canvas_lut must be int64 [H,W] on frame device")
    batch = ads_buckets[0].shape[0]
    palette_parts = [frames.new_zeros((batch, 4, 1))]
    for source, grids, opacity in zip(ads_buckets, warp_grids_buckets, transparencies_buckets):
        _, members, _, src_h, src_w = source.shape
        length = grids.shape[2]
        if not legacy_alpha:
            source_alpha = source[:, :, 3:4] * (1.0 / 255.0)
            source = torch.cat((source[:, :, :3]*source_alpha, source_alpha), dim=2)
        warped = F.grid_sample(
            source.reshape(batch*members, 4, src_h, src_w),
            grids.reshape(batch*members, 1, length, 2),
            mode="bilinear", padding_mode=warp_padding_mode,
            align_corners=warp_align_corners,
        ).reshape(batch, members, 4, length)
        if legacy_alpha:
            alpha = warped[:, :, 3:4] * (1.0 / 255.0) * opacity
            patch = torch.cat((warped[:, :, :3]*alpha, alpha), dim=2)
        else:
            patch = warped*opacity
        # 不能直接reshape(B,4,M*L)，必须先交换M/C维。
        palette_parts.append(patch.permute(0, 2, 1, 3).reshape(batch, 4, members*length))

    palette = torch.cat(palette_parts, dim=2)
    height, width = frames.shape[-2:]
    # 每个画布像素只读其最终所有者，没有重复目的地址的scatter竞争。
    canvas = torch.index_select(palette, 2, canvas_lut.reshape(-1)).reshape(batch, 4, height, width)
    return _finish_canvas(
        canvas, frames, distortion_grid, motion_kernel, mask_canvas,
        distortion_align_corners, distortion_padding_mode, legacy_alpha,
        acceleration=acceleration, blur_backend=blur_backend, blur_factors=blur_factors,
    )


@torch.inference_mode()
def stack_img_groups(
    imgs_raw_groups: Sequence[dict],
    ad_params: Sequence[dict],
    *,
    device,
    image_index: int = 0,
) -> tuple[torch.Tensor, ...]:
    """把多份原 self.imgs_raw 格式的数据按广告位堆叠，仅在素材变化时调用。

    imgs_raw_groups[b] 与原 self.imgs_raw 结构一致：
      group[(ad_placement_id, ad_id)] 或 group[ad_placement_id]
      -> {'img': [HWC_RGBA_tensor_for_device_0, ...]}

    image_index 对应原代码 (way_id-1) % len(self.device)。各组采用同样的
    slot 顺序；同一 slot 的 h,w 必须一致。允许 numpy 或 torch 图像。
    这里的初始化循环不会逐组运行 warp/blur/composite 整条渲染流程。
    """
    if not imgs_raw_groups:
        raise ValueError("need at least one image group")
    slots = []
    for param in ad_params:
        placement_id, slot_id = param["ad_placement_id"], param["ad_id"]
        images = []
        for group in imgs_raw_groups:
            # 不使用 dict.get(key, group[placement_id])，避免提前计算 fallback。
            key = (placement_id, slot_id)
            record = group[key] if key in group else group[placement_id]
            source = torch.as_tensor(record["img"][image_index], device=device, dtype=torch.float32)
            if source.ndim != 3 or source.shape[-1] != 4:
                raise ValueError("source images must be HWC RGBA/BGRA")
            if images and source.shape != images[0].shape:
                raise ValueError(f"image shapes differ across groups for slot {key}")
            images.append(source)
        slots.append(torch.stack(images, dim=0).permute(0,3,1,2).contiguous())
    return tuple(slots)


def validate_inputs(args: tuple, bboxes: Sequence[Sequence[int]]) -> None:
    """图外检查结构；不读取 GPU 数值，不检查值域或核归一化。"""
    frames, ads, grids, distortion, kernel, mask, opacities = args
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must be [1|B,3,H,W]")
    if not len(ads) == len(grids) == len(opacities) == len(bboxes):
        raise ValueError("ads, grids, transparencies and bboxes must have equal lengths")
    batch = ads[0].shape[0] if ads else frames.shape[0]
    height, width = frames.shape[-2:]
    if batch < 1 or frames.shape[0] not in (1, batch):
        raise ValueError("invalid batch size")
    for tensor in _tensor_leaves(args):
        if tensor.device != frames.device or tensor.dtype != torch.float32:
            raise ValueError("all tensors must be float32 on the same device")
    for i, (ad, grid, opacity, box) in enumerate(zip(ads, grids, opacities, bboxes)):
        x1, y1, x2, y2 = box
        if not all(isinstance(v, int) for v in box):
            raise ValueError("bboxes must contain Python integers")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"bbox {i} must be nonempty and inside canvas")
        if ad.ndim != 4 or ad.shape[:2] != (batch, 4):
            raise ValueError(f"ad {i} must be [B,4,h,w]")
        if grid.ndim != 4 or grid.shape[0] not in (1, batch) or grid.shape[1:] != (y2-y1, x2-x1, 2):
            raise ValueError(f"grid {i} must be [1|B,roi_h,roi_w,2]")
        if opacity.ndim != 4 or opacity.shape[0] not in (1, batch) or opacity.shape[1] != 1 or opacity.shape[2] not in (1, y2-y1) or opacity.shape[3] not in (1, x2-x1):
            raise ValueError(f"transparency {i} must broadcast to [B,1,roi_h,roi_w]")
    if distortion.ndim != 4 or distortion.shape[0] not in (1, batch) or distortion.shape[1:] != (height, width, 2):
        raise ValueError("distortion_grid must be [1|B,H,W,2]")
    if mask.ndim != 4 or mask.shape[0] not in (1, batch) or mask.shape[1] not in (1, 3) or mask.shape[-2:] != (height, width):
        raise ValueError("mask_canvas must be [1|B,1|3,H,W]")
    if kernel.ndim == 4:
        good_kernel = kernel.shape[:2] == (4, 1)
    elif kernel.ndim == 5:
        good_kernel = kernel.shape[:3] == (batch, 4, 1)
    else:
        good_kernel = False
    if not good_kernel or any(k <= 0 or k % 2 == 0 for k in kernel.shape[-2:]):
        raise ValueError("kernel must be [4,1,odd_h,odd_w] or [B,4,1,odd_h,odd_w]")


def homography_roi_grid(
    inverse_matrix: torch.Tensor,
    source_hw: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    matrix_uses_canvas_xy: bool,
    align_corners: bool = False,
    eps: float = 1e-8,
    acceleration: str = "eager",
) -> torch.Tensor:
    """无缓存 grid 时的可选建网格函数，可批量处理 m_inv。

    inverse_matrix: float32 GPU [3,3] 或 [B,3,3]，把输出坐标映射到素材像素。
    matrix_uses_canvas_xy=True: m_inv 接收全画布坐标 (x+x1,y+y1)。
    False: m_inv 接收 ROI 局部坐标 (x,y)。必须核对旧 warp 函数的约定！
    返回 [1或B,roi_h,roi_w,2]，与上述 grid_sample 直接配合。
    不负责复刻未知的第三方 warp 实现；已有 grid 时优先直接复用。
    acceleration='auto'/'triton' 可用单个 kernel 建网格，矩阵值/ROI 尺寸/
    bbox 坐标均为动态参数；矩阵变化不要求每帧重新编译 Triton kernel。
    """
    if inverse_matrix.dtype != torch.float32:
        raise ValueError("inverse_matrix must be float32")
    matrix = inverse_matrix.unsqueeze(0) if inverse_matrix.ndim == 2 else inverse_matrix
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("inverse_matrix must be [3,3] or [B,3,3]")
    src_h, src_w = source_hw
    if src_h < 1 or src_w < 1 or (align_corners and min(src_h, src_w) < 2):
        raise ValueError("positive source sizes required; align_corners=True requires both >=2")
    x1, y1, x2, y2 = bbox
    roi_h, roi_w = y2-y1, x2-x1
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError("bbox must be nonempty")
    accelerator = _select_accelerator(acceleration, matrix)
    if accelerator is not None:
        return accelerator.build_homography_grid(
            matrix, source_hw, bbox, matrix_uses_canvas_xy=matrix_uses_canvas_xy,
            align_corners=align_corners, eps=eps,
        )
    y, x = torch.meshgrid(
        torch.arange(roi_h, device=matrix.device, dtype=matrix.dtype),
        torch.arange(roi_w, device=matrix.device, dtype=matrix.dtype), indexing="ij"
    )
    if matrix_uses_canvas_xy:
        x, y = x+x1, y+y1
    coordinates = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(1, 3, -1)
    projected = matrix @ coordinates
    z = projected[:, 2:3]
    valid = (z.abs() > eps) & torch.isfinite(projected).all(dim=1, keepdim=True)
    safe_z = torch.where(valid, z, torch.ones_like(z))
    xy = projected[:, :2] / safe_z
    sx, sy = xy[:, 0], xy[:, 1]
    if align_corners:
        gx, gy = 2*sx/(src_w-1)-1, 2*sy/(src_h-1)-1
    else:
        gx, gy = (2*sx+1)/src_w-1, (2*sy+1)/src_h-1
    grid = torch.stack((gx, gy), dim=-1)
    # 避免把无穷远/NaN 误变成图像内采样；使用 zeros padding 才会置零。
    valid_grid = valid.transpose(1, 2) & torch.isfinite(grid).all(dim=-1, keepdim=True)
    grid = torch.where(valid_grid, grid, torch.full_like(grid, 1e6))
    return grid.reshape(matrix.shape[0], roi_h, roi_w, 2)


def _tensor_leaves(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, (tuple, list)):
        for item in tree:
            yield from _tensor_leaves(item)
    else:
        raise TypeError("argument tree must contain only tensors and tuple/list containers")


def _clone_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone(memory_format=torch.contiguous_format)
    return tuple(_clone_tree(item) for item in tree)


def _copy_tree(destination, source):
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor) or destination.shape != source.shape or destination.dtype != source.dtype or destination.device != source.device:
            raise ValueError("CUDA Graph replay requires unchanged tensor shape, dtype and device")
        destination.copy_(source)
    else:
        if not isinstance(source, (tuple, list)) or len(destination) != len(source):
            raise ValueError("CUDA Graph replay requires unchanged tuple/list structure")
        for dst, src in zip(destination, source):
            _copy_tree(dst, src)


class StaticPasteGraph:
    """可选的显式 CUDA Graph 包装器；通常先试 torch.compile 即可。

    fn 使用 make_paste_callable 返回的 eager callable。不要在这里嵌套
    reduce-overhead / 自动 CUDA Graph 编译模式。捕获期间其他线程不能
    在此进程提交 CUDA 工作。对象非线程安全，需在创建时同一流上串行调用。

    __call__ 会在图外复制新输入；仅支持固定结构/shape/bbox/kernel size。
    图捕获后修改 fn 或其 Python 配置无效，必须重建对象。
    copy_output=False 返回内部缓冲区，下次 replay 会覆写；默认 True 为
    独立结果（多一次 GPU 拷贝）。copy_input 开销也应包含在基准测试中。
    """

    @torch.inference_mode()
    def __init__(self, fn: Callable, example_args: tuple, warmup: int = 3):
        tensors = tuple(_tensor_leaves(example_args))
        if not tensors or any(t.device.type != "cuda" for t in tensors):
            raise ValueError("StaticPasteGraph requires CUDA tensors")
        self.device = tensors[0].device
        if any(t.device != self.device for t in tensors):
            raise ValueError("all inputs must be on one CUDA device")
        if warmup < 1:
            raise ValueError("at least one warmup iteration required")
        self.fn = fn
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.current_stream(self.device)
            self.inputs = _clone_tree(example_args)
            side_stream = torch.cuda.Stream(device=self.device)
            side_stream.wait_stream(self.stream)
            with torch.cuda.stream(side_stream):
                for _ in range(warmup):
                    warm_output = self.fn(*self.inputs)
            self.stream.wait_stream(side_stream)
            torch.cuda.synchronize(self.device)
            del warm_output
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side_stream):
                self.output = self.fn(*self.inputs)
            self.stream.wait_stream(side_stream)

    @torch.inference_mode()
    def __call__(self, *new_args, copy_output: bool = True):
        with torch.cuda.device(self.device):
            if torch.cuda.current_stream(self.device) != self.stream:
                raise RuntimeError("use the same CUDA stream as construction; object is not concurrent")
            _copy_tree(self.inputs, new_args)
            self.graph.replay()
            return self.output.clone() if copy_output else self.output


BUCKET_EXAMPLE = r'''
# 复用 EXAMPLE 中准备的 frames/ads_by_slot/grids/distortion/kernel/mask/
# transparencies/bboxes；源数据结构仍与旧版相同。
import torch
from functools import partial
from paste_gpu_batch import (
    make_bucket_layout, pack_bucket_ads, pack_bucket_grids,
    pack_bucket_transparencies, paste_gpu_bucketed,
)

B = ads_by_slot[0].shape[0]
layout = make_bucket_layout(ads_by_slot, bboxes, frames.shape[-2:])
# padding_ratio 过大时可另试面积子桶，例如：
# layout = make_bucket_layout(ads_by_slot, bboxes, frames.shape[-2:],
#                             area_caps=(16384, 65536, 262144))
print('buckets:', len(layout['groups']), 'padding ratio:', layout['padding_ratio'])

# 以下在初始化或相应内容变化时执行。不要每帧重复pack不变的素材。
ads_buckets = pack_bucket_ads(ads_by_slot, layout)
grids_buckets = pack_bucket_grids(grids, layout, B)
opacity_buckets = pack_bucket_transparencies(transparencies, layout, B)
args_bucketed = (
    frames, ads_buckets, grids_buckets, distortion, kernel, mask,
    opacity_buckets, layout['canvas_lut'],
)

bucket_paste = partial(paste_gpu_bucketed,
    warp_align_corners=False, distortion_align_corners=False, legacy_alpha=True)
with torch.inference_mode():
    output_gpu = bucket_paste(*args_bucketed)

# 可选：先验证eager，再编译；需与旧版相同的align_corners/padding/alpha配置。
fast_bucket_paste = torch.compile(bucket_paste, fullgraph=True,
                                  dynamic=False, mode='reduce-overhead')
with torch.inference_mode():
    output_gpu = fast_bucket_paste(*args_bucketed)

# 与旧版比较时，把pack网格、LUT生成、D2H成本也按实际缓存策略计入。
# LUT为int64全画布：1080p约16.6MB，动态bbox不要忽略逐帧CPU重建/上传。
# 高性能动态布局需固定slot桶归属/容量，并在GPU上更新grid+LUT内容。
# 当前make_bucket_layout只提供初始化/布局变化时的CPU规划工具。
'''


EXAMPLE = r'''
# 假设现有 frame_input 为 CUDA HWC float/uint8，B 组素材已按广告位堆好。
# 素材缓存: ads_by_slot[i] = [B,4,h_i,w_i], float32, CUDA。
# 不要在每帧重复 stack 不变素材；换素材时更新/重建该缓存。

import torch
from paste_gpu_batch import stack_img_groups, make_paste_callable, validate_inputs, StaticPasteGraph

B = len(imgs_raw_groups)  # 每个元素与原 self.imgs_raw 的结构相同。
ad_params = data_dict.get(f"ad_params_{way_id}")
if ad_params is None:
    ad_params = data_dict.get("ad_params")
if not ad_params:
    # B 为期望的结果数；无广告时直接批量返回背景。
    result = frame_input.unsqueeze(0).expand(B, -1, -1, -1).clamp(0,255).byte().contiguous()
else:
    device = frame_input.device
    # 仅初始化或素材变化时运行，正常逐帧处理复用此缓存。
    ads_by_slot = stack_img_groups(imgs_raw_groups, ad_params, device=device,
                                  image_index=(way_id-1) % len(self.device))
    bboxes = tuple(tuple(int(v) for v in p["bbox"]) for p in ad_params)
    frames = frame_input.permute(2,0,1).unsqueeze(0).to(torch.float32).contiguous()

    # 原 paste_args 中已有的两个 grid 都必须是反向采样的 normalized grid。
    grids = tuple(ad_warp_grids_dict[p["ad_id"]].to(device=device, dtype=torch.float32)
                  for p in ad_params)
    # 若某个 grid 是 None，先在图外用原 warp 的网格构造逻辑补全，
    # 或使用 homography_roi_grid（先确认 canvas/ROI 坐标约定）。
    distortion = distortion_grid.to(device=device, dtype=torch.float32)
    kernel = motion_blur_kernel_4d.to(device=device, dtype=torch.float32).contiguous()

    # 假设原 mask 为 [H,W,1] 或 [H,W,3]；[H,W] 要先 unsqueeze(-1)。
    old_mask = mask_canvas if mask_canvas.ndim == 3 else mask_canvas.unsqueeze(-1)
    mask = old_mask.permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float32).contiguous()

    # 示例为各广告位一个共享标量。逐组标量改成 [B,1,1,1]；
    # 原 [roi_h,roi_w,1] 透明度图需要转换成 [1,1,roi_h,roi_w]。
    transparencies = tuple(torch.as_tensor(p.get("transparency",1.0),device=device,
                                          dtype=torch.float32).reshape(1,1,1,1)
                           for p in ad_params)
    args = (frames, ads_by_slot, grids, distortion, kernel, mask, transparencies)
    validate_inputs(args, bboxes)

    # 初始化/配置变化时构造；每帧不要重新构造和 compile。
    paste = make_paste_callable(bboxes,
        warp_align_corners=False,       # 改成原 warp_perspective_torch 的值！
        distortion_align_corners=False, # 改成原 apply_distortion 的值！
        legacy_alpha=True)

    # 方案 A：批处理 eager。
    with torch.inference_mode():
        result = paste(*args)

    # 方案 B：编译。编译/预热只在初始化做，首次耗时不能算稳态延迟。
    fast_paste = torch.compile(paste, fullgraph=True, dynamic=False, mode="reduce-overhead")
    with torch.inference_mode():
        for _ in range(3):
            result = fast_paste(*args)
        # 实际每帧更新 args 中的 tensor 后：
        result = fast_paste(*args)

    # 方案 C：显式 CUDA Graph（与 B 分开测试，别嵌套 capture）。
    graph_paste = StaticPasteGraph(paste, args)
    result = graph_paste(*args)  # 默认复制输出，避免下次 replay 覆盖。

    # GPU uint8 [B,H,W,3]；如果后续仍在 GPU 上处理就直接使用 result。
    # 必须 NumPy 时只在整个 batch 最后做一次 D2H：
    result_numpy = result.cpu().numpy()
'''

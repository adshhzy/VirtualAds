"""Public frame rendering API: initialize once, render each frame, release at shutdown.

SharedParameterBuilder computes live camera/geometry/kernel inputs. RenderExecutor
owns material preparation, bounded caches, execution policy and independent output.
The existing owner attributes remain the single storage locations so legacy and
new entry points use exactly the same objects, rather than parallel worker state.
"""

from dataclasses import replace

if __package__:
    from .paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
    from .paste_render_executor import (
        RENDERER_VERSION as RENDERER_VERSION,
        RenderExecutor as RenderExecutor,
        RenderOptions as RenderOptions,
        _image_plan,
    )
    from .paste_shared_params import SharedParameterBuilder
else:
    from paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
    from paste_render_executor import (
        RENDERER_VERSION as RENDERER_VERSION,
        RenderExecutor as RenderExecutor,
        RenderOptions as RenderOptions,
        _image_plan,
    )
    from paste_shared_params import SharedParameterBuilder


def initialize_renderer(
    owner,
    base_options,
    *,
    use_separable=False,
    normalization="legacy",
    camera_index_policy="legacy",
    distortion_inverse=True,
    max_cached_sizes=3,
):
    """Initialize one serial GPU worker, preserving the existing shared setup choices.

    base_options describes the business's image layout, scaling, alpha, padding
    and alignment conventions. As in setup_shared_paste, this integration selects
    batch/Triton, ROI-local matrices and independent pinned NumPy output. Its
    batch_chunk_size is preserved, including the default offline 'auto' policy.

    Set use_separable only after checking the real business Gaussian callback.
    No production motion/camera/Gaussian law or camera history is replaced here.
    Return the executor for cache/execution diagnostics. The two legacy owner
    attributes remain authoritative; callers need not manipulate them to render.
    """
    builder = SharedParameterBuilder(
        acceleration="triton",
        normalization=normalization,
        distortion_inverse=distortion_inverse,
        iterations=10,
        distortion_align_corners=base_options.distortion_align_corners,
        camera_index_policy=camera_index_policy,
        enable_separable=use_separable,
    )
    options = replace(
        base_options,
        backend="batch",
        acceleration="triton",
        blur_backend="separable" if use_separable else "triton",
        warp_grids_builder=builder.build_warp_grids,
        grid_source="matrix",
        matrix_coordinates="roi",
        pinned_output=True,
        return_numpy=True,
        squeeze_single=True,
        max_cached_sizes=max_cached_sizes,
        profile_every=0,
    )
    executor = RenderExecutor(options)
    owner._paste_batch_adapter = executor
    owner._shared_paste_builder = builder
    return executor


def _prepare_shared_parameters(
    owner, frame_input, data_dict, frame_id, mask_canvas, *, device=None, camera_result=None
):
    """Build the original six-item tuple once; ROI grids follow material collection.

    Kept separate for callers of the old explicit shared-preparation interface,
    including callers that share camera_result across several GPU workers.
    """
    if frame_input.ndim not in (3, 4):
        raise ValueError("frame_input must be HWC/BHWC or CHW/BCHW")
    layout = owner._paste_batch_adapter.options.frame_layout
    frame_plan = _image_plan(frame_input.shape, 3, layout, "frame_input")
    height, width = frame_plan[1][-2:]
    if device is None:
        device = getattr(frame_input, "device", None)
        if device is None:
            raise ValueError("device is required for frames without a tensor device")
    return owner._shared_paste_builder.prepare(
        owner,
        data_dict,
        frame_id,
        height=height,
        width=width,
        device=device,
        mask_canvas=mask_canvas,
        camera_result=camera_result,
    )


def render_frame(
    owner,
    frame_input,
    data_dict,
    frame_id,
    *,
    mask_canvas,
    material_groups=None,
    camera_history=None,
    device=None,
    way_id=1,
    camera_result=None,
):
    """Render one frame through shared preparation, execution and NumPy export.

    Initialize the worker first. Pass the current nonempty material_groups to
    change B; None leaves owner.imgs_raw_groups/imgs_raw unchanged. B=0 is a
    scheduler decision and an explicit empty group list is rejected. In-place
    material changes still require owner.imgs_raw_version to advance.

    camera_history=None keeps the existing history. A supplied history is held
    by the business owner; this function neither appends records nor trims it.
    The business may explicitly use BoundedCameraHistory and add each real
    camera record before calling. camera_result can reuse an earlier CPU filter
    result across devices, following SharedParameterBuilder's existing contract.

    Non-None ad_params_{way_id} takes precedence. No-ad frames bypass all shared
    camera/kernel preparation, preserving the background-only path. With ads,
    preparation runs once for the full B, and the executor builds ROI grids only
    after actual material sizes are known. No geometry is cached by frame_id.

    Standard initialization returns independent ready-to-read NumPy uint8:
    HWC for B1 and BHWC for B>1. Use one owner per serial worker/compute stream.
    """
    if material_groups is not None:
        if not isinstance(material_groups, (list, tuple)) or not material_groups:
            raise ValueError(
                "material_groups must be a nonempty list/tuple; skip rendering for B=0"
            )
        owner.imgs_raw_groups = material_groups
    if camera_history is not None:
        owner.paste_data_stack = camera_history
    executor = owner._paste_batch_adapter
    parameters = data_dict.get(f"ad_params_{way_id}")
    if parameters is None:
        parameters = data_dict.get("ad_params")
    if not isinstance(parameters, (list, tuple)) or not parameters:
        # Preserve the executor's original validation errors and no-ad fast path.
        return executor(owner, frame_input, data_dict, frame_id, way_id, None)

    paste_args = _prepare_shared_parameters(
        owner,
        frame_input,
        data_dict,
        frame_id,
        mask_canvas,
        device=device,
        camera_result=camera_result,
    )
    # Keep the existing callable dispatch, including business subclasses' hooks.
    # The standard executor forwards to render_prepared without additional state.
    return executor(owner, frame_input, data_dict, frame_id, way_id, paste_args)


def release_renderer(owner):
    """Release executor and shared-builder resources between calls or at shutdown.

    Keeps the initialized objects reusable and leaves caller-owned input, camera
    history and returned NumPy arrays valid. Do not call this in the frame loop.
    Release downstream queues/history according to their own business lifetime.
    """
    executor = getattr(owner, "_paste_batch_adapter", None)
    builder = getattr(owner, "_shared_paste_builder", None)
    try:
        if executor is not None:
            executor.release_resources()
    finally:
        if builder is not None:
            builder.release_resources()


__all__ = [
    "BoundedCameraHistory",
    "RENDERER_VERSION",
    "RenderExecutor",
    "RenderOptions",
    "initialize_renderer",
    "render_frame",
    "release_renderer",
]

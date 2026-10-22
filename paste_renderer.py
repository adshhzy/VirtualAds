"""Public frame rendering API: initialize once, render each frame, release at shutdown.

SharedParameterBuilder computes live camera/geometry/kernel inputs. RenderExecutor
owns material preparation, bounded caches, execution policy and independent output.
The existing owner attributes remain the single storage locations so legacy and
new entry points use exactly the same objects, rather than parallel worker state.
"""

from dataclasses import replace
from collections.abc import Mapping

import numpy as np
import torch

if __package__:
    from .paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
    from .paste_render_executor import (
        RENDERER_VERSION as RENDERER_VERSION,
        RenderExecutor as RenderExecutor,
        RenderOptions as RenderOptions,
        _as_tensor,
        _image_plan,
    )
    from .paste_shared_params import SharedParameterBuilder
    from .paste_shared_mask import SharedMaskBuilder
    from .paste_route_plan import build_route_plan
else:
    from paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
    from paste_render_executor import (
        RENDERER_VERSION as RENDERER_VERSION,
        RenderExecutor as RenderExecutor,
        RenderOptions as RenderOptions,
        _as_tensor,
        _image_plan,
    )
    from paste_shared_params import SharedParameterBuilder
    from paste_shared_mask import SharedMaskBuilder
    from paste_route_plan import build_route_plan


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


class ImageRenderer:
    """One live worker with the three-argument ``render_img`` business entry.

    Initialize once with the business's existing motion/Gaussian callbacks and
    input conventions. The owner is dedicated to this serial worker. Per-frame
    ad_params_N lists choose routes; imgs_raw maps exact (placement_id, ad_id)
    pairs to images (or legacy {'img': image} records). No global worker is used.

    Shared camera preparation and mask union/feathering run at most once per
    frame. Only compatible ordered layouts batch together; other layouts reuse
    the same shared inputs through separate executor calls. GPU math, slot
    overwrite order and the existing batch auto table remain unchanged.
    """

    def __init__(
        self,
        owner,
        base_options=None,
        *,
        device="cuda:0",
        use_separable=False,
        mask_sigma=1.5,
        mask_radius=3,
        history_capacity=9,
    ):
        for name in (
            "compute_camera_motion_rate",
            "motion_rate_to_sigma",
            "create_gaussian_kernel",
        ):
            if not callable(getattr(owner, name, None)):
                raise TypeError(f"owner.{name} must be the existing business callback")
        self.owner = owner
        self.device = SharedParameterBuilder._device(device)
        self.history = BoundedCameraHistory(capacity=history_capacity)
        self.mask_builder = SharedMaskBuilder(sigma=mask_sigma, radius=mask_radius)
        # The mask is built here in normalized HW form, irrespective of legacy
        # external mask conventions. Image/alpha/sampling options stay intact.
        options = replace(
            base_options if base_options is not None else RenderOptions(),
            mask_layout="HW",
            mask_scale=1.0,
        )
        self.executor = initialize_renderer(
            owner,
            options,
            use_separable=use_separable,
            max_cached_sizes=options.max_cached_sizes,
        )
        owner.paste_data_stack = self.history
        owner.imgs_raw_groups = []
        self.route_ids = ()
        self.last_execution = ()
        self._active = False
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise RuntimeError("ImageRenderer is closed; initialize a new worker")

    def _camera_inputs(self, data_dict, *, required):
        """Snapshot small CPU records without retaining a frame/config dictionary."""
        names = ("K", "w2c", "dist_coeffs")
        if not required and not all(name in data_dict for name in ("frame_id", *names)):
            return None
        record = {}
        for name in names:
            value = data_dict[name]
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    raise TypeError(f"{name} must be CPU camera data")
                value = value.detach().numpy()
            record[name] = np.array(value, copy=True)
        self.history[data_dict["frame_id"]] = record
        # The camera filter may replace current dist_coeffs. Neither the input
        # dictionary nor the history record is used as its mutable work dictionary.
        return {"K": record["K"], "dist_coeffs": record["dist_coeffs"].copy()}

    @torch.inference_mode()
    def render_img(self, data_dict, frame, imgs_raw):
        """Return ready NumPy uint8: HWC for B1, BHWC for B>1, (0,H,W,3) for B0.

        Routes are ordered by numeric N; ``route_ids`` gives their identities.
        Empty ad_params_N means a background output for that route. No route
        keys means pause: release active resources once and skip GPU rendering.
        In-place material edits require advancing data_dict['imgs_raw_version']
        or calling invalidate_assets(); replacing an image is detected directly.
        brightness, ad_paste_data, way_ad_map and image_path are not consulted.
        """
        self._require_open()
        if not isinstance(data_dict, Mapping):
            raise TypeError("data_dict must be a mapping")
        frame_plan = _image_plan(frame.shape, 3, self.executor.options.frame_layout, "frame")
        if frame_plan[1][0] != 1:
            raise ValueError("render_img expects one original frame, not a frame batch")
        height, width = frame_plan[1][-2:]
        plan = build_route_plan(data_dict, imgs_raw, (height, width), self.executor.options)
        has_ads = any(group.params for group in plan.groups)
        camera = self._camera_inputs(data_dict, required=has_ads)
        self.route_ids = plan.route_ids
        self.last_execution = ()
        if not self.route_ids:
            if self._active:
                self.release_resources()
            return np.empty((0, height, width, 3), dtype=np.uint8)

        frame_id = data_dict.get("frame_id", 0)
        self.owner.imgs_raw_version = data_dict.get("imgs_raw_version", 0)
        # Mark ownership before preparation so a later B0 also releases working
        # sets left by a failed frame when the caller elects to recover.
        self._active = True
        shared = None
        shared_frame = frame
        if has_ads:
            # Upload/convert the original background once even when routes need
            # multiple layout groups. Executors then reuse this float32 tensor.
            shared_frame = _as_tensor(frame, self.device)
            mask = self.mask_builder.build(
                data_dict.get("mask_coor_list", []),
                height=height,
                width=width,
                device=self.device,
            )
            shared = _prepare_shared_parameters(
                self.owner, frame, camera, frame_id, mask, device=self.device
            )

        # Most traffic has one compatible layout and returns the executor's
        # original independent output directly. Mixed layouts need one final
        # NumPy batch, with each group copied back into numeric route order.
        assembled = None
        if len(plan.groups) > 1:
            assembled = np.empty((len(self.route_ids), height, width, 3), dtype=np.uint8)
        decisions = []
        for group in plan.groups:
            self.owner.imgs_raw_groups = group.material_groups
            # Generated data has only the selected ordered slots; no legacy way
            # key can override it. Per-route geometry remains live in the plan.
            output = self.executor(
                self.owner, shared_frame, {"ad_params": group.params}, frame_id, 1, shared
            )
            decisions.append(dict(self.executor.last_execution))
            if assembled is None:
                self.last_execution = tuple(decisions)
                return output
            batch_output = output[None] if output.ndim == 3 else output
            try:
                for local_index, route_index in enumerate(group.indices):
                    assembled[route_index] = batch_output[local_index]
            finally:
                del output
                # Do not retain the previous group's entire batch while the next
                # executor call creates its own NumPy result.
                del batch_output
        self.last_execution = tuple(decisions)
        return assembled

    def invalidate_assets(self):
        """Explicitly invalidate cached pixels after editing source images in place."""
        self._require_open()
        self.executor.invalidate_assets()

    def release_resources(self):
        """Release GPU/cache working sets between calls; keep this worker reusable."""
        try:
            release_renderer(self.owner)
        finally:
            try:
                self.mask_builder.release_resources()
            finally:
                self.owner.imgs_raw_groups = []
                self._active = False
                self.last_execution = ()

    def reset_history(self):
        """Call at a seek/new camera stream before restarting frame IDs."""
        self._require_open()
        try:
            self.release_resources()
        finally:
            self.history.reset()

    def close(self):
        """Release this worker's caches/history; caller-owned images/outputs survive."""
        if self._closed:
            return
        try:
            self.release_resources()
        finally:
            self.history.clear()
            if self.owner.paste_data_stack is self.history:
                self.owner.paste_data_stack = None
            self.route_ids = ()
            self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


__all__ = [
    "BoundedCameraHistory",
    "RENDERER_VERSION",
    "RenderExecutor",
    "RenderOptions",
    "ImageRenderer",
    "initialize_renderer",
    "render_frame",
    "release_renderer",
]

"""Integration helpers. Initialize once INSIDE each GPU worker, then prepare once/frame.

The owner must provide the existing motion_rate_to_sigma, create_gaussian_kernel,
compute_camera_motion_rate callbacks and paste_data_stack. Their production laws
are deliberately not replaced by the benchmark's synthetic examples.
"""

from dataclasses import replace

if __package__:
    from .paste_gpu_batch_adapter import PasteBatchAdapter, _image_plan
    from .paste_shared_params import SharedParameterBuilder
    from .paste_camera_filter import compute_camera_filter_shared as compute_camera_filter_shared
    from .paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
else:
    from paste_gpu_batch_adapter import PasteBatchAdapter, _image_plan
    from paste_shared_params import SharedParameterBuilder
    from paste_camera_filter import compute_camera_filter_shared as compute_camera_filter_shared
    from paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory


def setup_shared_paste(
    owner,
    base_options,
    *,
    use_separable=False,
    normalization="legacy",
    camera_index_policy="legacy",
    distortion_inverse=True,
    max_cached_sizes=3,
):
    """Preserve existing align/padding/alpha/resize settings; select new preparation.

    Set use_separable only after the benchmark and the REAL callback's generated
    kernels pass validation. Dense triton supports odd kernel sizes <=15;
    separable triton supports verified rank-one shared kernels, odd sizes <=63.
    This helper does not replace or trim owner.paste_data_stack. For a live
    stream, explicitly create a BoundedCameraHistory once and add each raw camera
    record before preparation. Keep ordinary mappings for existing history rules.
    At shutdown call builder.release_resources(); it releases kernel/owner
    references but leaves caller-owned history and previously returned data alone.
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
    adapter_options = replace(
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
    owner._paste_batch_adapter = PasteBatchAdapter(adapter_options)
    owner._shared_paste_builder = builder
    return builder


def prepare_shared_paste_args(
    owner, frame_input, data_dict, frame_id, mask_canvas, *, device=None, camera_result=None
):
    """Call once for one frame/calibration/device/resolution; reuse returned tuple.

    For several GPUs, compute camera_result once with compute_camera_filter_shared,
    then pass it to each device's builder. Never cache by frame_id alone: callers
    may update camera estimates or image contents in-place for the same frame.
    Frame layout is read from the adapter: HWC/BHWC or CHW/BCHW is accepted.
    The returned six-item tuple contains a shared ``[1,H,W,2]`` distortion grid;
    each slot's current ROI grid is built later after actual source sizes are known.
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


# One-device integration:
# setup_shared_paste(self, self._paste_batch_adapter.options, use_separable=False)
# ... once per frame, replacing old camera/grid/kernel preparation ...
# paste_args = prepare_shared_paste_args(self, frame_tensor, data_dict, frame_id, mask_canvas)
# frame = paste_gpu_batch_adapter(self, frame_tensor, data_dict, frame_id, way_id, paste_args)
# self.send_ad_img(raw_frame_id, frame, way_id, data_dict)
#
# If several way_id calls truly share device, K, distortion, H/W and motion:
# reuse paste_args for them. Their ROI grids are still built from each way's
# current params and ACTUAL source sizes inside the adapter.
# A different mask can be supplied with:
# way_args = (*paste_args[:4], mask_for_way, paste_args[5])
#
# Multiple devices:
# camera_result = compute_camera_filter_shared(self, data_dict, frame_id, index_policy="legacy")
# ... each device's builder.prepare(..., camera_result=camera_result) ...
#
# Optional bounded live history, created once by the business (not per frame):
# camera_history = BoundedCameraHistory(capacity=9)
# self.paste_data_stack = camera_history
# ... each frame, in increasing ID order, BEFORE shared preparation ...
# camera_history[frame_id] = {
#     "K": current_K, "w2c": current_w2c, "dist_coeffs": current_cpu_coefficients,
# }
# Use a separate record dict from data_dict if retaining unfiltered coefficients.
# Copy producer-owned arrays at ingestion if their storage will be overwritten.
# Radius 4 needs 5 retained records when rendering the latest frame, or 9 for
# a centered window with 4-frame lookahead. Delayed consumers need more capacity.
# For a seek/new camera stream: camera_history.reset(); builder.clear_cache().
# At worker shutdown: builder.release_resources(); camera_history.clear().

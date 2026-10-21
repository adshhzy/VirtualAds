"""Compatibility helpers for the former explicit prepare-then-render integration.

New integrations use initialize_renderer/render_frame/release_renderer from
paste_renderer. These helpers retain their signatures and original return values.

The owner must provide the existing motion_rate_to_sigma, create_gaussian_kernel,
compute_camera_motion_rate callbacks and paste_data_stack. Their production laws
are deliberately not replaced by the benchmark's synthetic examples.
"""

if __package__:
    from .paste_gpu_batch_adapter import PasteBatchAdapter as PasteBatchAdapter
    from .paste_shared_params import SharedParameterBuilder as SharedParameterBuilder
    from .paste_renderer import _prepare_shared_parameters, initialize_renderer
    from .paste_camera_filter import compute_camera_filter_shared as compute_camera_filter_shared
    from .paste_camera_filter import BoundedCameraHistory as BoundedCameraHistory
else:
    from paste_gpu_batch_adapter import PasteBatchAdapter as PasteBatchAdapter
    from paste_shared_params import SharedParameterBuilder as SharedParameterBuilder
    from paste_renderer import _prepare_shared_parameters, initialize_renderer
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
    At shutdown call paste_renderer.release_renderer(owner) to release both the
    builder and executor; caller-owned history and earlier outputs remain valid.
    """
    initialize_renderer(
        owner,
        base_options,
        use_separable=use_separable,
        normalization=normalization,
        camera_index_policy=camera_index_policy,
        distortion_inverse=distortion_inverse,
        max_cached_sizes=max_cached_sizes,
    )
    # The old helper returns the builder; initialize_renderer returns the executor.
    return owner._shared_paste_builder


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
    return _prepare_shared_parameters(
        owner,
        frame_input,
        data_dict,
        frame_id,
        mask_canvas,
        device=device,
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
# At worker shutdown: paste_renderer.release_renderer(owner); camera_history.clear().

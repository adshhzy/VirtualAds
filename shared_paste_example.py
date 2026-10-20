"""Integration helpers. Initialize once INSIDE each GPU worker, then prepare once/frame.

The owner must provide the existing motion_rate_to_sigma, create_gaussian_kernel,
compute_camera_motion_rate callbacks and paste_data_stack. Their production laws
are deliberately not replaced by the benchmark's synthetic examples.
"""
from dataclasses import replace

if __package__:
    from .paste_gpu_batch_adapter import PasteBatchAdapter, _image_plan
    from .paste_shared_params import SharedParameterBuilder
    from .paste_camera_filter import compute_camera_filter_shared
else:
    from paste_gpu_batch_adapter import PasteBatchAdapter, _image_plan
    from paste_shared_params import SharedParameterBuilder
    from paste_camera_filter import compute_camera_filter_shared


def setup_shared_paste(owner, base_options, *, use_separable=False,
                       normalization="legacy", camera_index_policy="legacy",
                       distortion_inverse=True, max_cached_sizes=3):
    """Preserve existing align/padding/alpha/resize settings; select new preparation.

    Set use_separable only after the benchmark and the REAL callback's generated
    kernels pass validation. Dense triton supports odd kernel sizes <=15;
    separable triton supports verified rank-one shared kernels, odd sizes <=63.
    """
    builder = SharedParameterBuilder(
        acceleration="triton", normalization=normalization,
        distortion_inverse=distortion_inverse, iterations=10,
        distortion_align_corners=base_options.distortion_align_corners,
        camera_index_policy=camera_index_policy, enable_separable=use_separable,
    )
    owner._paste_batch_adapter = PasteBatchAdapter(replace(
        base_options, backend="batch", acceleration="triton",
        blur_backend="separable" if use_separable else "triton",
        warp_grids_builder=builder.build_warp_grids, grid_source="matrix", matrix_coordinates="roi",
        pinned_output=True, return_numpy=True, squeeze_single=True,
        max_cached_sizes=max_cached_sizes, profile_every=0,
    ))
    owner._shared_paste_builder = builder
    return builder


def prepare_shared_paste_args(owner, frame_input, data_dict, frame_id, mask_canvas,
                              *, device=None, camera_result=None):
    """Call once for one frame/calibration/device/resolution; reuse returned tuple.

    For several GPUs, compute camera_result once with compute_camera_filter_shared,
    then pass it to each device's builder. Never cache by frame_id alone: callers
    may update camera estimates or image contents in-place for the same frame.
    """
    if frame_input.ndim not in (3, 4):
        raise ValueError("frame_input must be HWC/BHWC or CHW/BCHW")
    layout = owner._paste_batch_adapter.options.frame_layout
    plan = _image_plan(frame_input.shape, 3, layout, "frame_input")
    height, width = plan[1][-2:]
    if device is None:
        device = getattr(frame_input, "device", None)
        if device is None:
            raise ValueError("device is required for frames without a tensor device")
    return owner._shared_paste_builder.prepare(
        owner, data_dict, frame_id, height=height, width=width,
        device=device, mask_canvas=mask_canvas, camera_result=camera_result,
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

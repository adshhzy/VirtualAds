"""Compatibility entry points for changing the number of material groups.

New integrations use paste_renderer.render_frame directly. These wrappers keep
the old signatures and initialization return value for existing business code.

The caller supplies its existing owner and real camera/motion/Gaussian callbacks.
Use a separate owner per worker and call it serially on one CUDA compute stream.
This module does not generate camera history, replace callbacks, or send images.
For bounded live history, the business creates BoundedCameraHistory once, writes
each real camera record before rendering, and explicitly resets it on a seek.
"""

from dataclasses import replace

if __package__:
    from .paste_renderer import render_frame
    from .shared_paste_example import BoundedCameraHistory as BoundedCameraHistory
    from .shared_paste_example import prepare_shared_paste_args as prepare_shared_paste_args
    from .shared_paste_example import setup_shared_paste
else:
    from paste_renderer import render_frame
    from shared_paste_example import BoundedCameraHistory as BoundedCameraHistory
    from shared_paste_example import prepare_shared_paste_args as prepare_shared_paste_args
    from shared_paste_example import setup_shared_paste


def initialize_dynamic_worker(owner, base_options, *, use_separable=False, max_cached_sizes=3):
    """Call once inside the GPU worker, preserving its existing image conventions.

    base_options is the business's existing AdapterOptions: keep its channel
    layouts, alpha, padding, alignment, scaling and resizing settings. Shared
    preparation expects ROI-local inverse matrices, as in shared_paste_example.
    Set use_separable=True only after validating the real business kernel.
    Dense mode still supports changing B; auto chunking currently requires the
    measured separable configuration. No synthetic Gaussian callback is installed.
    Create history once separately: ``history = BoundedCameraHistory(capacity=9)``.
    Initialization does not convert, truncate or clear an existing business history.
    """
    return setup_shared_paste(
        owner,
        replace(base_options, batch_chunk_size="auto"),
        use_separable=use_separable,
        normalization="legacy",
        camera_index_policy="legacy",
        max_cached_sizes=max_cached_sizes,
    )


def render_dynamic_frame(
    owner,
    frame_input,
    data_dict,
    frame_id,
    *,
    material_groups,
    camera_history,
    mask_canvas,
    device,
    way_id=1,
):
    """Render one frame for the current nonempty list of material groups.

    B = len(material_groups), independent of way_id. Every group maps
    (ad_placement_id, ad_id) -> {"img": RGBA_tensor_or_array}; a list of
    per-device images is also accepted by the original adapter. Keep static
    image objects alive and reuse them across frames. Changing B or order does
    not require manual cache invalidation; in-place pixel edits do.

    data_dict contains current K, dist_coeffs and ad_params. A non-None
    ad_params_{way_id} list takes precedence over the common ad_params list.
    Each ad_param supplies IDs, bbox, ROI-local m_inv and optional transparency.
    camera_history maps integer frame IDs to K/w2c/dist_coeffs records. Include
    the current frame when filtering at least three eligible records; median
    coefficients must be CPU data. Data preparation may replace
    data_dict['dist_coeffs']; use a fresh frame dictionary when retaining raw data.
    A BoundedCameraHistory is optional: add ``history[frame_id] = camera_record``
    before this call in monotonic order. The record must contain the real CPU
    coefficients and K/w2c; it is held by reference, so copy producer buffers if
    they will be overwritten. With radius 4, retain at least 5 records for latest
    frame filtering, or 9 for a centered window with four future frames. Sparse
    histories may still take the original fewer-than-three-neighbors fallback.
    A seek/new stream requires history.reset() and builder.clear_cache(); this
    function never evicts records from an external ordinary mapping.

    The scheduler should skip this call for B=0. An empty ad_params list instead
    means B requested background outputs; the adapter bypasses shared preparation.
    Returns the original independent NumPy uint8 contract: HWC for B1, BHWC for B>1.
    """
    if not isinstance(material_groups, (list, tuple)) or not material_groups:
        raise ValueError("material_groups must be a nonempty list/tuple; skip rendering for B=0")

    owner.imgs_raw_groups = material_groups
    owner.paste_data_stack = camera_history
    # Assign history even when it is None, preserving this legacy API's behavior.
    # New render_frame callers may omit groups/history to reuse the owner's state.
    return render_frame(
        owner,
        frame_input,
        data_dict,
        frame_id,
        mask_canvas=mask_canvas,
        device=device,
        way_id=way_id,
    )


# Worker setup, once:
# builder = initialize_dynamic_worker(owner, existing_options)
# history = BoundedCameraHistory(capacity=9)
# Each incoming frame, including frames skipped by the rendering scheduler:
# history[frame_id] = {"K": current_K, "w2c": current_w2c,
#                      "dist_coeffs": current_cpu_coefficients}
# output = render_dynamic_frame(..., camera_history=history, ...)
# On shutdown, after the last render returns:
# paste_renderer.release_renderer(owner)
# history.clear()

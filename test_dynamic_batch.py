"""Exact-output regression checks for opt-in CUDA batch chunking.

Run serially with other GPU work stopped:
    python test_dynamic_batch.py --device cuda:0
    python test_dynamic_batch.py --device cuda:0 --profile-checks
    python test_dynamic_batch.py --device cpu

CPU execution checks option validation, the documented CUDA-only rejection,
and the no-ad early return; it explicitly skips CUDA equivalence checks.
Fixtures and Gaussian callbacks are synthetic test data, not production laws.
This script measures correctness only and reports no performance claims.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from benchmark_paste_gpu_batch import adapter_options, make_adapter_case, make_case
from paste_gpu_batch_adapter import AdapterOptions, PasteBatchAdapter
from paste_shared_types import MotionKernelSpec


HEIGHT, WIDTH = 37, 53  # Deliberately not divisible by typical GPU tile sizes.


def exact(actual, expected, label):
    """No uint8 tolerance: chunking must preserve the selected whole path."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: shape/dtype {actual.shape}/{actual.dtype} != "
            f"{expected.shape}/{expected.dtype}"
        )
    if not np.array_equal(actual, expected):
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        raise AssertionError(
            f"{label}: {np.count_nonzero(difference)} different values, "
            f"maximum uint8 difference={difference.max()}"
        )


def output_contract(value, batch, *, squeeze=True, height=HEIGHT, width=WIDTH):
    shape = (height, width, 3) if batch == 1 and squeeze else (batch, height, width, 3)
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise AssertionError("output must be a NumPy uint8 array")
    if value.shape != shape or not value.flags.c_contiguous:
        raise AssertionError(
            f"output must be contiguous {shape}, got {value.shape}, strides={value.strides}"
        )


def strided_copy(tensor):
    """Keep values and shape, but force non-unit stride in the final dimension."""
    storage = tensor.new_empty((*tensor.shape[:-1], tensor.shape[-1] * 2))
    view = storage[..., ::2]
    view.copy_(tensor)
    assert tensor.numel() == 1 or not view.is_contiguous()
    return view


def options(cli, **overrides):
    base = adapter_options(
        cli.acceleration, blur_backend=cli.dense_blur_backend, max_cached_sizes=3
    )
    return replace(base, **dict(dict(pinned_output=True), **overrides))


def require_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: expected an explicit ValueError")


def check_options():
    count = 0
    assert AdapterOptions().batch_chunk_size == "auto"
    for value in (-1, True, False, 1.5, None, "2", "AUTO", np.int64(2)):
        require_value_error(
            lambda: PasteBatchAdapter(AdapterOptions(batch_chunk_size=value, pinned_output=True)),
            f"chunk size {value!r}",
        )
        count += 1
    for change in ({"backend": "bucketed"}, {"return_numpy": False}, {"pinned_output": False}):
        args = dict(batch_chunk_size=2, pinned_output=True)
        args.update(change)
        require_value_error(lambda: PasteBatchAdapter(AdapterOptions(**args)), str(change))
        count += 1
    # Auto must safely fall back for the existing modes, without rejecting options.
    for mode in (0, "auto"):
        for change in (
            {},
            {"backend": "bucketed"},
            {"return_numpy": False},
            {"pinned_output": False},
        ):
            PasteBatchAdapter(AdapterOptions(batch_chunk_size=mode, **change))
            count += 1
    return count


def check_auto_fallback(cli, device):
    """Unmeasured shapes and unsupported output/backend modes retain whole behavior."""
    cpu = device.type == "cpu"
    base = options(
        cli,
        batch_chunk_size="auto",
        acceleration="eager" if cpu else cli.acceleration,
        blur_backend="torch" if cpu else cli.dense_blur_backend,
    )
    args, boxes = make_case(device, batch=3, height=HEIGHT, width=WIDTH, slots=3, variant=0)
    call = make_adapter_case(args, boxes)
    count = 0
    for change in ({}, {"backend": "bucketed"}, {"return_numpy": False}, {"pinned_output": False}):
        opt = replace(base, **change)
        renderer = PasteBatchAdapter(opt)
        expected = PasteBatchAdapter(replace(opt, batch_chunk_size=0))(*call)
        actual = renderer(*call)
        if isinstance(actual, torch.Tensor):
            assert isinstance(expected, torch.Tensor)
            actual, expected = actual.cpu().numpy(), expected.cpu().numpy()
        output_contract(actual, 3)
        exact(actual, expected, f"auto fallback {device}, {change}")
        execution = renderer.last_execution
        if execution["chunk_size"] != 0 or execution["blur_launch_config"] is not None:
            raise AssertionError(f"unknown configuration must use whole legacy launch: {execution}")
        import paste_execution_policy as policy

        if execution["reason"] == policy.PROFILE_ID:
            raise AssertionError("small fixture unexpectedly matched measured 1080p profile")
        count += 1
    return count


class TensorMetadata:
    """Only metadata consumed by choose_execution; never creates a CUDA tensor."""

    def __init__(self, shape, strides=None, device="cuda:0", contiguous=True, dtype=torch.float32):
        self.shape = tuple(shape)
        if strides is None:
            reversed_strides, stride = [], 1
            for size in reversed(self.shape):
                reversed_strides.append(stride)
                stride *= size
            strides = tuple(reversed(reversed_strides))
        self._strides = tuple(strides)
        self.device = torch.device(device)
        self.dtype = dtype
        self._contiguous = contiguous

    def stride(self):
        return self._strides

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return int(np.prod(self.shape))


def policy_fixture(batch=8, device="cuda:0"):
    import paste_execution_policy as policy

    def tensor(shape, **kw):
        return TensorMetadata(shape, device=device, **kw)

    frames = tensor((1, 3, 1080, 1920), strides=(6220800, 1, 5760, 3), contiguous=False)
    ads = tuple(tensor((batch, 4, *policy.SOURCE_SIZES[i % 3])) for i in range(16))
    grids = tuple(tensor((1, 108, 320, 2)) for _ in range(16))
    distortion = tensor((1, 1080, 1920, 2))
    kernel, mask = tensor((4, 1, 9, 9)), tensor((1, 1, 1080, 1920))
    opacity = tuple(tensor((1, 1, 1, 1)) for _ in range(16))
    boxes = tuple((0, 0, 320, 108) for _ in range(16))
    return (
        (frames, ads, grids, distortion, kernel, mask, opacity),
        boxes,
        (tensor((9,)), tensor((9,))),
    )


def check_policy_metadata():
    """Exercise policy gates independently of hardware and the final timing table."""
    import paste_execution_policy as policy
    from paste_gpu_batch import validate_inputs

    base = AdapterOptions(acceleration="triton", blur_backend="separable", pinned_output=True)
    launch = (128, 4, 4, 128, 4, 4)
    args, boxes, factors = policy_fixture()
    count = 0
    with (
        patch.object(policy, "CHUNK_BY_BATCH", {8: 2}),
        patch.object(policy, "BLUR_BY_BATCH", {8: launch}),
        patch.object(policy, "_measured_environment", return_value=True),
    ):
        assert policy.choose_execution(args, boxes, base, factors) == (2, launch, policy.PROFILE_ID)
        count += 1

        def fallback(
            label, candidate=args, candidate_boxes=boxes, opt=base, candidate_factors=factors
        ):
            nonlocal count
            chunk, config, reason = policy.choose_execution(
                candidate, candidate_boxes, opt, candidate_factors
            )
            if chunk != 0 or config is not None or reason == policy.PROFILE_ID:
                raise AssertionError(
                    f"{label}: expected safe whole fallback, got {(chunk, config, reason)}"
                )
            count += 1

        for change in (
            {"backend": "bucketed"},
            {"acceleration": "eager"},
            {"blur_backend": "torch"},
            {"return_numpy": False},
            {"pinned_output": False},
            {"legacy_alpha": False},
            {"warp_align_corners": True},
            {"distortion_align_corners": True},
            {"warp_padding_mode": "border"},
            {"distortion_padding_mode": "reflection"},
        ):
            fallback(str(change), opt=replace(base, **change))
        fallback("unknown B", candidate=policy_fixture(batch=17)[0])
        fallback(
            "CPU even when environment mock accepts", candidate=policy_fixture(device="cpu")[0]
        )
        fallback("missing factors", candidate_factors=None)
        fallback("different factor size", candidate_factors=(TensorMetadata((7,)), factors[1]))
        for replacement, label in (
            (TensorMetadata((3, 3)), "factor shape with nine elements"),
            (TensorMetadata((9,), strides=(2,), contiguous=False), "strided factor"),
            (TensorMetadata((9,), device="cuda:1"), "factor on another CUDA device"),
            (TensorMetadata((9,), device="cpu"), "CPU factor"),
            (TensorMetadata((9,), dtype=torch.float64), "factor dtype mismatch"),
        ):
            fallback(label, candidate_factors=(replacement, factors[1]))
        with patch.object(policy, "BLUR_BY_BATCH", {}):
            fallback("chunk entry without corresponding blur measurement")
        fallback("different slot count", candidate_boxes=boxes[:-1])
        for index, replacement, label in (
            (0, TensorMetadata((1, 3, 1079, 1920)), "unknown frame size"),
            (0, TensorMetadata((1, 3, 1080, 1920)), "unknown frame storage"),
            (0, TensorMetadata((8, 3, 1080, 1920)), "per-member background"),
            (3, TensorMetadata((8, 1080, 1920, 2)), "per-member distortion"),
            (3, TensorMetadata((1, 1080, 1920, 2), contiguous=False), "strided distortion"),
            (4, TensorMetadata((4, 1, 3, 3)), "unknown kernel size"),
            (5, TensorMetadata((1, 3, 1080, 1920)), "three-channel mask"),
            (5, TensorMetadata((1, 1, 1080, 1920), strides=(0, 0, 0, 1)), "broadcast mask rows"),
        ):
            candidate = list(args)
            candidate[index] = replacement
            fallback(label, candidate=tuple(candidate))
        for index, replacement, label in (
            (1, TensorMetadata((8, 4, 201, 200)), "unknown source size"),
            (1, TensorMetadata((8, 4, 200, 200), contiguous=False), "strided source"),
            (2, TensorMetadata((8, 108, 320, 2)), "per-member warp grid"),
            (2, TensorMetadata((1, 108, 320, 2), contiguous=False), "strided warp grid"),
            (6, TensorMetadata((8, 1, 1, 1)), "per-member opacity"),
        ):
            candidate = list(args)
            candidate[index] = (replacement, *candidate[index][1:])
            fallback(label, candidate=tuple(candidate))
        fallback("unknown ROI size", candidate_boxes=((0, 0, 319, 108), *boxes[1:]))
        # A nonzero CUDA index is checked against its own environment, not cuda:0.
        with patch.object(
            policy, "_measured_environment", side_effect=lambda index: index == 0
        ) as environment:
            other_args, _, other_factors = policy_fixture(device="cuda:1")
            fallback(
                "unmeasured CUDA device", candidate=other_args, candidate_factors=other_factors
            )
            environment.assert_called_once_with(1)
        with patch.object(policy, "_measured_environment", return_value=False):
            fallback("unmeasured software/hardware")

    # Adapter validation precedes policy selection; mixed-device inputs must fail
    # there. Meta tensors check that invariant without using a second GPU.
    cpu_args, cpu_boxes = make_case(torch.device("cpu"), batch=1)
    mixed = list(cpu_args)
    mixed[1] = (cpu_args[1][0].to("meta"), *cpu_args[1][1:])
    require_value_error(
        lambda: validate_inputs(tuple(mixed), cpu_boxes), "mixed-device input validation"
    )
    return count + 1


def check_auto_override_reset():
    """Mock a measured selection, then test the real override/early-return routing on CPU."""
    import paste_execution_policy as policy

    cli = SimpleNamespace(acceleration="eager", dense_blur_backend="torch")
    call = case_inputs(torch.device("cpu"), 3, 0, "separable")
    override = (64, 4, 4, 128, 4, 4)
    opt = options(
        cli, batch_chunk_size="auto", blur_backend="separable", blur_launch_config=override
    )
    renderer = PasteBatchAdapter(opt)
    expected = PasteBatchAdapter(replace(opt, batch_chunk_size=0))(*call)
    # This deliberately mocks selection only. No CUDA tensors or execution are
    # involved; a missing override fallback would incorrectly attempt chunking.
    with patch.object(
        policy, "choose_execution", return_value=(2, None, policy.PROFILE_ID)
    ) as choose:
        actual = renderer(*call)
        choose.assert_called_once()
    output_contract(actual, 3)
    exact(actual, expected, "auto with unmeasured manual blur override")
    execution = renderer.last_execution
    if execution != dict(
        batch=3, chunk_size=0, blur_launch_config=override, reason="unmeasured_blur_override"
    ):
        raise AssertionError(
            f"manual blur override did not disable automatic chunking: {execution}"
        )
    retained = actual.copy()
    owner, frame, _, frame_id, way, _ = call
    no_ads = renderer(owner, frame, {}, frame_id, way, None)
    background = frame.clamp(0, 255).to(torch.uint8).numpy()
    exact(no_ads, np.broadcast_to(background, (3, HEIGHT, WIDTH, 3)), "no ads after override")
    exact(actual, retained, "retained output after no-ad frame")
    if renderer.last_execution != dict(
        batch=3, chunk_size=0, blur_launch_config=None, reason="no_ads"
    ):
        raise AssertionError(
            f"no-ad frame retained stale execution metadata: {renderer.last_execution}"
        )
    return 2


def check_no_ads(cli, device):
    count = 0
    # A shared background must expand to every current member, even with no ads.
    frame = torch.rand((HEIGHT, WIDTH, 3), device=device) * 300 - 20
    expected_frame = frame.clamp(0, 255).to(torch.uint8).cpu().numpy()
    for batch in (1, 5):
        for squeeze in (True, False):
            owner = SimpleNamespace(imgs_raw_groups=[{} for _ in range(batch)])
            renderer = PasteBatchAdapter(options(cli, batch_chunk_size=2, squeeze_single=squeeze))
            result = renderer(owner, frame, {}, 77, 1, None)
            expected = np.broadcast_to(expected_frame, (batch, HEIGHT, WIDTH, 3))
            if batch == 1 and squeeze:
                expected = expected[0]
            output_contract(result, batch, squeeze=squeeze)
            exact(result, expected, f"no ads B={batch}, squeeze={squeeze}")
            if renderer.last_execution != dict(
                batch=batch, chunk_size=0, blur_launch_config=None, reason="no_ads"
            ):
                raise AssertionError(
                    f"incorrect no-ad execution metadata: {renderer.last_execution}"
                )
            count += 1
    return count


def check_cpu_rejection(cli):
    # Even B <= chunk size must not silently claim CUDA chunk support on CPU.
    args, boxes = make_case(torch.device("cpu"), batch=1, height=HEIGHT, width=WIDTH)
    call = make_adapter_case(args, boxes)
    renderer = PasteBatchAdapter(options(cli, batch_chunk_size=2))
    require_value_error(lambda: renderer(*call), "CPU chunk mode with ads")
    return 1


def case_inputs(device, batch, variant, kernel_mode):
    args, boxes = make_case(
        device,
        batch=batch,
        height=HEIGHT,
        width=WIDTH,
        slots=3,
        variant=variant,
        source_sizes=((11, 17), (19, 23), (13, 29)),
    )
    frames, ads, grids, distortion, kernel, _, opacity = args
    # variant 0: shared background/mask/distortion; variant 1: per-member data.
    # Both fixtures contain shared and per-member warp grids and opacities.
    channels = 1 if variant == 0 else 3
    mask = torch.rand((batch if variant else 1, channels, HEIGHT, WIDTH), device=device) * 0.35
    if kernel_mode == "dense_shared":
        kernel = kernel[0] if kernel.ndim == 5 else kernel
    elif kernel_mode == "dense_per_member":
        if kernel.ndim == 4:
            kernel = torch.rand((batch, 4, 1, 3, 3), device=device)
            kernel /= kernel.sum(dim=(-2, -1), keepdim=True)
    else:
        ky = torch.tensor([0.25, 0.5, 0.25], device=device)
        kx = torch.tensor([0.125, 0.75, 0.125], device=device)
        dense = torch.outer(ky, kx)[None, None].repeat(4, 1, 1, 1)
        kernel = MotionKernelSpec(dense, (ky, kx))
    args = (
        frames,
        ads,
        tuple(strided_copy(grid) for grid in grids),
        strided_copy(distortion),
        kernel,
        strided_copy(mask),
        tuple(strided_copy(value) for value in opacity),
    )
    return make_adapter_case(args, boxes)


def check_equivalence(cli, device):
    count = 0
    for batch in (1, 2, 3, 5, 8):
        for variant in (0, 1):
            for kernel_mode in ("dense_shared", "dense_per_member", "separable"):
                call = case_inputs(device, batch, variant, kernel_mode)
                blur = "separable" if kernel_mode == "separable" else cli.dense_blur_backend
                for squeeze in (True, False) if batch == 1 else (True,):
                    whole_options = options(
                        cli, blur_backend=blur, squeeze_single=squeeze, batch_chunk_size=0
                    )
                    expected = PasteBatchAdapter(whole_options)(*call)
                    output_contract(expected, batch, squeeze=squeeze)
                    for chunk_size in cli.chunk_sizes:
                        renderer = PasteBatchAdapter(
                            replace(whole_options, batch_chunk_size=chunk_size)
                        )
                        label = (
                            f"B={batch}, C={chunk_size}, variant={variant}, "
                            f"kernel={kernel_mode}, squeeze={squeeze}"
                        )
                        result = renderer(*call)
                        output_contract(result, batch, squeeze=squeeze)
                        exact(result, expected, label)
                        snapshot = result.copy()
                        # A cache-hit call must not alias or overwrite a live NumPy result.
                        again = renderer(*call)
                        exact(again, expected, label + " repeated")
                        exact(result, snapshot, label + " retained output")
                        if np.shares_memory(result, again):
                            raise AssertionError(label + ": outputs share storage")
                        count += 1
    return count


def check_retained(retained):
    for index, (array, snapshot) in enumerate(retained):
        exact(array, snapshot, f"retained NumPy output {index}")


def check_dynamic_members(cli, device):
    """One adapter/owner, changing B and order, with a fresh uncached oracle."""
    args, boxes = make_case(
        device,
        batch=12,
        height=HEIGHT,
        width=WIDTH,
        slots=3,
        variant=0,
        source_sizes=((11, 17), (19, 23), (13, 29)),
    )
    frames, ads, grids, distortion, kernel, mask, opacity = args
    # Shared geometry/background makes the expected member permutation explicit.
    args = (
        frames,
        ads,
        tuple(grid[:1] for grid in grids),
        distortion,
        kernel,
        mask,
        tuple(value[:1] for value in opacity),
    )
    call = make_adapter_case(args, boxes)
    owner, frame, data, _, way, paste_args = call
    for index, param in enumerate(data["ad_params"]):
        param["transparency"] = 0.61 + 0.08 * index
    member_pool = list(owner.imgs_raw_groups)
    renderer = PasteBatchAdapter(options(cli, batch_chunk_size=3))
    retained, count = [], 0

    def render_and_compare(label):
        nonlocal count
        current = (owner, frame, data, 1234, way, paste_args)
        expected = PasteBatchAdapter(options(cli, batch_chunk_size=0, cache_ads=False))(*current)
        actual = renderer(*current)
        output_contract(actual, len(owner.imgs_raw_groups))
        exact(actual, expected, label)
        check_retained(retained)
        if any(np.shares_memory(actual, previous) for previous, _ in retained):
            raise AssertionError(label + ": live output storage was reused")
        retained.append((actual, actual.copy()))
        count += 1
        return actual

    for step, batch in enumerate((8, 1, 5, 12, 3, 8)):
        owner.imgs_raw_groups = [member_pool[(step + index) % 12] for index in range(batch)]
        before = render_and_compare(f"dynamic step={step}, B={batch}")
        if batch > 1:
            owner.imgs_raw_groups.reverse()
            reordered = render_and_compare(f"reordered members B={batch}")
            exact(reordered, before[::-1], f"member order B={batch}")
        before_update = retained[-1][0]
        # Same image objects, changed pixels, explicit cache version increment.
        for record in owner.imgs_raw_groups[0].values():
            image = record["img"]
            image[..., :3].fill_(30 + step * 25)
            image[..., 3].fill_(255)
        owner.imgs_raw_version += 1
        updated = render_and_compare(f"asset version={owner.imgs_raw_version}, B={batch}")
        if np.array_equal(updated, before_update):
            raise AssertionError("asset update fixture did not affect any output pixel")
    check_retained(retained)
    return count


def check_same_frame_matrices(cli, device):
    args, boxes = make_case(device, batch=5, height=HEIGHT, width=WIDTH, slots=3, variant=1)
    call = list(make_adapter_case(args, boxes))
    matrices = []
    for index, (ad, box, param) in enumerate(zip(args[1], boxes, call[2]["ad_params"])):
        sh, sw = ad.shape[-2:]
        x1, y1, x2, y2 = box
        matrix = torch.tensor(
            [
                [(sw - 1) / (x2 - x1), 0.01, 0.0],
                [0.02, (sh - 1) / (y2 - y1), 0.0],
                [0.001, 0.0, 1.0],
            ],
            device=device,
        )
        # Mix shared and per-member matrices, retaining their object identities.
        matrix = matrix.unsqueeze(0).repeat(1 if index % 2 == 0 else 5, 1, 1)
        param["m_inv"] = matrix
        matrices.append(matrix)
    call[3] = 5678
    whole_options = options(cli, grid_source="matrix", matrix_coordinates="roi")
    renderer = PasteBatchAdapter(replace(whole_options, batch_chunk_size=2))
    retained = []
    for step in range(3):
        if step:
            for matrix in matrices:
                matrix[:, 0, 2].add_(2.7)
                matrix[:, 1, 2].sub_(0.8)
        expected = PasteBatchAdapter(replace(whole_options, cache_ads=False))(*call)
        actual = renderer(*call)
        output_contract(actual, 5)
        exact(actual, expected, f"same frame_id, matrix mutation {step}")
        check_retained(retained)
        if retained and np.array_equal(actual, retained[-1][0]):
            raise AssertionError("in-place matrix update did not change output")
        retained.append((actual, actual.copy()))
    return 3


def check_shared_geometry(cli, device):
    """Real preparation path with explicitly synthetic callbacks and fixed frame_id."""
    from benchmark_shared_paste import RenderPath, SyntheticFixture

    fixture_cli = SimpleNamespace(
        height=HEIGHT,
        width=WIDTH,
        batch=5,
        kernel_size=3,
        sigma_pattern="continuous",
        acceleration=cli.acceleration,
        normalization="legacy",
        forward=False,
        inverse_iterations=10,
        camera_index_policy="legacy",
        align_corners=False,
        dense_blur_backend=cli.dense_blur_backend,
        ad_sizes=((11, 17), (19, 23), (13, 29)),
    )
    args, boxes = make_case(
        device,
        batch=5,
        height=HEIGHT,
        width=WIDTH,
        slots=3,
        variant=0,
        source_sizes=fixture_cli.ad_sizes,
    )
    fixture = SyntheticFixture(fixture_cli, device, args, boxes)
    count = 0
    for separable in (False, True):
        fixture.set_frame(0)
        frame_id = fixture.frame_id
        whole = RenderPath("whole", fixture, optimized=True, separable=separable)
        chunk = RenderPath("chunk", fixture, optimized=True, separable=separable)
        chunk.adapter = PasteBatchAdapter(replace(chunk.adapter.options, batch_chunk_size=2))
        retained = []
        previous_grid = None
        for step in ("initial", "K", "dist_coeffs", "matrix"):
            if step == "K":
                fixture.K[0, 0] *= 0.6
                fixture.K[0, 2] += 4.0
            elif step == "dist_coeffs":
                for record in fixture.records:
                    record["dist_coeffs"][0] -= 0.12
                fixture.raw_coeff[0] -= 0.12
            elif step == "matrix":
                for matrix in fixture.matrices:
                    matrix[0, 2] += 2.7
                    matrix[1, 2] -= 0.8
            assert fixture.frame_id == frame_id
            prepared = chunk.prepare(include_warps=True)
            distortion = prepared[1]
            if step in ("K", "dist_coeffs") and torch.equal(distortion, previous_grid):
                raise AssertionError(f"same frame_id {step} update reused a stale distortion grid")
            previous_grid = distortion.clone()
            expected, actual = whole.render(), chunk.render()
            output_contract(actual, 5)
            exact(actual, expected, f"shared geometry {step}, separable={separable}")
            check_retained(retained)
            if retained and np.array_equal(actual, retained[-1][0]):
                raise AssertionError(f"same frame_id {step} update did not change output")
            retained.append((actual, actual.copy()))
            count += 1
    return count


def check_exception_recovery(cli, device):
    """A failed later chunk must finish pending copies and leave the adapter usable."""
    import paste_gpu_batch_adapter as adapter_module

    call = case_inputs(device, 5, 1, "dense_per_member")
    whole_options = options(cli, batch_chunk_size=0)
    expected = PasteBatchAdapter(whole_options)(*call)
    renderer = PasteBatchAdapter(replace(whole_options, batch_chunk_size=2))
    retained = renderer(*call)
    snapshot = retained.copy()
    original = adapter_module._paste_gpu_batch
    calls = 0

    class InjectedChunkError(RuntimeError):
        pass

    def fail_second_chunk(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InjectedChunkError("intentional second-chunk failure")
        return original(*args, **kwargs)

    adapter_module._paste_gpu_batch = fail_second_chunk
    try:
        try:
            renderer(*call)
        except InjectedChunkError:
            pass
        else:
            raise AssertionError("second-chunk failure did not propagate")
    finally:
        adapter_module._paste_gpu_batch = original
    if calls != 2:
        raise AssertionError(f"expected failure on second chunk, got {calls} core calls")
    recovered = renderer(*call)
    output_contract(recovered, 5)
    exact(recovered, expected, "adapter recovery after second-chunk failure")
    exact(retained, snapshot, "retained output after second-chunk failure")
    return 1


def check_spatial_broadcast(cli, device):
    """Legal zero spatial strides must compile and match materialized inputs."""
    count = 0
    args, boxes = make_case(device, batch=5, height=HEIGHT, width=WIDTH, slots=3, variant=0)
    for spatial in ((1, WIDTH), (HEIGHT, 1), (1, 1)):
        for target in ("background", "mask"):
            for stored_batch in (1, 5):
                for channels in (1, 3):
                    frame_hw = spatial if target == "background" else (HEIGHT, WIDTH)
                    mask_hw = spatial if target == "mask" else (HEIGHT, WIDTH)
                    frames = torch.rand((stored_batch, 3, *frame_hw), device=device) * 255
                    # Adapter image geometry stays full H/W; its view may have zero strides.
                    frames = frames.expand(stored_batch, 3, HEIGHT, WIDTH)
                    mask = torch.rand((stored_batch, channels, *mask_hw), device=device) * 0.35
                    for separable in (False, True):
                        kernel = args[4]
                        if separable:
                            factor = torch.tensor([0.25, 0.5, 0.25], device=device)
                            dense = torch.outer(factor, factor)[None, None].repeat(4, 1, 1, 1)
                            kernel = MotionKernelSpec(dense, (factor, factor))
                        broadcast = (frames, args[1], args[2], args[3], kernel, mask, args[6])
                        materialized = (
                            frames.contiguous(),
                            args[1],
                            args[2],
                            args[3],
                            kernel,
                            mask.expand(stored_batch, channels, HEIGHT, WIDTH).contiguous(),
                            args[6],
                        )
                        opt = options(
                            cli, blur_backend=("separable" if separable else cli.dense_blur_backend)
                        )
                        expected = PasteBatchAdapter(opt)(*make_adapter_case(materialized, boxes))
                        call = make_adapter_case(broadcast, boxes)
                        label = (
                            f"{target} spatial={spatial}, stored B={stored_batch}, "
                            f"mask channels={channels}, separable={separable}"
                        )
                        for chunk_size in (0, 2):
                            actual = PasteBatchAdapter(replace(opt, batch_chunk_size=chunk_size))(
                                *call
                            )
                            output_contract(actual, 5)
                            exact(actual, expected, label + f", chunk={chunk_size}")
                        count += 1
    return count


def check_measured_profile(cli, device):
    """Optional 1080p dynamic-B integration using persistent adapters/builders."""
    import paste_execution_policy as policy
    from benchmark_shared_paste import RenderPath, SyntheticFixture

    fixture_cli = SimpleNamespace(
        height=1080,
        width=1920,
        batch=12,
        kernel_size=9,
        sigma_pattern="continuous",
        acceleration=cli.acceleration,
        normalization="legacy",
        forward=False,
        inverse_iterations=10,
        camera_index_policy="legacy",
        align_corners=False,
        dense_blur_backend=cli.dense_blur_backend,
        ad_sizes=policy.SOURCE_SIZES,
    )
    args, boxes = make_case(
        device,
        batch=12,
        height=1080,
        width=1920,
        slots=16,
        variant=0,
        kernel_size=9,
        realistic=True,
        source_sizes=fixture_cli.ad_sizes,
    )
    fixture = SyntheticFixture(fixture_cli, device, args, boxes)
    pool = list(fixture.owner.imgs_raw_groups)
    whole = RenderPath("forced whole legacy launch", fixture, optimized=True, separable=True)
    whole.adapter = PasteBatchAdapter(
        replace(whole.adapter.options, batch_chunk_size=0, blur_launch_config=None)
    )
    automatic = RenderPath("automatic measured profile", fixture, optimized=True, separable=True)
    automatic.adapter = PasteBatchAdapter(
        replace(automatic.adapter.options, batch_chunk_size="auto", blur_launch_config=None)
    )
    retained, executions = [], []
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    measured_environment = policy._measured_environment(device_index)
    for step, batch in enumerate((8, 1, 5, 12, 3, 8)):
        fixture.cli.batch = batch
        fixture.owner.imgs_raw_groups = [pool[(step + index) % len(pool)] for index in range(batch)]
        fixture.set_frame(step)
        expected, actual = whole.render(), automatic.render()
        output_contract(actual, batch, height=1080, width=1920)
        exact(actual, expected, f"measured auto profile step={step}, B={batch}")
        check_retained(retained)
        if any(np.shares_memory(actual, old) for old, _ in retained):
            raise AssertionError("auto profile reused storage belonging to an earlier output")
        retained.append((actual, actual.copy()))
        execution = automatic.adapter.last_execution.copy()
        if execution["batch"] != batch:
            raise AssertionError(f"last_execution has stale B: {execution}")
        if measured_environment and batch in policy.CHUNK_BY_BATCH:
            expected_selection = (
                policy.CHUNK_BY_BATCH[batch],
                policy.BLUR_BY_BATCH.get(batch),
                policy.PROFILE_ID,
            )
            selection = (
                execution["chunk_size"],
                execution["blur_launch_config"],
                execution["reason"],
            )
            if selection != expected_selection:
                raise AssertionError(
                    f"measured profile was not applied: {selection} != {expected_selection}"
                )
        elif execution["chunk_size"] != 0 or execution["blur_launch_config"] is not None:
            raise AssertionError(f"unmeasured environment/batch did not fall back: {execution}")
        if (
            whole.adapter.last_execution["chunk_size"] != 0
            or whole.adapter.last_execution["blur_launch_config"] is not None
        ):
            raise AssertionError(
                "forced whole baseline did not retain its original launch configuration"
            )
        executions.append(dict(step=step, **execution))
        print(f"PASS: measured-profile B={batch}, selection={execution}", flush=True)

    # Keep both adapters/builders alive, switch the full workload to an unknown
    # canvas shape, and ensure policy fallback does not affect correctness.
    small_cli = SimpleNamespace(**vars(fixture_cli))
    small_cli.height, small_cli.width, small_cli.batch = HEIGHT, WIDTH, 3
    small_args, small_boxes = make_case(
        device,
        batch=3,
        height=HEIGHT,
        width=WIDTH,
        slots=16,
        variant=0,
        kernel_size=9,
        realistic=True,
        source_sizes=small_cli.ad_sizes,
    )
    small = SyntheticFixture(small_cli, device, small_args, small_boxes)
    whole.fixture = automatic.fixture = small
    expected, actual = whole.render(), automatic.render()
    output_contract(actual, 3)
    exact(actual, expected, "auto profile fallback after canvas-size change")
    check_retained(retained)
    execution = automatic.adapter.last_execution.copy()
    if (
        execution["chunk_size"] != 0
        or execution["blur_launch_config"] is not None
        or execution["reason"] == policy.PROFILE_ID
    ):
        raise AssertionError(f"unknown shape did not use safe whole fallback: {execution}")
    executions.append(dict(step="unknown_shape", **execution))
    return 7, executions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acceleration", choices=("eager", "auto", "triton"), default=None)
    parser.add_argument("--dense-blur-backend", choices=("torch", "triton"), default=None)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--profile-checks",
        action="store_true",
        help="also check the measured 1080p/16-slot/k9 profile during dynamic B changes",
    )
    cli = parser.parse_args()
    device = torch.device(cli.device)
    if any(value < 1 for value in cli.chunk_sizes):
        parser.error("--chunk-sizes must be positive integers")
    cli.acceleration = cli.acceleration or ("triton" if device.type == "cuda" else "eager")
    cli.dense_blur_backend = cli.dense_blur_backend or (
        "triton" if device.type == "cuda" else "torch"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for contract checks only")
    torch.manual_seed(20261021)
    summary = {
        "device": str(device),
        "acceleration": cli.acceleration,
        "dense_blur_backend": cli.dense_blur_backend,
        "uint8_tolerance": 0,
        "checks": {},
    }
    checks = summary["checks"]
    checks["option_validation"] = check_options()
    checks["policy_metadata"] = check_policy_metadata()
    checks["auto_cpu_fallback"] = check_auto_fallback(cli, torch.device("cpu"))
    checks["auto_blur_override_and_reset"] = check_auto_override_reset()
    checks["cpu_rejection"] = check_cpu_rejection(cli)
    checks["no_ads"] = check_no_ads(cli, device)
    if device.type == "cuda":
        checks["auto_cuda_fallback"] = check_auto_fallback(cli, device)
        checks["whole_vs_chunk"] = check_equivalence(cli, device)
        print(
            "PASS: whole/chunk exact equivalence, tail blocks, broadcast/strided inputs", flush=True
        )
        checks["dynamic_members_and_assets"] = check_dynamic_members(cli, device)
        checks["same_frame_matrix_updates"] = check_same_frame_matrices(cli, device)
        checks["shared_live_geometry"] = check_shared_geometry(cli, device)
        checks["exception_recovery"] = check_exception_recovery(cli, device)
        print("PASS: dynamic members/assets, live geometry and exception recovery", flush=True)
        checks["spatial_broadcast"] = check_spatial_broadcast(cli, device)
        if cli.profile_checks:
            checks["measured_profile"], summary["profile_execution"] = check_measured_profile(
                cli, device
            )
        summary["gpu"] = torch.cuda.get_device_name(device)
        summary["cuda_checks"] = "passed"
    else:
        summary["cuda_checks"] = "SKIPPED: chunk execution requires CUDA"
        if cli.profile_checks:
            summary["profile_execution"] = "SKIPPED: measured-profile integration requires CUDA"
    summary["total_checks"] = sum(checks.values())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

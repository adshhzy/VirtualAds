"""Measure ONCE-per-frame shared geometry preparation plus batch compositing.

    python benchmark_shared_paste.py --batch 8 --diagnose
    python benchmark_shared_paste.py --batch 8 --check-only
    python benchmark_shared_paste.py --batch 8 --forward --normalization matched
    python benchmark_shared_paste.py --batch 8 --sigma-pattern repeated --diagnose

The three matched paths use original shared preparation + dense blur, optimized
shared preparation + the SAME dense blur, and optimized preparation + separable
blur. All return independent pinned NumPy images. Preparation is shared ONCE
per frame even in the baseline; no artificial B-times preprocessing baseline.
Frame/material/mask pixels start on GPU, but calibration and homographies start
on CPU and are updated every sample. Timings include filtering, staging, all grids,
Gaussian creation/cache lookup, adapter and CPU output. Synthetic input updates
and compilation/warmup are excluded. No video decode, segmentation, encode or send.

Motion estimation, sigma mapping and Gaussian generation below are SYNTHETIC
callbacks because the production callbacks were not supplied. Their timings and
cache hit rates are not claims about production. Gaussian sigma changes smoothly
by default; --sigma-pattern repeated cycles four values to exercise cache hits.
K, coefficients and homographies change every sample. Each path
has independent mutable camera history and calibration arrays.

Default legacy camera-index selection and grid normalization preserve the supplied
original behavior. corrected/matched are explicit semantic changes, applied to
BOTH reference and optimized paths. Grid tolerance is fixed at absolute 2e-6;
final uint8 tolerance is one level. Correctness failures abort the benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

BENCHMARK_VERSION = "2026-10-21-perf4.5-streaming"
GRID_ATOL = 2e-6


def reference_camera_filter(owner, data_dict, frame_id, index_policy="legacy"):
    """Original full SciPy median, with explicitly selected current-index policy."""
    import numpy as np
    from scipy.ndimage import median_filter

    neighbors, legacy_index = [], 0
    for key, value in owner.paste_data_stack.items():
        if abs(key - frame_id) < 5 and len(value) != 1:
            neighbors.append((key, value["K"], value["w2c"], value["dist_coeffs"]))
            legacy_index += key != frame_id
    neighbors.sort(key=lambda value: value[0])
    if len(neighbors) < 3:
        return data_dict["dist_coeffs"], 0.0
    index = (
        legacy_index
        if index_policy == "legacy"
        else next(i for i, value in enumerate(neighbors) if value[0] == frame_id)
    )
    filtered = median_filter(
        np.stack([value[3] for value in neighbors]), size=(7, 1), mode="reflect"
    )[index]
    data_dict["dist_coeffs"] = filtered
    return filtered, owner.compute_camera_motion_rate([value[2] for value in neighbors])


def reference_distortion_grid(
    height,
    width,
    K,
    coeff,
    *,
    align_corners=False,
    inverse=True,
    iterations=10,
    normalization="legacy",
):
    """Supplied original formulas, including the GPU .tolist() synchronization."""
    import torch

    xs = torch.linspace(0, width - 1, width, device=K.device, dtype=torch.float32)
    ys = torch.linspace(0, height - 1, height, device=K.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xn, yn = (pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy
    k1 = k2 = p1 = p2 = k3 = 0.0
    if len(coeff) >= 5:
        k1, k2, p1, p2, k3 = coeff[:5].tolist()
    elif len(coeff) >= 2:
        k1, k2 = coeff[:2].tolist()
    elif len(coeff) == 1:
        k1 = coeff[0]
    else:
        raise ValueError("distortion coefficients must not be empty")
    if inverse:
        x, y = xn.clone(), yn.clone()
        for _ in range(iterations):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if p1 != 0 or p2 != 0:
                tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
                ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
                x, y = (xn - tx) / radial, (yn - ty) / radial
            else:
                x, y = xn / radial, yn / radial
    else:
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x, y = xn * radial, yn * radial
        if p1 != 0 or p2 != 0:
            x += 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            y += p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    sx, sy = fx * x + cx, fy * y + cy
    if normalization == "legacy" or align_corners:
        gx, gy = 2.0 * sx / (width - 1) - 1, 2.0 * sy / (height - 1) - 1
    else:
        gx, gy = (2.0 * sx + 1) / width - 1, (2.0 * sy + 1) / height - 1
    return torch.stack((gx, gy), dim=-1).reshape(1, height, width, 2)


def reference_roi_grid(
    source_hw, matrix, roi_hw, *, align_corners=False, normalization="legacy", eps=1e-8
):
    """Original ROI-local projection and z+eps division, without a z clamp."""
    import torch

    sh, sw = source_hw
    rh, rw = roi_hw
    ys = torch.linspace(0, rh - 1, rh, device=matrix.device, dtype=torch.float32)
    xs = torch.linspace(0, rw - 1, rw, device=matrix.device, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    coordinates = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(-1, 3)
    projected = matrix @ coordinates.T
    if projected.ndim == 2:
        projected = projected.unsqueeze(0)
    projected = projected.transpose(-1, -2)
    sx = projected[..., 0] / (projected[..., 2] + eps)
    sy = projected[..., 1] / (projected[..., 2] + eps)
    if normalization == "legacy" or align_corners:
        gx, gy = 2.0 * sx / (sw - 1) - 1, 2.0 * sy / (sh - 1) - 1
    else:
        gx, gy = (2.0 * sx + 1) / sw - 1, (2.0 * sy + 1) / sh - 1
    return torch.stack((gx, gy), dim=-1).reshape(projected.shape[0], rh, rw, 2)


class SyntheticFixture:
    """Independent mutable CPU camera state; shared immutable GPU image pixels."""

    def __init__(self, cli, device, image_args, boxes):
        import numpy as np
        from benchmark_paste_gpu_batch import make_adapter_case

        self.cli, self.device, self.image_args, self.boxes = cli, device, image_args, boxes
        self.owner, self.frame, self.data, _, self.way_id, self.unused_paste_args = (
            make_adapter_case(image_args, boxes)
        )
        self.K = np.eye(3, dtype=np.float32)
        self.raw_coeff = np.zeros(5, dtype=np.float32)
        self.records = [
            {
                "K": self.K,
                "w2c": np.eye(4, dtype=np.float32),
                "dist_coeffs": np.zeros(5, dtype=np.float32),
            }
            for _ in range(9)
        ]
        self.owner.paste_data_stack = {}
        self.owner.gaussian_kernel_cache_version = 0
        # Explicit synthetic fixture callbacks, NOT inferred production behavior.
        self.owner.compute_camera_motion_rate = lambda poses: float(
            np.mean(np.abs(np.diff(np.asarray([pose[0, 3] for pose in poses], dtype=np.float32))))
        )
        self.owner.motion_rate_to_sigma = lambda rate: 1.0 + float(rate)

        def gaussian(sigma):
            x = np.arange(cli.kernel_size, dtype=np.float32) - cli.kernel_size // 2
            vector = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
            vector /= vector.sum()
            return np.outer(vector, vector).astype(np.float32)

        self.owner.create_gaussian_kernel = gaussian
        self.matrices = [np.eye(3, dtype=np.float32) for _ in boxes]
        for param, matrix in zip(self.data["ad_params"], self.matrices):
            param["m_inv"] = matrix
            param["transparency"] = 0.83
        self.data["K"] = self.K
        self.set_frame(0)

    def set_frame(self, step):
        """Synthetic source updates occur outside measured rendering intervals."""
        self.frame_id = 1000 + step
        height, width = self.cli.height, self.cli.width
        self.K[0, 0] = width * (0.9 + (step % 13) * 0.0005)
        self.K[1, 1] = height * (1.2 + (step % 11) * 0.0005)
        self.K[0, 2], self.K[1, 2] = width * 0.49, height * 0.51
        self.raw_coeff[:] = (-0.04 + step * 1e-6, 0.004, 0.0001, -0.00015, 0.0002)
        self.data["dist_coeffs"] = self.raw_coeff
        self.owner.paste_data_stack.clear()
        speed = (
            (step % 4 + 1) / 8.0
            if self.cli.sigma_pattern == "repeated"
            else 0.5 + 0.3 * math.sin(0.017 * step)
        )
        for offset, record in zip(range(-4, 5), self.records):
            record["dist_coeffs"][:] = self.raw_coeff
            record["dist_coeffs"][0] += offset * 0.0002
            record["w2c"][0, 3] = offset * speed
            self.owner.paste_data_stack[self.frame_id + offset] = record
        for i, (matrix, box, ad) in enumerate(zip(self.matrices, self.boxes, self.image_args[1])):
            x1, y1, x2, y2 = box
            matrix[0, 0], matrix[1, 1] = (
                (ad.shape[-1] - 1) / (x2 - x1),
                (ad.shape[-2] - 1) / (y2 - y1),
            )
            matrix[0, 1], matrix[1, 0] = 0.002, -0.001
            matrix[0, 2], matrix[1, 2] = (step % 17) * 0.03 + i * 0.01, (step % 7) * 0.02
            matrix[2, 0], matrix[2, 1] = 1e-5, -1e-5


class RenderPath:
    def __init__(self, name, fixture, *, optimized, separable=False, eager=False):
        from benchmark_paste_gpu_batch import adapter_options
        from paste_gpu_batch_adapter import PasteBatchAdapter
        from paste_shared_params import SharedParameterBuilder

        self.name, self.fixture, self.optimized = name, fixture, optimized
        cli = fixture.cli
        self.builder = SharedParameterBuilder(
            acceleration=cli.acceleration,
            normalization=cli.normalization,
            distortion_inverse=not cli.forward,
            iterations=cli.inverse_iterations,
            camera_index_policy=cli.camera_index_policy,
            distortion_align_corners=cli.align_corners,
            enable_separable=separable,
        )
        options = adapter_options(
            "eager" if eager else cli.acceleration,
            align=cli.align_corners,
            blur_backend="torch"
            if eager
            else ("separable" if separable else cli.dense_blur_backend),
        )
        options = replace(
            options,
            pinned_output=True,
            matrix_coordinates="roi",
            max_cached_sizes=max(2, len(set(cli.ad_sizes))),
            warp_grids_builder=self.builder.build_warp_grids if optimized else None,
        )
        self.adapter = PasteBatchAdapter(options)

    def prepare(self, include_warps=False):
        import torch

        fixture, cli = self.fixture, self.fixture.cli
        if self.optimized:
            values = self.builder.prepare(
                fixture.owner,
                fixture.data,
                fixture.frame_id,
                height=cli.height,
                width=cli.width,
                device=fixture.device,
                mask_canvas=fixture.image_args[5],
            )
            if include_warps:
                grids = self.builder.build_warp_grids(
                    params=fixture.data["ad_params"],
                    ads=fixture.image_args[1],
                    boxes=fixture.boxes,
                    device=fixture.device,
                    batch=cli.batch,
                    warp_align_corners=cli.align_corners,
                )
                values = (
                    *values[:5],
                    {param["ad_id"]: grid for param, grid in zip(fixture.data["ad_params"], grids)},
                )
            return values
        coefficients, rate = reference_camera_filter(
            fixture.owner, fixture.data, fixture.frame_id, cli.camera_index_policy
        )
        K = torch.as_tensor(fixture.K, device=fixture.device, dtype=torch.float32)
        coeff = torch.as_tensor(coefficients, device=fixture.device, dtype=torch.float32)
        distortion = reference_distortion_grid(
            cli.height,
            cli.width,
            K,
            coeff,
            inverse=not cli.forward,
            iterations=cli.inverse_iterations,
            normalization=cli.normalization,
            align_corners=cli.align_corners,
        )
        sigma = fixture.owner.motion_rate_to_sigma(rate)
        kernel = (
            torch.from_numpy(fixture.owner.create_gaussian_kernel(sigma)).float().to(fixture.device)
        )
        kernel = kernel.reshape(1, 1, cli.kernel_size, cli.kernel_size).repeat(4, 1, 1, 1)
        grids = {}
        for param, ad, box in zip(fixture.data["ad_params"], fixture.image_args[1], fixture.boxes):
            x1, y1, x2, y2 = box
            matrix = torch.as_tensor(param["m_inv"], device=fixture.device, dtype=torch.float32)
            grids[param["ad_id"]] = reference_roi_grid(
                ad.shape[-2:],
                matrix,
                (y2 - y1, x2 - x1),
                normalization=cli.normalization,
                align_corners=cli.align_corners,
            )
        return fixture.device, distortion, coeff, kernel, fixture.image_args[5], grids

    def render(self):
        fixture = self.fixture
        return self.adapter(
            fixture.owner,
            fixture.frame,
            fixture.data,
            fixture.frame_id,
            fixture.way_id,
            self.prepare(),
        )


def make_paths(cli, device):
    import torch
    from benchmark_paste_gpu_batch import make_case

    torch.manual_seed(24680)
    image_args, boxes = make_case(
        device,
        batch=cli.batch,
        height=cli.height,
        width=cli.width,
        slots=cli.slots,
        kernel_size=cli.kernel_size,
        realistic=True,
        source_sizes=cli.ad_sizes,
    )
    configurations = [
        ("original_shared_plus_dense", False, False, False),
        ("optimized_shared_plus_dense", True, False, False),
        ("optimized_shared_plus_separable", True, True, False),
    ]
    if cli.include_eager:
        configurations.append(("original_shared_plus_eager", False, False, True))
    return [
        RenderPath(
            name,
            SyntheticFixture(cli, device, image_args, boxes),
            optimized=optimized,
            separable=separable,
            eager=eager,
        )
        for name, optimized, separable, eager in configurations
    ]


def run_checks(paths):
    import numpy as np
    import torch
    from benchmark_paste_gpu_batch import assert_close

    worst_grid, worst_image = 0.0, 0
    for step in (0, 1, 5):
        prepared = []
        for path in paths:
            path.fixture.set_frame(step)
            prepared.append(path.prepare(include_warps=True))
        reference = prepared[0]
        for path, candidate in zip(paths[1:], prepared[1:]):
            np.testing.assert_array_equal(
                path.fixture.data["dist_coeffs"], paths[0].fixture.data["dist_coeffs"]
            )
            grid_pairs = [(reference[1], candidate[1])]
            grid_pairs += [(reference[5][key], candidate[5][key]) for key in reference[5]]
            for expected, actual in grid_pairs:
                error = (actual - expected).abs().max().item()
                worst_grid = max(worst_grid, error)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=GRID_ATOL,
                    msg=f"{path.name}: geometry differs after input update {step}",
                )
        expected = paths[0].render()
        for path in paths[1:]:
            worst_image = max(
                worst_image,
                assert_close(
                    path.render(), expected, f"{path.name}, live calibration update {step}"
                ),
            )
    print(
        json.dumps(
            {
                "correctness": "passed",
                "dynamic_frames": 3,
                "max_grid_coordinate_error": worst_grid,
                "grid_absolute_tolerance": GRID_ATOL,
                "max_uint8_difference": worst_image,
            }
        )
    )


def measure_path(path, *, shared_only=False):
    import torch
    from benchmark_paste_gpu_batch import summarize

    cli, device = path.fixture.cli, path.fixture.device
    invoke = (lambda: path.prepare(include_warps=True)) if shared_only else path.render
    for step in range(cli.warmup):
        path.fixture.set_frame(step)
        _result = invoke()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cache_before = path.builder.cache_info() if path.optimized else None
    values = []
    for step in range(cli.iterations):
        path.fixture.set_frame(cli.warmup + step)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _result = invoke()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000)
    report = {"wall_ms": summarize(values)}
    if cache_before is not None:
        cache_after = path.builder.cache_info()
        report["timed_kernel_cache_hits"] = cache_after["hits"] - cache_before["hits"]
        report["timed_kernel_cache_misses"] = cache_after["misses"] - cache_before["misses"]
        report["kernel_cache_after"] = cache_after
    return report


def parse_arguments():
    from benchmark_paste_gpu_batch import parse_ad_size

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {BENCHMARK_VERSION}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acceleration", choices=("eager", "auto", "triton"), default="triton")
    parser.add_argument("--dense-blur-backend", choices=("torch", "triton"), default="triton")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument(
        "--ad-sizes",
        type=parse_ad_size,
        nargs="+",
        default=[(200, 200), (500, 500)],
        help="source HEIGHTxWIDTH sizes, e.g. 200x200 500x500 300x600",
    )
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--inverse-iterations", type=int, default=10)
    parser.add_argument(
        "--forward",
        action="store_true",
        help="use forward distortion instead of inverse fixed-point iterations",
    )
    parser.add_argument("--normalization", choices=("legacy", "matched"), default="legacy")
    parser.add_argument("--camera-index-policy", choices=("legacy", "corrected"), default="legacy")
    parser.add_argument(
        "--sigma-pattern",
        choices=("continuous", "repeated"),
        default="continuous",
        help="continuous changes sigma every frame; repeated exercises four-value kernel caching",
    )
    parser.add_argument("--align-corners", action="store_true")
    parser.add_argument(
        "--include-eager",
        action="store_true",
        help="also time the original shared preparation with eager compositing",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="also time shared preparation alone, including all ROI grids",
    )
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args()
    if min(cli.height, cli.width) < 8 or min(cli.batch, cli.slots, cli.iterations, cli.warmup) < 1:
        parser.error("height/width >=8; batch/slots/iterations/warmup >=1 required")
    if (
        min(min(shape) for shape in cli.ad_sizes) < 2
        or cli.kernel_size < 1
        or cli.kernel_size % 2 != 1
        or cli.kernel_size > 63
    ):
        parser.error("ad sizes >=2 and positive odd kernel-size <=63 required")
    if cli.inverse_iterations < 0:
        parser.error("inverse-iterations must be nonnegative")
    if cli.dense_blur_backend == "triton" and (cli.acceleration == "eager" or cli.kernel_size > 15):
        parser.error(
            "dense Triton blur requires acceleration auto/triton and kernel-size <=15; select --dense-blur-backend torch otherwise"
        )
    return parser, cli


def main():
    parser, cli = parse_arguments()
    try:
        import torch
        import scipy
    except ModuleNotFoundError as exc:
        parser.exit(
            2, f"UNABLE TO TEST: missing {exc.name}; this benchmark requires PyTorch and SciPy.\n"
        )
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.exit(
            2, "UNABLE TO TEST: a working CUDA GPU is required; no timings were produced.\n"
        )
    torch.cuda.set_device(device)
    from paste_gpu_batch_adapter import ADAPTER_VERSION

    if ADAPTER_VERSION != BENCHMARK_VERSION:
        parser.error(
            f"requires adapter {BENCHMARK_VERSION}; loaded {ADAPTER_VERSION}; extract the complete matching bundle"
        )
    print(
        json.dumps(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "benchmark_path": str(Path(__file__).resolve()),
                "python": sys.version,
                "python_executable": sys.executable,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "scipy": scipy.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "arguments": vars(cli),
                "fixture": "synthetic motion/gaussian; shared preparation once per frame; GPU image pixels; live CPU calibration",
            },
            indent=2,
        )
    )
    with torch.inference_mode():
        paths = make_paths(cli, device)
        run_checks(paths)
        if cli.check_only:
            return
        report = {path.name: measure_path(path) for path in paths}
        old = report["original_shared_plus_dense"]["wall_ms"]["mean"]
        dense = report["optimized_shared_plus_dense"]["wall_ms"]["mean"]
        separable = report["optimized_shared_plus_separable"]["wall_ms"]["mean"]
        report["shared_preparation_optimization_saving_ms"] = round(old - dense, 3)
        report["additional_separable_saving_ms"] = round(dense - separable, 3)
        report["total_saving_vs_original_shared_dense_ms"] = round(old - separable, 3)
        if cli.diagnose:
            report["isolated_shared_preparation"] = {
                path.name: measure_path(path, shared_only=True) for path in paths[:3]
            }
            report["last_builder_info"] = {
                path.name: path.builder.last_info for path in paths if path.optimized
            }
        print(json.dumps(report, indent=2))
        print(
            "All full-path timings include filtering, staging, grids, kernel preparation/cache lookup, adapter and pinned NumPy output. "
            "Isolated preparation timings are not additive to full-path timings. Synthetic callbacks and cache hits may differ from production."
        )


if __name__ == "__main__":
    main()

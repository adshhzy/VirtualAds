"""CPU-only contracts for isolated sources and complete-output benchmark checks."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import subprocess
import json
import hashlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import benchmark_upload_comparison as benchmark
from benchmarks.source_layout import REPOSITORY_ROOT

POLICY = [{"chunk_size": 0, "reason": "unsupported_backend_or_output"}]


def minimal_sources(directory, renderer_source=None):
    for name in benchmark.REQUIRED_MODULES:
        (directory / f"{name}.py").write_text("", encoding="utf-8")
    (directory / "paste_render_executor.py").write_text(
        "class RenderOptions: pass\n", encoding="utf-8"
    )
    (directory / "paste_renderer.py").write_text(
        renderer_source or "from .paste_render_executor import RenderOptions\n",
        encoding="utf-8",
    )


def small_fixture():
    template = {"frame_id": 0}
    for route in range(1, 9):
        template[f"ad_params_{route}"] = [{"bbox": (10, 10, 12, 12)} for _ in range(16)]
    frame = np.zeros((2, 3, 3), np.uint8)
    references = [np.full(frame.shape, value, np.uint8) for value in (10, 9)]
    return template, frame, references


class FakeWorker:
    def __init__(self, *, reuse=False):
        self.reuse = reuse
        self.buffer = None
        self.last_execution = POLICY
        self.calls = 0
        self.closed = False

    def render_img(self, data, frame, assets):
        self.calls += 1
        value = data["ad_params_1"][0]["bbox"][0]
        if self.buffer is None or not self.reuse:
            self.buffer = np.empty((8, *frame.shape), np.uint8)
        self.buffer.fill(value)
        return self.buffer

    def close(self):
        self.closed = True
        self.buffer = None


class SourceLoadingTests(unittest.TestCase):
    def test_relative_classes_are_isolated_and_cleanup_is_scoped(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            minimal_sources(Path(first))
            minimal_sources(Path(second))
            with benchmark.load_sources(first) as left, benchmark.load_sources(second) as right:
                self.assertIsNot(left.api.RenderOptions, right.api.RenderOptions)
                self.assertTrue(left.api.RenderOptions.__module__.startswith(left.package_name))
                self.assertTrue(right.api.RenderOptions.__module__.startswith(right.package_name))
                left_name, right_name = left.package_name, right.package_name
                left.close()
                self.assertNotIn(left_name, sys.modules)
                self.assertIn(right_name, sys.modules)
                right.verify()
            self.assertFalse(any(name.startswith(right_name) for name in sys.modules))
            self.assertIn("virtualads.paste_renderer", sys.modules)

    def test_reexport_from_current_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            minimal_sources(
                Path(directory), "from virtualads.paste_renderer import ImageRenderer\n"
            )
            before = set(sys.modules)
            with self.assertRaisesRegex(RuntimeError, "outside its package"):
                benchmark.load_sources(directory)
            self.assertFalse(
                any(
                    name.startswith("_paste_upload_baseline_") for name in set(sys.modules) - before
                )
            )

    def test_incomplete_snapshot_names_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            minimal_sources(Path(directory), "if False:\n    from .paste_missing import value\n")
            with self.assertRaisesRegex(ValueError, "paste_missing"):
                benchmark.load_sources(directory)

    def test_source_edit_after_loading_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            minimal_sources(path)
            with benchmark.load_sources(path) as loaded:
                (path / "paste_gpu_batch.py").write_text("# changed\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "changed after loading"):
                    loaded.verify()

    def test_loaded_module_path_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            minimal_sources(Path(directory))
            with benchmark.load_sources(directory) as loaded:
                with patch.object(loaded.api, "__file__", str(Path(directory) / "wrong.py")):
                    with self.assertRaisesRegex(RuntimeError, "wrong source"):
                        loaded.verify()

    def test_packaged_snapshot_loads_implementation_without_current_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "virtualads"
            runtime.mkdir()
            (runtime / "__init__.py").write_text("# snapshot package\n", encoding="utf-8")
            minimal_sources(runtime)
            with benchmark.load_sources(root) as loaded:
                self.assertEqual(loaded.directory, runtime)
                self.assertTrue(loaded.api.RenderOptions.__module__.startswith(loaded.package_name))
                self.assertIsNot(loaded.api.RenderOptions, benchmark.current_api.RenderOptions)
                loaded.verify()
            result = benchmark.source_hashes(root)
            self.assertIn("virtualads/paste_renderer.py", result["files"])
            self.assertIn("virtualads/__init__.py", result["files"])

    def test_current_provenance_hashes_real_runtime_and_layout_helper(self):
        result = benchmark.source_hashes(REPOSITORY_ROOT, include_harness=True)
        for name in (
            "virtualads/paste_renderer.py",
            "virtualads/paste_gpu_batch.py",
            "virtualads/shared_paste_example.py",
            "benchmarks/source_layout.py",
        ):
            expected = hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(result["files"][name], expected)
        self.assertFalse(any(name.startswith("compat/") for name in result["files"]))

    def test_dynamic_source_dir_selects_requested_flat_or_packaged_fixtures(self):
        # A fresh process is intentional: this standalone command rejects mixing
        # already-imported runtimes from unrelated snapshots in one namespace.
        for packaged in (False, True):
            with self.subTest(packaged=packaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = root / "virtualads" if packaged else root
                harness = root / "benchmarks" if packaged else root
                runtime.mkdir(exist_ok=True)
                harness.mkdir(exist_ok=True)
                if packaged:
                    (runtime / "__init__.py").write_text("", encoding="utf-8")
                    (harness / "__init__.py").write_text("", encoding="utf-8")
                for name in (
                    "paste_gpu_batch_adapter",
                    "paste_gpu_batch",
                    "paste_gpu_kernels",
                    "paste_gaussian_kernels",
                    "paste_shared_params",
                    "paste_shared_kernels",
                    "paste_shared_types",
                    "paste_camera_filter",
                ):
                    (runtime / (name + ".py")).write_text("MARKER = 'snapshot'\n", encoding="utf-8")
                for name in ("benchmark_paste_gpu_batch", "benchmark_shared_paste"):
                    (harness / (name + ".py")).write_text("MARKER = 'snapshot'\n", encoding="utf-8")
                code = (
                    "from pathlib import Path; import json, sys; "
                    "from benchmarks.benchmark_dynamic_batch import "
                    "validate_source_directory, load_source_modules; "
                    "root=Path(sys.argv[1]); names=validate_source_directory(root); "
                    "modules=load_source_modules(root,names); "
                    "print(json.dumps([(m.MARKER,m.__file__) for m in modules]))"
                )
                result = subprocess.run(
                    [sys.executable, "-B", "-c", code, str(root)],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                rows = json.loads(result.stdout)
                self.assertEqual(len(rows), 3)
                for marker, filename in rows:
                    self.assertEqual(marker, "snapshot")
                    self.assertTrue(Path(filename).is_relative_to(root))


class CompleteOutputTests(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (2, 3, 3)
        self.output = np.zeros((8, *self.frame_shape), np.uint8)

    def validate(self, output, reference, execution=POLICY):
        benchmark.validate_output(output, reference, execution, POLICY, self.frame_shape)

    def test_compacted_reference_checks_the_eighth_route(self):
        reference, compacted = benchmark.compact_reference(self.output)
        self.assertTrue(compacted)
        self.assertEqual(reference.shape, self.frame_shape)
        self.validate(self.output, reference)
        self.output[7, 1, 2, 2] = 1
        with self.assertRaisesRegex(AssertionError, "changed_channels.*1"):
            self.validate(self.output, reference)
        self.assertEqual(int(reference.max()), 0)

    def test_distinct_routes_keep_full_reference(self):
        self.output[7] = 4
        reference, compacted = benchmark.compact_reference(self.output)
        self.assertFalse(compacted)
        self.assertEqual(reference.shape, self.output.shape)
        self.validate(self.output, reference)
        self.output[0, 0, 0, 0] = 1
        with self.assertRaises(AssertionError):
            self.validate(self.output, reference)

    def test_wrong_output_protocol_is_rejected(self):
        for output in (self.output.astype(np.float32), self.output[:7], self.output[..., ::-1]):
            with self.subTest(shape=output.shape, dtype=output.dtype):
                with self.assertRaises(AssertionError):
                    self.validate(output, self.output.copy())

    def test_pixel_equality_does_not_hide_changed_policy(self):
        with self.assertRaisesRegex(AssertionError, "execution policy"):
            self.validate(self.output, self.output.copy(), [{"chunk_size": 4}])

    def test_ownership_rejects_reused_buffer(self):
        template, frame, references = small_fixture()
        inputs = benchmark.Inputs(template, ((0, 0), (-1, 0)))
        with self.assertRaisesRegex(AssertionError, "share storage"):
            benchmark.retain_output_check(
                FakeWorker(reuse=True), inputs, frame, {}, references, [POLICY, POLICY]
            )

    def test_ownership_retains_valid_output_after_close(self):
        template, frame, references = small_fixture()
        worker = FakeWorker()
        inputs = benchmark.Inputs(template, ((0, 0), (-1, 0)))
        held, snapshot = benchmark.retain_output_check(
            worker, inputs, frame, {}, references, [POLICY, POLICY]
        )
        worker.close()
        self.assertTrue(np.array_equal(held, snapshot))

    def test_cpu_phase_checks_every_measured_call_and_closes(self):
        template, frame, references = small_fixture()
        worker = FakeWorker()
        args = SimpleNamespace(device="cuda:0", warmup=0, steps=3)
        empty = {
            "assets": {"retained_stack_bytes": 0},
            "mask": {"workspace_bytes": 0},
            "gaussian": {"entries": 0},
            "history_entries": 0,
            "source_references": 0,
            "owner_material_groups": 0,
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, "make_worker", return_value=worker))
            stack.enter_context(patch.object(benchmark, "check_plan"))
            stack.enter_context(patch.object(benchmark, "effective_options", return_value={}))
            stack.enter_context(patch.object(benchmark, "worker_memory", return_value=empty))
            stack.enter_context(patch.object(benchmark.torch.cuda, "synchronize"))
            stack.enter_context(patch.object(benchmark.torch.cuda, "reset_peak_memory_stats"))
            validation = stack.enter_context(
                patch.object(benchmark, "validate_output", wraps=benchmark.validate_output)
            )
            phase = benchmark.run_phase(
                None,
                "current",
                0,
                template,
                ((0, 0), (-1, 0)),
                frame,
                {},
                references,
                [POLICY, POLICY],
                args,
            )
        self.assertTrue(worker.closed)
        self.assertEqual(worker.calls, 9 + 3 + 2)
        self.assertEqual(validation.call_count, 3 + 2)
        self.assertEqual(phase["checked_calls"], 3)
        self.assertEqual(phase["checked_route_images"], 24)
        self.assertEqual(phase["pattern_indices"], [0, 1, 0])
        self.assertEqual(len(phase["samples_ms"]), 3)
        self.assertTrue(phase["ownership"]["unchanged_after_close"])


if __name__ == "__main__":
    unittest.main()

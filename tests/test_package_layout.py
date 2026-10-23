"""Package/import identity contracts after grouping the repository by responsibility."""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class PackageLayoutTests(unittest.TestCase):
    def test_legacy_module_names_alias_the_single_runtime(self):
        root = Path(__file__).resolve().parents[1]
        names = [path.stem for path in sorted((root / "compat").glob("*.py"))]
        self.assertEqual(len(names), 20)
        for name in names:
            with self.subTest(name=name):
                legacy = importlib.import_module(name)
                canonical = importlib.import_module("virtualads." + name)
                self.assertIs(legacy, canonical)
                self.assertEqual(Path(canonical.__file__).resolve().parent, root / "virtualads")

    def test_patch_through_old_module_reaches_implementation(self):
        canonical = importlib.import_module("virtualads.paste_shared_params")
        original = canonical.SharedParameterBuilder
        marker = object()
        with patch("paste_shared_params.SharedParameterBuilder", marker):
            self.assertIs(canonical.SharedParameterBuilder, marker)
        self.assertIs(canonical.SharedParameterBuilder, original)

    def test_public_exports_are_the_existing_api(self):
        package = importlib.import_module("virtualads")
        api = importlib.import_module("virtualads.paste_renderer")
        for name in package.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(package, name), getattr(api, name))
        with self.assertRaises(AttributeError):
            getattr(package, "missing_public_name")

    def test_installed_imports_work_outside_checkout_without_gpu_initialization(self):
        code = (
            "import sys, virtualads; assert 'torch' not in sys.modules; "
            "from virtualads import ImageRenderer, RenderOptions; "
            "import paste_renderer, virtualads.paste_renderer as actual; "
            "assert paste_renderer is actual; "
            "assert ImageRenderer is actual.ImageRenderer; "
            "import torch; assert not torch.cuda.is_initialized()"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

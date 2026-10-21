"""CPU-only checks that the public API retains its Triton input guards."""

import unittest
from unittest.mock import patch

import torch

import paste_gpu_kernels as kernels


class PublicWarpGuards(unittest.TestCase):
    def setUp(self):
        self.arguments = (
            torch.zeros((1, 4, 3, 5)),
            torch.zeros((1, 2, 4, 2)),
            torch.ones((1, 1, 1, 1)),
            torch.zeros((1, 4, 6, 8)),
            (1, 1, 5, 3),
        )

    def test_cpu_input_rejected_before_launch(self):
        with patch.object(kernels, "_warp_paste_into_supported") as launch:
            with self.assertRaisesRegex(ValueError, "float32 CUDA"):
                kernels.warp_paste_into(*self.arguments)
            launch.assert_not_called()

    def test_unsupported_padding_rejected_before_launch(self):
        with patch.object(kernels, "supported", return_value=True):
            with patch.object(kernels, "_warp_paste_into_supported") as launch:
                with self.assertRaisesRegex(ValueError, "zeros/border"):
                    kernels.warp_paste_into(*self.arguments, padding_mode="reflection")
                launch.assert_not_called()

    def test_validated_wrapper_preserves_sampling_options(self):
        with patch.object(kernels, "supported", return_value=True):
            with patch.object(kernels, "_warp_paste_into_supported") as launch:
                kernels.warp_paste_into(
                    *self.arguments, align_corners=True, padding_mode="border", legacy_alpha=False
                )
                launch.assert_called_once_with(
                    *self.arguments, align_corners=True, padding_mode="border", legacy_alpha=False
                )


if __name__ == "__main__":
    unittest.main()

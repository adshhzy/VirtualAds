"""Explicit per-frame motion kernel payload used by the shared preparation path."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MotionKernelSpec:
    """Dense reference kernel plus optional shared (vertical, horizontal) factors.

    The builder supplies float32 ``kernel[4, 1, kernel_height, kernel_width]``. Optional
    ``factors`` contain ``ky[kernel_height]`` and ``kx[kernel_width]`` on the same
    device. The factors are shared by all batch members and RGBA channels.
    Factors are an explicit contract: outer(ky,kx) must reconstruct every RGBA
    channel of kernel to the chosen numerical tolerance. The builder checks this
    on the CPU kernel before upload; the rendering loop does not scan GPU values.
    """

    kernel: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor] | None = None

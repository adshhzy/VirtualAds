"""Explicit per-frame motion kernel payload used by the shared preparation path."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MotionKernelSpec:
    """Dense reference kernel plus optional shared (vertical, horizontal) factors.

    Factors are an explicit contract: outer(ky,kx) must reconstruct every RGBA
    channel of kernel to the chosen numerical tolerance. The builder checks this
    on the CPU kernel before upload; the rendering loop does not scan GPU values.
    """
    kernel: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor] | None = None

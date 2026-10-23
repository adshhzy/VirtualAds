"""Compatibility alias; implementation lives in virtualads.paste_gpu_batch."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.paste_gpu_batch")

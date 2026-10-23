"""Compatibility alias; implementation lives in virtualads.paste_shared_mask."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.paste_shared_mask")

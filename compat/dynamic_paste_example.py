"""Compatibility alias; implementation lives in virtualads.dynamic_paste_example."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.dynamic_paste_example")

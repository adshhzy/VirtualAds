"""Compatibility alias; implementation lives in virtualads.shared_paste_example."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.shared_paste_example")

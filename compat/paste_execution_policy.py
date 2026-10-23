"""Compatibility alias; implementation lives in virtualads.paste_execution_policy."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.paste_execution_policy")

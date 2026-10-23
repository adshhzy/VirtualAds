"""Compatibility alias; implementation lives in virtualads.paste_asset_cache."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("virtualads.paste_asset_cache")

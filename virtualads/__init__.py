"""VirtualAds public API; the renderer is loaded only when an export is requested."""

from importlib import import_module

__all__ = [
    "ImageRenderer",
    "RenderOptions",
    "RenderExecutor",
    "BoundedCameraHistory",
    "RENDERER_VERSION",
    "initialize_renderer",
    "render_frame",
    "release_renderer",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".paste_renderer", __name__), name)
    globals()[name] = value
    return value

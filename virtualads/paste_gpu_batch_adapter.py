"""Compatibility names and original binding function for the render executor.

New code can import RenderExecutor and RenderOptions from paste_render_executor.
AdapterOptions and PasteBatchAdapter remain aliases of those exact classes; the
original binding function still owns its instance at owner._paste_batch_adapter.
Private helper reads/imports remain available for existing integrations. Patching
implementation globals belongs in paste_render_executor, where they are executed.
"""

from __future__ import annotations

if __package__:
    from . import paste_render_executor as _executor
else:
    import paste_render_executor as _executor

RenderOptions = _executor.RenderOptions
RenderExecutor = _executor.RenderExecutor
AdapterOptions = RenderOptions
PasteBatchAdapter = RenderExecutor
ADAPTER_VERSION = _executor.RENDERER_VERSION

# Existing tests and integrations import these helpers directly.
_image_plan = _executor._image_plan
_SizeAssetCache = _executor._SizeAssetCache


def __getattr__(name):
    """Keep reads of former module helpers compatible after their move."""
    return getattr(_executor, name)


def paste_gpu_batch_adapter(
    self, frame_input, data_dict: dict, frame_id: int, way_id: int, paste_args
):
    """可直接绑定到原类；入参顺序与原paste_gpu完全相同。

    class YourRenderer:
        paste_gpu_batch = paste_gpu_batch_adapter

    显式配置：self._paste_batch_adapter = PasteBatchAdapter(AdapterOptions(...))
    默认返回numpy uint8：单组[H,W,3]，多组[B,H,W,3]。
    squeeze_single=False可让单组也保留batch维；不自动编译，保留原Alpha逻辑。
    """
    adapter = getattr(self, "_paste_batch_adapter", None)
    if adapter is None:
        adapter = PasteBatchAdapter()
        self._paste_batch_adapter = adapter
    if not isinstance(adapter, PasteBatchAdapter):
        raise TypeError("self._paste_batch_adapter must be a PasteBatchAdapter instance")
    return adapter(self, frame_input, data_dict, frame_id, way_id, paste_args)

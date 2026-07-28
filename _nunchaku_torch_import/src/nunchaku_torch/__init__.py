from importlib import import_module

from .__version__ import __version__


_EXPORTS = {
    "GenerationConfig": ("nunchaku_torch.zimage", "GenerationConfig"),
    "generate_image": ("nunchaku_torch.zimage", "generate_image"),
    "save_image": ("nunchaku_torch.zimage", "save_image"),
    "resolve_device": ("nunchaku_torch.device", "resolve_device"),
    "default_dtype": ("nunchaku_torch.device", "default_dtype"),
    "NunchakuZImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuZImageTransformer2DModel",
    ),
    "NunchakuQwenImageTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuQwenImageTransformer2DModel",
    ),
    "NunchakuFluxTransformer2DModel": (
        "nunchaku_torch.models.transformers",
        "NunchakuFluxTransformer2DModel",
    ),
}

_DEFERRED = {
    "NunchakuFluxTransformer2DModelV2": "Flux V2 runtime is not enabled in this standalone build yet.",
    "NunchakuSanaTransformer2DModel": "Sana runtime still depends on accelerator-backed components in this standalone build.",
    "NunchakuT5EncoderModel": "T5 encoder quantized runtime is not enabled in this standalone build yet.",
}

__all__ = [*sorted(_EXPORTS.keys()), *sorted(_DEFERRED.keys()), "__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    if name in _DEFERRED:
        from ._unsupported import missing_runtime_feature

        missing_runtime_feature(name, _DEFERRED[name])
    raise AttributeError(name)

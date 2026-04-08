from importlib import import_module


_EXPORTS = {
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
    "SVDQW4A4Linear": ("nunchaku_torch.models.linear", "SVDQW4A4Linear"),
}

_DEFERRED = {
    "NunchakuFluxTransformer2DModelV2": "Flux V2 runtime is not enabled in this standalone build yet.",
    "NunchakuSanaTransformer2DModel": "Sana runtime still depends on accelerator-backed components in this standalone build.",
    "NunchakuT5EncoderModel": "T5 encoder quantized runtime is not enabled in this standalone build yet.",
}

__all__ = [*sorted(_EXPORTS.keys()), *sorted(_DEFERRED.keys())]


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    if name in _DEFERRED:
        from .._unsupported import missing_runtime_feature

        missing_runtime_feature(name, _DEFERRED[name])
    raise AttributeError(name)

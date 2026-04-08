from importlib import import_module


_EXPORTS = {
    "NunchakuZImageTransformer2DModel": (
        "nunchaku_torch.models.transformers.transformer_zimage",
        "NunchakuZImageTransformer2DModel",
    ),
    "NunchakuQwenImageTransformer2DModel": (
        "nunchaku_torch.models.transformers.transformer_qwenimage",
        "NunchakuQwenImageTransformer2DModel",
    ),
    "NunchakuFluxTransformer2DModel": (
        "nunchaku_torch.models.transformers.transformer_flux",
        "NunchakuFluxTransformer2DModel",
    ),
}

_DEFERRED = {
    "NunchakuFluxTransformer2DModelV2": "Flux V2 runtime is not enabled in this standalone build yet.",
    "NunchakuSanaTransformer2DModel": "Sana runtime still depends on accelerator-backed components in this standalone build.",
}

__all__ = [
    *sorted(_EXPORTS.keys()),
    *sorted(_DEFERRED.keys()),
    "transformer_flux",
    "transformer_qwenimage",
    "transformer_zimage",
]


def __getattr__(name: str):
    if name == "transformer_zimage":
        return import_module("nunchaku_torch.models.transformers.transformer_zimage")
    if name == "transformer_qwenimage":
        return import_module(
            "nunchaku_torch.models.transformers.transformer_qwenimage"
        )
    if name == "transformer_flux":
        return import_module("nunchaku_torch.models.transformers.transformer_flux")
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    if name in _DEFERRED:
        from ..._unsupported import missing_runtime_feature

        missing_runtime_feature(name, _DEFERRED[name])
    raise AttributeError(name)

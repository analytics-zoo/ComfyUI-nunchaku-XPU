from importlib import import_module


_EXPORTS = {
    "NunchakuZSingleStreamAttnProcessor": (
        "nunchaku_torch.models.attention_processors.zimage",
        "NunchakuZSingleStreamAttnProcessor",
    ),
    "NunchakuQwenImageNaiveFA2Processor": (
        "nunchaku_torch.models.attention_processors.qwenimage",
        "NunchakuQwenImageNaiveFA2Processor",
    ),
}

_EXPORTS["NunchakuFluxAttnProcessor"] = (
    "nunchaku_torch.models.attention_processors.flux",
    "NunchakuFluxAttnProcessor",
)
_EXPORTS["NunchakuFluxSingleAttnProcessor"] = (
    "nunchaku_torch.models.attention_processors.flux",
    "NunchakuFluxSingleAttnProcessor",
)

_DEFERRED = {}

__all__ = [*sorted(_EXPORTS.keys()), *sorted(_DEFERRED.keys())]


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name), attr_name)
    if name in _DEFERRED:
        from ..._unsupported import missing_runtime_feature

        missing_runtime_feature(name, _DEFERRED[name])
    raise AttributeError(name)

def missing_runtime_feature(name: str, detail: str) -> None:
    raise ImportError(
        f"{name} is not available in the current standalone runtime build. {detail}".strip()
    )

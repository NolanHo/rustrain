from pathlib import Path


def require_complete_qwen_model_path(path: str, summary_path: Path | str) -> str:
    model_path = Path(path)
    missing = [
        name
        for name in ("config.json", "tokenizer.json", "model.safetensors")
        if not (model_path / name).exists()
    ]
    if missing:
        raise SystemExit(
            f"{summary_path} model_path {model_path} is not a complete Qwen checkpoint; missing {missing}"
        )
    return str(model_path)


def require_complete_qwen_base_model_path(manifest: dict, manifest_path: Path | str) -> str:
    return require_complete_qwen_model_path(manifest["base_model_path"], manifest_path)


def require_complete_qwen_manifest_paths(manifest: dict, manifest_path: Path | str) -> str:
    base_model_path = Path(require_complete_qwen_base_model_path(manifest, manifest_path))
    tokenizer_path = Path(manifest["tokenizer_path"])
    if not tokenizer_path.exists():
        raise SystemExit(f"{manifest_path} tokenizer_path {tokenizer_path} does not exist")
    if tokenizer_path.parent != base_model_path:
        raise SystemExit(
            f"{manifest_path} tokenizer_path {tokenizer_path} must live under base_model_path {base_model_path}"
        )
    return str(base_model_path)

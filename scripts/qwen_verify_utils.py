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

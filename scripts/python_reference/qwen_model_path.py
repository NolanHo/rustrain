"""Shared local Qwen checkpoint path resolution for Python fixtures."""

from __future__ import annotations

from pathlib import Path


def resolve_qwen_model_path(model_path: Path) -> Path:
    if _qwen_model_path_is_complete(model_path):
        return model_path
    model_dir_name = model_path.name
    if not model_dir_name:
        raise FileNotFoundError(
            f"Qwen model path {model_path} is incomplete and has no model directory name"
        )
    root = model_path.parent
    hub_root = root / "hub"
    hub_suffix = f"--{model_dir_name}"
    hub_model_dirs = sorted(
        path
        for path in hub_root.glob("models--*")
        if path.name.startswith("models--") and path.name.endswith(hub_suffix)
    )
    if not hub_model_dirs:
        raise FileNotFoundError(
            f"Qwen model path {model_path} is incomplete and no matching HF hub cache "
            f"entry was found under {hub_root}"
        )
    candidates = []
    for hub_model_dir in hub_model_dirs:
        snapshots_dir = hub_model_dir / "snapshots"
        if snapshots_dir.is_dir():
            candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
    for candidate in sorted(candidates, reverse=True):
        if _qwen_model_path_is_complete(candidate):
            return candidate
    raise FileNotFoundError(
        f"Qwen model path {model_path} is incomplete and no complete HF hub snapshot "
        f"exists under {hub_root}"
    )


def _qwen_model_path_is_complete(model_path: Path) -> bool:
    return all(
        (model_path / name).exists()
        for name in ("config.json", "tokenizer.json", "model.safetensors")
    )

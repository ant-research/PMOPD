"""Utilities for applying LoRA adapters or PMOPD linear-matrix snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Tuple

import torch


def merge_lora_adapter_(model, adapter_path: str) -> Tuple[int, float]:
    """Merge a standard PEFT LoRA adapter without importing PEFT."""
    path = Path(adapter_path)
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    safe_weights = path / "adapter_model.safetensors"
    if safe_weights.exists():
        from safetensors.torch import load_file
        state = load_file(str(safe_weights), device="cpu")
    else:
        state = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)

    named = dict(model.named_parameters())
    rank = float(config["r"])
    scale = float(config.get("lora_alpha", rank)) / rank
    pairs = {}
    for key, value in state.items():
        if key.endswith(".lora_A.weight"):
            side, stem = "A", key[: -len(".lora_A.weight")]
        elif key.endswith(".lora_B.weight"):
            side, stem = "B", key[: -len(".lora_B.weight")]
        else:
            continue
        for prefix in ("base_model.model.", "base_model."):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        pairs.setdefault(stem, {})[side] = value

    merged = 0
    with torch.no_grad():
        for stem, matrices in pairs.items():
            weight_name = stem + ".weight"
            if weight_name not in named or set(matrices) != {"A", "B"}:
                continue
            weight = named[weight_name]
            delta = matrices["B"].float() @ matrices["A"].float()
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype), alpha=scale)
            merged += 1
    if merged != len(pairs):
        raise RuntimeError(f"LoRA merge incomplete: merged={merged}, pairs={len(pairs)}")
    return merged, scale


def load_linear_snapshot_(model, snapshot_dir: str, predicate: Callable[[str], bool] | None = None) -> int:
    """Apply ``model.safetensors`` matrices to matching model parameters in place."""
    from safetensors.torch import load_file

    path = Path(snapshot_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"snapshot not found: {path}")
    state = load_file(str(path), device="cpu")
    named = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for name, value in state.items():
            if name in named and (predicate is None or predicate(name)):
                target = named[name]
                target.copy_(value.to(device=target.device, dtype=target.dtype))
                loaded += 1
    if loaded == 0:
        raise RuntimeError(f"no matching matrices were loaded from {path}")
    return loaded


def apply_model_override_(model, path: str) -> str:
    """Apply either a LoRA adapter directory or a linear snapshot directory."""
    root = Path(path)
    if (root / "adapter_config.json").is_file():
        merged, scale = merge_lora_adapter_(model, str(root))
        return f"merged LoRA matrices={merged}, scale={scale:g}"
    loaded = load_linear_snapshot_(model, str(root))
    return f"loaded snapshot matrices={loaded}"

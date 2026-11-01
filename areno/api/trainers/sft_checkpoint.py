"""Dataset cursor metadata for resumable, deterministic text SFT loops."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FILENAME = "sft_training_state.json"
FORMAT = "areno-sft-cursor-v1"


def dataset_signature(config, dataset):
    """Bind the ordered normalized rows and the filtering/batching contract."""
    digest = hashlib.sha256()
    try:
        for index in range(len(dataset)):
            raw = json.dumps(dict(dataset[index]), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("SFT cursor resume supports JSON-serializable text/encoded dataset rows") from exc
    loader = getattr(config, "dataset_loader_fn", None)
    loader_name = None
    loader_path = None
    if loader:
        if ":" in loader:
            loader_text, loader_name = loader.rsplit(":", 1)
            if not loader_text or not loader_name:
                raise ValueError("SFT dataset loader must be a path or path:function")
        else:
            loader_text, loader_name = loader, "load_training_dataset"
        loader_path = Path(loader_text)
    return {
        "dataset_rows": len(dataset),
        "dataset_sha256": digest.hexdigest(),
        "loader_sha256": hashlib.sha256(loader_path.read_bytes()).hexdigest() if loader_path else None,
        "loader_function": loader_name,
        "batch_size": config.batch_size,
        "mini_bs": config.mini_bs,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_prompt_tokens": config.max_prompt_tokens,
        "max_new_tokens": config.max_new_tokens,
        "chat_template_enable_thinking": getattr(config, "chat_template_enable_thinking", None),
    }


def load_cursor(directory, signature):
    """Reject a missing/mismatched loop cursor before initializing workers."""
    path = Path(directory)
    try:
        value = json.loads((path / FILENAME).read_text(encoding="utf-8"))
        native = json.loads((path / "training_state.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("SFT resume requires a full training checkpoint with sft_training_state.json") from exc
    if value.get("format") != FORMAT or value.get("signature") != signature:
        raise ValueError("SFT checkpoint dataset, loader, filtering or batching settings differ")
    for key in ("epoch", "next_batch", "completed_steps"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"SFT checkpoint {key} must be a non-negative integer")
    if value["completed_steps"] < 1 or value["next_batch"] < 1:
        raise ValueError("SFT checkpoint must describe at least one completed batch")
    if native.get("format") != "areno-training-v1" or type(native.get("global_step")) is not int:
        raise ValueError("SFT checkpoint requires valid native training-state metadata")
    if native["global_step"] < value["completed_steps"]:
        raise ValueError("SFT checkpoint native clock precedes completed Trainer calls")
    if (
        type(native.get("trainer_global_step")) is not int
        or native["trainer_global_step"] != value["completed_steps"] - 1
    ):
        raise ValueError("SFT checkpoint public Trainer clock differs from its dataset cursor")
    if not (path / "training_state.pt").is_file():
        raise ValueError("SFT resume requires native optimizer training_state.pt")
    return value


def save_cursor(directory, signature, *, epoch, next_batch, completed_steps):
    """Commit the next deterministic batch only after native checkpoint save."""
    value = {
        "format": FORMAT,
        "signature": signature,
        "epoch": epoch,
        "next_batch": next_batch,
        "completed_steps": completed_steps,
    }
    path = Path(directory) / FILENAME
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)

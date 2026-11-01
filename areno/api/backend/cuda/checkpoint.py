"""CUDA checkpoint adapter and public Trainer clock metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_trainer_step(path: str) -> int:
    """Read the public call clock independently of native optimizer updates.

    Older checkpoints lack this clock. Preserve their historical update-based
    fallback, with an explicit warning: a multi-update call count cannot be
    reconstructed from the optimizer count alone.
    """

    metadata = json.loads((Path(path) / "training_state.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "areno-training-v1":
        raise ValueError("unsupported training-state checkpoint format")
    native_step = metadata.get("global_step")
    if type(native_step) is not int or native_step < 0:
        raise ValueError("training-state global_step must be a non-negative integer")
    if "trainer_global_step" in metadata:
        trainer_step = metadata["trainer_global_step"]
        if type(trainer_step) is not int or trainer_step < -1:
            raise ValueError("training-state trainer_global_step must be an integer >= -1")
        return trainer_step
    logger.warning(
        "Legacy checkpoint has no trainer_global_step; public logging uses optimizer count - 1. "
        "The original Trainer call count cannot be reconstructed when calls contain multiple optimizer updates."
    )
    return native_step - 1


def save_checkpoint(engine: Any, path: str, *, trainer_global_step: int | None = None) -> str:
    """Persist native state, then append its separate public call clock."""

    if trainer_global_step is not None and (type(trainer_global_step) is not int or trainer_global_step < -1):
        raise ValueError("trainer_global_step must be an integer >= -1")
    saved_path = engine.save_checkpoint(path)
    metadata_path = Path(saved_path) / "training_state.json"
    # Weight-only and adapter checkpoints retain their existing format.
    if trainer_global_step is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["trainer_global_step"] = trainer_global_step
        pending = metadata_path.with_name("training_state.json.tmp")
        pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pending.replace(metadata_path)
    return saved_path


__all__ = ["load_trainer_step", "save_checkpoint"]

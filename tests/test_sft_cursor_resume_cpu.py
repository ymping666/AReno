"""SFT resume continues the filtered dataset cursor and total call budget."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from areno.api.trainer_config import TrainerConfig
from areno.api.trainers.sft import SFTTrainer
from areno.api.trainers.sft_checkpoint import FILENAME, dataset_signature, load_cursor


class LoopBackend:
    """CPU loop spy: optimizer update count can exceed the call count."""

    def __init__(self, resume=None):
        self.resume = resume
        self.initialized = False
        self.closed = False
        self.batches = []
        self.updates = 0
        self.weight = 0
        self._ctx = SimpleNamespace(global_step=-1)

    def init(self):
        self.initialized = True
        if self.resume:
            value = json.loads((Path(self.resume) / "loop-spy.json").read_text())
            self.weight = value["weight"]
            self.updates = value["updates"]
            self._ctx.global_step = value["step"]

    def close(self):
        self.closed = True

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)

    def get_processor(self):
        return None

    def train(self, batch, loss, *, mini_bs, gradient_accumulation_steps):
        del loss, gradient_accumulation_steps
        ids = [tuple(row.tokens) for row in batch]
        self.batches.append(ids)
        self._ctx.global_step += 1
        for start in range(0, len(batch), mini_bs):
            self.updates += 1
            self.weight += sum(sum(row.tokens) for row in batch[start : start + mini_bs])
        return {"loss": 0.5}

    def save_checkpoint(self, directory):
        path = Path(directory)
        path.mkdir(parents=True)
        (path / "loop-spy.json").write_text(
            json.dumps({"weight": self.weight, "updates": self.updates, "step": self._ctx.global_step})
        )
        (path / "training_state.json").write_text(
            json.dumps(
                {
                    "format": "areno-training-v1",
                    "global_step": self.updates,
                    "trainer_global_step": self._ctx.global_step,
                }
            )
        )
        (path / "training_state.pt").write_bytes(b"CPU loop spy; no native optimizer payload")
        return str(path)


def rows():
    result = [{"tokens": [1, i, 3], "prompt_mask": [True, False, False]} for i in range(2, 7)]
    result.insert(1, {"tokens": [1] * 12, "prompt_mask": [True] * 10 + [False] * 2})
    return result


def config(**kwargs):
    return TrainerConfig(
        algo="sft",
        backend="cuda",
        ckpt="unused",
        dataset_path="unused",
        world_size=1,
        tp_size=1,
        galore_rank=2,
        epochs=3,
        batch_size=2,
        mini_bs=1,
        max_prompt_tokens=4,
        max_new_tokens=4,
        **kwargs,
    )


def fit(cfg, backend, dataset=None):
    trainer = SFTTrainer(
        cfg, instance=backend, dataset=rows() if dataset is None else dataset, reward_fn=None, loss_fn=lambda *_: None
    )
    trainer.fit()
    return trainer


def make_checkpoint(tmp_path):
    first = LoopBackend()
    fit(config(max_steps=2, save_path=str(tmp_path), save_interval=2), first)
    return tmp_path / "step_000002", first


def test_resume_crosses_filtered_tail_and_epoch_without_replay_cpu(tmp_path):
    whole = LoopBackend()
    fit(config(max_steps=5), whole)
    checkpoint, first = make_checkpoint(tmp_path)
    second = LoopBackend(checkpoint)
    fit(config(max_steps=5, resume_from_checkpoint=str(checkpoint)), second)
    assert first.batches + second.batches == whole.batches
    assert second.weight == whole.weight
    assert second.updates == whole.updates == 9
    assert second._ctx.global_step == whole._ctx.global_step == 4
    assert len(second.batches) == 3
    assert first.closed and second.closed
    value = json.loads((checkpoint / FILENAME).read_text())
    assert (value["epoch"], value["next_batch"], value["completed_steps"]) == (0, 2, 2)


@pytest.mark.parametrize("mutation", ["data", "batch", "token_limit", "clock", "missing_native", "missing_cursor"])
def test_resume_rejects_changed_contract_before_worker_initialization_cpu(tmp_path, mutation):
    checkpoint, _ = make_checkpoint(tmp_path)
    cfg = config(max_steps=5, resume_from_checkpoint=str(checkpoint))
    dataset = rows()
    if mutation == "data":
        dataset[0]["tokens"][1] += 1
    elif mutation == "batch":
        cfg.batch_size = 1
    elif mutation == "token_limit":
        cfg.max_prompt_tokens = 5
    elif mutation == "clock":
        p = checkpoint / "training_state.json"
        value = json.loads(p.read_text())
        value["trainer_global_step"] = 4
        p.write_text(json.dumps(value))
    elif mutation == "missing_native":
        (checkpoint / "training_state.pt").unlink()
    else:
        (checkpoint / FILENAME).unlink()
    backend = LoopBackend(checkpoint)
    with pytest.raises(ValueError):
        fit(cfg, backend, dataset)
    assert not backend.initialized and not backend.batches


def test_completed_total_call_budget_does_not_repeat_training_cpu(tmp_path):
    checkpoint, _ = make_checkpoint(tmp_path)
    backend = LoopBackend(checkpoint)
    fit(config(max_steps=2, resume_from_checkpoint=str(checkpoint)), backend)
    assert backend.batches == [] and backend.closed
    assert backend._ctx.global_step == 1


@pytest.mark.parametrize("key,value", [("epoch", True), ("next_batch", 0), ("completed_steps", -1)])
def test_invalid_cursor_types_rejected_cpu(tmp_path, key, value):
    checkpoint, _ = make_checkpoint(tmp_path)
    cfg = config()
    metadata = json.loads((checkpoint / FILENAME).read_text())
    metadata[key] = value
    (checkpoint / FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_cursor(checkpoint, dataset_signature(cfg, rows()))


def test_out_of_range_cursor_cannot_skip_an_epoch_cpu(tmp_path):
    checkpoint, _ = make_checkpoint(tmp_path)
    metadata = json.loads((checkpoint / FILENAME).read_text())
    metadata["next_batch"] = 5
    (checkpoint / FILENAME).write_text(json.dumps(metadata))
    backend = LoopBackend(checkpoint)
    with pytest.raises(ValueError, match="filtered dataset position"):
        fit(config(max_steps=6, resume_from_checkpoint=str(checkpoint)), backend)
    assert not backend.batches and backend.closed


def test_dataset_fingerprint_is_order_sensitive_and_relocation_independent_cpu():
    cfg = config()
    first = dataset_signature(cfg, rows())
    moved = copy.copy(cfg)
    moved.dataset_path = "another-location"
    assert first == dataset_signature(moved, rows())
    assert first != dataset_signature(cfg, list(reversed(rows())))


def test_loader_path_function_identity_matches_cli_spec_cpu(tmp_path):
    loader = tmp_path / "loader.py"
    loader.write_text("def custom(*args): return []\ndef other(*args): return []\n")
    cfg = config()
    cfg.dataset_loader_fn = str(loader) + ":custom"
    first = dataset_signature(cfg, rows())
    assert first["loader_function"] == "custom"
    cfg.dataset_loader_fn = str(loader) + ":other"
    assert first != dataset_signature(cfg, rows())
    cfg.dataset_loader_fn = str(loader)
    implicit = dataset_signature(cfg, rows())
    cfg.dataset_loader_fn += ":load_training_dataset"
    assert implicit == dataset_signature(cfg, rows())

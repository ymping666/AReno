"""Public SFT diagnostics and restored Trainer call clocks, without CUDA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import areno
import areno.api.trainer as trainer_module
import areno.api.trainers.sft as sft_module
from areno import Trainer
from areno.api.backend.common import TrainMetric
from areno.api.backend.cuda.backend import CudaBackend
from areno.api.backend.cuda.checkpoint import load_trainer_step, save_checkpoint
from areno.api.backend.cuda.losses import sft_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.api.config import CudaConfig
from areno.api.models import BackendType, TrainSequence
from areno.api.trainers.sft import _record_to_train_sequence
from areno.engine.runtime.train_step import _pack_train_data


@pytest.mark.parametrize("schema", ["text", "encoded", "image"])
def test_standard_sft_has_no_scored_logprob_diagnostic_cpu(schema, monkeypatch):
    tokenizer = SimpleNamespace(eos_token_id=0, encode=lambda text, **kwargs: [1, 2])
    record = (
        {"tokens": [1, 2, 3], "prompt_mask": [True, False, False]}
        if schema == "encoded"
        else {"prompt": "p", "response": "r"}
    )
    if schema == "image":
        monkeypatch.setattr(sft_module, "record_has_image", lambda _: True)
        monkeypatch.setattr(sft_module, "encode_multimodal_prompt", lambda *args: ([1, 2], None))
    sequence = _record_to_train_sequence(record, tokenizer, max_prompt_tokens=16, max_new_tokens=16)
    assert sequence.logprobs == []
    packed = _pack_train_data(make_train_pack([sequence]))
    current = torch.full_like(packed["packed_logprobs"], -0.7, requires_grad=True)
    loss, metrics = sft_loss_fn(packed, current)
    assert TrainMetric.LOGP_ABS_DIFF_MEAN not in metrics
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    # The old dummy-filled path had the same objective and gradient; only its
    # misleading consistency fields disappear, with no new scoring pass.
    dummy_sequence = sequence.model_copy(update={"logprobs": [0.0] * len(sequence.tokens)})
    old_pack = _pack_train_data(make_train_pack([dummy_sequence]))
    old_current = current.detach().clone().requires_grad_()
    old_loss, old_metrics = sft_loss_fn(old_pack, old_current)
    old_loss.backward()
    assert torch.equal(loss, old_loss)
    assert torch.equal(current.grad, old_current.grad)
    assert TrainMetric.LOGP_ABS_DIFF_MEAN in old_metrics


def test_explicit_prescored_sft_still_uses_native_diagnostic_cpu():
    sequence = TrainSequence(tokens=[1, 2, 3], prompt_len=1, logprobs=[0.0, -0.5, -0.75])
    pack = _pack_train_data(make_train_pack([sequence]))
    _, metrics = sft_loss_fn(pack, torch.tensor([-0.5, -0.75]))
    assert metrics[TrainMetric.LOGP_ABS_DIFF_MEAN] == 0


class EngineStub:
    """Two packs per public call advance the native optimizer twice."""

    def __init__(self, model_path="unused", resume=False):
        self.native_updates = 0
        if resume:
            self.native_updates = json.loads((Path(model_path) / "training_state.json").read_text())["global_step"]

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return cls(model_path, kwargs["optimizer_config"].resume_training_state)

    def step(self, packs, gradient_accumulation_steps=None):
        assert gradient_accumulation_steps == 1
        self.native_updates += len(packs)
        return [SimpleNamespace(loss=1.0, metrics={}, stepped=True, adapter_version=None) for _ in packs]

    def save_checkpoint(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"format": "areno-training-v1", "optimizer_class": "FakeAdamW", "global_step": self.native_updates}
        (directory / "training_state.json").write_text(json.dumps(metadata))
        return str(directory)

    def close(self):
        pass


def make_public_trainer(monkeypatch, model_path="unused", *, resume=False):
    monkeypatch.setitem(areno.__dict__, "ArenoEngine", EngineStub)
    config = CudaConfig(devices=[0], optimizer={"resume_training_state": resume})
    trainer = Trainer(world_size=1, model_path=str(model_path), backend_type=BackendType.CUDA, custom_config=config)
    monkeypatch.setattr(trainer_module, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(trainer_module, "load_processor", lambda path: None)
    monkeypatch.setattr(trainer_module, "eos_token_ids", lambda path, tokenizer: ())
    trainer.init()
    assert isinstance(trainer._backend, CudaBackend)
    return trainer


def train_two_optimizer_updates(trainer):
    rows = [TrainSequence(tokens=[1, 2, 3], prompt_len=1) for _ in range(2)]
    trainer.train(rows, sft_loss_fn, mini_bs=1, gradient_accumulation_steps=1)


def test_public_save_resume_retains_call_clock_separate_from_updates_cpu(monkeypatch, tmp_path):
    original = make_public_trainer(monkeypatch)
    recorded_steps = []
    recorder = SimpleNamespace(
        record_train_step=lambda **kwargs: recorded_steps.append(kwargs["step"]), close=lambda: None
    )
    original._metrics = recorder
    train_two_optimizer_updates(original)
    train_two_optimizer_updates(original)
    assert recorded_steps == [0, 1]
    assert original._ctx.global_step == 1
    assert original._backend._train_engine.native_updates == 4
    original.save_checkpoint(str(tmp_path))
    original.close()
    resumed = make_public_trainer(monkeypatch, tmp_path, resume=True)
    assert resumed._ctx.global_step == 1
    assert resumed._backend._train_engine.native_updates == 4
    resumed._metrics = recorder
    train_two_optimizer_updates(resumed)
    assert recorded_steps == [0, 1, 2]
    assert resumed._ctx.global_step == 2
    assert resumed._backend._train_engine.native_updates == 6
    metadata = json.loads((tmp_path / "training_state.json").read_text())
    assert metadata["global_step"] == 4
    assert metadata["trainer_global_step"] == 1
    resumed.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("trainer_global_step", -2),
        ("trainer_global_step", 1.5),
        ("trainer_global_step", True),
        ("trainer_global_step", "1"),
        ("trainer_global_step", None),
        ("global_step", -1),
        ("global_step", 1.5),
        ("global_step", True),
        ("global_step", "1"),
        ("format", "unknown"),
    ],
)
def test_invalid_clock_rejected_before_starting_engine_cpu(monkeypatch, tmp_path, field, value):
    metadata = {"format": "areno-training-v1", "global_step": 4, "trainer_global_step": 1}
    metadata[field] = value
    (tmp_path / "training_state.json").write_text(json.dumps(metadata))

    def forbidden_engine(*args, **kwargs):
        pytest.fail("malformed metadata must fail before engine construction")

    monkeypatch.setattr(EngineStub, "from_pretrained", classmethod(forbidden_engine))
    with pytest.raises(ValueError, match="training-state"):
        make_public_trainer(monkeypatch, tmp_path, resume=True)


def test_legacy_clock_fallback_is_explicit_cpu(tmp_path, caplog):
    (tmp_path / "training_state.json").write_text(json.dumps({"format": "areno-training-v1", "global_step": 4}))
    assert load_trainer_step(str(tmp_path)) == 3
    assert "cannot be reconstructed" in caplog.text


def test_checkpoint_before_training_retains_minus_one_clock_cpu(monkeypatch, tmp_path):
    original = make_public_trainer(monkeypatch)
    original.save_checkpoint(str(tmp_path))
    original.close()
    resumed = make_public_trainer(monkeypatch, tmp_path, resume=True)
    assert resumed._ctx.global_step == -1
    train_two_optimizer_updates(resumed)
    assert resumed._ctx.global_step == 0
    assert resumed._backend._train_engine.native_updates == 2
    resumed.close()


def test_weight_only_checkpoint_does_not_gain_optimizer_metadata_cpu(tmp_path):
    def save_weights(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return str(path)

    engine = SimpleNamespace(save_checkpoint=save_weights)
    assert save_checkpoint(engine, str(tmp_path), trainer_global_step=7) == str(tmp_path)
    assert not (tmp_path / "training_state.json").exists()

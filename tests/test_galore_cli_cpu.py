"""Public GaLore entry points, explicit policies and legacy-route equivalence."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from click.testing import CliRunner

from areno.api.trainer_config import DPOTrainerConfig, PolicyTrainerConfig, PPOTrainerConfig, TrainerConfig
from areno.cli import train as train_cli
from areno.engine.config import EngineConfig, ModelConfig, OptimizerConfig, RuntimeConfig
from areno.engine.modeling import build_optimizer
from areno.engine.optim.galore import GaLoreAdamW

BASE = [
    "--backend",
    "cuda",
    "--algo",
    "sft",
    "--ckpt",
    "model",
    "--dataset-path",
    "data",
    "--dataset-loader-fn",
    "examples/sft/alpaca/dataset_loader.py",
    "--world-size",
    "1",
    "--tp-size",
    "1",
    "--batch-size",
    "4",
    "--mini-bs",
    "4",
    "--attn-backend",
    "native",
]


def config(**kwargs):
    defaults = dict(
        algo="sft",
        ckpt="model",
        dataset_path="data",
        backend="cuda",
        world_size=1,
        tp_size=1,
        batch_size=4,
        mini_bs=4,
        attn_backend="native",
    )
    return TrainerConfig(**(defaults | kwargs))


def invoke(monkeypatch, options):
    calls = []
    monkeypatch.setattr(train_cli, "run", calls.append)
    monkeypatch.setattr(train_cli, "_model_config_for_summary", lambda config: None)
    monkeypatch.setattr("areno.cli.dashboard_registry.register_dashboard_job", lambda **kwargs: None)
    result = CliRunner().invoke(train_cli.train_command, BASE + options)
    return result, calls


def identical(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            identical(x, y)
    else:
        assert a == b


def test_default_cli_and_sdk_do_not_enable_galore_cpu(monkeypatch):
    result, calls = invoke(monkeypatch, [])
    assert result.exit_code == 0, result.output
    cfg = calls[0]
    assert cfg.galore_rank is None
    assert not any(key.startswith("galore_") for key in cfg.optimizer_config())
    assert "compile_model" not in cfg.cuda_config().runtime
    assert config().optimizer_config() == cfg.optimizer_config()


def test_strict_public_cli_reproduces_frozen_recipe_without_environment_cpu(monkeypatch):
    monkeypatch.delenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", raising=False)
    monkeypatch.delenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK", raising=False)
    result, calls = invoke(
        monkeypatch,
        [
            "--galore-rank",
            "128",
            "--galore-recipe",
            "strict",
            "--galore-deferred-projection",
            "--galore-fused-scale-writeback",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = calls[0]
    optimizer = cfg.optimizer_config()
    expected = {
        "galore_rank": 128,
        "galore_update_proj_gap": 200,
        "galore_scale": 1.0,
        "galore_projector_backend": "recycled",
        "galore_projector_oversampling": 16,
        "galore_projector_iterations": 1,
        "galore_projector_seed": 0,
        "galore_projector_final_qr": True,
        "galore_compact_master": True,
        "galore_tile_rows": 16384,
        "galore_release_gradients": True,
        "galore_fused_writeback": True,
        "galore_transport_moments": True,
        "galore_exclude_vocab": False,
        "galore_deferred_projection": True,
        "galore_fused_scale_writeback": True,
    }
    assert {key: optimizer[key] for key in expected} == expected
    assert cfg.cuda_config().runtime["compile_model"] is False
    assert cfg.cuda_config().runtime["keep_rollout_state"] is True
    parameter = torch.nn.Parameter(torch.zeros(8, 6))
    actual = build_optimizer([parameter], OptimizerConfig(**optimizer), SimpleNamespace(world_size=1, dp_size=1))
    assert actual._deferred_projection and actual._fused_scale_writeback
    assert actual.projector_backend == "recycled" and actual.compact_master


def test_standard_public_route_overrides_ambient_private_flags_cpu(monkeypatch):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    monkeypatch.setenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK", "1")
    result, calls = invoke(monkeypatch, ["--galore-rank", "2"])
    assert result.exit_code == 0, result.output
    parameter = torch.nn.Parameter(torch.zeros(8, 6))
    actual = build_optimizer(
        [parameter], OptimizerConfig(**calls[0].optimizer_config()), SimpleNamespace(world_size=1, dp_size=1)
    )
    assert not actual._deferred_projection and not actual._fused_scale_writeback


@pytest.mark.parametrize(
    "options,message",
    [
        (["--galore-rank", "0"], "Invalid value"),
        (["--galore-update-proj-gap", "0"], "Invalid value"),
        (["--galore-rank", "2", "--galore-scale", "nan"], "must be positive"),
        (["--galore-recipe", "strict"], "requires --galore-rank"),
        (["--galore-deferred-projection"], "require galore_rank"),
        (["--galore-rank", "2", "--adam-8bit"], "low-bit"),
        (["--galore-rank", "2", "--adam-4bit"], "low-bit"),
        (["--galore-rank", "2", "--world-size", "2"], "single-GPU"),
        (["--galore-rank", "2", "--optimizer-state-offload", "cpu"], "resident"),
        (["--galore-rank", "2", "--galore-deferred-projection", "--mini-bs", "2"], "one microbatch"),
        (
            ["--galore-rank", "2", "--galore-deferred-projection", "--gradient-accumulation-steps", "2"],
            "one microbatch",
        ),
        (["--galore-rank", "2", "--galore-fused-scale-writeback"], "requires deferred projection"),
    ],
)
def test_invalid_public_combinations_fail_before_run_cpu(monkeypatch, options, message):
    result, calls = invoke(monkeypatch, options)
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert calls == []


def test_resume_cli_preserves_base_reference_and_requests_training_state_cpu(monkeypatch, tmp_path):
    result, calls = invoke(monkeypatch, ["--resume-from-checkpoint", str(tmp_path), "--galore-rank", "2"])
    assert result.exit_code == 0, result.output
    assert calls[0].ckpt == "model" and calls[0].resume_from_checkpoint == str(tmp_path)
    assert calls[0].optimizer_config()["resume_training_state"] is True


@pytest.mark.parametrize("kwargs", [{"backend": "mlx"}, {"algo": "gspo"}, {"resume_from_checkpoint": "missing"}])
def test_resume_unsupported_routes_rejected_cpu(tmp_path, kwargs):
    options = {"resume_from_checkpoint": str(tmp_path)} | kwargs
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        config(**options)


@pytest.mark.parametrize(
    "override,message",
    [
        ({"model": ModelConfig(model_type="llama")}, "dense Qwen3"),
        ({"runtime": RuntimeConfig(attn_backend="native", compile_model=True)}, "compile_model=False"),
    ],
)
def test_lowlevel_explicit_deferred_domain_rejected_cpu(override, message):
    settings = dict(
        model=ModelConfig(model_type="qwen3"),
        devices=[0],
        runtime=RuntimeConfig(attn_backend="native", compile_model=False),
        optimizer=OptimizerConfig(galore_rank=2, galore_deferred_projection=True),
    )
    with pytest.raises(ValueError, match=message):
        EngineConfig(**(settings | override))


def test_explicit_policy_matches_legacy_route_and_serialized_resume_cpu(monkeypatch):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    options = dict(
        lr=0.003,
        rank=2,
        update_proj_gap=2,
        projector_backend="recycled",
        projector_seed=17,
        compact_master=True,
        tile_rows=8,
        release_gradients=True,
        fused_writeback=True,
        transport_moments=True,
    )
    initial = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 100
    a, b = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
    legacy = GaLoreAdamW([a], **options)
    monkeypatch.delenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION")
    explicit = GaLoreAdamW([b], **options, deferred_projection=True, fused_scale_writeback=False)
    for step in range(5):
        for param, optimizer in ((a, legacy), (b, explicit)):
            optimizer.zero_grad()
            optimizer.begin_deferred_projection()
            (param.square() * (step + 1)).sum().backward()
            optimizer.finish_deferred_backward()
            optimizer.clip_deferred_gradients(optimizer.deferred_grad_norms()["global"], 1.0)
            optimizer.step()
        assert torch.equal(a, b)
        identical(legacy.state_dict(), explicit.state_dict())
        if step == 1:
            snapshot = copy.deepcopy(explicit.state_dict())
            b = torch.nn.Parameter(torch.zeros_like(b))
            explicit = GaLoreAdamW([b], **options, deferred_projection=True, fused_scale_writeback=False)
            explicit.load_state_dict(snapshot)
            assert torch.equal(a, b)
            assert explicit._deferred_projection and not explicit._fused_scale_writeback


@pytest.mark.parametrize("name", ["galore_deferred_projection", "galore_fused_scale_writeback"])
def test_public_sdk_flags_require_boolean_cpu(name):
    with pytest.raises(ValueError, match="must be a boolean"):
        config(galore_rank=2, **{name: None})


def test_run_constructs_public_trainer_from_resume_model_cpu(monkeypatch, tmp_path):
    import areno
    import areno.api.trainer_factory

    calls = []
    engine = object()
    monkeypatch.setattr(areno, "Trainer", lambda *args, **kwargs: calls.append((args, kwargs)) or engine)
    monkeypatch.setattr(train_cli, "resolve_model_refs_for_config", lambda value: value)
    monkeypatch.setattr(train_cli, "_write_dashboard_run_config", lambda value: None)
    monkeypatch.setattr(
        train_cli, "_load_dataset_for_training", lambda *args, **kwargs: [{"prompt": "a", "response": "b"}]
    )
    monkeypatch.setattr(
        areno.api.trainer_factory,
        "build_trainer",
        lambda *args, **kwargs: SimpleNamespace(fit=lambda: calls.append("fit")),
    )
    cfg = config(resume_from_checkpoint=str(tmp_path), galore_rank=2)
    train_cli.run(cfg)
    args, kwargs = calls[0]
    assert args == (1, str(tmp_path))
    assert kwargs["custom_config"].optimizer["resume_training_state"] is True
    assert kwargs["custom_config"].optimizer["save_training_state"] is True
    assert cfg.ckpt == "model"
    assert calls[-1] == "fit"


@pytest.mark.parametrize(
    "cls,algo",
    [
        (DPOTrainerConfig, "dpo"),
        (PolicyTrainerConfig, "gspo"),
        (PolicyTrainerConfig, "grpo"),
        (PPOTrainerConfig, "ppo"),
    ],
)
def test_public_deferred_rejects_unvalidated_algorithm_before_engine_cpu(cls, algo):
    with pytest.raises(ValueError, match="only non-agentic SFT"):
        cls(
            algo=algo,
            ckpt="model",
            dataset_path="data",
            backend="cuda",
            world_size=1,
            tp_size=1,
            batch_size=2,
            mini_bs=2,
            galore_rank=2,
            galore_deferred_projection=True,
        )
    # The restriction concerns deferred lifecycle support, not core GaLore.
    ordinary = cls(
        algo=algo,
        ckpt="model",
        dataset_path="data",
        backend="cuda",
        world_size=1,
        tp_size=1,
        batch_size=2,
        mini_bs=2,
        galore_rank=2,
    )
    assert ordinary.optimizer_config()["galore_rank"] == 2
    assert ordinary.optimizer_config()["galore_deferred_projection"] is False


def test_public_deferred_rejects_agentic_entry_before_engine_cpu():
    with pytest.raises(ValueError, match="only non-agentic SFT"):
        config(galore_rank=2, galore_deferred_projection=True, agent_fn="agent.py:agent_fn")


def test_new_galore_config_fields_do_not_shift_existing_positional_parameters_cpu():
    import inspect

    from areno.api.trainer_config import TrainerConfig
    from areno.engine.config import OptimizerConfig

    trainer_signature = inspect.signature(TrainerConfig)
    for name, parameter in trainer_signature.parameters.items():
        if name.startswith("galore_") or name == "resume_from_checkpoint":
            assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    optimizer_signature = inspect.signature(OptimizerConfig)
    assert optimizer_signature.parameters["galore_deferred_projection"].kind == inspect.Parameter.KEYWORD_ONLY
    assert optimizer_signature.parameters["galore_fused_scale_writeback"].kind == inspect.Parameter.KEYWORD_ONLY

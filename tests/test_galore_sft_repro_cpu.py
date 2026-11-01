"""CPU contract tests; optional full offline replay uses caller-supplied assets."""

import ast
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from areno.engine.training import TrainingManager

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "galore_sft"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# CLI imports install a process-local profiler. Restore it after loading for tests
# so collecting this test cannot alter the rest of the repository CPU suite.
_native_manager_train = TrainingManager.train
try:
    common = load_script("galore_sft_repro_common", "common.py")
finally:
    TrainingManager.train = _native_manager_train
prepare_data = load_script("galore_sft_repro_prepare", "prepare_data.py")


def _frozen_ast_sha256(node):
    # The immutable manifest uses Python3.12's AST representation. Earlier
    # parsers omit this empty metadata field; add it per node without changing
    # bodies, existing type parameters, global AST classes or frozen hashes.
    for child in ast.walk(node):
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and "type_params" not in child._fields
        ):
            child._fields += ("type_params",)
            if not hasattr(child, "type_params"):
                child.type_params = []
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _remove_type_params_metadata(node):
    """Simulate Python3.10/3.11 AST nodes without changing their bodies."""
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            child._fields = tuple(name for name in child._fields if name != "type_params")
            if hasattr(child, "type_params"):
                del child.type_params
    return node


@pytest.mark.parametrize("legacy_metadata", [False, True], ids=["native", "pre312"])
def test_frozen_mathematical_ast(legacy_metadata):
    reference = json.loads((SCRIPTS / "frozen-ast-manifest.json").read_text())
    for filename, functions in reference.items():
        if filename.startswith("_"):
            continue
        nodes = {
            n.name: n for n in ast.walk(ast.parse((SCRIPTS / filename).read_text())) if isinstance(n, ast.FunctionDef)
        }
        for name, expected in functions.items():
            node = _remove_type_params_metadata(nodes[name]) if legacy_metadata else nodes[name]
            actual = _frozen_ast_sha256(node)
            assert actual == expected, (filename, name)


def test_frozen_ast_normalizes_all_definition_kinds_without_global_mutation():
    source = "async def outer(x):\n    class Box:\n        def value(self):\n            return x + 1\n    return Box\n"
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    original_fields = {kind: kind._fields for kind in kinds}
    legacy = _remove_type_params_metadata(ast.parse(source))
    # Captured from Python3.12 ast.dump(include_attributes=False), independently
    # of normalization. Includes AsyncFunctionDef, ClassDef and FunctionDef.
    assert _frozen_ast_sha256(legacy) == "7b8cfda4547712e9003d73d97ed318d1d16e756e3d630b77601401765f11b069"
    assert {kind: kind._fields for kind in kinds} == original_fields


def test_frozen_ast_preserves_existing_type_parameters():
    node = ast.parse("def identity(value): return value").body[0]
    if "type_params" not in node._fields:
        node._fields += ("type_params",)
    node.type_params = [ast.Name(id="T", ctx=ast.Load())]
    expected = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert _frozen_ast_sha256(node) == expected
    assert len(node.type_params) == 1 and node.type_params[0].id == "T"


def test_frozen_ast_rejects_changed_mathematical_body():
    reference = json.loads((SCRIPTS / "frozen-ast-manifest.json").read_text())
    node = next(
        n
        for n in ast.walk(ast.parse((SCRIPTS / "common.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "encode"
    )
    assert _frozen_ast_sha256(node) == reference["common.py"]["encode"]
    # Alter response-reservation arithmetic while keeping portability metadata
    # unchanged. A runtime version fix must not permit this scientific change.
    division = next(
        n
        for n in ast.walk(node)
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.FloorDiv)
        and isinstance(n.right, ast.Constant)
        and n.right.value == 2
    )
    division.right.value = 3
    assert _frozen_ast_sha256(_remove_type_params_metadata(node)) != reference["common.py"]["encode"]


class Tokenizer:
    eos_token_id = 999
    bos_token_id = 998

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


@pytest.mark.parametrize(
    ("record", "context", "tokens", "prompt_len", "targets"),
    [
        ({"prompt": "ABCDEFGHIJKLMNOP", "response": "xyz"}, 8, [65, 66, 79, 80, 120, 121, 122, 999], 4, 4),
        ({"prompt": "AB", "response": "abcdefghijklmnop"}, 8, [65, 66, 97, 98, 99, 100, 101, 102], 2, 6),
        ({"prompt": "", "response": "a"}, 8, [998, 97, 999], 1, 2),
        ({"text": "abcdefghi"}, 8, list(range(97, 105)), 1, 7),
    ],
)
def test_response_encoding_mask(tmp_path, record, context, tokens, prompt_len, targets):
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n")
    rows, stats = common.encode(Tokenizer(), SimpleNamespace(data_dir=tmp_path, context=context), "train")
    assert rows[0].tokens == tokens and rows[0].prompt_len == prompt_len
    assert stats["targets"] == targets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_absolute_resume_order(seed):
    size, batch = 5, 4
    actual = sum([common.batch_indices(s, seed, size, batch, {}) for s in range(5)], [])
    expected = []
    for epoch in range(4):
        order = list(range(size))
        random.Random(seed + 1000003 * epoch).shuffle(order)
        expected.extend(order)
    assert actual == expected
    assert common.batch_indices(3, seed, size, batch, {}) == actual[12:16]


def test_response_token_weighted_loss():
    examples = [SimpleNamespace(tokens=[0, 2, 4], prompt_len=1), SimpleNamespace(tokens=[0, 9], prompt_len=1)]

    class Engine:
        def score_logprobs(self, role, rows, **kwargs):
            return [[-float(v) for v in row] for row in rows]

    assert common.evaluate(Engine(), examples, 1) == dict(loss=5.0, target_tokens=3, examples=2)


def test_update_profile(monkeypatch):
    worker = SimpleNamespace(_global_step=10)

    def native(self, payload):
        worker._global_step += 1
        return [
            {"metrics": dict(cuda_step_time_s=0.2, cuda_peak_allocated_bytes=100, cuda_peak_reserved_bytes=120)},
            {
                "metrics": dict(
                    cuda_step_time_s=0.3,
                    cuda_peak_allocated_bytes=80,
                    cuda_peak_reserved_bytes=150,
                    galore_projection_refresh=True,
                )
            },
        ]

    monkeypatch.setattr(common, "_native_train", native)
    rows = common._profile_update(SimpleNamespace(worker=worker), None)
    for r in rows:
        m = r["metrics"]
        assert m["update_cuda_time_s"] == pytest.approx(0.5)
        assert m["update_peak_allocated_bytes"] == 100 and m["update_peak_reserved_bytes"] == 150
        assert m["update_optimizer_steps"] == 1 and m["update_worker_step"] == 11 and m["update_refresh_any"] == 1


@pytest.mark.parametrize("optimizer", ["adamw", "adam8", "galore"])
def test_optimizer_runtime_extra_flags(monkeypatch, tmp_path, optimizer):
    args = common.parser("test").parse_args(
        [
            "--model",
            "local-model",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--optimizer",
            optimizer,
            "--extra-config-json",
            '{"optimizer":{"galore_exclude_vocab":true},"runtime":{"activation_checkpointing":false}}',
        ]
    )
    monkeypatch.setattr(common, "CudaConfig", lambda **kw: kw)
    monkeypatch.setattr(common, "Trainer", lambda **kw: kw)
    c = common.trainer(args)
    o = c["custom_config"]["optimizer"]
    runtime = c["custom_config"]["runtime"]
    assert (
        o["lr"] == o["min_lr"] == 1e-5
        and o["betas"] == (0.9, 0.95)
        and o["weight_decay"] == 0.01
        and o["grad_clip_norm"] == 1
    )
    assert o["galore_exclude_vocab"] is True and runtime["activation_checkpointing"] is False
    assert c["model_path"] == "local-model" and c["world_size"] == 1
    assert o.get("adam_8bit", False) == (optimizer == "adam8")
    assert o.get("galore_rank") == (128 if optimizer == "galore" else None)
    if optimizer == "galore":
        assert o["galore_scale"] == 4 and o["galore_update_proj_gap"] == 200 and o["galore_transport_moments"] is True
    assert common.trainer(args, resume=True)["model_path"] == str(tmp_path / "ckpt")


def test_model_argument_required():
    with pytest.raises(SystemExit):
        common.parser("test").parse_args(["--data-dir", "d", "--output-dir", "o", "--checkpoint-dir", "c"])


def test_group_dedup_disjointness():
    row = dict(instruction="Same", context="Document", response="Answer", category="qa")
    duplicate = dict(instruction=" SAME ", context=" DOCUMENT ", response=" ANSWER ", category="qa")
    other = dict(instruction="Question 2", context="Document", response="New answer", category="qa")
    outputs, manifests, dropped = prepare_data.prepare_rows([row, duplicate, other])
    assert sum(map(len, outputs.values())) == 2 and len(dropped) == 1
    populated = [x for x in manifests.values() if x]
    assert len(populated) == 1 and len({r["group_id"] for r in populated[0]}) == 1


def test_bad_raw_rejected_before_tokenizer(monkeypatch, tmp_path):
    (tmp_path / "databricks-dolly-15k.jsonl").write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--raw-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"), "--model", "absent"],
    )
    with pytest.raises(ValueError, match="Pinned source mismatch"):
        prepare_data.main()


def test_train_entrypoint_cpu_smoke(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "common", common)
    spec = importlib.util.spec_from_file_location("portable_sft_train", SCRIPTS / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for split in ["train", "validation"]:
        (tmp_path / (split + ".jsonl")).write_text(
            "".join(json.dumps({"prompt": "AB", "response": str(i)}) + "\n" for i in range(5))
        )

    class Engine:
        def __init__(self):
            self._ctx = SimpleNamespace(global_step=-1)
            self.updates = 0
            self.closed = False

        def init(self):
            pass

        def get_tokenizer(self):
            return Tokenizer()

        def score_logprobs(self, role, rows, **kwargs):
            return [[-1.0] * len(row) for row in rows]

        def train(self, batch, loss_fn, mini_bs):
            self.updates += 1
            self._ctx.global_step += 1
            assert mini_bs == 2 and loss_fn is module.sft_loss_fn
            return dict(
                loss=1.0,
                logp_abs_diff_mean=0.0,
                update_optimizer_steps=1,
                update_refresh_any=0,
                update_cuda_time_s=0.1,
                update_peak_allocated_bytes=100,
                update_peak_reserved_bytes=120,
            )

        def save_checkpoint(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            return path

        def close(self):
            self.closed = True

    engine = Engine()
    monkeypatch.setattr(module, "trainer", lambda *a, **k: engine)
    monkeypatch.setattr(module, "metadata", lambda args: {"cpu_smoke_only": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model",
            "local",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--steps",
            "3",
            "--context",
            "8",
            "--eval-every",
            "1",
        ],
    )
    module.main()
    result = json.loads((tmp_path / "out/summary.json").read_text())
    assert engine.updates == 3 and engine.closed and result["completed_updates"] == result["next_step"] == 3
    assert result["native_logp_abs_diff_max"] == 0 and result["target_tokens"] == 24 and result["measured_steps"] == 1
    assert not (tmp_path / "out/recovery").exists()


def test_offline_full_pinned_rebuild(tmp_path):
    raw = os.getenv("ARENO_DOLLY_RAW_DIR")
    model = os.getenv("ARENO_DOLLY_MODEL_DIR")
    if not raw or not model:
        pytest.skip("Set ARENO_DOLLY_RAW_DIR and ARENO_DOLLY_MODEL_DIR for full pinned offline replay")
    output = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_data.py"),
            "--raw-dir",
            raw,
            "--model",
            model,
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    expected = json.loads((SCRIPTS / "data-manifest.json").read_text())
    proof = {}
    for name, meta in expected["files"].items():
        p = output / name
        assert p.stat().st_size == meta["bytes"]
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == meta["sha256"]
        proof[name] = {
            "bytes": p.stat().st_size,
            "expected_sha256": meta["sha256"],
            "actual_sha256": actual,
            "byte_identical": True,
        }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["raw_rows"] == 15011 and summary["retained_rows"] == 14995
    assert [summary["splits"][s]["rows"] for s in ["train", "validation", "test"]] == [13544, 708, 743]
    original = json.loads((SCRIPTS / "expected-summary.json").read_text())
    for k, v in original.items():
        assert summary[k] == v, k

    if os.getenv("ARENO_DOLLY_PROOF"):
        Path(os.environ["ARENO_DOLLY_PROOF"]).write_text(
            json.dumps(
                {
                    "status": "passed",
                    "files": proof,
                    "stable_summary_identical": True,
                    "raw_sha256": prepare_data.RAW_FILES["databricks-dolly-15k.jsonl"],
                    "modelscope_revision": prepare_data.REVISION,
                },
                indent=2,
            )
            + "\n"
        )

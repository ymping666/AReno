"""P0 integration proof replay and corruption rejection, without CUDA imports."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/reports/galore-p0-evidence"
SPEC = importlib.util.spec_from_file_location("p0_evidence_verifier", EVIDENCE / "verify.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture(scope="module")
def bundle():
    return VERIFIER.load_bundle(EVIDENCE)


def test_native_public_entry_and_recorded_fork_checks_cpu(bundle):
    result = VERIFIER.audit_bundle(*bundle)
    assert result["native_training_calls"] == 202
    assert result["cli"]["resume"]["public_steps"] == [200, 201]
    assert result["cli"]["resume"]["refresh_steps"] == [200]
    assert (result["checkpoint_files"], result["checkpoint_bytes"]) == (44, 11_320_251_085)
    assert result["checkpoint_tensor_payload_rehashed_offline"] is False
    assert result["live_restored_tensor_identity_observed"] is False
    assert result["cli"]["prefix"]["logprob_consistency"] == "absent, not zero"
    assert len(result["fork_cpu_matrix"]) == 3
    assert all(job["passed"] == 1855 and job["skipped"] == 146 for job in result["fork_cpu_matrix"])
    assert result["upstream_ci_or_maintainer_approval_claim"] is False
    assert result["new_quality_or_performance_claim"] is False


@pytest.mark.parametrize(
    "mutation", ["clock", "owner", "refresh", "diagnostic", "checkpoint", "dataset", "ci_counts", "docs", "recipe"]
)
def test_corrupted_p0_records_rejected_cpu(bundle, mutation):
    manifest, original = bundle
    files = dict(original)
    name = "qualification/stage1/outputs/resume/native-metrics.json"
    if mutation in ("clock", "owner", "refresh", "diagnostic"):
        value = json.loads(files[name])
        if mutation == "clock":
            value["series"]["train/loss"][0]["step"] = 202
        elif mutation == "owner":
            value["series"]["train/galore_deferred_owner_count"][0]["value"] = 225
        elif mutation == "refresh":
            value["series"]["train/galore_projection_refresh"][0]["value"] = 0
        else:
            value["keys"].append("train/logp_abs_diff_mean")
            value["series"]["train/logp_abs_diff_mean"] = [dict(row, value=0) for row in value["series"]["train/loss"]]
        files[name] = json.dumps(value).encode()
    elif mutation == "checkpoint":
        name = "qualification/stage1/checkpoints/step_000200/training_state.json"
        value = json.loads(files[name])
        value["trainer_global_step"] = 200
        files[name] = json.dumps(value).encode()
    elif mutation == "dataset":
        name = "qualification/stage1/data/train.jsonl"
        files[name] = files[name].replace(b"Paris.", b"Lyon.")
    elif mutation == "ci_counts":
        name = "ci/CI_JOB_124803913534.log"
        files[name] = files[name].replace(b"1855 passed", b"1856 passed")
    elif mutation == "recipe":
        name = next(
            key
            for key in files
            if key.startswith("qualification/stage1/outputs/prefix/metrics/areno_run_config.") and key.endswith(".json")
        )
        value = json.loads(files[name])
        item = next(
            item
            for section in value["settings"]["sections"]
            for item in section["items"]
            if item["key"] == "optimizer_lr"
        )
        item["value"] = 1e-4
        files[name] = json.dumps(value).encode()
    else:
        name = "ci/DOCS_FINAL.log"
        files[name] = files[name].replace(b"build succeeded", b"build failed")
    assert files[name] != original[name]
    with pytest.raises(ValueError):
        VERIFIER.audit_bundle(manifest, files)


def test_p0_runtime_drift_rejected_cpu(bundle, tmp_path):
    with pytest.raises(ValueError, match="current runtime differs"):
        VERIFIER.audit_bundle(*bundle, repo=tmp_path)


def test_optional_current_runtime_identity_remains_strict_cpu(bundle):
    digest = hashlib.sha256()
    paths = sorted((REPO / "areno").rglob("*.py"))
    for path in paths:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    if len(paths) == 222 and digest.hexdigest() == VERIFIER.SOURCE:
        assert VERIFIER.audit_bundle(*bundle, repo=REPO)["current_repo_checked"] is True
    else:
        with pytest.raises(ValueError, match="current runtime differs"):
            VERIFIER.audit_bundle(*bundle, repo=REPO)


def test_p0_corrupted_archive_rejected_cpu(bundle, tmp_path):
    manifest, _ = bundle
    bad = copy.deepcopy(manifest)
    part = bad["parts"][0]
    part["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(bad))
    (tmp_path / part["archive"]).write_bytes((EVIDENCE / part["archive"]).read_bytes())
    with pytest.raises(ValueError, match="archive identity"):
        VERIFIER.load_bundle(tmp_path)


def test_p0_verifier_runs_with_only_standard_library_cpu(tmp_path):
    output = tmp_path / "review.json"
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(EVIDENCE / "verify.py"),
            str(EVIDENCE),
            "--json-output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text())["status"] == "passed"

"""Replay P0 native scalar exports and checkpoint/CI records; no CUDA imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

SOURCE = "47c75ef8a4e76cdbda5770a3875830d56e1a5420faf9c41ea9fd07f1972776ff"
MEASURED = "8af25b43b3837e96206de928d9dc6eb210c3b0e4"
CI_COMMIT = "487fe9ea538665b8ebe7f34ff056191139219680"
NATIVE = "f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226"
Q = "qualification/"


def container():
    path = Path(__file__).resolve().parents[1] / "galore-capacity-evidence/verify.py"
    spec = importlib.util.spec_from_file_location("p0_container_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTAINER = container()
check, sha = CONTAINER.check, CONTAINER.sha


def load_bundle(directory):
    return CONTAINER.load_bundle(directory, schema="areno-galore-p0-evidence-v1")


def read(files, name):
    return json.loads(files[name])


def normalized_dataset(raw):
    rows = []
    for line in raw.splitlines():
        record = json.loads(line)
        prompt = f"Instruction: {str(record['instruction']).strip()}\n"
        context = str(record.get("input") or "").strip()
        if context:
            prompt += f"Input: {context}\n"
        rows.append({"prompt": prompt + "Response:", "response": str(record["output"])})
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return rows, digest.hexdigest()


def audit_cli(files, mode, prepared):
    base = Q + f"stage1/outputs/{mode}/"
    meta, summary, metrics = (
        read(files, base + name) for name in ("metadata.json", "summary.json", "native-metrics.json")
    )
    expected = list(range(200)) if mode == "prefix" else [200, 201]
    check(
        meta["commit"] == MEASURED and meta["source_sha256"] == SOURCE and meta["native_sha256"] == NATIVE,
        "CLI source differs",
    )
    check(meta["experiment_environment"] == {} and meta["cuda_visible_devices"] == "1", "nonpublic policy/device")
    check(meta["dataset_sha256"] == prepared["data_sha256"], "CLI input differs")
    check(
        meta["model_manifest_sha256"] == sha(files[Q + "repo/scripts/galore_deferred/model-manifest.json"]),
        "model manifest differs",
    )
    check(summary["status"] == "passed" and summary["mode"] == mode, "incomplete console run")
    check(
        summary["public_console_completed"] and summary["no_training_or_scoring_override"], "console contract differs"
    )
    check(summary["training_calls"] == len(expected) and summary["public_steps"] == expected, "public clock differs")
    check(metrics["ok"] is True and metrics["unknown"] == [], "native metric export incomplete")
    series = metrics["series"]
    check(
        len(metrics["keys"]) == len(set(metrics["keys"])) and set(metrics["keys"]) == set(series),
        "metric inventory differs",
    )
    check(
        "train/logp_abs_diff_mean" not in series and summary["native_logp_abs_diff_mean"] == "absent",
        "unscored SFT diagnostic invented",
    )
    for key, rows in series.items():
        check([row["step"] for row in rows] == expected, f"metric clock differs: {key}")
        check(
            all(
                type(row["step"]) is int
                and type(row["value"]) in (int, float)
                and math.isfinite(row["value"])
                and math.isfinite(row["wall_time"])
                for row in rows
            ),
            "invalid/nonfinite native scalar",
        )
    for key, value in {
        "train/galore_deferred_owner_count": 226,
        "train/galore_deferred_owners_once": 1,
        "train/galore_deferred_barrier_versions_unchanged": 1,
        "train/galore_deferred_full_leaf_bytes_after_backward": 0,
        "rollout/num_sequences": 4,
    }.items():
        check(
            key in series and all(row["value"] == value for row in series[key]),
            f"native lifecycle guard differs: {key}",
        )
    for key in ("train/loss", "train/grad_norm", "train/galore_master_bytes", "train/sft_target_tokens", "time/train"):
        check(key in series, f"required native metric missing: {key}")
    check(
        all(
            row["value"] > 0
            for key in ("train/galore_master_bytes", "train/sft_target_tokens", "time/train")
            for row in series[key]
        ),
        "invalid native byte/token/time scalar",
    )
    refresh = series["train/galore_projection_refresh"]
    check(all(row["value"] == float(row["step"] % 200 == 0) for row in refresh), "refresh schedule differs")
    argv = meta["argv"]
    for flag in ("--galore-deferred-projection", "--galore-fused-scale-writeback"):
        check(flag in argv, "public opt-in missing")
    check(argv[argv.index("--galore-recipe") + 1] == "strict", "recipe differs")
    configs = [name for name in files if name.startswith(base + "metrics/areno_run_config.") and name.endswith(".json")]
    check(len(configs) == 1, "UI run configuration ambiguous")
    run_config = read(files, configs[0])
    check(type(run_config["pid"]) is int and run_config["pid"] > 0, "console PID missing")
    config = {item["key"]: item["value"] for section in run_config["settings"]["sections"] for item in section["items"]}
    for key, value in {
        "algo": "sft",
        "world_size": 1,
        "tp_size": 1,
        "batch_size": 4,
        "mini_bs": 4,
        "max_prompt_tokens": 256,
        "max_new_tokens": 256,
        "gradient_accumulation_steps": None,
        "optimizer_lr": 1e-5,
        "optimizer_min_lr": 1e-5,
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.95,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "attn_backend": "native",
        "activation_checkpointing": True,
        "keep_rollout_state": True,
        "optimizer_state_offload": "none",
        "lora": None,
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
        "galore_transport_moments": True,
        "galore_release_gradients": True,
        "galore_fused_writeback": True,
        "galore_exclude_vocab": False,
        "galore_deferred_projection": True,
        "galore_fused_scale_writeback": True,
        "max_steps": 200 if mode == "prefix" else 202,
    }.items():
        check(config[key] == value, f"public configuration differs: {key}")
    if mode == "resume":
        check(
            config["resume_from_checkpoint"] == argv[argv.index("--resume-from-checkpoint") + 1],
            "resume source differs",
        )
    else:
        check(config["resume_from_checkpoint"] is None, "prefix unexpectedly resumed")
    times = {row["step"]: row["value"] for row in series["time/train"]}
    timing = {}
    for label, selected in (
        ("ordinary", [r["step"] for r in refresh if r["value"] == 0]),
        ("refresh", [r["step"] for r in refresh if r["value"] == 1]),
    ):
        timing[label] = {"calls": len(selected), "native_train_seconds": sum(times[step] for step in selected)}
    return {
        "calls": len(expected),
        "console_pid": run_config["pid"],
        "public_steps": expected,
        "refresh_steps": [r["step"] for r in refresh if r["value"] == 1],
        "native_train_timing_for_audit_only": timing,
        "logprob_consistency": "absent, not zero",
    }


def audit_bundle(manifest, files, repo=None):
    check(manifest["schema"] == "areno-galore-p0-evidence-v1", "unexpected schema")
    prepared, freeze = read(files, Q + "PREPARED_stage1.json"), read(files, Q + "FREEZE_stage1.json")
    check(
        prepared["git_commit"] == MEASURED
        and prepared["source_sha256"] == SOURCE
        and prepared["native_sha256"] == NATIVE,
        "frozen source differs",
    )
    check(
        prepared["model_revision"] == "4855588ea1a12789f2e965e5f52a9e4a24c94b2a"
        and prepared["runtime_python_files"] == 222,
        "model/runtime preparation differs",
    )
    check(
        freeze["parent_approved"] is True
        and freeze["source_sha256"] == SOURCE
        and freeze["prepared_sha256"] == sha(files[Q + "PREPARED_stage1.json"]),
        "freeze differs",
    )
    check(
        freeze["parent_review_sha256"] == sha(files[Q + "PARENT_HARNESS_REVIEW.json"])
        and freeze["parent_cpu_log_sha256"] == sha(files[Q + "PARENT_HARNESS_CPU_SOURCE_BOUND.log"]),
        "approval proof differs",
    )
    for name, expected in prepared["files"].items():
        key = Q + name + (".txt" if name.endswith(".py") else "")
        if key in files:
            check(
                len(files[key]) == expected["bytes"] and sha(files[key]) == expected["sha256"],
                f"frozen file differs: {name}",
            )
    identities = read(files, "SOURCE_IDENTITIES.json")
    check(
        identities["runtime_sha256"] == SOURCE and len(identities["runtime_files"]) == 222, "runtime inventory differs"
    )
    check(
        identities["measured_commit"] == MEASURED and identities["ci_commit"] == CI_COMMIT, "source attribution differs"
    )
    check(
        set(identities["ci_changes_since_native_run"])
        == {"tests/test_config_data_cpu.py", "tests/test_galore_sft_repro_cpu.py"},
        "CI repair changes runtime",
    )
    for name, expected in identities["runtime_files"].items():
        check(prepared["files"]["repo/" + name]["sha256"] == expected, "runtime inventory not bound to native run")
    if repo is not None:
        repo = Path(repo)
        paths = sorted((repo / "areno").rglob("*.py"))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(path.read_bytes())
        check(
            len(paths) == 222 and digest.hexdigest() == SOURCE, "current runtime differs from native-qualified source"
        )
    records = read(files, Q + "stage1/CASE_RESULTS.json")
    check(
        [r["case_id"] for r in records] == ["p0_full_cpu_selection", "p0_cli_prefix", "p0_cli_resume"]
        and all(r["outcome"] == "complete" and r["return_code"] == 0 for r in records),
        "qualification queue incomplete",
    )
    cli = {mode: audit_cli(files, mode, prepared) for mode in ("prefix", "resume")}
    check(
        cli["prefix"]["public_steps"] + cli["resume"]["public_steps"] == list(range(202)),
        "resume public clock discontinuity",
    )
    check(cli["prefix"]["console_pid"] != cli["resume"]["console_pid"], "resume did not use a fresh console process")
    analysis, review = read(files, Q + "stage1/ANALYSIS.json"), read(files, Q + "stage1/PARENT_RESULT_REVIEW.json")
    check(review["analysis_sha256"] == sha(files[Q + "stage1/ANALYSIS.json"]), "parent audit identity differs")
    check(
        review["live_restored_tensor_identity_observed"] is False and analysis["new_quality_or_speed_claim"] is False,
        "unsupported numerical claim",
    )
    before = read(files, Q + "stage1/CHECKPOINT_BEFORE_RESUME.json")
    check(
        before == analysis["checkpoint"] and len(before["files"]) == 44 and before["total_bytes"] == 11_320_251_085,
        "checkpoint inventories differ",
    )
    check(
        sum(value["bytes"] for value in before["files"].values()) == before["total_bytes"],
        "checkpoint byte arithmetic differs",
    )
    for name, value in before["files"].items():
        CONTAINER.safe_name(name)
        check(
            type(value["bytes"]) is int and value["bytes"] > 0 and re.fullmatch(r"[a-f0-9]{64}", value["sha256"]),
            "invalid checkpoint inventory",
        )
    checkpoint = Q + "stage1/checkpoints/step_000200/"
    for name in ("training_state.json", "sft_training_state.json"):
        check(
            len(files[checkpoint + name]) == before["files"][name]["bytes"]
            and sha(files[checkpoint + name]) == before["files"][name]["sha256"],
            "checkpoint metadata identity differs",
        )
    native, cursor = (
        read(files, checkpoint + "training_state.json"),
        read(files, checkpoint + "sft_training_state.json"),
    )
    check(
        native
        == {
            "format": "areno-training-v1",
            "optimizer_class": "GaLoreAdamW",
            "global_step": 200,
            "trainer_global_step": 199,
        },
        "checkpoint clocks differ",
    )
    check(
        cursor["format"] == "areno-sft-cursor-v1"
        and cursor["completed_steps"] == 200
        and cursor["epoch"] == 199
        and cursor["next_batch"] == 1,
        "SFT cursor differs",
    )
    raw = files[Q + "stage1/data/train.jsonl"]
    rows, normalized_sha = normalized_dataset(raw)
    signature = cursor["signature"]
    check(
        sha(raw) == prepared["data_sha256"]
        and len(rows) == signature["dataset_rows"] == 4
        and normalized_sha == signature["dataset_sha256"],
        "normalized dataset differs",
    )
    check(
        signature["loader_sha256"] == sha(files[Q + "repo/examples/sft/alpaca/dataset_loader.py.txt"])
        and signature["loader_function"] == "load_training_dataset",
        "loader identity differs",
    )
    check(
        signature["batch_size"] == signature["mini_bs"] == 4
        and signature["max_prompt_tokens"] == signature["max_new_tokens"] == 256
        and signature["gradient_accumulation_steps"] is None,
        "cursor batching/filter differs",
    )
    cpu = read(files, Q + "stage1/outputs/full_suite/summary.json")
    cpu_meta = read(files, Q + "stage1/outputs/full_suite/metadata.json")
    check(
        cpu_meta["commit"] == MEASURED
        and cpu_meta["source_sha256"] == SOURCE
        and cpu_meta["native_sha256"] == NATIVE
        and cpu_meta["cuda_visible_devices"] == "2",
        "local suite provenance differs",
    )
    check(
        cpu["pytest_exit_code"] == 0 and cpu["status"] == "passed" and cpu["cuda_visible"] is True, "local suite failed"
    )
    check(
        [cpu["counts"][key] for key in ("passed", "skipped", "subtests passed")] == [1966, 28, 12],
        "local suite counts differ",
    )
    ci = read(files, "ci/CPU_CI_RESULT.json")
    check(
        ci["conclusion"] == "success" and ci["status"] == "completed" and ci["headSha"] == CI_COMMIT,
        "fork CPU CI incomplete",
    )
    check(
        len(ci["jobs"]) == 3
        and {job["name"] for job in ci["jobs"]} == {f"unittest / python {v}" for v in ("3.10", "3.11", "3.12")},
        "fork matrix differs",
    )
    ci_counts = []
    for job in ci["jobs"]:
        log = files[f"ci/CI_JOB_{job['databaseId']}.log"].decode()
        check(
            job["conclusion"] == "success" and job["status"] == "completed" and CI_COMMIT in log,
            "fork CPU job failed/different SHA",
        )
        counts = re.findall(r"(\d+) passed, (\d+) skipped, (\d+) deselected", log)
        check(counts and tuple(map(int, counts[-1])) == (1855, 146, 2301), "fork test counts differ")
        ci_counts.append({"job": job["name"], "passed": 1855, "skipped": 146, "url": job["url"]})
    lint = read(files, "ci/LINT_CI_RESULT.json")
    check(
        lint["headSha"] == CI_COMMIT and lint["conclusion"] == "success" and lint["status"] == "completed",
        "fork lint failed",
    )
    check(
        any(step["name"] == "Run pre-commit" and step["conclusion"] == "success" for step in lint["jobs"][0]["steps"]),
        "lint hooks did not run",
    )
    lint_log = files["ci/LINT_JOB_124803921172.log"].decode()
    check(
        CI_COMMIT in lint_log
        and "Run Linter Check (Ruff)" in lint_log
        and "Run Formatter (Ruff)" in lint_log
        and "Passed" in lint_log,
        "lint raw proof differs",
    )
    docs = read(files, "ci/DOCS_FINAL_VALIDATION.json")
    docs_log = files["ci/DOCS_FINAL.log"].decode()
    check(
        "build succeeded." in docs_log and "WARNING" not in docs_log and "ERROR" not in docs_log,
        "strict docs build did not pass",
    )
    check(
        docs["status"] == "passed"
        and docs["commit"] == CI_COMMIT
        and docs["warnings"] == 0
        and docs["source_sha256"] == SOURCE
        and docs["log_sha256"] == sha(files["ci/DOCS_FINAL.log"]),
        "docs check differs",
    )
    return {
        "status": "passed",
        "native_commit": MEASURED,
        "fork_ci_commit": CI_COMMIT,
        "runtime_sha256": SOURCE,
        "current_repo_checked": repo is not None,
        "native_training_calls": 202,
        "cli": cli,
        "checkpoint_files": 44,
        "checkpoint_bytes": before["total_bytes"],
        "recorded_checkpoint_inventories_equal": True,
        "checkpoint_tensor_payload_rehashed_offline": False,
        "live_restored_tensor_identity_observed": False,
        "normalized_public_dataset_rows_replayed": 4,
        "tokenization_replayed_offline": False,
        "cache_and_worker_step_native_metrics": "not emitted; no substituted values",
        "local_cpu_named_selection": {"passed": 1966, "skipped": 28, "subtests_passed": 12, "CUDA_visible": True},
        "fork_cpu_matrix": ci_counts,
        "fork_lint_url": lint["url"],
        "recorded_docs_warnings": 0,
        "upstream_ci_or_maintainer_approval_claim": False,
        "new_quality_or_performance_claim": False,
    }


def verify(directory, repo=None):
    return audit_bundle(*load_bundle(directory), repo=repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args.directory, args.repo), indent=2, allow_nan=False) + "\n"
    if args.json_output:
        args.json_output.write_text(result)
    print(result, end="")

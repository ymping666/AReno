"""Pack only the bounded P0 integration proof, never weights or event binaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


def load_packer():
    path = Path(__file__).resolve().parents[1] / "galore-capacity-evidence/pack_evidence.py"
    spec = importlib.util.spec_from_file_location("p0_container_packer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(root, ci_root, repo):
    files = {}

    def add(base, name, label):
        path = base / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/nonregular allowlisted file: {name}")
        raw = path.read_bytes()
        raw.decode("utf-8")
        files[f"{label}/{name}" + (".txt" if path.suffix == ".py" else "")] = raw

    for name in (
        "SESSION.json",
        "PREPARED_stage1.json",
        "FREEZE_stage1.json",
        "QUALIFICATION_stage1.json",
        "PARENT_HARNESS_REVIEW.json",
        "PARENT_HARNESS_CPU_SOURCE_BOUND.log",
        "contract.py",
        "guards.py",
        "launch_registered.py",
        "watchdog.py",
        "prepare_p0.py",
        "docs-requirements-lock.txt",
        "stage1/checkpoints/step_000200/training_state.json",
        "stage1/checkpoints/step_000200/sft_training_state.json",
        "repo/examples/sft/alpaca/dataset_loader.py",
        "repo/.agents/skills/areno-run-training/scripts/read_metrics.py",
        "repo/scripts/galore_deferred/model-manifest.json",
    ):
        add(root, name, "qualification")
    for path in sorted((root / "stage1").iterdir()):
        if path.is_file() and path.suffix in (".py", ".json", ".jsonl", ".md", ".log"):
            add(root, path.relative_to(root).as_posix(), "qualification")
    add(root, "stage1/data/train.jsonl", "qualification")
    for case in ("prefix", "resume", "full_suite"):
        for name in ("metadata.json", "summary.json"):
            add(root, f"stage1/outputs/{case}/{name}", "qualification")
        if case != "full_suite":
            add(root, f"stage1/outputs/{case}/native-metrics.json", "qualification")
            for path in sorted((root / f"stage1/outputs/{case}/metrics").glob("areno_run_config.*.json")):
                add(root, path.relative_to(root).as_posix(), "qualification")
    for case in ("p0_full_cpu_selection", "p0_cli_prefix", "p0_cli_resume"):
        for name in ("launcher.log", "launch.json", "cleanup.json", "monitor.jsonl"):
            add(root, f"stage1/control/{case}/{name}", "qualification")
    for name in (
        "CPU_CI_RESULT.json",
        "LINT_CI_RESULT.json",
        "CI_TEST_COUNTS.json",
        "DOCS_FINAL_VALIDATION.json",
        "DOCS_FINAL.log",
        "CI_JOB_124803913534.log",
        "CI_JOB_124803913615.log",
        "CI_JOB_124803913632.log",
        "LINT_JOB_124803921172.log",
        "BEFORE_SOURCE.json",
        "PROOF.json",
        "CHANGES.md",
        "BEFORE_REGRESSIONS.log",
        "AFTER_TARGETED.log",
        "RUFF_CHECK.log",
        "RUFF_FORMAT.log",
    ):
        add(ci_root, name, "ci")
    for name in ("CI_JOB_124799447095_raw.log", "CI_8AF_JOB_124800414101.log", "CI_8AF_JOB_124800414310.log"):
        add(root, name, "failures")
    prepared = json.loads(files["qualification/PREPARED_stage1.json"])
    runtime = {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repo / "areno").rglob("*.py"))
    }
    for name, digest in runtime.items():
        if prepared["files"]["repo/" + name]["sha256"] != digest:
            raise ValueError("current source differs from actual native qualification")
    files["SOURCE_IDENTITIES.json"] = (
        json.dumps(
            {
                "measured_commit": prepared["git_commit"],
                "ci_commit": "487fe9ea538665b8ebe7f34ff056191139219680",
                "runtime_sha256": prepared["source_sha256"],
                "runtime_files": runtime,
                "ci_changes_since_native_run": subprocess.check_output(
                    ["git", "diff", "--name-only", prepared["git_commit"], "487fe9ea538665b8ebe7f34ff056191139219680"],
                    cwd=repo,
                    text=True,
                ).splitlines(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return files


def pack(root, ci_root, repo, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or any(output.glob("part-*.tar.xz")):
        raise FileExistsError("output already contains evidence")
    helper = load_packer()
    files = collect(root, ci_root, repo)
    raw = helper.logical_tar(files)
    chunks = [
        (f"chunks/u000-{i // helper.CHUNK_BYTES:04d}.bin", raw[i : i + helper.CHUNK_BYTES])
        for i in range(0, len(raw), helper.CHUNK_BYTES)
    ]
    parts, inventory = [], {}

    def emit(group):
        packed = helper.make_tar(dict(group), compressed=True)
        if len(packed) >= helper.MAX_ARCHIVE_BYTES:
            if len(group) == 1:
                raise ValueError("single chunk exceeds archive limit")
            emit(group[: len(group) // 2])
            emit(group[len(group) // 2 :])
            return
        name = f"part-{len(parts):03d}.tar.xz"
        (output / name).write_bytes(packed)
        parts.append(
            {"archive": name, "bytes": len(packed), "sha256": helper.digest(packed), "members": [k for k, _ in group]}
        )
        for key, value in group:
            inventory[key] = {"archive": name, "bytes": len(value), "sha256": helper.digest(value)}

    emit(chunks)
    manifest = {
        "schema": "areno-galore-p0-evidence-v1",
        "chunk_bytes": helper.CHUNK_BYTES,
        "max_archive_bytes": helper.MAX_ARCHIVE_BYTES,
        "parts": parts,
        "chunks": inventory,
        "units": [
            {
                "name": "p0-integration",
                "bytes": len(raw),
                "sha256": helper.digest(raw),
                "files": len(files),
                "chunks": [key for key, _ in chunks],
            }
        ],
        "exclusions": [
            "weights",
            "optimizer tensor payloads",
            "native binaries",
            "TensorBoard event binaries",
            "caches",
            "tokenized inputs",
            "Dolly/ablation data",
            "credentials",
            "full source-tree copies",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "parts": len(parts),
        "files": len(files),
        "compressed_bytes": sum(p["bytes"] for p in parts),
        "largest_part": max(p["bytes"] for p in parts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ci-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.ci_root, args.repo, args.output), indent=2))

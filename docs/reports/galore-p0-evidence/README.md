# P0 public-entry and CI evidence

This small, tensor-free package qualifies public CLI integration and fresh-process save/resume on runtime `47c75ef8…`. It is separate from frozen `337c7612…` quality/capacity studies. It contains one bounded archive, not a copy of the whole source tree.

```bash
python -I -S docs/reports/galore-p0-evidence/verify.py \
  docs/reports/galore-p0-evidence --repo .
python -m pytest tests/test_galore_p0_evidence_cpu.py -q
```

The standard-library verifier reuses the existing capacity container reader with an explicit P0 schema. It checks every archive/chunk/file identity; frozen approval records and included protocol source identities; every native scalar's public clock and finiteness; owner/barrier/released-leaf guards; refresh steps 0/200; public opt-in configuration; checkpoint clock/cursor metadata; before/after checkpoint inventory/count/byte arithmetic; normalized four-row public data and loader digest; recorded local test counts; fork CI head/job/log/count records; and recorded strict-docs log identity. Native train timing is recomputed only to audit refresh/ordinary records, not to certify a speed comparison.

`--repo` additionally hashes all 222 runtime Python files and requires the actual native-qualified source `47c75ef8a4e76cdbda5770a3875830d56e1a5420faf9c41ea9fd07f1972776ff`. Omit it to audit this frozen package from a later source. Native jobs ran at `8af25b4`; fork CI ran at `487fe9e` with only two test files changed. Individual runtime source hashes bind to the original943-file preparation inventory without copying those 943 files. Archived `.py.txt` harness/reader/loader sources are inert and never executed.

The package preserves two native TensorBoard **scalar exports**, metadata/UI configuration, three case launch/resource/stdout logs, protocol/harness/approval records, and one checkpoint's SHA256 inventory plus its two small metadata sidecars. The original binary event files and model/optimizer tensor payloads remain external. Therefore offline replay does not independently decode TensorBoard binaries or rehash the 11.32 GB checkpoint payload. The original before/after scan identities are retained and checked against each other. The CLI flow did not observe live/restored full-tensor identity; no such claim is made. Native cache and worker-step metrics were not emitted and are not synthesized. Normal SFT's consistency metric is absent because no scored old-policy reference is supplied; it is not reported as zero.

Four generic demonstration rows are retained (306 bytes), along with the existing Alpaca loader. Normalization/digest arithmetic is replayable, but model tokenization is not rerun offline. No Dolly/ablation payload, token stream, weights, `.pt`/`.so`, event binary, cache, credential or build HTML is included. Logs are forensic context; native scalar exports are the metric source. Scope is 202 public training calls and reliable user flow, not convergence or memory/throughput superiority.

Fork CPU CI and lint links belong to `ymping666/AReno` and exact commit `487fe9e`; they are not upstream PR checks or maintainer approval. Each Python 3.10/3.11/3.12 job has 1855 passed / 146 skipped; counts are not summed across interpreters. The local CPU-named selection has 1966 passed / 28 skipped / 12 subtests with CUDA visible, so it is not claimed CPU-only. Earlier `710663b` budget-handling and `8af25b4` test-portability CI failures, plus the test-only repair's before/after proofs, remain in the package. Frozen mathematical hashes and warning predicates were not loosened.

Deterministic repack from the original qualification and CI artifact directories:

```bash
python docs/reports/galore-p0-evidence/pack_evidence.py \
  --root /path/to/original-p0-round --ci-root /path/to/p0-ci-artifacts \
  --repo /path/to/runtime47-checkout --output /path/to/new-empty-package
```

The packer uses a fixed text allowlist and refuses to overwrite a package. The manifest inventories the exact original bytes and exclusions. Original host paths in archived records are informational; the portable verifier never resolves them to private files.

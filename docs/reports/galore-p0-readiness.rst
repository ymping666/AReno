GaLore public-entry qualification
=================================

The P0 integration adds explicit standard-CLI GaLore/deferred/fused options,
a deterministic text-SFT resume cursor, truthful optional SFT diagnostics and
a separate restored public logging clock. This report qualifies that user
flow and repository checks. It makes no new convergence, memory or speed claim.
Public option names remain a proposal awaiting maintainer feedback.

Sources and fixed workload
--------------------------

Native qualification uses commit ``8af25b43b3837e96206de928d9dc6eb210c3b0e4``,
runtime SHA256
``47c75ef8a4e76cdbda5770a3875830d56e1a5420faf9c41ea9fd07f1972776ff``
(222 Python files), and existing native extension SHA256
``f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226``.
Fork CI at ``487fe9ea538665b8ebe7f34ff056191139219680`` changes only two
test files for interpreter portability; all runtime bytes remain identical.
The earlier ``337c7612...`` quality/capacity records retain their own source
identity and are not relabeled as P0 measurements.

The actual console entry runs Qwen3-1.7B ModelScope revision
``4855588ea1a12789f2e965e5f52a9e4a24c94b2a`` on one A100-SXM4-40GB, using
four generic public SFT rows (306 bytes), batch/microbatch 4, prompt/response
limits 256 each, rank 128 / gap 200 / scale 1, strict recycled/compact FP32 recipe,
deferred projection and rounded scale/writeback fusion. LR=min LR=1e-5,
betas=(0.9,0.95), decay 0.01 and clip 1.0 are explicit. No experimental environment
flags or training/scoring overrides are used. Exact command, source/model/data
identities and software versions accompany each case.

Observed checks
---------------

.. list-table:: Separate execution and repository checks
   :header-rows: 1
   :widths: 28 72

   * - Check
     - Result
   * - Standard CLI prefix
     - 200 completed calls; native public steps 0--199; first refresh at step 0.
   * - Fresh console resume
     - Two further calls at public steps 200/201; production refresh at step 200
       completes in the resumed process, followed by an ordinary update.
   * - Native lifecycle exports
     - Each call records 226 owners once, unchanged versions at the backward
       barrier and zero full-leaf bytes after backward; finite loss/gradient
       and expected refresh schedule.
   * - One retained checkpoint
     - 44 files, 11,320,251,085 bytes; before/after inventory identities match.
       Native clock 200, public clock 199 and 200 completed SFT calls agree.
   * - Local CPU-named selection
     - 1966 passed, 28 skipped, 12 subtests passed. CUDA was visible and
       permitted GPU parameterizations ran; this is not a CPU-only claim.
   * - Fork CPU matrix at ``487fe9e``
     - Python 3.10/3.11/3.12 each: 1855 passed, 146 skipped. Interpreter runs
       are separate compatibility checks, not summed scientific replications.
   * - Fork lint / isolated docs
     - Pre-commit hooks passed. Strict Sphinx 8.2.3 / Shibuya 2026.7.12 HTML build
       passed with zero warnings; training dependencies were unchanged.

The `fork CPU matrix <https://github.com/ymping666/AReno/actions/runs/41291634418>`_
and `fork lint run <https://github.com/ymping666/AReno/actions/runs/41291637118>`_
are tied to ``487fe9e``. They are not upstream PR checks, a Ready transition or
maintainer approval. Earlier CI failures on optional budget handling and then
AST/warning-context portability are retained alongside their repairs. The
Python-version repair preserves all frozen mathematical hashes and the same
warning type/regex assertions.

What save/resume establishes
----------------------------

The public configuration selects the saved weights and native optimizer state;
``sft_training_state.json`` also restores the next filtered batch/epoch. Its
ordered normalized-dataset and loader hashes, token limits and batching
settings are bound to the checkpoint. In this four-row/one-batch-per-epoch
case, epoch 199 / next batch 1 / completed calls 200 is consistent with native and
public clocks; continuation advances directly to calls 200/201.

The CLI flow did not observe live/restored complete-tensor equality. The
checkpoint tensor payload is omitted from this portable package, so offline
verification checks recorded scan identities, not a new tensor rehash. Native
cache and worker-step fields were not emitted; no replacement values are
invented. Normal SFT has no old-policy reference, so native
``train/logp_abs_diff_mean`` is absent, not a measured zero. Historical scored
SFT/reference/restore evidence remains separately attributed.

Reproduce and inspect
---------------------

``examples/sft/galore/README.md`` gives the ordinary-user local JSONL command,
explicit flags and bounded save/resume example. Public deferred support is
currently single-GPU dense Qwen3 text SFT with one nonreentrant microbatch;
unsupported topology, LoRA, offload and accumulation combinations fail
explicitly. This tiny repeated dataset qualifies wiring and cursor behavior,
not learning or throughput. Native timing exports are forensic audit data.

The :download:`small tensor-free package <galore-p0-evidence/README.md>` retains
scalar exports, configs, original logs/protocols, source inventories and CI
proofs. It excludes tensor/event binaries, caches and private datasets. No
archived code is executed::

    python -I -S docs/reports/galore-p0-evidence/verify.py \
      docs/reports/galore-p0-evidence --repo .
    python -m pytest tests/test_galore_p0_evidence_cpu.py -q

The optional ``--repo`` check requires the exact runtime ``47c75ef8...`` source; omit it when
replaying frozen records from a later runtime. Start code review with
:doc:`the reviewer guide <../concepts/galore-reviewer-guide>` before consulting
raw archives.

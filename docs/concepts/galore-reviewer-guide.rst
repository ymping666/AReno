Reviewing the GaLore contribution
=================================

Start with the runtime and its focused tests below. Raw archives preserve
reproducibility and historical failures; they do not need to be read before
reviewing the optimizer. This contribution targets single-GPU full-parameter
training. Deferred projection changes gradient lifetime while preserving the
complete-backward/global-clipping barrier before persistent state updates.

Runtime review order
--------------------

.. list-table:: Owning code and the invariant to review
   :header-rows: 1
   :widths: 18 42 40

   * - Area
     - Files
     - Invariant
   * - Core GaLore
     - ``areno/engine/optim/galore.py``, ``recycled_subspace.py``
     - Projection rule, low-rank moments, ordinary fallback, basis refresh,
       parameter steps and explicit moment-transport policy.
   * - Selection/configuration
     - ``areno/engine/config.py``, ``modeling.py``,
       ``areno/api/trainer_config.py``, ``areno/cli/train.py``
     - Opt-in defaults and rejection of unsupported topology, offload,
       low-bit/adapter and deferred microbatch combinations.
   * - Deferred projection
     - ``areno/engine/optim/deferred_projection.py``,
       ``areno/engine/training.py``
     - Each completed leaf is projected once and released; model/state
       versions remain unchanged until complete backward; global clipping
       precedes optimizer publication.
   * - Rounded writeback
     - ``areno/engine/optim/compact_writeback.py``
     - Separate FP32 scale rounding before writeback; unsupported scalar
       specializations retain the checked path. Logical masters remain FP32.
   * - Checkpoint state
     - ``areno/engine/checkpoints/training_state.py``,
       ``areno/api/backend/cuda/checkpoint.py``
     - Canonical masters, moments, bases, RNG, schedule, ownership and
       hyperparameters restore before the next update.
   * - Public training behavior
     - ``areno/api/trainer.py``, ``areno/api/backend/cuda/training.py``,
       ``areno/api/trainers/sft.py``
     - SFT diagnostic availability and restored logging clocks are distinct
       from the optimizer update count and application data cursor.

The lossless master codec in ``areno/engine/optim/master_storage.py`` is existing
infrastructure. The deferred hook lifetime and optional rounded scale/writeback
fusion are separate contributions. Whole eligible parameter matrices are
projected; logical-block projection is a different policy. Numerical or speed
superiority over another implementation cannot be inferred from this choice.

Focused correctness checks
--------------------------

Run these before the complete repository suite::

    python -m pytest tests/test_galore_cli_cpu.py -q
    python -m pytest tests/test_galore_cpu.py \
      tests/test_galore_checkpoint_policy_cpu.py \
      tests/test_recycled_galore_cpu.py tests/test_integrated_galore_cpu.py -q
    python -m pytest tests/test_deferred_projection_reference_cpu.py \
      tests/test_deferred_projection_cpu.py -q
    python -m pytest tests/test_sft_logprob_diagnostics_cpu.py \
      tests/test_trainer_api_cpu.py -q

The independent small-matrix reference keeps ``atol=1e-6, rtol=1e-4`` and
includes clipping, production-style refresh and serialized continuation.
CPU tests do not establish native CUDA execution. On a compatible CUDA host::

    python -m pytest tests/test_deferred_projection_reference_cuda.py \
      tests/test_fused_scale_writeback_cuda.py -q

The public training entry is ``areno train``; :doc:`galore` describes the SDK
optimizer configuration and checkpoint contract. The P0 CLI proposal adds
``--galore-rank``, ``--galore-update-proj-gap``, ``--galore-scale`` and
``--galore-recipe {default,strict}``, plus explicit
``--galore-deferred-projection`` and ``--galore-fused-scale-writeback`` options.
``--resume-from-checkpoint`` restores training state from a saved directory;
``--ckpt`` remains the base-model reference. Ordinary training defaults remain
unchanged. The public deferred route currently supports SFT only, with
dense Qwen3, eager execution and one microbatch. Fused scaling additionally requires compact masters and tiled
writeback; the strict preset supplies that recipe. Public option naming
remains subject to maintainer review. Native logprob consistency is meaningful
only when real old-model scores are available; SFT placeholder zeros are not
a consistency reference. A resumed optimizer clock does not reconstruct a
user application's dataset cursor by itself.

Which evidence supports which claim?
------------------------------------

* :doc:`../reports/galore-deferred-qwen3` retains historical strict-runtime
  quality and three-card performance, the separate fusion ABBA experiment,
  source identities, repairs and original failures. Historical results are
  not relabeled as later source runs.
* :doc:`../reports/galore-final-quality-qwen3` directly measures runtime
  ``337c7612...`` for three methods, three seeds and 4096 updates. It retains
  every final response-target logprob. All candidate seeds satisfy the
  full-AdamW+2% gate; average NLL is 0.4603% above 8-bit. At context512/batch4,
  observed allocated training peaks fall 28.76--29.90% versus 8-bit, with
  0.63--1.76% whole score-plus-train time overhead. These timings are not
  repeated ABBA qualification.
* :doc:`../reports/galore-capacity-qwen3` uses separate exact-length packed
  inputs and worker peak instrumentation. The registered confirmed batch
  and context limits are twice full AdamW's, equal to 8-bit's; boundary
  memory savings versus 8-bit are about 1.9%. These runs contain no learning
  endpoint and do not extend the context512 result to every workload.

The P0 integration changes runtime source identity. The existing
``337c7612...`` measurements remain valid for that frozen source; they are
not automatically quality or performance qualifications of the later P0
implementation. New focused regressions and native entry/save/resume checks
must be reported separately before assigning new measured claims.

Review numerical tables before opening raw archives. For an independent
standard-library replay of final NLL and capacity records, from the repo root::

    python -I -S docs/reports/galore-final-quality-evidence/verify.py \
      docs/reports/galore-final-quality-evidence
    python -I -S docs/reports/galore-capacity-evidence/verify.py \
      docs/reports/galore-capacity-evidence

Omit ``--repo`` when auditing the archived frozen source from a later runtime:
that option intentionally rejects current code with a different identity.
The archives exclude weights and raw/tokenized dataset payloads. They allow
recorded input-identity checks and scalar arithmetic, rather than regeneration
of omitted tokens or future CUDA trajectories from digests.

Repository checks and review status
-----------------------------------

The normal CPU suite, pinned repository Ruff checks and a strict Sphinx build
are separate checks::

    python -m pytest tests/ -k cpu
    ruff check .
    ruff format --check .
    python -m sphinx -n -W --keep-going -b html \
      -D html_copy_source=0 docs /path/to/docs-build

Sphinx needs only ``docs/requirements.txt``; it can run in an isolated docs
virtual environment without installing the training runtime. Actual command,
versions, return code and logs should accompany each reported run. A fork
workflow run and upstream PR checks are distinct; a missing check is not a
successful check. Public API naming remains subject to maintainer feedback.
Mark the PR ready only after its listed integration checks are complete.

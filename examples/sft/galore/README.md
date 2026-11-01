# GaLore SFT through the standard CLI

This Bash example trains, saves and resumes a single-GPU dense Qwen3 model with the normal `areno train` entry. It exercises GaLore, deferred gradient projection and fused rounded scale/writeback through explicit options, without experimental environment variables. Public flag names remain a proposal for maintainer review.

Use an existing Linux/CUDA AReno installation and a local Qwen3-1.7B checkpoint/tokenizer. The measured-study ModelScope revision is `4855588ea1a12789f2e965e5f52a9e4a24c94b2a`; obtain pinned assets separately if reproducing that study. This four-row example checks the user flow; it is not the Dolly quality/performance protocol or a new benchmark result.

## Prepare a small local dataset

Run from the repository root, select a GPU you are authorized to use, and choose an output filesystem with room for a full model plus optimizer checkpoint. The example saves one checkpoint; checkpoints contain much more than low-rank moments.

```bash
MODEL_DIR=/path/to/local/Qwen3-1.7B
RUN_DIR=./runs/galore-cli-demo
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/train.jsonl" <<'JSONL'
{"instruction":"Name the capital of France.","input":"","output":"Paris."}
{"instruction":"Add the numbers.","input":"2 and 3","output":"5."}
{"instruction":"Give one primary color.","input":"","output":"Blue."}
{"instruction":"Name the largest planet in our solar system.","input":"","output":"Jupiter."}
JSONL

python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --algo sft --model-hub modelscope \
  --dataset-path "$RUN_DIR/train.jsonl" \
  --loader examples/sft/alpaca/dataset_loader.py
```

The existing Alpaca loader converts `instruction`/`input`/`output` to `prompt`/`response`. Inspection must return `"ok": true` before training. Local model/data paths require no dataset download. For repository references, explicitly select ModelScope rather than changing hubs silently.

## Train and save

Keep the common arguments for both invocations:

```bash
galore_args=(
  --algo sft
  --ckpt "$MODEL_DIR"
  --model-hub modelscope
  --dataset-path "$RUN_DIR/train.jsonl"
  --dataset-loader-fn examples/sft/alpaca/dataset_loader.py
  --world-size 1 --tp-size 1
  --batch-size 4 --mini-bs 4
  --epochs 16
  --max-prompt-tokens 256 --max-new-tokens 256
  --attn-backend native
  --lr 1e-5 --min-lr 1e-5
  --adam-beta1 0.9 --adam-beta2 0.95
  --weight-decay 0.01 --grad-clip-norm 1.0
  --galore-rank 128 --galore-update-proj-gap 200 --galore-scale 1.0
  --galore-recipe strict
  --galore-deferred-projection
  --galore-fused-scale-writeback
  --save-path "$RUN_DIR/checkpoints" --save-interval 12
)

CUDA_VISIBLE_DEVICES=0 areno train "${galore_args[@]}" \
  --max-steps 12 --metrics-log-dir "$RUN_DIR/initial-metrics"
```

`strict` selects the recycled projector, compact lossless logical FP32 masters, tile 16384, released gradients, tiled fused writeback and moment transport. Rank, gap, scale, learning rate and betas are explicit. Deferred projection and fused scaling are separate default-off opt-ins; omit them to use ordinary GaLore, or omit all GaLore options for ordinary optimizer selection.

The public deferred route currently supports SFT only and requires one nonreentrant microbatch. The command uses batch=microbatch=4; smaller microbatches, extra accumulation, non-Qwen3 models, compilation, active multimodal optimizer groups, low-bit AdamW, LoRA, distributed topology and optimizer offload are outside the supported combination and fail explicitly.

A successful run reaches 12 completed Trainer calls and creates `checkpoints/step_000012/`, containing model/tokenizer assets, native `training_state.pt`/`training_state.json` and `sft_training_state.json`. The native state contains masters, bases, moments, hyperparameters, schedule and RNG; metadata separately preserves the public call clock. The JSON SFT cursor binds the ordered normalized dataset, loader bytes and filtering/batching settings. Check these files and the actual training logs before proceeding; model load alone is insufficient.

## Resume to a total of 16 calls

```bash
CUDA_VISIBLE_DEVICES=0 areno train "${galore_args[@]}" \
  --resume-from-checkpoint "$RUN_DIR/checkpoints/step_000012" \
  --max-steps 16 --metrics-log-dir "$RUN_DIR/resumed-metrics"
```

`--ckpt` remains the required base-model reference; restored weights come from `--resume-from-checkpoint`. `--max-steps 16` means 16 total Trainer calls, including the saved 12, so this resumes for four further calls at the next filtered batch/epoch. It does not replay the training prefix. The 16 total epochs provide enough batches for this tiny dataset. Save interval remains 12, so this bounded continuation does not write a second checkpoint.

Keep normalized dataset contents/order, loader code, token limits, batch/microbatch and accumulation settings unchanged, and use the same model and optimizer recipe. Missing state, changed dataset/filter/batch signatures and incompatible optimizer policies are rejected. `--max-steps` and the total epoch cap may be extended; they do not redefine the saved cursor. This initial resume convenience is CUDA text SFT; no LoRA/distributed/RL reward or rollout-state recovery is claimed.

## Metrics and scope

Normal SFT has no old-policy score reference. Its input old logprobs are empty, so the optional native `train/logp_abs_diff_mean` is absent; absence is not a measured zero error. The public SDK can emit this native diagnostic when callers supply actual complete same-model logprobs immediately before training. The original measured protocols do this explicitly.

The 12+4-call demonstration does not cross the production gap200 refresh boundary and does not establish convergence, memory saving, speed or future CUDA bitwise trajectory equivalence. The independent reference/refresh tests and frozen three-seed evidence are linked from `docs/concepts/galore-reviewer-guide.rst`; new P0 source qualifications must be recorded separately.

# Debugging Guide

A practical guide to debugging accuracy issues and isolating training/inference components in Relax.

## Accuracy Debugging

### 1. Check Rollout Responses

Search for `Finish rollout` in the logs to locate the current rollout ID and the generated responses. First, determine whether the responses are coherent and sensible.

- **If garbled output appears from the very first step**, this usually indicates a checkpoint loading error or model conversion issue. A thorough method is to save all parameters inside the model's SGLang `load_weights` implementation and compare them against the loaded checkpoint. If all parameter updates are correct but the issue persists, it may be caused by special buffers in SGLang being released during the `release_memory_occupation` phase. If you are testing with a pretrain model, try switching to the instruct version of the same architecture to check whether the garbled output is specific to pretrain models.

- **If the first step produces normal responses but later steps degrade**, the training has diverged. You need to carefully inspect each step's reward computation, hyperparameter settings, and other factors.

### 2. Verify log_probs and ref_log_probs

Check the rollout stats printed at the first step. Verify that `log_probs` and `ref_log_probs` are exactly equal (i.e., KL = 0 at step 1) and the values are small.

- **If they are not exactly equal**, this is typically caused by non-deterministic kernels in Transformer Engine. For example, in some versions of TE, Megatron requires `--attention-backend flash` to force Flash Attention usage and avoid numerical instability of fused attention under Context Parallelism (CP).

- **If the values are large** (e.g., > 1), there are generally two possibilities:
  - Very large values usually indicate a training configuration error.
  - Values only slightly higher than the SFT loss baseline (e.g., logprob around 0.8 for an instruct model) may mean the data doesn't match the training chat template, or doesn't match the cold-start distribution.

### 3. Verify KL and grad_norm at Step 1

With one-step-per-rollout (`num_steps_per_rollout == 1`), check whether KL is 0 and `grad_norm` is small at step 1.

Issues at this stage are typically Megatron / Transformer Engine related bugs. For example:

- MoE models require `--moe-permute-fusion` to be enabled.

## SGLang Runtime Crashes

### 1. Stop Strings Trigger IMA Under Agentic High Concurrency

**Symptom**: Mid-rollout (after the rollout has been running for tens of seconds to a few minutes), the SGLang scheduler raises an exception, followed by `SIGQUIT received. ... It usually means one child failed`. The engine is killed and the rollout stalls:

```
[TP0] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File "sglang/srt/managers/scheduler_output_processor_mixin.py", line 371, in process_batch_result_decode
    next_token_ids = next_token_ids.tolist()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Root cause**: The `stop` strings on an OpenAI-compatible request are handled by the SGLang detokenizer subprocess via per-token incremental decode plus substring match. Under agentic workloads with 100+ concurrent sessions (each running multiple chat completions), the detokenizer's per-token work scales linearly with the number of stop strings and can starve its 20s heartbeat. The GPU-side async kernel error then surfaces as an illegal memory access.

**Workaround**: Prefer `stop_token_ids` over `stop` whenever possible. `stop_token_ids` is matched on the GPU against generated token ids and bypasses the detokenizer subprocess entirely.

- Single-token terminators (e.g. `</tool_call>`, `<|im_end|>`): look up their ids via the tokenizer and pass them as `stop_token_ids`.
- Multi-token sequences (e.g. `</answer>` = `[510, 8944, 29]`) cannot be expressed as `stop_token_ids`. Prefer terminating on the chat-template turn end (`<|im_end|>`) and extracting the answer via regex after generation, so `stop` can still be avoided.

Pass `stop_token_ids` from the agent app via `extra_body`:

```python
extra_body = {"stop_token_ids": [248059, 248046]}  # </tool_call>, <|im_end|>
resp = await client.chat.completions.create(
    model=...,
    messages=messages,
    extra_body=extra_body,
    ...
)
```

See the comments in `examples/deepeyes_v2_agentic/app/agent.py` and `examples/deepeyes_v2_agentic/app/deepeyes_v2_config.yaml` for a working example.

## Isolated Debugging

Relax supports running the training and inference components independently, which allows:

- Debugging the inference pipeline with minimal GPU resources.
- Debugging the training pipeline with fixed inputs, eliminating rollout randomness.

### Available Debug Flags

The following CLI arguments enable isolated debugging:

| Flag | Description |
|---|---|
| `--debug-rollout-only` | Only initialize SGLang (skip Megatron). Use for inference debugging. |
| `--debug-train-only` | Only initialize Megatron (skip SGLang). Use for training debugging. |
| `--save-debug-rollout-data <path>` | Save rollout results to the specified path for later replay. |
| `--load-debug-rollout-data <path>` | Load rollout data from the specified path. Automatically sets `--debug-train-only`. |
| `--dump-details <dir>` | Dump all training details (automatically enables rollout data saving). |
| `--load-forge-rollout-data <path>` | Replay a saved rollout dump **while keeping SGLang live** (does not set `--debug-train-only`). Requires `--rollout-function-path relax.engine.rollout.forge_load.generate_rollout`. |

### Workflow 1: Debug Inference Only

Use `--debug-rollout-only` to skip Megatron initialization entirely. Only SGLang engines will be started, allowing you to test inference with fewer GPUs.

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    # ... other rollout args
```

You can combine this with `--save-debug-rollout-data` to capture rollout results:

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... other rollout args
```

### Workflow 2: Debug Training Only

Use `--load-debug-rollout-data` to load pre-saved rollout data and run only the training pipeline. This automatically sets `debug_train_only=True`, so SGLang will not be initialized.

```bash
python3 relax/entrypoints/train.py \
    --load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... other training args
```

This approach is especially useful for:

- Tuning parallelism configurations (TP, PP, EP, CP) without waiting for rollout.
- Reproducing and fixing training-specific issues with deterministic inputs.
- Iterating quickly on loss computation or optimizer changes.

### Workflow 3: Full Detail Dump

Use `--dump-details` to save all training details for post-hoc analysis. When set, it automatically enables:

- `--save-debug-rollout-data` at `<dir>/rollout_data/{rollout_id}.pt`
- `--save-debug-train-data` at `<dir>/train_data/{rollout_id}_{rank}.pt`

```bash
python3 relax/entrypoints/train.py \
    --dump-details /path/to/dump/dir \
    # ... other args
```

::: tip
`--dump-details` is also useful for collecting data for bug reports — it captures everything needed to reproduce an issue.
:::

### Workflow 4: Replay Rollout with SGLang Live (Memory Profiling)

`--load-debug-rollout-data` (Workflow 2) skips SGLang entirely, so it measures the training loop in isolation. To profile the **real colocate memory footprint** — training peak, weight sync, and the offload/onload transitions — you need SGLang, the router, and weight update to stay live. `forge_load` replays a saved rollout dump instead of generating, keeping the full pipeline alive while skipping only the (slow) real generation.

First capture a dump (e.g. via Workflow 1 or `--dump-details`), then replay it:

```bash
python3 relax/entrypoints/train.py \
    --rollout-function-path relax.engine.rollout.forge_load.generate_rollout \
    --load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt \
    # ... keep the SAME model / colocate / batch args as the run that produced the dump
```

`--load-forge-rollout-data` accepts two forms, mirroring the `format(rollout_id=...)` convention of `--save-debug-rollout-data`:

- **Literal path** — `--load-forge-rollout-data /path/to/rollout_data/0.pt`
  No `{rollout_id}` placeholder: the same file is reused for **every** rollout step. Best for a memory test that replays one dump across `--num-rollout > 1`.
- **Template path** — `--load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt`
  Contains `{rollout_id}`: each step loads its **own** file (`data_0.pt`, `data_1.pt`, …). If a step's file is missing, the training path falls back to the `0` dump.

::: warning
Keep batch-related args (`--global-batch-size`, `--n-samples-per-prompt`, parallelism) identical to the dump run — the dump must contain exactly one step's worth of samples. Do **not** combine with `--debug-rollout-only` or `--load-debug-rollout-data`; both would defeat the purpose (no training, or SGLang skipped). `forge_load` does not exercise decode-time memory, so it is not a substitute for measuring the generation-phase peak.
:::

## Local RL traces

Training launched through `relax.entrypoints.train` records RL stages to disk by default. Each launch gets a new `rl.run.id`, including checkpoint restarts. Files are written under `<save>/otel-traces/<run_id>/`; without `--save`, the root is `otel-traces` in the driver working directory. Each process writes its own OTLP JSONL file. Each line is an OTLP trace export request, so concatenate the files for analysis rather than parsing the entire directory as one JSON document.

Set `RELAX_OTEL_DIR` before submission to choose another output root, or `RELAX_OTEL_ENABLED=0` to disable RL traces. `OTEL_SDK_DISABLED=true` also disables them. Use a shared, persistent filesystem when you want to read all workers' output from one machine; on node-local disks, collect files from each worker separately. No collector endpoint is required.

Batch stages use independent traces. Join stages using `rl.run.id` and `rl.step.id`; `rl.iteration.index` preserves the framework's zero-based rollout counter. Each batch has its own `rl.batch.id` and actual input count. Member spans share the batch trace and name their parent explicitly: `rl.rollout.sample` records a complete input rollout, and `rl.reward.score` belongs to `rl.reward.batch`. Dependencies between different stages use Links. Agent calls retain independent traces and Link to their sample.

Structured results, sample inputs and responses are stored directly as complete JSON strings in span attributes. New traces do not require `_ref` attachments. Only actual per-sample rollout and scoring calls create child spans, marked `rl.sample.timing=operation`. Batched scoring, forward, advantage, filtering, transfer and training keep their member results on the batch span; they do not create zero-duration result spans. A singleton OPD forward uses one span for its actual request. Training uses reference update outcomes by update ID. Legacy 0.10 result spans can still be read separately from execution timelines. Do not add parent and child durations. Agentic runs also enable the existing observation records, retain Session requests, attempts, states and exports, and associate training consumption with Export IDs. Missing external tool timings are left unknown. These records include model inputs and outputs, so choose output permissions and retention appropriate to the training data.

A completed stage is appended when it ends. A process killed during an active stage can leave that stage absent. An observation write failure is logged without failing training; incomplete rank summaries use `relax.observation.partial` instead of claiming complete training results. Trace collection does not add distributed collectives; compact GPU summaries are copied asynchronously and read off the training thread.

To check a completed run's file format, inline results, parent relationships and stage coverage, run:

```bash
python scripts/tools/validate_rl_traces.py /path/to/otel-traces/RUN_ID --min-train-steps 2
```

The check fails on failed spans, incomplete rank summaries, invalid results, missing member parents or insufficient training steps. It summarizes recorded stages and optimizer decisions; it does not assert that algorithms or paths absent from the run were exercised.

#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-30B-A3B two-iteration OTel verification on 16 GPUs.
# Set OTEL_SMOKE_AGENTIC=1 for the local two-turn agent.
# Set SGLANG_TRACE_ENDPOINT to an OTLP/gRPC receiver to collect backend spans.
#
# Usage:
#   bash scripts/training/text/run-qwen3-30B-A3B-16xgpu-otel-smoke.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-dapo-16xgpu-otel-smoke}"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# Preserve the network interface selected by the experiment environment.
# Trace collection uses framework defaults; this script does not enable it.
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    RUNTIME_ENV_JSON=$(python3 - <<'INNER_PY'
import json, os
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {})["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
print(json.dumps(runtime))
INNER_PY
)
    export RUNTIME_ENV_JSON
    export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}GLOO_SOCKET_IFNAME"
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-30B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3-30B-A3B"
   --megatron-to-hf-mode bridge

   --load "${SAVE_DIR}/${EXP_NAME}"
   --save "${SAVE_DIR}/${EXP_NAME}"
   --save-interval 2
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-temperature 1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480

   # use deepep for megatron
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-dtype fp32
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-expert-parallel-size 8
)

if [[ -n "${SGLANG_TRACE_ENDPOINT:-}" ]]; then
   SGLANG_ARGS+=(--sglang-enable-trace --sglang-otlp-traces-endpoint "${SGLANG_TRACE_ENDPOINT}")
   # Multi-turn decoding can outpace the SDK's small default export queue.
   RUNTIME_ENV_JSON=$(python3 - <<'INNER_PY'
import json, os
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime.setdefault("env_vars", {})
trace_defaults = {
    "SGLANG_TRACE_LEVEL": "2",
    "OTEL_BSP_MAX_QUEUE_SIZE": "65536",
    "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "512",
}
for name, default in trace_defaults.items():
    env.setdefault(name, os.environ.get(name, default))
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""))
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(filter(None, [propagate, *trace_defaults]))
print(json.dumps(runtime))
INNER_PY
)
   export RUNTIME_ENV_JSON
fi

if [[ "${OTEL_SMOKE_AGENTIC:-0}" == 1 ]]; then
   OTEL_SMOKE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
   ROLLOUT_ARGS+=(
      --use-agentic-rollout
      --agent-command "python3 ${OTEL_SMOKE_ROOT}/examples/otel_tracing/agent.py"
      --agent-cwd "${OTEL_SMOKE_ROOT}"
      --agent-timeout 600
      --agentic-concurrency 4
      --agentic-reasoning-parser qwen3
   )
   if [[ -n "${SGLANG_TRACE_ENDPOINT:-}" ]]; then
      agent_otlp_endpoint=${SGLANG_TRACE_ENDPOINT}
      [[ ${agent_otlp_endpoint} == *://* ]] || agent_otlp_endpoint="http://${agent_otlp_endpoint}"
      ROLLOUT_ARGS+=(--agent-env "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${agent_otlp_endpoint}")
   fi
fi

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}' \
   --colocate \
   --max-staleness 0 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Any

from relax.utils.misc import load_function
from relax.utils.rl_tracing import rollout_call
from relax.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    config = args[0] if args else kwargs.get("args")
    rollout_id = args[1] if len(args) > 1 else kwargs.get("rollout_id")
    with rollout_call(config, rollout_id, evaluation) as rollout:
        output = fn(*args, **kwargs, evaluation=evaluation)
        if rollout is not None:
            rollout.span.freeze_end()
            # Resident group tasks can survive a pause. Attribute their exports to the actual consuming call.
            if getattr(config, "use_agentic_rollout", False) and isinstance(output, RolloutFnTrainOutput):
                for group in output.samples:
                    rollout.exported_versions(group)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    # Apply --rollout-sample-filter-path (train only). The filter sets
    # sample.remove_sample=True in-place; downstream (relax/utils/utils.py:126)
    # zeros loss_mask for those samples so they don't contribute gradient, while
    # keeping reward in GRPO group-normalization. Reloaded every call
    # (ReloadScope.IMMEDIATE per reload_utils.py:180) to support hot-swap.
    if not evaluation and isinstance(output, RolloutFnTrainOutput):
        train_args = args[0] if args else kwargs.get("args")
        filter_path = getattr(train_args, "rollout_sample_filter_path", None)
        if filter_path:
            load_function(filter_path)(train_args, output.samples)

    return output

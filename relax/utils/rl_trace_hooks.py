# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapters at real RL operation boundaries; no synthetic parent spans."""

from __future__ import annotations

import inspect
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_tracing import enabled, iteration_context, model_name, stage


_PURPOSE: ContextVar[str] = ContextVar("rl_purpose", default="training")


def trace_iteration(function: Callable) -> Callable:
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        iteration = signature.bind_partial(*args, **kwargs).arguments.get("rollout_id")
        with iteration_context(iteration):
            return function(*args, **kwargs)

    @wraps(function)
    async def async_call(*args: Any, **kwargs: Any) -> Any:
        iteration = signature.bind_partial(*args, **kwargs).arguments.get("rollout_id")
        token = _PURPOSE.set("evaluation" if "eval" in function.__name__ else "training")
        try:
            with iteration_context(iteration):
                return await function(*args, **kwargs)
        finally:
            _PURPOSE.reset(token)

    return async_call if inspect.iscoroutinefunction(function) else call


def export_id(sample: Any) -> str:
    value = getattr(sample, "sample_export_id", -1)
    if value is not None and value >= 0:
        return str(value)
    # Custom generators may export multiple rows with the same sample.index.
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    return sample.metadata.setdefault("_rl_export_id", uuid.uuid4().hex)


def _samples(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return [value]
    return [sample for item in value for sample in _samples(item)]


def _sources(samples: list[Any]) -> list[dict[str, Any]]:
    return [
        s.metadata["_rl_source"]
        for s in samples
        if isinstance(getattr(s, "metadata", None), dict) and s.metadata.get("_rl_source")
    ]


@best_effort
def _publish_source(samples: list[Any], observation: Any) -> None:
    carrier = observation.context()
    if carrier is not None:
        for sample in samples:
            if not isinstance(sample.metadata, dict):
                sample.metadata = {}
            sample.metadata["_rl_source"] = carrier


def trace_generation(function: Callable) -> Callable:
    @wraps(function)
    async def call(state: Any, args: Any, sample: Any, *rest: Any, **kwargs: Any) -> Any:
        if not enabled(args):
            return await function(state, args, sample, *rest, **kwargs)
        attrs = {
            "rl.batch.id": uuid.uuid4().hex,
            "rl.batch.size": 1,
            "rl.purpose": "evaluation" if kwargs.get("evaluation", False) else "training",
            "rl.rollout.mode": "multi_turn"
            if getattr(sample, "generate_function_path", None) or getattr(args, "custom_generate_function_path", None)
            else "single_turn",
            "rl.model.name": model_name(args),
            "rl.backend.name": "sglang",
        }
        with stage(args, "rl.rollout.generate", attributes=attrs) as observation:
            result = await function(state, args, sample, *rest, **kwargs)
            _publish_source(_samples(result), observation)
            return result

    return call


def reward_attributes(args: Any, evaluation: bool = False) -> dict[str, Any]:
    name = getattr(args, "custom_rm_path", None) or getattr(args, "rm_type", None) or "reward"
    # Custom scoring can declare its category. Never infer a model from Ray.
    kind = getattr(args, "rl_reward_type", None)
    if kind is None:
        kind = "model" if str(name) in ("dapo-genrm", "genrm", "remote_rm") else "rule"
    return {
        "rl.purpose": "evaluation" if evaluation else _PURPOSE.get(),
        "rl.reward.name": str(name),
        "rl.reward.type": kind,
    }


@best_effort
def record_rewards(observation: Any, samples: list[Any], values: list[Any], args: Any) -> None:
    results = []
    for sample, value in zip(samples, values, strict=True):
        item = {
            "export_id": export_id(sample),
            "source": observation.span.attributes.get("rl.reward.name", reward_attributes(args)["rl.reward.name"])
            if observation.span is not None
            else reward_attributes(args)["rl.reward.name"],
            "status": "valid",
            "value": value,
        }
        if isinstance(value, dict) and getattr(args, "reward_key", None):
            item["scalar_key"] = args.reward_key
        if value is None:
            item.update(status="invalid", reason="scorer returned no reward")
        results.append(item)
    observation.data("rl.reward.results", results)
    _publish_source(samples, observation)


def trace_reward(function: Callable) -> Callable:
    @wraps(function)
    async def call(args: Any, sample: Any, **kwargs: Any) -> Any:
        if not enabled(args):
            return await function(args, sample, **kwargs)
        samples = _samples(sample)
        with stage(
            args,
            "rl.reward.score",
            attributes=reward_attributes(args, kwargs.get("evaluation", False)),
            links=_sources(samples),
        ) as observation:
            result = await function(args, sample, **kwargs)
            record_rewards(observation, samples, result if isinstance(sample, list) else [result], args)
            return result

    return call


def trace_transfer(function: Callable) -> Callable:
    @wraps(function)
    async def call(args: Any, batch_samples: Any, batch_count: int, rollout_id: int, *rest: Any, **kwargs: Any) -> Any:
        if not batch_samples or not enabled(args):
            return await function(args, batch_samples, batch_count, rollout_id, *rest, **kwargs)
        samples = _samples(batch_samples)
        data = {"export_ids": [export_id(sample) for sample in samples], "send_start_time": time.time()}
        with stage(
            args,
            "rl.data.transfer",
            attributes={
                "rl.transfer.target": f"train_{rollout_id}",
                "rl.transfer.target.step_id": f"iteration:{rollout_id}",
            },
            links=_sources(samples),
        ) as observation:
            try:
                result = await function(args, batch_samples, batch_count, rollout_id, *rest, **kwargs)
                data["receive_ack_time"] = time.time()
                return result
            finally:
                observation.data("rl.transfer.data", data)

    return call


def callable_name(function: Any) -> str:
    function = getattr(function, "func", function)
    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


def trace_filter(function: Callable) -> Callable:
    @wraps(function)
    def call(fn: Callable | None, *args: Any, **kwargs: Any) -> Any:
        if fn is None or not enabled():
            return function(fn, *args, **kwargs)
        config = args[0] if len(args) > 1 else None
        samples = _samples(args[1] if len(args) > 1 else args[0])
        with stage(
            config,
            "rl.data.filter",
            attributes={"rl.filter.name": callable_name(fn)},
            links=_sources(samples),
        ) as observation:
            result = function(fn, *args, **kwargs)
            observation.data(
                "rl.filter.results",
                [
                    {
                        "export_id": export_id(sample),
                        "decision": "keep" if result.keep else "drop",
                        **({"reason": result.reason or "filter rejected sample"} if not result.keep else {}),
                    }
                    for sample in samples
                ],
            )
            _publish_source(samples, observation)
            return result

    return call


def _main_rank() -> bool:
    from relax.backends.megatron.initialize import is_megatron_main_rank

    return is_megatron_main_rank()


def trace_forward(function: Callable) -> Callable:
    @wraps(function)
    def call(
        owner: Any, data_iterator: Any, num_microbatches: Any, store_prefix: str = "", collect_topk: bool = False
    ) -> Any:
        role = {"ref_": "reference", "teacher_": "teacher"}.get(store_prefix, "actor")
        attrs = {"rl.model.name": model_name(owner.args, role), "rl.backend.name": "megatron"}
        if role == "actor":
            attrs["rl.forward.purpose"] = "distillation" if getattr(owner.args, "use_opd", False) else "old_policy"
        elif role == "teacher":
            attrs["rl.forward.output"] = "topk_distribution" if collect_topk else "sampled_token_logprob"
        with stage(owner.args, f"rl.{role}.forward", attributes=attrs, active=_main_rank()):
            return function(
                owner, data_iterator, num_microbatches, store_prefix=store_prefix, collect_topk=collect_topk
            )

    return call


def trace_advantage(function: Callable) -> Callable:
    @wraps(function)
    def call(args: Any, rollout_data: Any, **kwargs: Any) -> Any:
        from relax.utils.rl_tracing import iteration_attributes
        from relax.utils.rl_train_trace import window

        iteration = iteration_attributes().get("rl.iteration.index")
        if all(rollout_data.get(key) is None for key in ("log_probs", "rollout_log_probs", "values")):
            return function(args, rollout_data, **kwargs)
        with window(args, "advantage", iteration, estimator=args.advantage_estimator):
            result = function(args, rollout_data, **kwargs)
            return result

    return call


def trace_train(function: Callable) -> Callable:
    @wraps(function)
    def call(
        rollout_id: int,
        model: Any,
        optimizer: Any,
        opt_param_scheduler: Any,
        data_iterator: Any,
        num_microbatches: Any,
    ) -> Any:
        from megatron.training import get_args

        from relax.utils.rl_train_trace import window

        args = get_args()
        role = getattr(model[0], "role", "actor")
        with window(args, role, rollout_id, data=getattr(data_iterator[0], "rollout_data", {})):
            return function(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches)

    return call


def optimizer_result(rollout_id: int, step_id: int, successful: bool, reason: str | None = None) -> None:
    from relax.utils.rl_train_trace import optimizer_result as record

    record(successful, reason)


def trace_agentic_generation(function: Callable) -> Callable:
    @wraps(function)
    async def call(owner: Any, stream: Any) -> Any:
        with stage(
            owner.args,
            "rl.rollout.generate",
            inherit_iteration=False,
            attributes={
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": len(stream.session_results),
                "rl.purpose": "evaluation" if owner.rollout_mode == "eval" else "training",
                "rl.rollout.mode": "agentic",
                "rl.model.name": model_name(owner.args),
                "rl.backend.name": "sglang",
            },
        ) as observation:
            try:
                return await function(owner, stream)
            finally:
                if len(stream.trace_iterations) == 1:
                    from relax.utils.rl_tracing import iteration_attributes

                    for key, value in iteration_attributes(next(iter(stream.trace_iterations))).items():
                        observation.set(key, value)

    return call


def trace_agentic_transfer(function: Callable) -> Callable:
    @wraps(function)
    async def call(*, args: Any, batch_samples: Any, partition_id: int, **kwargs: Any) -> Any:
        if not enabled(args):
            return await function(args=args, batch_samples=batch_samples, partition_id=partition_id, **kwargs)
        samples = _samples(batch_samples)
        data = {"export_ids": [export_id(sample) for sample in samples], "send_start_time": time.time()}
        with stage(
            args,
            "rl.data.transfer",
            attributes={
                "rl.transfer.target": f"train_{partition_id}",
                "rl.transfer.target.step_id": f"iteration:{partition_id}",
            },
            links=_sources(samples),
        ) as observation:
            try:
                result = await function(args=args, batch_samples=batch_samples, partition_id=partition_id, **kwargs)
                data["receive_ack_time"] = time.time()
                return result
            finally:
                observation.data("rl.transfer.data", data)

    return call


@best_effort
def _record_custom_advantage(observation: Any, group: Any, accepted: bool) -> None:
    rows = []
    if accepted:
        for sample in group.samples:
            value = sample.custom_advantage
            values = value if isinstance(value, list) else [value] * sample.response_length
            mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            count = sum(bool(m) for m in mask)
            row = {"export_id": export_id(sample), "token_count": count}
            if count:
                row["mean"] = sum(float(v) * m for v, m in zip(values, mask, strict=True) if m) / sum(mask)
            rows.append(row)
    observation.data("rl.advantage.results", rows)


def trace_custom_advantage(function: Callable) -> Callable:
    @wraps(function)
    async def call(owner: Any, group: Any) -> Any:
        if not enabled(owner.args):
            return await function(owner, group)
        with stage(
            owner.args,
            "rl.advantage.compute",
            attributes={"rl.advantage.estimator": callable_name(owner.custom_advantage_func)},
        ) as observation:
            result = await function(owner, group)
            _record_custom_advantage(observation, group, result)
            return result

    return call


def trace_opd_forward(role: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        async def call(owner: Any, sample: Any, *args: Any, **kwargs: Any) -> Any:
            if not sample.response_length:
                return await function(owner, sample, *args, **kwargs)
            attrs = {"rl.model.name": model_name(owner.args, role), "rl.backend.name": "sglang"}
            if role == "teacher":
                attrs["rl.forward.output"] = (
                    "topk_distribution" if owner.topk_worker is not None else "sampled_token_logprob"
                )
            else:
                attrs["rl.forward.purpose"] = "distillation"
            with stage(owner.args, f"rl.{role}.forward", attributes=attrs) as observation:
                result = await function(owner, sample, *args, **kwargs)
                if result is False:
                    observation.end(RuntimeError("forward returned no usable scoring result"))
                return result

        return call

    return decorate


@best_effort
def record_cpu_advantage(observation: Any, data: Any, result: Any) -> None:
    """Observe the standalone CPU advantage service without device
    transfers."""
    import math

    import torch

    rows = []
    for identity, advantage, mask in zip(
        data["sample_export_ids"], result["advantages"], data["loss_masks"], strict=True
    ):
        tensors = [value for value in (identity, advantage, mask) if isinstance(value, torch.Tensor)]
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("Standalone advantage observation requires CPU tensors")
        count = int((mask != 0).sum())
        row = {"export_id": str(int(identity)), "token_count": count}
        if count:
            mean = float((torch.where(mask != 0, advantage, 0.0) * mask).sum() / mask.sum())
            if math.isfinite(mean):
                row["mean"] = mean
        rows.append(row)
    observation.data("rl.advantage.results", rows)

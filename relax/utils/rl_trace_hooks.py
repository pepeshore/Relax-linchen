# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapters at real RL operation boundaries; no synthetic parent spans."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_tracing import Stage, current_stage, enabled, iteration_context, model_name, stage


_PURPOSE: ContextVar[str] = ContextVar("rl_purpose", default="training")
_GENERATION_BATCH: ContextVar[GenerationBatch | None] = ContextVar("rl_generation_batch", default=None)
_REWARD_BATCH: ContextVar[RewardBatch | None] = ContextVar("rl_reward_batch", default=None)


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


@best_effort
def _record_generation_version(observation: Any, samples: list[Any]) -> None:
    histories = [getattr(sample, "weight_versions", []) for sample in samples]
    if not histories or any(not history for history in histories):
        return
    versions = [version for history in histories for version in history]
    if any(version is None or version == "" for version in versions):
        return
    unique = {str(version) for version in versions}
    if len(unique) == 1:
        observation.set("rl.model.version", next(iter(unique)))


class GenerationBatch:
    """Observe admitted tasks without waiting for their reward/transfer
    tails."""

    def __init__(self, args: Any, evaluation: bool) -> None:
        self.args = args
        self.evaluation = evaluation
        self.observation: Stage | None = None
        self.members: dict[int, Any] = {}
        self.pending: set[int] = set()
        self.owners: dict[int, asyncio.Task] = {}
        self.versions: set[str] = set()
        self.unknown_version = False
        self.error: BaseException | None = None
        self.end_time: int | None = None
        self.sealed = False

    @best_effort
    def track(self, task: asyncio.Task, samples: Any) -> None:
        identities = []
        for sample in _samples(samples):
            # The existing completed-result fast path performs no generation.
            if getattr(getattr(sample, "status", None), "name", None) in ("COMPLETED", "TRUNCATED"):
                continue
            identity = id(sample)
            if identity not in self.members:
                self.members[identity] = sample
                self.pending.add(identity)
            identities.append(identity)
            # A child replaces its group task as owner. A failing gather must
            # not close siblings that are still generating.
            self.owners[identity] = task
        if identities and self.observation is None:
            self.observation = Stage(
                self.args,
                "rl.rollout.generate",
                attributes={
                    "rl.batch.id": uuid.uuid4().hex,
                    "rl.purpose": "evaluation" if self.evaluation else "training",
                    "rl.model.name": model_name(self.args),
                    "rl.backend.name": "sglang",
                },
            )
        if self.observation is not None:
            self.observation.set("rl.batch.size", len(self.members))
            multi_turn = getattr(self.args, "custom_generate_function_path", None) or any(
                getattr(sample, "generate_function_path", None) for sample in self.members.values()
            )
            self.observation.set("rl.rollout.mode", "multi_turn" if multi_turn else "single_turn")

        @best_effort
        def done(finished: asyncio.Task) -> None:
            remaining = [i for i in identities if i in self.pending and self.owners.get(i) is finished]
            if remaining:
                error = asyncio.CancelledError() if finished.cancelled() else finished.exception()
                for identity in remaining:
                    # Only a pre-dispatch failure/cancellation/abort reaches
                    # this fallback; successful generation finishes below.
                    self.finish(self.members[identity], [], error)

        task.add_done_callback(done)

    @best_effort
    def finish(self, sample: Any, results: list[Any], error: BaseException | None) -> None:
        identity = id(sample)
        if identity not in self.pending:
            return
        self.end_time = time.time_ns()
        self.pending.remove(identity)
        self.owners.pop(identity, None)
        if error is not None and self.error is None:
            self.error = error
        histories = [getattr(result, "weight_versions", []) for result in results]
        if not histories or any(not history for history in histories):
            self.unknown_version = True
        for history in histories:
            for version in history or []:
                if version is None or version == "":
                    self.unknown_version = True
                else:
                    self.versions.add(str(version))
        if self.observation is not None:
            _publish_source(results, self.observation)
        self.close_if_finished()

    @best_effort
    def close_if_finished(self) -> None:
        if self.sealed and not self.pending and self.observation is not None:
            if not self.unknown_version and len(self.versions) == 1:
                self.observation.set("rl.model.version", next(iter(self.versions)))
            self.observation.end(self.error, end_time=self.end_time)
            self.members.clear()
            self.owners.clear()


@contextmanager
def generation_batch(args: Any, *, evaluation: bool = False) -> Iterator[None]:
    batch = GenerationBatch(args, evaluation) if enabled(args) else None
    token = _GENERATION_BATCH.set(batch)
    try:
        yield
    finally:
        if batch is not None:
            batch.sealed = True
            batch.close_if_finished()
        _GENERATION_BATCH.reset(token)


def track_generation_task(task: asyncio.Task, samples: Any) -> asyncio.Task:
    batch = _GENERATION_BATCH.get()
    if batch is not None:
        batch.track(task, samples)
    return task


def seal_generation_batch() -> None:
    batch = _GENERATION_BATCH.get()
    if batch is not None:
        batch.sealed = True
        batch.close_if_finished()


def trace_generation_window(function: Callable) -> Callable:
    @wraps(function)
    async def call(args: Any, *rest: Any, **kwargs: Any) -> Any:
        with generation_batch(args, evaluation=True):
            return await function(args, *rest, **kwargs)

    return call


def trace_generation(function: Callable) -> Callable:
    @wraps(function)
    async def call(state: Any, args: Any, sample: Any, *rest: Any, **kwargs: Any) -> Any:
        if not enabled(args):
            return await function(state, args, sample, *rest, **kwargs)
        batch = _GENERATION_BATCH.get()
        if batch is not None:
            results = []
            error = None
            try:
                result = await function(state, args, sample, *rest, **kwargs)
                results = _samples(result)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                batch.finish(sample, results, error)
        # Standalone callers have a genuine singleton input operation.
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
            _record_generation_version(observation, _samples(result))
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
        batch = _REWARD_BATCH.get()
        if batch is not None:
            batch.active += 1
            value, error = None, None
            try:
                value = await function(args, sample, **kwargs)
                return value
            except BaseException as exc:
                error = exc
                raise
            finally:
                batch.record(sample, value, error)
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


@best_effort
def record_reward_route(sample: Any, name: str, kind: str) -> None:
    batch = _REWARD_BATCH.get()
    if batch is not None:
        batch.routes[id(sample)] = (name, kind)
        return
    observation = current_stage()
    if observation is not None and observation.span is not None and observation.span.name == "rl.reward.score":
        observation.set("rl.reward.name", name)
        observation.set("rl.reward.type", kind)


class RewardBatch:
    """Keep fanout results even when gather returns an early failure."""

    def __init__(self, args: Any, samples: list[Any], evaluation: bool) -> None:
        self.args = args
        self.samples = samples
        self.routes: dict[int, tuple[str, str]] = {}
        self.values: dict[int, Any] = {}
        self.failures: dict[int, str] = {}
        self.active = 0
        self.sealed = False
        self.error: BaseException | None = None
        self.end_time: int | None = None
        self.observation = Stage(
            args, "rl.reward.score", attributes=reward_attributes(args, evaluation), links=_sources(samples)
        )

    @best_effort
    def record(self, sample: Any, value: Any, error: BaseException | None) -> None:
        self.active -= 1
        self.end_time = time.time_ns()
        if error is not None:
            self.failures[id(sample)] = f"{type(error).__name__}: {error}"
        else:
            self.values[id(sample)] = value
        self.close_if_finished()

    @best_effort
    def close(self, values: Any, error: BaseException | None) -> None:
        self.sealed = True
        self.error = error
        self.end_time = time.time_ns()
        if error is None:
            self.values.update((id(sample), value) for sample, value in zip(self.samples, values, strict=True))
        self.close_if_finished()

    @best_effort
    def close_if_finished(self) -> None:
        if not self.sealed or self.active:
            return
        kinds = set(self.routes.values())
        if len(kinds) == 1:
            name, kind = next(iter(kinds))
            self.observation.set("rl.reward.name", name)
            self.observation.set("rl.reward.type", kind)
        elif kinds:
            self.observation.set("rl.reward.name", "router")
            self.observation.set("rl.reward.type", "hybrid")
        results = []
        default_source = reward_attributes(self.args)["rl.reward.name"]
        for sample in self.samples:
            identity = id(sample)
            if identity not in self.values and identity not in self.failures:
                continue  # Never invent a value/status for an unfinished child.
            item = {
                "export_id": export_id(sample),
                "source": self.routes.get(identity, (default_source, ""))[0],
            }
            if identity in self.failures:
                item.update(status="error", reason=self.failures[identity])
            else:
                value = self.values[identity]
                item.update(status="valid" if value is not None else "invalid", value=value)
                if value is None:
                    item["reason"] = "scorer returned no reward"
                if isinstance(value, dict) and getattr(self.args, "reward_key", None):
                    item["scalar_key"] = self.args.reward_key
            results.append(item)
        self.observation.data("rl.reward.results", results)
        _publish_source(self.samples, self.observation)
        self.observation.end(self.error, end_time=self.end_time)


def trace_batched_reward(function: Callable) -> Callable:
    @wraps(function)
    async def call(args: Any, samples: list[Any], **kwargs: Any) -> Any:
        if not enabled(args) or not samples or _REWARD_BATCH.get() is not None:
            return await function(args, samples, **kwargs)
        batch = RewardBatch(args, samples, kwargs.get("evaluation", False))
        token = _REWARD_BATCH.set(batch)
        values, error = None, None
        try:
            values = await function(args, samples, **kwargs)
            return values
        except BaseException as exc:
            error = exc
            raise
        finally:
            _REWARD_BATCH.reset(token)
            batch.close(values, error)

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
            start_time=getattr(stream, "trace_admitted_at", None),
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
                result = await function(owner, stream)
                _record_generation_version(observation, getattr(result, "samples", []))
                return result
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

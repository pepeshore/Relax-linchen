# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Batch operations and their actual member calls or confirmed results."""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_sample_trace import admit_sample, sample_attributes
from relax.utils.rl_tracing import (
    Stage,
    current_rollout,
    current_stage,
    enabled,
    inference_context,
    iteration_context,
    model_name,
    stage,
)


_PURPOSE: ContextVar[str] = ContextVar("rl_purpose", default="training")
_GENERATION_BATCH: ContextVar[GenerationBatch | None] = ContextVar("rl_generation_batch", default=None)
_REWARD_BATCH: ContextVar[RewardBatch | None] = ContextVar("rl_reward_batch", default=None)

_RESIDENT_REWARD_PARENT: ContextVar[Callable[[], Stage | None] | None] = ContextVar(
    "rl_resident_reward_parent", default=None
)


@contextmanager
def resident_reward_parent(get_parent: Callable[[], Stage | None]) -> Iterator[None]:
    """Resolve a resident Group's owner when scoring actually starts after a
    resume."""
    token = _RESIDENT_REWARD_PARENT.set(get_parent)
    try:
        yield
    finally:
        _RESIDENT_REWARD_PARENT.reset(token)


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
        self.rollout = current_rollout()
        self.observation: Stage | None = self.rollout.span if self.rollout else None
        self.members: dict[int, Any] = {}
        self.pending: set[int] = set()
        self.owners: dict[int, asyncio.Task] = {}
        self.versions: set[str] = set()
        self.unknown_version = False
        self.error: BaseException | None = None
        self.end_time: int | None = None
        self.sealed = False
        self.pipeline_tasks: set[asyncio.Task] = set()
        self.sample_observations: dict[int, Any] = {}
        self.reward: RewardBatch | None = None

    def reward_batch(self) -> RewardBatch:
        if self.reward is None:
            self.reward = RewardBatch(self.args, [], self.evaluation, parent=self.observation)
        return self.reward

    def close_rewards(self) -> None:
        if self.sealed and not self.pipeline_tasks and self.reward is not None:
            self.reward.close(None, None)

    @best_effort
    def track(self, task: asyncio.Task, samples: Any) -> None:
        self.pipeline_tasks.add(task)
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
            self.observation.set("rl.backend.name", "sglang")
            if self.rollout is None:
                self.observation.set("rl.batch.size", len(self.members))
            multi_turn = getattr(self.args, "custom_generate_function_path", None) or any(
                getattr(sample, "generate_function_path", None) for sample in self.members.values()
            )
            if multi_turn or self.rollout is None:
                self.observation.set("rl.rollout.mode", "multi_turn" if multi_turn else "single_turn")
            for identity in identities:
                if identity not in self.sample_observations:
                    self.sample_observations[identity] = admit_sample(
                        self.args, self.members[identity], self.observation
                    )

        @best_effort
        def done(finished: asyncio.Task) -> None:
            remaining = [i for i in identities if i in self.pending and self.owners.get(i) is finished]
            if remaining:
                error = asyncio.CancelledError() if finished.cancelled() else finished.exception()
                for identity in remaining:
                    # Only a pre-dispatch failure/cancellation/abort reaches
                    # this fallback; successful generation finishes below.
                    self.finish(self.members[identity], [], error)
            self.pipeline_tasks.discard(finished)
            self.close_rewards()

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
        observation = self.sample_observations.pop(identity, None)
        if observation is not None:
            observation.finish(sample, results, error, end_time=self.end_time)
        self.close_if_finished()

    @best_effort
    def close_if_finished(self) -> None:
        self.close_rewards()
        if self.sealed and not self.pending and self.observation is not None:
            if self.rollout is None:
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
                detail = batch.sample_observations.get(id(sample))
                with inference_context(detail.span if detail is not None else None):
                    result = await function(state, args, sample, *rest, **kwargs)
                results = _samples(result)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                batch.finish(sample, results, error)
        # If a custom callback generates directly, retain its framework invocation.
        rollout = current_rollout()
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
        scope = nullcontext(rollout.span) if rollout else stage(args, "rl.rollout.generate", attributes=attrs)
        with scope as observation:
            detail = admit_sample(args, sample, observation)
            results, error = [], None
            try:
                with inference_context(detail.span if detail is not None else None):
                    result = await function(state, args, sample, *rest, **kwargs)
                results = _samples(result)
                if rollout is None:
                    observation.freeze_end()
                    _record_generation_version(observation, results)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                end_time = time.time_ns() if rollout else observation.freeze_end()
                if detail is not None:
                    detail.finish(sample, results, error, end_time=end_time)

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
        generation = _GENERATION_BATCH.get()
        own = batch is None and generation is None
        if batch is None:
            batch = (
                generation.reward_batch()
                if generation is not None
                else RewardBatch(args, [], kwargs.get("evaluation", False))
            )
        batch.admit(sample)
        batch.active += 1
        source = _sources([sample])
        attrs = {
            **batch.observation.child_attributes(),
            **sample_attributes(sample),
            "rl.export.id": export_id(sample),
            "rl.sample.timing": "operation",
        }
        value, error = None, None
        token = _REWARD_BATCH.set(batch)
        with stage(args, "rl.reward.score", parent=batch.observation, attributes=attrs, links=source) as observation:
            try:
                value = await function(args, sample, **kwargs)
                return value
            except BaseException as exc:
                error = exc
                raise
            finally:
                _REWARD_BATCH.reset(token)
                batch.record(sample, value, error, observation)
                if own:
                    batch.close(None, error)

    return call


@best_effort
def record_reward_route(sample: Any, name: str, kind: str) -> None:
    batch = _REWARD_BATCH.get()
    if batch is not None:
        batch.routes[id(sample)] = (name, kind)
    observation = current_stage()
    if observation is not None and observation.name == "rl.reward.score":
        observation.set("rl.reward.name", name)
        observation.set("rl.reward.type", kind)


class RewardBatch:
    """One actual scoring window, with independently timed member calls."""

    def __init__(self, args: Any, samples: list[Any], evaluation: bool, *, parent: Stage | None = None) -> None:
        self.args = args
        self.samples: list[Any] = []
        self.routes: dict[int, tuple[str, str]] = {}
        self.results: dict[int, dict[str, Any]] = {}
        self.active = 0
        self.sealed = False
        self.error: BaseException | None = None
        self.end_time: int | None = None
        if parent is None:
            resident_parent = _RESIDENT_REWARD_PARENT.get()
            if resident_parent is not None:
                parent = resident_parent()
            elif (rollout := current_rollout()) is not None:
                parent = rollout.span
        ownership = parent.child_attributes() if parent is not None else {}
        self.observation = Stage(
            args,
            "rl.reward.batch",
            parent=parent,
            attributes={
                **reward_attributes(args, evaluation),
                **{
                    key: ownership[key]
                    for key in ("rl.purpose", "rl.step.id", "rl.iteration.index")
                    if key in ownership
                },
                "rl.batch.id": uuid.uuid4().hex,
            },
        )
        for sample in samples:
            self.admit(sample)

    @best_effort
    def admit(self, sample: Any) -> None:
        if all(value is not sample for value in self.samples):
            self.samples.append(sample)
            self.observation.set("rl.batch.size", len(self.samples))
            for source in _sources([sample]):
                self.observation.add_link(source)

    @best_effort
    def record(
        self,
        sample: Any,
        value: Any,
        error: BaseException | None,
        observation: Stage | None = None,
        *,
        end_time: int | None = None,
    ) -> None:
        self.end_time = max(self.end_time or 0, time.time_ns() if end_time is None else end_time)
        source = self.routes.get(id(sample), (reward_attributes(self.args)["rl.reward.name"], ""))[0]
        row = {
            "export_id": export_id(sample),
            "source": source,
            **({"sample_index": str(sample.index)} if getattr(sample, "index", None) is not None else {}),
            **(
                {"sample_id": sample_attributes(sample)["rl.sample.id"]}
                if "rl.sample.id" in sample_attributes(sample)
                else {}
            ),
        }
        if error is not None:
            row.update(status="error", reason=f"{type(error).__name__}: {error}")
            self.error = self.error or error
        else:
            row.update(status="valid" if value is not None else "invalid", value=copy.deepcopy(value))
            if value is None:
                row["reason"] = "scorer returned no reward"
            if isinstance(value, dict) and getattr(self.args, "reward_key", None):
                row["scalar_key"] = self.args.reward_key
        self.results[id(sample)] = row
        if observation is not None:
            self.active -= 1
            observation.data("rl.reward.results", [row])
            observation.end(error, end_time=self.end_time)
            _publish_source([sample], observation)
        else:
            _publish_source([sample], self.observation)
        self.close_if_finished()

    @best_effort
    def close(self, values: Any, error: BaseException | None) -> None:
        self.sealed = True
        self.error = self.error or error
        if values is not None:
            for sample, value in zip(self.samples, values, strict=True):
                if id(sample) not in self.results:
                    self.record(sample, value, None)
        self.close_if_finished()

    @best_effort
    def close_if_finished(self) -> None:
        if not self.sealed or self.active or self.observation._ended:
            return
        kinds = set(self.routes.values())
        if len(kinds) == 1:
            name, kind = next(iter(kinds))
            self.observation.set("rl.reward.name", name)
            self.observation.set("rl.reward.type", kind)
        elif kinds:
            self.observation.set("rl.reward.name", "router")
            self.observation.set("rl.reward.type", "hybrid")
        self.observation.data(
            "rl.reward.results", [self.results[id(s)] for s in self.samples if id(s) in self.results]
        )
        self.observation.end(self.error, end_time=self.end_time)


@best_effort
def _record_batched_values(
    batch: RewardBatch, samples: list[Any], values: Any, error: BaseException | None, end_time: int
) -> None:
    if values is not None:
        if len(values) != len(samples):
            batch.observation.data(
                "relax.observation.partial", {"errors": ["scorer result count differs from input count"]}
            )
        for sample, value in zip(samples, values, strict=False):
            previous = batch.results.get(id(sample))
            if previous is None or previous.get("value") != value:
                if previous is not None:
                    route = reward_attributes(batch.args)
                    batch.routes[id(sample)] = (route["rl.reward.name"], route["rl.reward.type"])
                batch.record(sample, value, None, end_time=end_time)
    elif error is not None and batch.active == 1:
        for sample in samples:
            if id(sample) not in batch.results:
                batch.record(sample, None, error, end_time=end_time)


def trace_batched_reward(function: Callable) -> Callable:
    @wraps(function)
    async def call(args: Any, samples: list[Any], **kwargs: Any) -> Any:
        if not enabled(args) or not samples:
            return await function(args, samples, **kwargs)
        batch = _REWARD_BATCH.get()
        generation = _GENERATION_BATCH.get()
        own = batch is None and generation is None
        if batch is None:
            batch = (
                generation.reward_batch()
                if generation is not None
                else RewardBatch(args, [], kwargs.get("evaluation", False))
            )
        for sample in samples:
            batch.admit(sample)
        batch.active += 1
        token = _REWARD_BATCH.set(batch)
        values, error = None, None
        try:
            values = await function(args, samples, **kwargs)
            return values
        except BaseException as exc:
            error = exc
            raise
        finally:
            end_time = time.time_ns()
            custom = getattr(args, "custom_rm_path", None) and not kwargs.get("ignore_custom", False)
            # Built-in fanout ends with its last child, not after child export.
            # Custom scorers may do real batch work after those child calls.
            if custom or not batch.results:
                batch.end_time = max(batch.end_time or 0, end_time)
            _REWARD_BATCH.reset(token)
            _record_batched_values(batch, samples, values, error, end_time)
            batch.active -= 1
            if own:
                batch.close(None, error)
            else:
                batch.error = batch.error or error
                batch.close_if_finished()

    return call


@best_effort
def _record_transfer_results(observation: Stage, samples: list[Any], data: dict[str, Any]) -> None:
    data["members"] = [
        {
            "export_id": export_id(sample),
            **({"sample_index": str(sample.index)} if getattr(sample, "index", None) is not None else {}),
        }
        for sample in samples
    ]
    observation.data("rl.transfer.data", data)
    _publish_source(samples, observation)


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
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": len(samples),
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
                observation.freeze_end()
                _record_transfer_results(observation, samples, data)

    return call


def callable_name(function: Any) -> str:
    function = getattr(function, "func", function)
    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


@best_effort
def _record_filter_results(observation: Stage, samples: list[Any], result: Any) -> None:
    rows = [
        {
            "export_id": export_id(sample),
            "decision": "keep" if result.keep else "drop",
            **({"reason": result.reason or "filter rejected sample"} if not result.keep else {}),
            **({"sample_index": str(sample.index)} if getattr(sample, "index", None) is not None else {}),
        }
        for sample in samples
    ]
    observation.data("rl.filter.results", rows)
    _publish_source(samples, observation)


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
            attributes={
                "rl.filter.name": callable_name(fn),
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": len(samples),
            },
            links=_sources(samples),
        ) as observation:
            result = function(fn, *args, **kwargs)
            observation.freeze_end()
            _record_filter_results(observation, samples, result)
            return result

    return call


def _main_rank() -> bool:
    from relax.backends.megatron.initialize import is_megatron_main_rank

    return is_megatron_main_rank()


@best_effort
def _record_forward_members(observation: Stage, data: Any) -> None:
    from relax.utils.rl_train_trace import _cpu_list

    key = "export_id" if data.get("sample_export_ids") is not None else "sample_index"
    values = data.get("sample_export_ids") if key == "export_id" else data.get("sample_indices")
    if values is not None:
        rows = [{key: str(value), "status": "valid"} for value in _cpu_list(values) if value >= 0]
        observation.set("rl.batch.size", len(rows))
        observation.data("rl.forward.results", rows)


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
        from relax.utils.rl_tracing import iteration_attributes
        from relax.utils.rl_train_trace import _WINDOWS, _owner, window

        data = getattr(data_iterator[0], "rollout_data", {})
        name = f"rl.{role}.forward"
        if _WINDOWS.get() is not None:
            with window(
                owner.args,
                role,
                iteration_attributes().get("rl.iteration.index"),
                data=data,
                name=name,
                attributes=attrs,
            ):
                return function(
                    owner, data_iterator, num_microbatches, store_prefix=store_prefix, collect_topk=collect_topk
                )
        from megatron.core import mpu

        active = _owner() and mpu.get_context_parallel_rank() == 0
        attrs.update(
            {
                "rl.batch.id": uuid.uuid4().hex,
                "rl.partition.dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
            }
        )
        with stage(owner.args, name, attributes=attrs, active=active) as observation:
            result = function(
                owner, data_iterator, num_microbatches, store_prefix=store_prefix, collect_topk=collect_topk
            )
            observation.freeze_end()
            if active:
                _record_forward_members(observation, data)
            return result

    return call


def trace_advantage(function: Callable) -> Callable:
    @wraps(function)
    def call(args: Any, rollout_data: Any, **kwargs: Any) -> Any:
        from megatron.core import mpu

        from relax.utils.rl_tracing import iteration_attributes
        from relax.utils.rl_train_trace import window

        iteration = iteration_attributes().get("rl.iteration.index")
        # Match compute_advantages_and_returns: other PP stages return without
        # computing, even when their input contains rollout_log_probs.
        if not mpu.is_pipeline_last_stage() or all(
            rollout_data.get(key) is None for key in ("log_probs", "rollout_log_probs", "values")
        ):
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
    """Compatibility decorator; the SessionShard now owns the full batch."""
    return function


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
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": len(samples),
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
                observation.freeze_end()
                _record_transfer_results(observation, samples, data)

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
    observation.set("rl.batch.id", uuid.uuid4().hex)
    observation.set("rl.batch.size", len(rows))
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
            observation.freeze_end()
            _record_custom_advantage(observation, group, result)
            return result

    return call


def trace_opd_forward(role: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        async def call(owner: Any, sample: Any, *args: Any, **kwargs: Any) -> Any:
            if not enabled(owner.args) or not sample.response_length:
                return await function(owner, sample, *args, **kwargs)
            attrs = {
                "rl.model.name": model_name(owner.args, role),
                "rl.backend.name": "sglang",
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": 1,
                **sample_attributes(sample),
                "rl.export.id": export_id(sample),
            }
            if role == "teacher":
                attrs["rl.forward.output"] = (
                    "topk_distribution" if owner.topk_worker is not None else "sampled_token_logprob"
                )
            else:
                attrs["rl.forward.purpose"] = "distillation"
            with stage(owner.args, f"rl.{role}.forward", attributes=attrs, links=_sources([sample])) as observation:
                result = await function(owner, sample, *args, **kwargs)
                observation.freeze_end()
                row = {"export_id": export_id(sample), "status": "invalid" if result is False else "valid"}
                observation.data("rl.forward.results", [row])
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
    identity_key = "export_id" if "sample_export_ids" in data else "sample_index"
    identities = data["sample_export_ids"] if identity_key == "export_id" else data["sample_indices"]
    for identity, advantage, mask in zip(identities, result["advantages"], data["loss_masks"], strict=True):
        tensors = [value for value in (identity, advantage, mask) if isinstance(value, torch.Tensor)]
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("Standalone advantage observation requires CPU tensors")
        count = int((mask != 0).sum())
        row = {identity_key: str(int(identity)), "token_count": count}
        if count:
            mean = float((torch.where(mask != 0, advantage, 0.0) * mask).sum() / mask.sum())
            if math.isfinite(mean):
                row["mean"] = mean
        rows.append(row)
    observation.set("rl.batch.id", uuid.uuid4().hex)
    observation.set("rl.batch.size", len(rows))
    observation.data("rl.advantage.results", rows)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Trace a distributed training request at dispatch and at each worker RPC.

Rank-local stages retain their native timing. Accelerator results are copied on
the producer stream and read after the operation.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator

from relax.utils.agentic_obs.writers import BoundedWorker, best_effort
from relax.utils.logging_utils import get_logger
from relax.utils.rl_tracing import Stage, enabled, iteration_attributes, model_name, stage


logger = get_logger(__name__)


@dataclass
class _RankRequest:
    span: Stage
    global_rank: int
    pending: list[tuple[Stage, dict, BaseException | None]] = field(default_factory=list)


_REQUEST: ContextVar[_RankRequest | None] = ContextVar("rl_rank_request", default=None)
_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar("rl_train_window", default=None)
_UPDATE: ContextVar[str | None] = ContextVar("rl_update_id", default=None)
_READER: BoundedWorker | None = None


def _reader() -> BoundedWorker:
    global _READER
    if _READER is None:
        _READER = BoundedWorker(capacity=4096)
    return _READER


def _owner() -> bool:
    from megatron.core import mpu

    return mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage(ignore_virtual=True)


def _cpu_list(value: Any) -> list:
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("RL observation identities must remain on CPU")
        return value.tolist()
    result = []
    for item in value:
        if isinstance(item, torch.Tensor) and item.device.type != "cpu":
            raise ValueError("RL observation identities must remain on CPU")
        result.append(int(item))
    return result


def dispatch_training(group: Any, method: str, rollout_id: int) -> list:
    """Fan out one request, preserving the existing nonblocking ObjectRef
    API."""
    workers = group._actor_handlers
    if getattr(group.args, "train_backend", "megatron") != "megatron" or not enabled(group.args):
        return [getattr(worker, method).remote(rollout_id) for worker in workers]

    parent = Stage(
        group.args,
        f"rl.{group.role}.train_step",
        producer_role=f"{group.role}_service",
        iteration=rollout_id,
        attributes={
            "rl.execution.side": "training",
            "rl.backend.name": "megatron",
            "rl.rank.world_size": len(workers),
        },
    )
    headers = parent.headers()
    kwargs = {"trace_headers": headers} if headers else {}
    try:
        refs = [getattr(worker, method).remote(rollout_id, **kwargs) for worker in workers]
    except BaseException as error:
        parent.end(error)
        raise

    # RayTrainGroup also has fire-and-forget callers. Observe acknowledgements
    # without blocking them, reading training results or changing the returned refs.
    remaining = len(refs)
    failure = None
    lock = threading.Lock()

    def acknowledge(future: Future) -> None:
        nonlocal remaining, failure
        error = CancelledError() if future.cancelled() else future.exception()
        with lock:
            if error is not None and failure is None:
                failure = error
            remaining -= 1
            if remaining == 0:
                parent.end(failure)

    try:
        if not refs:
            parent.end()
        for ref in refs:
            ref.future().add_done_callback(acknowledge)
    except Exception as error:
        logger.warning("RL OTel could not observe training acknowledgements: %s", error)
        parent.end(error)
    return refs


def current_rank_span() -> Stage | None:
    request = _REQUEST.get()
    return request.span if request is not None else None


def current_window_span() -> Stage | None:
    """The rank-local training phase, scoped to its actual execution."""
    window = _CURRENT.get()
    return window["span"] if window is not None else None


def training_rpc(function: Callable) -> Callable:
    @wraps(function)
    def call(owner: Any, rollout_id: int, *args: Any, trace_headers: dict | None = None, **kwargs: Any) -> Any:
        if not trace_headers or not enabled(owner.args):
            return function(owner, rollout_id, *args, **kwargs)

        attrs = {
            "rl.execution.side": "training",
            "rl.backend.name": "megatron",
            "rl.timing.scope": "rank_local",
            "rl.rank.global": owner._rank,
            "rl.rank.world_size": owner._world_size,
        }
        with stage(
            owner.args,
            f"rl.{owner.role}.train_rank",
            iteration=rollout_id,
            attributes=attrs,
            parent_headers=trace_headers,
        ) as rank:
            request = _RankRequest(rank, owner._rank)
            token = _REQUEST.set(request)
            try:
                return function(owner, rollout_id, *args, **kwargs)
            finally:
                rank.freeze_end()
                _REQUEST.reset(token)
                deadline = time.monotonic() + 5.0
                for operation, local, failure in request.pending:
                    _finish_local(operation, local, deadline, failure)

    return call


@best_effort
def mark_incomplete(reason: str) -> None:
    current = _CURRENT.get()
    if current is not None:
        current.setdefault("capture_errors", []).append(reason)


@contextmanager
def window(
    args: Any,
    role: str,
    rollout_id: int,
    *,
    data: Any = None,
    estimator: str | None = None,
    name: str | None = None,
    attributes: dict | None = None,
) -> Iterator[None]:
    request = _REQUEST.get()
    if request is None:
        yield
        return
    from megatron.core import mpu

    name = name or ("rl.advantage.compute" if role == "advantage" else f"rl.{role}.train")
    coordinates = {
        "rl.timing.rank.dp": mpu.get_data_parallel_rank(with_context_parallel=False),
        "rl.timing.rank.cp": mpu.get_context_parallel_rank(),
        "rl.timing.rank.tp": mpu.get_tensor_model_parallel_rank(),
        "rl.timing.rank.pp": mpu.get_pipeline_model_parallel_rank(),
    }
    for key, value in coordinates.items():
        request.span.set(key, value)
    attrs = {
        **iteration_attributes(rollout_id),
        **(attributes or {}),
        **coordinates,
        "rl.timing.scope": "rank_local",
        "rl.execution.side": "training",
        "rl.backend.name": "megatron",
        "rl.batch.id": uuid.uuid4().hex,
        "rl.rank.global": request.global_rank,
    }
    if role == "advantage":
        attrs["rl.advantage.estimator"] = estimator
    else:
        attrs["rl.model.name"] = model_name(args, role)
        if role == "actor" and name.endswith(".train"):
            attrs["rl.algorithm.name"] = getattr(args, "algorithm", None) or args.advantage_estimator
    local = {"name": name, "updates": [], "futures": [], "owner": _owner()}
    if data is not None and local["owner"]:
        _collect_members(data, local)
    operation = Stage(args, name, parent=request.span, attributes=attrs)
    local["span"] = operation
    token = _CURRENT.set(local)
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        operation.freeze_end()
        _CURRENT.reset(token)
        if local["futures"]:
            request.pending.append((operation, local, failure))
        else:
            _finish_local(operation, local, time.monotonic(), failure)


@best_effort
def _collect_members(data: Any, local: dict) -> None:
    for key, kind in (("sample_export_ids", "export"), ("sample_indices", "sample")):
        if data.get(key) is not None:
            local["members"] = [f"{kind}:{i}" for i in _cpu_list(data[key]) if i >= 0]
            break


@contextmanager
def update(args: Any, rollout_id: int, step_id: int, role: str) -> Iterator[None]:
    identity = f"{getattr(args, 'agentic_obs_train_generation_id', 'generation')}:{role}:{rollout_id}:{step_id}"
    token = _UPDATE.set(identity)
    try:
        yield
    except BaseException as error:
        current = _CURRENT.get()
        if current is not None and not any(row["update_id"] == identity for row in current["updates"]):
            current["updates"].append({"update_id": identity, "status": "failed", "reason": str(error)})
        raise
    finally:
        _UPDATE.reset(token)


@best_effort
def optimizer_result(successful: bool, reason: str | None = None) -> None:
    current = _CURRENT.get()
    identity = _UPDATE.get()
    if current is None or identity is None:
        return
    item = {"update_id": identity, "status": "committed" if successful else "skipped"}
    if reason is not None:
        item["reason"] = reason
    current["updates"].append(item)


@best_effort
def prepare_batch(batch: Any) -> None:
    """Retain CPU identities before get_batch moves ordinary tensors to GPU."""
    current = _CURRENT.get()
    if current is None or not current["owner"] or batch.get("__is_dummy__", False):
        return
    try:
        batch["_rl_observation_ids"] = {
            key: _cpu_list(batch[key]) for key in ("sample_export_ids", "sample_indices") if batch.get(key) is not None
        }
    except Exception as error:
        mark_incomplete(str(error))
        logger.warning("RL OTel input identity capture failed: %s", error)


@best_effort
def capture(args: Any, batch: Any, masks: Any = None, *, local_gate: Any = None) -> None:
    current = _CURRENT.get()
    if current is None or not current["owner"] or batch.get("__is_dummy__", False):
        return
    identity_data = batch.get("_rl_observation_ids", batch)
    ids = identity_data.get("sample_export_ids")
    member_ids = ids if ids is not None else identity_data.get("sample_indices")
    try:
        if member_ids is not None:
            prefix = "export" if ids is not None else "sample"
            current.setdefault("members", []).extend(
                f"{prefix}:{value}" for value in _cpu_list(member_ids) if value >= 0
            )
        identity_key = "export_id" if ids is not None else "sample_index"
        ids = member_ids
        if ids is None:
            return
        import torch

        from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

        identities = _cpu_list(ids)
        masks = batch["loss_masks"] if masks is None else masks
        advantages = batch.get("advantages")
        rows, selected = [], []
        gate_offset = 0
        for index, (identity, mask) in enumerate(zip(identities, masks, strict=True)):
            if identity < 0:
                raise ValueError("Missing Export ID at actual training consumption")
            local_mask = slice_log_prob_with_cp(
                mask.detach().float(),
                int(batch["total_lengths"][index]),
                int(batch["response_lengths"][index]),
                args.qkv_format,
                batch["max_seq_lens"][index] if batch.get("max_seq_lens") is not None else None,
                batch["padded_total_lengths"][index] if batch.get("padded_total_lengths") is not None else None,
                dynamic_cp_size=batch.get("dynamic_cp_size"),
                dynamic_cp_rank=batch.get("dynamic_cp_rank"),
            )
            if local_gate is not None:
                local_mask = local_mask * local_gate[gate_offset : gate_offset + local_mask.numel()].detach()
                gate_offset += local_mask.numel()
            count = (local_mask != 0).sum().float()
            numerator = count.new_zeros(())
            advantage = advantages[index].detach() if advantages is not None else None
            if advantage is not None and advantage.numel() == mask.numel() and advantage.numel() != local_mask.numel():
                advantage = slice_log_prob_with_cp(
                    advantage,
                    int(batch["total_lengths"][index]),
                    int(batch["response_lengths"][index]),
                    args.qkv_format,
                    batch["max_seq_lens"][index] if batch.get("max_seq_lens") is not None else None,
                    batch["padded_total_lengths"][index] if batch.get("padded_total_lengths") is not None else None,
                    dynamic_cp_size=batch.get("dynamic_cp_size"),
                    dynamic_cp_rank=batch.get("dynamic_cp_rank"),
                )
            known_advantage = advantage is not None and advantage.numel() == local_mask.numel()
            if known_advantage:
                numerator = (torch.where(local_mask != 0, advantage, 0.0) * local_mask).sum()
            rows.append(torch.stack((count, numerator, local_mask.sum())))
            selected.append((str(identity), known_advantage))
        if not rows:
            return
        source = torch.stack(rows).detach()
        event = None
        if source.is_cuda:
            values = torch.empty_like(source, device="cpu", pin_memory=True)
            values.copy_(source, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(source.device))
        else:
            values = source.clone()
        future = Future()

        def read_snapshot() -> None:
            try:
                future.set_result(_read(source, values, event, selected, update_id, identity_key))
            except Exception as error:
                future.set_exception(error)

        update_id = _UPDATE.get()
        if not _reader().submit(read_snapshot, size_bytes=source.numel() * source.element_size() * 2):
            future.set_exception(RuntimeError("observation reader queue is full"))
        current["futures"].append(future)
    except Exception as error:
        current.setdefault("capture_errors", []).append(str(error))
        logger.warning("RL OTel sample capture failed: %s", error)


def _read(
    source: Any, values: Any, event: Any, identities: list, update_id: str | None, identity_key: str = "export_id"
) -> list[dict[str, Any]]:
    if event is not None:
        event.synchronize()
    del source
    result = []
    for (identity, known), (count, numerator, denominator) in zip(identities, values.tolist(), strict=True):
        if not math.isfinite(numerator):
            known, numerator = False, 0.0
        row = {
            identity_key: identity,
            "effective_tokens": int(count),
            "advantage_numerator": numerator,
            "advantage_denominator": denominator,
            "advantage_known": known,
        }
        if update_id is not None:
            row.update(update_id=update_id, used=True)
        result.append(row)
    return result


def _finish_local(operation: Stage, local: dict, deadline: float, failure: BaseException | None) -> None:
    try:
        _record_local_results(operation, local, deadline, failure)
    finally:
        operation.end(failure)


@best_effort
def _record_local_results(operation: Stage, local: dict, deadline: float, failure: BaseException | None) -> None:
    records = []
    for future in local["futures"]:
        try:
            records.extend(future.result(timeout=max(0.0, deadline - time.monotonic())))
        except Exception as error:
            local.setdefault("capture_errors", []).append(f"{type(error).__name__}: {error}")

    members = set(local["members"]) if "members" in local else None
    if members is not None:
        operation.set("rl.batch.size", len(members))
    if failure is not None:
        local.setdefault("capture_errors", []).append(f"{type(failure).__name__}: {failure}")
    if local.get("capture_errors"):
        operation.data(
            "relax.observation.partial",
            {
                "errors": local["capture_errors"],
                "records": records,
                "updates": local["updates"],
            },
        )
        return
    name = local["name"]
    if name.endswith(".forward"):
        if members is not None:
            rows = []
            for member in sorted(members):
                kind, identity = member.split(":", 1)
                rows.append({"export_id" if kind == "export" else "sample_index": identity, "status": "valid"})
            operation.data("rl.forward.results", rows)
        return
    # Multiple local microbatches can consume the same sample during an update.
    fragments = defaultdict(list)
    for record in records:
        key = "export_id" if "export_id" in record else "sample_index"
        fragments[(key, record[key], record.get("update_id"))].append(record)
    rows = []
    for (key, identity, update_id), items in fragments.items():
        row = {key: identity}
        row["effective_tokens" if update_id else "token_count"] = sum(item["effective_tokens"] for item in items)
        if update_id is not None:
            row.update(update_id=update_id, used=True)
        denominator = sum(item["advantage_denominator"] for item in items)
        if denominator and all(item["advantage_known"] for item in items):
            row["advantage_mean" if update_id else "mean"] = (
                sum(item["advantage_numerator"] for item in items) / denominator
            )
        rows.append(row)
    if name.endswith(".train"):
        result = {"updates": local["updates"]}
        if local["owner"] and (members is not None or rows):
            result["uses"] = rows
        if result["updates"] or "uses" in result:
            operation.data("rl.train.results", result)
    elif name == "rl.advantage.compute" and (members is not None or rows):
        operation.data("rl.advantage.results", rows)

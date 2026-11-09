# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Collect rank facts through the existing training RPC return path.

Only the coordinator emits a logical stage. GPU summaries are copied on the
producer stream and read by a background thread; no additional collective or
trace-context transport is introduced. Delayed export retains native timings.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator

from relax.utils.agentic_obs.writers import BoundedWorker, best_effort
from relax.utils.logging_utils import get_logger
from relax.utils.rl_tracing import Stage, enabled, iteration_attributes, model_name


logger = get_logger(__name__)
_WINDOWS: ContextVar[list[dict[str, Any]] | None] = ContextVar("rl_windows", default=None)
_CURRENT: ContextVar[dict[str, Any] | None] = ContextVar("rl_train_window", default=None)
_UPDATE: ContextVar[str | None] = ContextVar("rl_update_id", default=None)
_STREAM_START: ContextVar[int | None] = ContextVar("rl_stream_start", default=None)
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


def training_rpc(function: Callable) -> Callable:
    @wraps(function)
    def call(owner: Any, rollout_id: int, *args: Any, **kwargs: Any) -> Any:
        if not enabled(owner.args):
            return function(owner, rollout_id, *args, **kwargs)
        windows: list[dict[str, Any]] = []
        token = _WINDOWS.set(windows)
        stream_token = _STREAM_START.set(time.time_ns() if function.__name__ == "train_async" else None)
        try:
            result = function(owner, rollout_id, *args, **kwargs)
            payload = _finish_windows(windows)
            return {"rl_trace_windows": payload, "rl_trace_producer": _producer(windows)} if result is None else result
        except BaseException:
            # Preserve completed training when a later checkpoint/eval fails.
            producer = _producer(windows)
            if producer and producer["owner"] and producer["rank"]["dp"] == producer["rank"]["cp"] == 0:
                completed = [w for w in windows if "end_time" in w and "error" not in w]
                for item in _finish_windows(completed):
                    observation = Stage(
                        owner.args, item["name"], attributes=item["attributes"], start_time=item["start_time"]
                    )
                    observation.data("relax.observation.partial", item)
                    observation.end(end_time=item["end_time"])
            raise
        finally:
            _STREAM_START.reset(stream_token)
            _WINDOWS.reset(token)

    return call


def _producer(windows: list[dict[str, Any]]) -> dict[str, Any]:
    # Colocate sleep can destroy NCCL groups before the RPC returns. Preserve
    # the ranks captured inside the operation while those groups were live.
    if windows:
        return {"owner": windows[0]["owner"], "rank": windows[0]["rank"]}
    return {"owner": False, "rank": {}}


def _finish_windows(windows: list[dict]) -> list[dict]:
    deadline = time.monotonic() + 5.0
    for item in windows:
        records = []
        for future in item.pop("futures", []):
            try:
                records.extend(future.result(timeout=max(0.0, deadline - time.monotonic())))
            except Exception as error:
                item.setdefault("capture_errors", []).append(f"{type(error).__name__}: {error}")
        item["records"] = records
    return windows


@best_effort
def mark_incomplete(reason: str) -> None:
    current = _CURRENT.get()
    if current is not None:
        current.setdefault("capture_errors", []).append(reason)


@contextmanager
def window(args: Any, role: str, rollout_id: int, *, data: Any = None, estimator: str | None = None) -> Iterator[None]:
    windows = _WINDOWS.get()
    if windows is None or not enabled(args):
        yield
        return
    from megatron.core import mpu

    name = "rl.advantage.compute" if role == "advantage" else f"rl.{role}.train"
    attrs = iteration_attributes(rollout_id)
    if role == "advantage":
        attrs["rl.advantage.estimator"] = estimator
    else:
        attrs.update({"rl.model.name": model_name(args, role), "rl.backend.name": "megatron"})
        if role == "actor":
            attrs["rl.algorithm.name"] = getattr(args, "algorithm", None) or args.advantage_estimator
    observation = {
        "name": name,
        "sequence": sum(w["name"] == name for w in windows),
        "attributes": attrs,
        "start_time": (_STREAM_START.get() if role == "actor" else None) or time.time_ns(),
        "updates": [],
        "futures": [],
        "owner": _owner(),
        "rank": {
            "dp": mpu.get_data_parallel_rank(with_context_parallel=False),
            "cp": mpu.get_context_parallel_rank(),
            "tp": mpu.get_tensor_model_parallel_rank(),
            "pp": mpu.get_pipeline_model_parallel_rank(),
        },
    }
    if data is not None and _owner():
        ids = data.get("sample_export_ids")
        if ids is not None:
            observation["members"] = [f"export:{value}" for value in _cpu_list(ids) if value >= 0]
        elif data.get("sample_indices") is not None:
            observation["members"] = [f"sample:{value}" for value in _cpu_list(data["sample_indices"]) if value >= 0]
    windows.append(observation)
    token = _CURRENT.set(observation)
    try:
        yield
    except BaseException as error:
        observation["error"] = f"{type(error).__name__}: {error}"
        # An exceptional RPC cannot return its facts. Retain the actual failing
        # operation locally, rather than losing the error while Ray unwinds.
        if observation["owner"] and observation["rank"]["dp"] == observation["rank"]["cp"] == 0:
            failed = Stage(args, name, attributes=attrs, start_time=observation["start_time"])
            failed.end(error)
        raise
    finally:
        observation["end_time"] = time.time_ns()
        _CURRENT.reset(token)


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
                future.set_result(_read(source, values, event, selected, update_id))
            except Exception as error:
                future.set_exception(error)

        update_id = _UPDATE.get()
        if not _reader().submit(read_snapshot, size_bytes=source.numel() * source.element_size() * 2):
            future.set_exception(RuntimeError("observation reader queue is full"))
        current["futures"].append(future)
    except Exception as error:
        current.setdefault("capture_errors", []).append(str(error))
        logger.warning("RL OTel sample capture failed: %s", error)


def _read(source: Any, values: Any, event: Any, identities: list, update_id: str | None) -> list[dict[str, Any]]:
    if event is not None:
        event.synchronize()
    del source
    result = []
    for (identity, known), (count, numerator, denominator) in zip(identities, values.tolist(), strict=True):
        if not math.isfinite(numerator):
            known, numerator = False, 0.0
        row = {
            "export_id": identity,
            "effective_tokens": int(count),
            "advantage_numerator": numerator,
            "advantage_denominator": denominator,
            "advantage_known": known,
        }
        if update_id is not None:
            row.update(update_id=update_id, used=True)
        result.append(row)
    return result


@best_effort
def emit_windows(args: Any, reports: list[Any]) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for report in reports:
        if not isinstance(report, dict):
            continue
        for item in report.get("rl_trace_windows", []):
            grouped[(item["name"], item["sequence"])].append(item)
    for (name, _), windows in grouped.items():
        attrs = dict(windows[0]["attributes"])
        owners = [item for item in windows if item["owner"]]
        if name.endswith(".train"):
            attrs["rl.batch.id"] = uuid.uuid4().hex
            if owners and all("members" in item for item in owners):
                attrs["rl.batch.size"] = len({identity for item in owners for identity in item["members"]})
        observation = Stage(args, name, attributes=attrs, start_time=min(w["start_time"] for w in windows))
        errors = [error for item in owners for error in item.get("capture_errors", [])]
        expected = {
            tuple(sorted(report["rl_trace_producer"]["rank"].items()))
            for report in reports
            if isinstance(report, dict) and report.get("rl_trace_producer", {}).get("owner")
        }
        actual = {tuple(sorted(item["rank"].items())) for item in owners}
        if expected != actual or any(
            not isinstance(report, dict) or "rl_trace_producer" not in report for report in reports
        ):
            errors.append("missing observation owner")
        if getattr(args, "use_agentic_rollout", False) and any("members" not in item for item in owners):
            errors.append("missing Export membership")
        records: dict[tuple, list[dict]] = defaultdict(list)
        for item in owners:
            for row in item.get("records", []):
                records[(row["export_id"], row.get("update_id"))].append(row)
        uses = []
        for (identity, update_id), rows in records.items():
            count = sum(row["effective_tokens"] for row in rows)
            value = {"export_id": identity, "effective_tokens" if update_id else "token_count": count}
            if update_id is not None:
                value.update(update_id=update_id, used=True)
            denominator = sum(row["advantage_denominator"] for row in rows)
            if denominator and all(row["advantage_known"] for row in rows):
                value["advantage_mean" if update_id else "mean"] = (
                    sum(row["advantage_numerator"] for row in rows) / denominator
                )
            uses.append(value)
        if name.endswith(".train"):
            updates = {}
            for item in windows:
                for row in item["updates"]:
                    previous = updates.setdefault(row["update_id"], row)
                    if previous != row:
                        errors.append(f"Conflicting update results: {row['update_id']}")
            result = {"updates": list(updates.values())}
            if getattr(args, "use_agentic_rollout", False):
                result["uses"] = uses
            if not errors:
                observation.data("rl.train.results", result)
        elif getattr(args, "use_agentic_rollout", False) and not errors:
            observation.data("rl.advantage.results", uses)
        if errors:
            observation.data("relax.observation.partial", {"errors": errors, "fragments": windows})
            logger.warning("RL OTel incomplete structured data: %s", errors)
        observation.end(end_time=max(w["end_time"] for w in windows))

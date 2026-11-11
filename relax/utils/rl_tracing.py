# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RL semantic conventions 0.19.0, with process-local OTLP JSON files.

A framework rollout owns its samples and the scoring, filtering, queue writes,
custom advantages and OPD forwards it initiates. Links identify input sources.
Business identities cross Ray boundaries in the existing args Namespace; SDK
objects never do. No global tracer provider is replaced, so SGLang requests
explicitly inherit their responsible RL span via W3C headers.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import json
import os
import secrets
import socket
import threading
import time
import uuid
from argparse import Namespace
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_LOCK = threading.RLock()
_RUNTIMES: dict[tuple[int, str, str], Any] = {}
_ITERATION: ContextVar[int | None] = ContextVar("rl_iteration", default=None)
_CURRENT: ContextVar[Stage | None] = ContextVar("rl_stage", default=None)
_INFERENCE_PARENT: ContextVar[tuple[Stage | None, dict[str, str] | None]] = ContextVar(
    "rl_inference_parent", default=(None, None)
)
_ROLLOUT_CALL: ContextVar[RolloutCall | None] = ContextVar("rl_rollout_call", default=None)
_ENV_NAMES = ("RELAX_OTEL_ENABLED", "RELAX_OTEL_DIR", "RELAX_OTEL_RUN_ID", "OTEL_SDK_DISABLED")


# Ray workers keep their role across threads and asyncio tasks. A fork must
# register its own producer rather than inheriting the parent process's identity.
_PROCESS_ROLE: tuple[int, str] | None = None


def set_process_role(role: str) -> None:
    global _PROCESS_ROLE
    _PROCESS_ROLE = (os.getpid(), role)


def process_role() -> str:
    if _PROCESS_ROLE is not None and _PROCESS_ROLE[0] == os.getpid():
        return _PROCESS_ROLE[1]
    return "framework"


def enabled(args: Namespace | None = None) -> bool:
    """Enable launched runs by default, without activating during imports."""
    return (
        Envs.RELAX_OTEL_ENABLED
        and os.environ.get("OTEL_SDK_DISABLED", "false").lower() != "true"
        and bool(getattr(args, "rl_trace_run_id", None) or os.environ.get("RELAX_OTEL_RUN_ID"))
    )


def initialize_run(args: Namespace) -> None:
    """Stamp each actual driver launch, including checkpoint restarts."""
    args.rl_trace_run_id = uuid.uuid4().hex
    args.rl_trace_dir = str(
        Path(os.environ.get("RELAX_OTEL_DIR") or Path(getattr(args, "save", None) or Path.cwd()) / "otel-traces")
        .expanduser()
        .resolve()
    )
    os.environ["RELAX_OTEL_RUN_ID"] = args.rl_trace_run_id
    os.environ["RELAX_OTEL_DIR"] = args.rl_trace_dir
    if enabled(args) and getattr(args, "use_agentic_rollout", False):
        if not getattr(args, "dump_agentic_obs", None):
            args.dump_agentic_obs = str(Path(args.rl_trace_dir) / "agentic")
        args.agentic_obs_run_id = args.rl_trace_run_id
    if enabled(args):
        logger.info("RL OTel traces: %s/%s", args.rl_trace_dir, args.rl_trace_run_id)


def runtime_env() -> dict[str, str]:
    return {key: os.environ[key] for key in _ENV_NAMES if key in os.environ}


def run_id(args: Namespace | None = None) -> str:
    return str(getattr(args, "rl_trace_run_id", None) or os.environ.get("RELAX_OTEL_RUN_ID", ""))


def model_name(args: Namespace, role: str = "actor") -> str | None:
    keys = {
        "reference": ("ref_load", "hf_checkpoint"),
        "teacher": ("teacher_hf_checkpoint", "opd_teacher_load", "teacher_load", "teacher_model_path"),
        "critic": ("critic_load", "hf_checkpoint"),
    }.get(role, ("hf_checkpoint", "model_name"))
    for key in keys:
        value = getattr(args, key, None)
        if value:
            return Path(str(value).rstrip("/")).name
    return None


def iteration_attributes(iteration: int | None = None) -> dict[str, Any]:
    iteration = _ITERATION.get() if iteration is None else iteration
    return {} if iteration is None else {"rl.step.id": f"iteration:{iteration}", "rl.iteration.index": iteration}


@contextmanager
def iteration_context(iteration: int | None) -> Iterator[None]:
    token = _ITERATION.set(iteration)
    try:
        yield
    finally:
        _ITERATION.reset(token)


def _runtime(args: Namespace | None) -> Any:
    root = Path(getattr(args, "rl_trace_dir", None) or os.environ["RELAX_OTEL_DIR"]) / run_id(args)
    key = (os.getpid(), str(root), run_id(args))
    with _LOCK:
        if key in _RUNTIMES:
            return _RUNTIMES[key]
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanLimits, TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
        from opentelemetry.sdk.trace.id_generator import IdGenerator
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        instance = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{instance}.jsonl"

        class DiskExporter(SpanExporter):
            """One append-and-close per completed stage, before RPC returns.

            No queue sampling or shutdown-dependent buffer: worker termination
            cannot discard an already completed span. Files never share
            writers.
            """

            def export(self, spans: Any) -> Any:
                try:
                    payload = MessageToDict(encode_spans(spans), use_integers_for_enums=True)
                    _hex_trace_ids(payload)
                    line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    with _LOCK, path.open("a", encoding="utf-8") as stream:
                        stream.write(line + "\n")
                    return SpanExportResult.SUCCESS
                except Exception:
                    logger.exception("RL OTel trace write failed: %s", path)
                    return SpanExportResult.FAILURE

            def shutdown(self) -> None:
                pass

        class TrainingIndependentIds(IdGenerator):
            # Megatron intentionally seeds Python's default PRNG alike on multiple ranks.
            @staticmethod
            def nonzero(bits: int) -> int:
                while True:
                    value = secrets.randbits(bits)
                    if value:
                        return value

            def generate_span_id(self) -> int:
                return self.nonzero(64)

            def generate_trace_id(self) -> int:
                return self.nonzero(128)

            def is_trace_id_random(self) -> bool:
                return True

        provider = TracerProvider(
            id_generator=TrainingIndependentIds(),
            resource=Resource.create(
                {
                    "service.name": "relax",
                    "service.instance.id": instance,
                    "host.name": socket.gethostname(),
                    "process.pid": os.getpid(),
                }
            ),
            sampler=ALWAYS_ON,
            # Structured JSON and many real dependencies must not be truncated.
            span_limits=SpanLimits(
                max_attributes=128,
                max_span_attributes=128,
                max_attribute_length=SpanLimits.UNSET,
                max_span_attribute_length=SpanLimits.UNSET,
                max_links=100000,
            ),
        )
        provider.add_span_processor(SimpleSpanProcessor(DiskExporter()))
        tracer = provider.get_tracer("relax.rl", "0.19.0")
        result = (provider, tracer, root)
        _RUNTIMES[key] = result
        return result


def _hex_trace_ids(value: Any) -> None:
    # OTLP/JSON uses hex identities, unlike generic protobuf JSON bytes.
    if isinstance(value, list):
        for item in value:
            _hex_trace_ids(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ("traceId", "spanId", "parentSpanId") and isinstance(item, str):
                value[key] = base64.b64decode(item).hex()
            else:
                _hex_trace_ids(item)


class Stage:
    """A live RL operation with inline JSON and explicit member children."""

    def __init__(
        self,
        args: Namespace | None,
        name: str,
        *,
        iteration: int | None = None,
        attributes: dict[str, Any] | None = None,
        links: list[dict[str, Any]] | None = None,
        start_time: int | None = None,
        active: bool = True,
        inherit_iteration: bool = True,
        parent: Stage | None = None,
        parent_headers: dict[str, str] | None = None,
        producer_role: str | None = None,
    ) -> None:
        self.args = args
        self.name = name
        self.start_time = start_time if start_time is not None else time.time_ns()
        self.span: Any = None
        self.root: Path | None = None
        self._links: set[tuple[int, int]] = set()
        self._ended = False
        self._operation_end_time: int | None = None
        if not active or not enabled(args):
            return
        try:
            from opentelemetry.context import Context
            from opentelemetry.trace import set_span_in_context

            _, tracer, self.root = _runtime(args)
            attrs = {
                "rl.run.id": run_id(args),
                "rl.timing.clock": "host_wall",
                "rl.timing.scope": "operation",
                **(iteration_attributes(iteration) if inherit_iteration else {}),
                **(attributes or {}),
                "rl.producer.role": producer_role or process_role(),
            }
            if start_time is None:
                self.start_time = time.time_ns()
            context = set_span_in_context(parent.span, Context()) if parent and parent.span else Context()
            if parent_headers:
                from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

                context = TraceContextTextMapPropagator().extract(parent_headers, context=Context())
            self.span = tracer.start_span(
                name,
                context=context,
                attributes={k: v for k, v in attrs.items() if v is not None},
                start_time=self.start_time,
            )
            for link in links or []:
                self.add_link(link)
        except Exception:
            logger.exception("RL OTel span unavailable: %s", name)

    @best_effort
    def set(self, key: str, value: Any) -> None:
        if self.span is not None and value is not None and not self._ended:
            self.span.set_attribute(key, value)

    def data(self, key: str, value: Any) -> None:
        """Store full structured content directly in the exported span."""
        if self.span is None or self._ended:
            return
        try:
            self.set(key, json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        except Exception:
            logger.exception("RL OTel structured data write failed: %s", key)

    @best_effort
    def headers(self) -> dict[str, str]:
        if self.span is None:
            return {}
        from opentelemetry.context import Context
        from opentelemetry.trace import set_span_in_context
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier = {}
        TraceContextTextMapPropagator().inject(carrier, context=set_span_in_context(self.span, Context()))
        return carrier

    def child_attributes(self) -> dict[str, Any]:
        if self.span is None:
            return {}
        return {
            key: value
            for key, value in self.span.attributes.items()
            if key
            in (
                "rl.run.id",
                "rl.batch.id",
                "rl.step.id",
                "rl.iteration.index",
                "rl.purpose",
                "rl.model.name",
                "rl.backend.name",
                "rl.model.version",
                "rl.rollout.mode",
                "rl.reward.name",
                "rl.reward.type",
                "rl.filter.name",
                "rl.advantage.estimator",
                "rl.forward.purpose",
                "rl.forward.output",
                "rl.algorithm.name",
                "rl.transfer.target",
                "rl.transfer.target.step_id",
            )
        }

    @best_effort
    def result_detail(
        self, key: str, row: dict[str, Any], *, name: str | None = None, links: list[dict] | None = None
    ) -> Stage | None:
        """Legacy explicit result-span helper, retained for API compatibility.

        Built-in 0.11 instrumentation stores member rows with data() instead.
        """
        if self.span is None or self._ended:
            return None
        timestamp = time.time_ns()
        attrs = {**self.child_attributes(), "rl.sample.timing": "result"}
        for source, target in (
            ("export_id", "rl.export.id"),
            ("sample_index", "rl.sample.index"),
            ("sample_id", "rl.sample.id"),
            ("update_id", "rl.update.id"),
        ):
            if source in row:
                attrs[target] = str(row[source])
        child = Stage(
            self.args,
            name or self.name + ".sample",
            parent=self,
            attributes=attrs,
            inherit_iteration=False,
            start_time=timestamp,
            links=links,
        )
        value = {"uses": [row]} if key == "rl.train.results" else [row]
        child.data(key, value)
        error = RuntimeError(row.get("reason", "member operation failed")) if row.get("status") == "error" else None
        child.end(error, end_time=timestamp)
        return child

    @best_effort
    def context(self) -> dict[str, Any] | None:
        if self.span is None:
            return None
        context = self.span.get_span_context()
        return {
            "trace_id": f"{context.trace_id:032x}",
            "span_id": f"{context.span_id:016x}",
            "trace_flags": int(context.trace_flags),
        }

    def add_link(self, carrier: dict[str, Any] | None) -> None:
        if self.span is None or self._ended or not carrier:
            return
        from opentelemetry.trace import SpanContext, TraceFlags

        from relax.utils.fault_events.context import parse_trace_context

        try:
            context = parse_trace_context(carrier)
            identity = (context.trace_id, context.span_id)
            if identity not in self._links:
                self.span.add_link(SpanContext(*identity, is_remote=True, trace_flags=TraceFlags(context.trace_flags)))
                self._links.add(identity)
        except (ValueError, TypeError, AttributeError):
            logger.warning("RL OTel ignored invalid source context")

    def freeze_end(self, end_time: int | None = None) -> int:
        """Latch the operation boundary before observation-only result work."""
        if self._operation_end_time is None:
            self._operation_end_time = time.time_ns() if end_time is None else end_time
        return self._operation_end_time

    @best_effort
    def end(self, error: BaseException | None = None, *, end_time: int | None = None) -> None:
        if self.span is None or self._ended:
            return
        if error is not None:
            from opentelemetry.trace import Status, StatusCode

            self.span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {str(error)[:512]}"))
        self._ended = True
        self.span.end(end_time=end_time if end_time is not None else self._operation_end_time)


class RolloutCall:
    """Owned by one framework rollout invocation, including top-ups and eval
    datasets."""

    def __init__(self, span: Stage) -> None:
        self.span = span
        self.members: dict[str, str | None] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._pending = 0
        self._error: BaseException | None = None

    def admit(self, identities: list[str]) -> None:
        with self._lock:
            for identity in identities:
                self.members.setdefault(identity, None)

    def version(self, identity: str, value: str | None) -> None:
        with self._lock:
            if identity in self.members:
                self.members[identity] = value

    def exported_versions(self, samples: list[Any]) -> None:
        for sample in samples:
            metadata = getattr(sample, "metadata", None)
            if isinstance(metadata, dict) and "_rl_sample_id" in metadata:
                self.version(metadata["_rl_sample_id"], metadata.get("_rl_session_model_version"))

    def track_admission(self, *outcomes: asyncio.Future) -> None:
        """Hold export until this scheduled fetch/prepare decides which
        Sessions it admitted."""
        with self._lock:
            self._pending += 1
        resolved = False

        def resolve(_outcome: asyncio.Future) -> None:
            nonlocal resolved
            with self._lock:
                if resolved:
                    return
                resolved = True
                self._pending -= 1
                self._export_if_ready()
            for outcome in outcomes:
                outcome.remove_done_callback(resolve)

        for outcome in outcomes:
            outcome.add_done_callback(resolve)

    def close(self, error: BaseException | None) -> None:
        self.span.freeze_end()
        with self._lock:
            self._closed = True
            self._error = error
            self._export_if_ready()

    def _export_if_ready(self) -> None:
        if self._closed and not self._pending:
            self.span.set("rl.batch.size", len(self.members))
            versions = set(self.members.values())
            if len(versions) == 1:
                self.span.set("rl.model.version", versions.pop())
            self.span.end(self._error)
            self._error = None

    def carrier(self) -> dict[str, Any]:
        return {
            "headers": self.span.headers(),
            "context": self.span.context(),
            "attributes": self.span.child_attributes(),
        }


def current_rollout() -> RolloutCall | None:
    return _ROLLOUT_CALL.get()


@contextmanager
def rollout_call(args: Any, rollout_id: int | None, evaluation: bool) -> Iterator[RolloutCall | None]:
    if not enabled(args):
        yield None
        return
    agentic = getattr(args, "use_agentic_rollout", False)
    mode = (
        "agentic"
        if agentic
        else ("multi_turn" if getattr(args, "custom_generate_function_path", None) else "single_turn")
    )
    span = Stage(
        args,
        "rl.rollout.generate",
        iteration=rollout_id,
        attributes={
            "rl.batch.id": uuid.uuid4().hex,
            "rl.purpose": "evaluation" if evaluation else "training",
            "rl.model.name": model_name(args),
            "rl.rollout.mode": mode,
        },
    )
    call = RolloutCall(span)
    token = _ROLLOUT_CALL.set(call)
    stage_token = _CURRENT.set(span)
    failure = None
    try:
        yield call
    except BaseException as error:
        failure = error
        raise
    finally:
        call.close(failure)
        _CURRENT.reset(stage_token)
        _ROLLOUT_CALL.reset(token)


@contextmanager
def inference_context(parent: Stage | None, callback: dict[str, str] | None = None) -> Iterator[None]:
    """Bind one sample/Session during its generation calls, never across
    Ray."""
    token = _INFERENCE_PARENT.set((parent, callback))
    try:
        yield
    finally:
        _INFERENCE_PARENT.reset(token)


def inference_headers(headers: dict[str, str] | None = None) -> dict[str, str] | None:
    """Serialize the explicit RL parent before HTTP dispatch or Ray
    forwarding."""
    parent, callback = _INFERENCE_PARENT.get()
    if parent is None or parent.span is None or parent._ended or not enabled(parent.args):
        return headers
    try:
        from opentelemetry.context import Context
        from opentelemetry.trace import set_span_in_context
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        carrier = {
            key: value for key, value in (headers or {}).items() if key.lower() not in ("traceparent", "tracestate")
        }
        propagator = TraceContextTextMapPropagator()
        context = set_span_in_context(parent.span, Context())
        if callback:
            from opentelemetry.trace import get_current_span

            incoming = propagator.extract(callback, context=Context())
            candidate = get_current_span(incoming).get_span_context()
            if candidate.is_valid and candidate.trace_id == parent.span.get_span_context().trace_id:
                context = incoming
        propagator.inject(carrier, context=context)
        return carrier
    except Exception:
        logger.exception("RL OTel inference context unavailable")
        return headers


@contextmanager
def stage(args: Namespace | None, name: str, **kwargs: Any) -> Iterator[Stage]:
    observation = Stage(args, name, **kwargs)
    token = _CURRENT.set(observation)
    try:
        yield observation
    except BaseException as error:
        observation.end(error)
        raise
    finally:
        observation.end()
        _CURRENT.reset(token)


def current_stage() -> Stage | None:
    return _CURRENT.get()


def flush() -> None:
    with _LOCK:
        providers = [runtime[0] for key, runtime in _RUNTIMES.items() if key[0] == os.getpid()]
    for provider in providers:
        provider.force_flush(timeout_millis=1000)


atexit.register(flush)

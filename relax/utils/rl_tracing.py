# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Relax configuration and rollout ownership adapter for rl_lens.

Public imports remain compatible. All OTel providers, span state and
propagation are implemented by the standalone SDK, which never imports Relax.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from argparse import Namespace
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from rl_lens import (
    RLTracer,
    TraceConfig,
    current_stage,
    flush,
    inference_context,
    iteration_attributes,
    iteration_context,
    process_role,
    set_process_role,
    use_stage,
)
from rl_lens import Stage as SDKStage
from rl_lens.tracing import _hex_trace_ids as _hex_trace_ids  # Compatibility for the OTLP collector.
from rl_lens.tracing import inference_headers as sdk_inference_headers
from rl_lens.tracing import inference_parent

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_ROLLOUT_CALL: ContextVar[RolloutCall | None] = ContextVar("rl_rollout_call", default=None)
_ENV_NAMES = ("RELAX_OTEL_ENABLED", "RELAX_OTEL_DIR", "RELAX_OTEL_RUN_ID", "OTEL_SDK_DISABLED")

# These names remain public aliases to the SDK's single process-local state.
__all__ = [
    "Stage",
    "RolloutCall",
    "current_rollout",
    "rollout_call",
    "enabled",
    "initialize_run",
    "runtime_env",
    "run_id",
    "model_name",
    "stage",
    "telemetry",
    "current_stage",
    "flush",
    "inference_context",
    "inference_headers",
    "iteration_attributes",
    "iteration_context",
    "process_role",
    "set_process_role",
]


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


def _config(args: Namespace | None) -> TraceConfig:
    return TraceConfig(
        run_id=run_id(args),
        output_dir=str(getattr(args, "rl_trace_dir", None) or os.environ.get("RELAX_OTEL_DIR", "")),
        service_name="relax",
        scope_name="relax.rl",
        enabled=enabled(args),
    )


class Stage(SDKStage):
    """Compatibility entry point accepting Relax's existing Namespace."""

    def __init__(self, args: Namespace | None, name: str, **options: Any) -> None:
        super().__init__(_config(args), name, **options)
        self.args = args


def telemetry(args: Namespace | None = None) -> RLTracer:
    """Translate framework configuration once at the instrumentation
    boundary."""
    return RLTracer(_config(args), stage_factory=partial(Stage, args))


@contextmanager
def stage(args: Namespace | None, name: str, **options: Any) -> Iterator[Stage]:
    with Stage(args, name, **options) as observation:
        yield observation


def inference_headers(headers: dict[str, str] | None = None) -> dict[str, str] | None:
    parent = inference_parent()
    if isinstance(parent, Stage) and not enabled(parent.args):
        return headers
    return sdk_inference_headers(headers)


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
    span = telemetry(args).rollout_generate(
        iteration=rollout_id,
        purpose="evaluation" if evaluation else "training",
        model_name=model_name(args),
        mode=mode,
    )
    call = RolloutCall(span)
    token = _ROLLOUT_CALL.set(call)
    failure = None
    try:
        with use_stage(span):
            yield call
    except BaseException as error:
        failure = error
        raise
    finally:
        call.close(failure)
        _ROLLOUT_CALL.reset(token)

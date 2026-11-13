# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Driver-observed service lifecycle and process-local Serve constructor
attempts."""

from __future__ import annotations

import copy
from argparse import Namespace
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from inspect import signature
from typing import Any

from relax.utils.rl_tracing import Stage, current_stage, enabled, stage


def current() -> Stage | None:
    active = current_stage()
    return active if active is not None and active.name.startswith("rl.service.") else None


def operation(
    config: Namespace, name: str, *, role: str | None = None, attributes: dict[str, Any] | None = None
) -> AbstractContextManager[Stage]:
    span_name = None
    if name == "rl.service.startup":
        span_name = "start services"
    elif name == "rl.service.start":
        span_name = f"start {role}"
    return stage(
        config,
        name,
        span_name=span_name,
        parent=None if name == "rl.service.restart" else current(),
        inherit_iteration=False,
        producer_role="driver",
        attributes={"rl.service.role": role, **(attributes or {})},
    )


def restart_requested(config: Namespace, role: str, mode: str, attempt: int) -> None:
    with operation(
        config,
        "rl.service.restart.requested",
        role=role,
        attributes={"rl.service.restart.mode": mode, "rl.service.restart.attempt": attempt},
    ) as marker:
        marker.freeze_end(marker.start_time)


def phase_task(config: Namespace, phase: str, task: Callable) -> Callable:
    """Carry just the lifecycle parent into serial or parallel startup work."""
    if not enabled(config):
        return task
    parent = current()
    name = "rl.service.start" if phase == "deploy" else "rl.service.prepare"

    @wraps(task)
    def call(role, *args):
        resources = getattr(config, "resource", {}).get(role)
        with stage(
            config,
            name,
            span_name=f"start {role}" if phase == "deploy" else None,
            parent=parent,
            inherit_iteration=False,
            producer_role="driver",
            attributes={"rl.service.role": str(role), "rl.service.gpus": resources[1] if resources else None},
        ) as attempt:
            result = task(role, *args)
            # Controller workers return failures and peer cancellations as values.
            if result[2] is not None:
                attempt.end(RuntimeError(result[2]))
            return result

    return call


def deployment_config(config: Namespace, role: str) -> Namespace:
    """Bind immutable origin data to this Serve deployment, including future
    replicas."""
    start = current()
    origin = start.context() if start is not None and start.name == "rl.service.start" else None
    if origin is None:
        return config
    deployment_id = origin["span_id"]
    start.set("rl.service.deployment.id", deployment_id)
    bound = copy.copy(config)
    bound._rl_service_deployment = {"role": str(role), "origin": origin}
    return bound


def trace_replica_init(function: Callable) -> Callable:
    params = signature(function)

    @wraps(function)
    def call(self, *args, **kwargs):
        config = params.bind(self, *args, **kwargs).arguments["config"]
        deployment = getattr(config, "_rl_service_deployment", None)
        if not deployment or not enabled(config):
            return function(self, *args, **kwargs)
        role, origin = deployment["role"], deployment["origin"]
        attrs = {"rl.service.role": role, "rl.service.deployment.id": origin["span_id"]}
        options = {"attributes": attrs, "producer_role": f"{role}_service", "inherit_iteration": False}
        # Serve may invoke the same bound constructor again after the driver call has ended.
        with stage(config, f"init {role}_service", links=[origin], **options) as replica:
            # Persist entry even if the constructor hangs or the process is killed.
            with stage(config, f"enter {role}_service", parent=replica, **options) as marker:
                marker.freeze_end(marker.start_time)
            return function(self, *args, **kwargs)

    return call

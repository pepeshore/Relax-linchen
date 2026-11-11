# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Weight publication and activation boundaries, using existing ACKs only."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any, Callable
from uuid import uuid4

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_tracing import Stage, current_stage, enabled, model_name, stage
from relax.utils.rl_train_trace import current_rank_span


def trace_weight_sync(function: Callable) -> Callable:
    parameters = signature(function)

    @wraps(function)
    def call(owner: Any, *args: Any, **kwargs: Any) -> Any:
        rank = current_rank_span()
        if not enabled(owner.args) or (rank is None and owner._rank != 0):
            return function(owner, *args, **kwargs)
        if function.__name__ == "update_weights_fully_async":
            flags = parameters.bind(owner, *args, **kwargs).arguments
            if flags.get("actor_fwd_only") and flags.get("rollout_only"):
                return function(owner, *args, **kwargs)
        elif getattr(owner.args, "debug_train_only", False) or getattr(owner.args, "debug_rollout_only", False):
            return function(owner, *args, **kwargs)
        attrs = {"rl.execution.side": "training", "rl.rank.global": owner._rank, "rl.backend.name": "megatron"}
        with stage(owner.args, "rl.weight.sync", parent=rank, attributes=attrs):
            return function(owner, *args, **kwargs)

    return call


def trace_weight_call(function: Callable) -> Callable:
    @wraps(function)
    def call(owner: Any, *args: Any, **kwargs: Any) -> Any:
        if not enabled(owner.args):
            return function(owner, *args, **kwargs)
        previous = getattr(owner, "_rl_weight_stages", None)
        owner._rl_weight_stages = None
        error = None
        try:
            return function(owner, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            stages = owner._rl_weight_stages
            if stages is not None:
                publication, activation = stages
                end_time = publication.freeze_end()
                activation.freeze_end(end_time)
                publication.end(error)
                activation.end(error)
            owner._rl_weight_stages = previous

    return call


@best_effort
def begin_weights(owner: Any) -> None:
    if not enabled(owner.args):
        return
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    if dist.get_rank(get_gloo_group()) != 0:
        return
    engines = list(owner.rollout_engines) + list(getattr(owner, "distributed_rollout_engines", []))
    if not engines:
        return
    targets = [engine._actor_id.hex() for engine in engines]
    attrs = {
        "rl.model.name": model_name(owner.args),
        "rl.timing.scope": "source_sync",
        "rl.model.version": str(owner.weight_version),
        "rl.weight.transfer.id": uuid4().hex,
    }
    sync = current_stage()
    parent = sync if sync is not None and sync.name == "rl.weight.sync" else None
    publication = Stage(
        owner.args, "rl.actor.publish_weights", parent=parent, attributes={**attrs, "rl.backend.name": "megatron"}
    )
    activation = Stage(
        owner.args,
        "rl.rollout.load_weights",
        parent=parent,
        attributes={
            **attrs,
            "rl.backend.name": "sglang",
            "rl.weight.target.id": "ray-actors:" + ",".join(targets),
        },
        links=[publication.context()] if publication.context() else [],
    )
    owner._rl_weight_stages = (publication, activation)


@best_effort
def weights_delivered(owner: Any) -> None:
    stages = getattr(owner, "_rl_weight_stages", None)
    if stages is not None:
        stages[0].end()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Complete input rollouts, including continuation across collection
windows."""

from __future__ import annotations

import copy
import uuid
from typing import Any

from relax.utils.agentic_obs.source import sample_prompt_id
from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_tracing import Stage, current_rollout, iteration_attributes, run_id


_LIVE: dict[tuple[str, str], SampleTrace] = {}


def sample_attributes(sample: Any) -> dict[str, Any]:
    attrs = {}
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("_rl_sample_id"):
        attrs["rl.sample.id"] = metadata["_rl_sample_id"]
    index = getattr(sample, "index", None)
    if index is not None:
        attrs["rl.sample.index"] = str(index)
    return attrs


class SampleTrace:
    def __init__(self, args: Any, sample: Any, parent: Stage) -> None:
        if not isinstance(sample.metadata, dict):
            sample.metadata = {}
        identity = sample.metadata.setdefault("_rl_sample_id", uuid.uuid4().hex)
        self.key = (run_id(args), identity)
        self.prompt_id = sample_prompt_id(sample)
        attrs = {
            **parent.child_attributes(),
            **sample_attributes(sample),
            "rl.prompt.id": self.prompt_id,
            "rl.sample.timing": "operation",
        }
        group = sample.metadata.get("_rl_group_id", getattr(sample, "group_index", None))
        if group is not None:
            attrs["rl.group.id"] = str(group)
        attrs.pop("rl.step.id", None)
        attrs.pop("rl.iteration.index", None)
        self.span = Stage(args, "rl.rollout.sample", parent=parent, attributes=attrs, inherit_iteration=False)
        self.memberships: list[dict[str, Any]] = []
        self.version: str | None = None
        if getattr(sample, "prompt", None) is not None:
            self.span.data("rl.sample.input", copy.deepcopy(sample.prompt))
        self.admit(parent)

    def admit(self, parent: Stage) -> None:
        attrs = parent.child_attributes()
        rollout = current_rollout()
        if rollout is not None and rollout.span is parent:
            rollout.admit([self.key[1]])
        membership = {"batch_id": attrs.get("rl.batch.id")}
        iteration = iteration_attributes()
        if iteration:
            membership.update(step_id=iteration["rl.step.id"], iteration_index=iteration["rl.iteration.index"])
        if membership not in self.memberships:
            if self.memberships:
                self.span.add_link(parent.context())
            self.memberships.append(membership)

    @best_effort
    def finish(
        self, source: Any, results: list[Any], error: BaseException | None, *, end_time: int | None = None
    ) -> None:
        from relax.utils.rl_trace_hooks import _publish_source, export_id

        continuing = error is None and any(
            getattr(getattr(s, "status", None), "name", None) == "ABORTED" for s in results
        )
        if not continuing:
            self.span.freeze_end(end_time)
        for result in results:
            if not isinstance(result.metadata, dict):
                result.metadata = {}
            result.metadata["_rl_sample_id"] = self.key[1]
            if self.prompt_id is not None:
                result.prompt_id = self.prompt_id
        _publish_source(results, self.span)
        responses = [
            {
                "export_id": export_id(s),
                "text": s.response,
                "status": getattr(getattr(s, "status", None), "name", "unknown").lower(),
            }
            for s in (results or [source])
            if getattr(s, "response", None) is not None
        ]
        if responses:
            self.span.data("rl.sample.response", responses)
        histories = [getattr(result, "weight_versions", None) for result in results]
        versions = {str(v) for history in histories if history for v in history if v not in (None, "")}
        known = histories and all(history and all(v not in (None, "") for v in history) for history in histories)
        self.version = next(iter(versions)) if known and len(versions) == 1 else None
        rollout = current_rollout()
        if rollout is not None:
            rollout.version(self.key[1], None if continuing else self.version)
        # ABORTED can be carried to a later window even with partial_rollout=False.
        if continuing:
            return
        self.span.set("rl.model.version", self.version)
        self.span.data("rl.sample.memberships", self.memberships)
        steps = {m["iteration_index"] for m in self.memberships if "iteration_index" in m}
        if len(steps) == 1:
            for key, value in iteration_attributes(next(iter(steps))).items():
                self.span.set(key, value)
        self.span.end(error)
        _LIVE.pop(self.key, None)


@best_effort
def admit_sample(args: Any, sample: Any, parent: Stage) -> SampleTrace | None:
    metadata = getattr(sample, "metadata", None)
    identity = metadata.get("_rl_sample_id") if isinstance(metadata, dict) else None
    key = (run_id(args), identity)
    if key in _LIVE:
        observation = _LIVE[key]
        observation.admit(parent)
        return observation
    observation = SampleTrace(args, sample, parent)
    _LIVE[observation.key] = observation
    return observation


@best_effort
def discard_samples(args: Any, samples: list[Any] | None = None) -> None:
    identities = None if samples is None else {sample_attributes(s).get("rl.sample.id") for s in samples}
    for key, observation in list(_LIVE.items()):
        if key[0] == run_id(args) and (identities is None or key[1] in identities):
            observation.span.freeze_end()
            observation.span.set("rl.sample.cancelled", True)
            observation.span.set("rl.model.version", observation.version)
            observation.span.data("rl.sample.memberships", observation.memberships)
            observation.span.end()
            _LIVE.pop(key, None)

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Full Session-owned trace data, retained across generation pauses."""

from __future__ import annotations

import copy
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any

from relax.utils.agentic_obs.writers import best_effort
from relax.utils.rl_tracing import Stage, enabled, iteration_attributes, model_name


def _timestamps(value: Any) -> Any:
    if isinstance(value, list):
        return [_timestamps(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if item is None:
            continue
        if key in ("started_at", "finished_at"):
            if isinstance(item, str):
                item = datetime.fromisoformat(item.replace("Z", "+00:00")).timestamp()
            key = "start_time" if key == "started_at" else "end_time"
        result[key] = _timestamps(item)
    return result


_AGENT_BATCH: ContextVar[AgentBatch | None] = ContextVar("rl_agent_batch", default=None)


class AgentBatch:
    def __init__(self, args: Any, group_id: str, mode: str, sessions: set[str]) -> None:
        self.expected = sessions
        self.finished: set[str] = set()
        self.steps: set[int] = set()
        self.error: BaseException | None = None
        self.versions: dict[str, str | None] = {}
        self.span = Stage(
            args,
            "rl.rollout.generate",
            inherit_iteration=False,
            attributes={
                "rl.batch.id": uuid.uuid4().hex,
                "rl.batch.size": len(sessions),
                "rl.group.id": group_id,
                "rl.purpose": "evaluation" if mode == "eval" else "training",
                "rl.rollout.mode": "agentic",
                "rl.model.name": model_name(args),
                "rl.backend.name": "sglang",
            },
        )

    @best_effort
    def finish(self, identity: str, steps: set[int], error: BaseException | None, version: str | None = None) -> None:
        self.finished.add(identity)
        self.versions[identity] = version
        self.steps.update(steps)
        self.error = self.error or error
        if self.finished >= self.expected:
            self.close()

    @best_effort
    def close(self, error: BaseException | None = None) -> None:
        versions = set(self.versions.values())
        if len(self.versions) == len(self.expected) and len(versions) == 1 and None not in versions:
            self.span.set("rl.model.version", next(iter(versions)))
        if len(self.steps) == 1:
            for key, value in iteration_attributes(next(iter(self.steps))).items():
                self.span.set(key, value)
        self.span.data("rl.sample.memberships", [{"step_id": f"iteration:{step}"} for step in sorted(self.steps)])
        self.span.end(self.error or error)


def trace_agentic_group(function: Any) -> Any:
    @wraps(function)
    async def call(owner: Any, rollout_mode: str, group_id: str, session_specs: Any) -> Any:
        if not enabled(owner.args) or not session_specs:
            return await function(owner, rollout_mode, group_id, session_specs)
        batch = AgentBatch(owner.args, group_id, rollout_mode, {spec.session_id for spec in session_specs})
        token = _AGENT_BATCH.set(batch)
        try:
            return await function(owner, rollout_mode, group_id, session_specs)
        except BaseException as error:
            # start_group has already joined its failure cleanup at this point.
            batch.close(error)
            raise
        finally:
            _AGENT_BATCH.reset(token)

    return call


class AgentTrace:
    def __init__(self, args: Any, session: Any, session_input: Any) -> None:
        self.observation = session.observation
        self.session_id = session.session_id
        self.steps: set[int] = set()
        self.exports: list[dict[str, Any]] = []
        self.version: str | None = None
        self.states: list[dict[str, Any]] = []
        self.execution = {"start_time": time.time()}
        self.input = {
            "input_payload": copy.deepcopy(session_input.input_payload),
            "sampling_config": copy.deepcopy(session.session_sampling_params),
        }
        if session.forest is not None:
            self.input["label"] = copy.deepcopy(session.forest.label)
            self.input["tools"] = copy.deepcopy(session_input.input_payload.get("tools", []))
        self.external_links = copy.deepcopy(getattr(session.forest, "static_metadata", {}).get("external_links", []))
        self.batch = _AGENT_BATCH.get() or AgentBatch(
            args, session.group.group_id, session.group.rollout_mode, {session.session_id}
        )
        self.sample = Stage(
            args,
            "rl.rollout.sample",
            parent=self.batch.span,
            inherit_iteration=False,
            attributes={
                **self.batch.span.child_attributes(),
                "rl.sample.id": session.session_id,
                "rl.session.id": session.session_id,
                "rl.sample.timing": "operation",
            },
        )
        if session_input.input_payload.get("messages") is not None:
            self.sample.data("rl.sample.input", session_input.input_payload["messages"])
        self.span = Stage(
            args,
            "rl.agent.run",
            links=[self.sample.context()],
            inherit_iteration=False,
            attributes={
                "rl.sample.id": session.session_id,
                "rl.session.id": session.session_id,
                "rl.agent.name": args.agent_command,
                "rl.purpose": "evaluation" if session.group.rollout_mode == "eval" else "training",
                "rl.group.id": session.group.group_id,
                "rl.group.expected_sessions": len(session.group.result_cells),
            },
        )

    @best_effort
    def assign(self, iteration: int) -> None:
        self.steps.add(iteration)

    @best_effort
    def execution_end(self, error: BaseException | None) -> None:
        self.execution.update(end_time=time.time(), result="failed" if error else "completed")
        if error:
            self.execution["reason"] = f"{type(error).__name__}: {error}"

    @best_effort
    def capture_states(self, forest: Any) -> None:
        if forest is None:
            return
        self.states = [
            {
                "hash": node.state_hash,
                "parent_hash": node.parent_state_hash,
                "type": node.kind,
                "messages_delta": copy.deepcopy(node.messages_delta),
                "request_id": node.export_metadata_patch.get("request_id"),
                "tools": copy.deepcopy(node.tools),
            }
            for node in forest.nodes_by_hash.values()
        ]

    @best_effort
    def capture_exports(self, prepared: list[Any], start_time: float) -> None:
        end_time = time.time()
        self.exports = []
        histories = [payload["sample_payload"].get("weight_versions") for _, payload in prepared]
        versions = {
            str(version) for history in histories if history for version in history if version not in (None, "")
        }
        if (
            histories
            and all(history and all(v not in (None, "") for v in history) for history in histories)
            and len(versions) == 1
        ):
            self.version = next(iter(versions))
            self.sample.set("rl.model.version", self.version)
        for state_hash, payload in prepared:
            sample = payload["sample_payload"]
            metadata = sample.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                metadata = sample["metadata"] = {}
            metadata.update(_rl_sample_id=self.session_id, _rl_source=self.sample.context())
            item = {
                "export_id": str(sample["sample_export_id"]),
                "name": payload["name"],
                "leaf_state_hash": state_hash,
                "start_time": start_time,
                "end_time": end_time,
                "result": "completed",
                "response_tokens": sample["response_length"],
            }
            if sample.get("reward") is not None:
                item.update(reward=copy.deepcopy(sample["reward"]), reward_source="agent")
            self.exports.append(item)

    @best_effort
    def _finish_data(self, error: BaseException | None = None) -> None:
        requests = _timestamps(copy.deepcopy(list(self.observation.calls.values())))
        committed = [request for request in requests if request.get("response_messages") is not None]
        self.span.set("rl.rollout.assignment", "assigned" if self.steps else "unassigned")
        self.span.set("rl.session.turn_count", len(committed))
        self.span.set("rl.session.export_count", len(self.exports))
        if len(self.steps) == 1:
            for key, value in iteration_attributes(next(iter(self.steps))).items():
                self.span.set(key, value)
        if all("usage" in request for request in committed) and len(committed) == len(requests):
            self.span.set(
                "rl.session.output_tokens", sum(request["usage"]["completion_tokens"] for request in committed)
            )
        tools: dict[str, dict[str, Any]] = {}
        for state in self.states:
            for message in state["messages_delta"]:
                for call in message.get("tool_calls", []):
                    identity = call.get("id")
                    if identity:
                        tools.setdefault(identity, {"tool_call_id": identity}).update(
                            copy.deepcopy(call.get("function", {}))
                        )
                identity = message.get("tool_call_id")
                if identity:
                    tools.setdefault(identity, {"tool_call_id": identity})["return_message"] = copy.deepcopy(message)
        data = {
            "execution": self.execution,
            "input": self.input,
            "requests": requests,
            "states": self.states,
            "tools": list(tools.values()),
            "exports": self.exports,
            "step_memberships": [
                {"session_id": self.session_id, "step_id": f"iteration:{step}"} for step in sorted(self.steps)
            ],
        }
        if self.external_links:
            data["external_links"] = self.external_links
        self.span.data("rl.trajectory.data", data)

    @best_effort
    def finish(self, error: BaseException | None = None) -> None:
        try:
            self._finish_data(error)
            nodes = {node["hash"]: node for node in self.states}
            responses = []
            for exported in self.exports:
                node = nodes.get(exported["leaf_state_hash"])
                seen = set()
                while node is not None and node["hash"] not in seen:
                    seen.add(node["hash"])
                    messages = [
                        message
                        for message in node["messages_delta"]
                        if node["type"] == "resp" and message.get("role") == "assistant"
                    ]
                    if messages:
                        responses.append(
                            {"export_id": exported["export_id"], "name": exported["name"], "messages": messages}
                        )
                        break
                    node = nodes.get(node["parent_hash"])
            if responses:
                self.sample.data("rl.sample.response", responses)
            self.sample.data(
                "rl.sample.memberships",
                [
                    {"batch_id": self.batch.span.child_attributes().get("rl.batch.id"), "step_id": f"iteration:{step}"}
                    for step in sorted(self.steps)
                ],
            )
            if len(self.steps) == 1:
                for key, value in iteration_attributes(next(iter(self.steps))).items():
                    self.sample.set(key, value)
        finally:
            self.span.end(error)
            self.sample.end(error)
            self.batch.finish(self.session_id, self.steps, error, self.version)


@best_effort
def start_agent_trace(args: Any, session: Any, session_input: Any) -> None:
    if enabled(args) and session.observation is not None:
        session.observation.rl_trace = AgentTrace(args, session, session_input)

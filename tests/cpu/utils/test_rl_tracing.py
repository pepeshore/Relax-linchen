# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
import re
import subprocess
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest

from relax.utils import rl_trace_hooks as hooks
from relax.utils import rl_tracing as tracing


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("RELAX_OTEL_RUN_ID", raising=False)
    args = Namespace(hf_checkpoint="models/policy", rm_type="math", custom_rm_path=None, reward_key=None)
    tracing.initialize_run(args)
    return args, tmp_path / args.rl_trace_run_id


def spans(root):
    result = []
    for path in root.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            for resource in json.loads(line)["resourceSpans"]:
                for scope in resource["scopeSpans"]:
                    assert scope["scope"] == {"name": "relax.rl", "version": "0.9.0"}
                    result.extend(scope["spans"])
    return result


def attrs(span):
    return {entry["key"]: next(iter(entry["value"].values())) for entry in span.get("attributes", [])}


def artifact(span, key):
    uri = attrs(span)[key + "_ref"]
    assert urlparse(uri).scheme == "file"
    return json.loads(Path(unquote(urlparse(uri).path)).read_text())


def sample():
    return SimpleNamespace(metadata={}, sample_export_id=-1)


def test_otlp_roots_links_resources_and_full_artifact(run):
    args, root = run
    with tracing.stage(args, "rl.rollout.generate", iteration=0) as generation:
        with tracing.stage(
            args, "rl.reward.score", iteration=0, links=[generation.context(), generation.context()]
        ) as reward:
            reward.data("rl.reward.results", {"content": "全量" * 100000})
    records = spans(root)
    assert len(records) == 2
    assert len({s["traceId"] for s in records}) == 2
    for record in records:
        assert re.fullmatch("[0-9a-f]{32}", record["traceId"])
        assert re.fullmatch("[0-9a-f]{16}", record["spanId"])
        assert not record.get("parentSpanId")
        assert record["kind"] == 1
        assert record.get("status", {}).get("code", 0) == 0
        assert attrs(record)["rl.run.id"] == args.rl_trace_run_id
        assert attrs(record)["rl.step.id"] == "iteration:0"
    reward = next(s for s in records if s["name"] == "rl.reward.score")
    assert len(reward["links"]) == 1
    assert artifact(reward, "rl.reward.results") == {"content": "全量" * 100000}


def test_error_preserves_exception_and_marks_span(run):
    args, root = run
    original = ValueError("actual failure")
    with pytest.raises(ValueError) as error, tracing.stage(args, "rl.actor.train"):
        raise original
    assert error.value is original
    record = spans(root)[0]
    assert record["status"] == {"message": "ValueError: actual failure", "code": 2}


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_disabled_is_transparent(run, monkeypatch, value):
    _, root = run
    monkeypatch.setenv("RELAX_OTEL_ENABLED", value)
    value_returned = object()
    calls = []

    @hooks.trace_reward
    async def reward(args, value):
        calls.append(value)
        return value_returned

    assert asyncio.run(reward(Namespace(), object())) is value_returned
    assert len(calls) == 1
    assert not spans(root)


def test_concurrent_eval_and_training_contexts(run):
    args, root = run

    @hooks.trace_reward
    async def reward(args, value):
        await asyncio.sleep(0)
        return 0.0

    @hooks.trace_iteration
    async def eval_rollout(args, rollout_id):
        return await reward(args, sample())

    @hooks.trace_iteration
    async def generate_rollout_async(args, rollout_id):
        return await reward(args, sample())

    async def both():
        return await asyncio.gather(eval_rollout(args, 3), generate_rollout_async(args, 4))

    assert asyncio.run(both()) == [0.0, 0.0]
    assert {(attrs(s)["rl.purpose"], attrs(s)["rl.step.id"]) for s in spans(root)} == {
        ("evaluation", "iteration:3"),
        ("training", "iteration:4"),
    }
    assert tracing.iteration_attributes() == {}


def test_generation_finishes_before_reward(run):
    args, root = run

    @hooks.trace_generation
    async def generate(state, args, value):
        return value

    @hooks.trace_reward
    async def reward(args, value):
        return 0.0

    async def execute():
        value = await generate(None, args, sample())
        return await reward(args, value)

    asyncio.run(execute())
    generation, reward_span = spans(root)
    assert int(generation["endTimeUnixNano"]) <= int(reward_span["startTimeUnixNano"])
    assert reward_span["links"][0]["spanId"] == generation["spanId"]


def test_filter_callable_and_normal_drop(run):
    args, root = run

    @hooks.trace_filter
    def call(fn, *args):
        return fn(*args)

    class Filter:
        def __call__(self, values):
            return SimpleNamespace(keep=False, reason="constant group reward")

    values = [sample(), sample()]
    assert not call(Filter(), values).keep
    assert not call(partial(lambda args, values: Filter()(values)), args, values).keep
    for record in spans(root):
        assert record.get("status", {}).get("code", 0) == 0
        results = artifact(record, "rl.filter.results")
        assert len({r["export_id"] for r in results}) == 2
        assert all(r["decision"] == "drop" for r in results)


def test_transfer_backfill_keeps_source_iteration_and_real_ack(run):
    args, root = run

    @hooks.trace_transfer
    async def transfer(args, samples, batch_count, rollout_id, client):
        return "ack"

    with tracing.iteration_context(4):
        assert asyncio.run(transfer(args, [[sample()]], 0, 3, None)) == "ack"
    record = spans(root)[0]
    assert attrs(record)["rl.step.id"] == "iteration:4"
    assert attrs(record)["rl.transfer.target.step_id"] == "iteration:3"
    assert artifact(record, "rl.transfer.data")["receive_ack_time"] > 0


def test_failed_transfer_does_not_invent_ack(run):
    args, root = run

    @hooks.trace_transfer
    async def transfer(args, samples, batch_count, rollout_id, client):
        raise TimeoutError("no ACK")

    with pytest.raises(TimeoutError):
        asyncio.run(transfer(args, [[sample()]], 0, 1, None))
    assert "receive_ack_time" not in artifact(spans(root)[0], "rl.transfer.data")


def test_storage_failure_does_not_break_business(run, monkeypatch):
    args, _ = run
    monkeypatch.setattr(tracing, "_runtime", lambda args: (_ for _ in ()).throw(OSError("disk full")))
    with tracing.stage(args, "rl.actor.train") as observation:
        observation.data("rl.train.results", {"updates": []})
    assert observation.context() is None


def test_two_processes_have_separate_files_and_same_run(run):
    args, root = run
    code = "from relax.utils.rl_tracing import stage\nwith stage(None, 'rl.actor.train', iteration=2): pass"
    for _ in range(2):
        subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
    assert len(list(root.glob("*.jsonl"))) == 2
    assert {attrs(span)["rl.run.id"] for span in spans(root)} == {args.rl_trace_run_id}


def test_restart_has_new_run_identity(run):
    args, _ = run
    previous = args.rl_trace_run_id
    tracing.initialize_run(args)
    assert previous != args.rl_trace_run_id


def test_agent_trace_preserves_branches_attempts_and_cross_step_identity(run):
    from relax.utils.rl_agent_trace import AgentTrace

    args, root = run
    args.agent_command = "python agent.py"
    calls = {
        "request": {
            "request_id": "request",
            "started_at": "2026-11-09T00:00:00Z",
            "finished_at": "2026-11-09T00:00:01Z",
            "response_messages": [{"role": "assistant", "content": "done"}],
            "usage": {"completion_tokens": 3},
            "attempts": [{"attempt_index": 0, "status": "aborted"}, {"attempt_index": 1, "status": "completed"}],
        }
    }
    node = lambda identity, parent, messages: SimpleNamespace(
        state_hash=identity,
        parent_state_hash=parent,
        kind="resp",
        messages_delta=messages,
        export_metadata_patch={"request_id": "request"},
        tools=[],
    )
    forest = SimpleNamespace(
        label="answer",
        static_metadata={},
        nodes_by_hash={
            "a": node(
                "a",
                None,
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "tool-1", "function": {"name": "search", "arguments": "{}"}}],
                    }
                ],
            ),
            "b": node("b", "a", [{"role": "tool", "tool_call_id": "tool-1", "content": "final tool result"}]),
            "c": node("c", "a", [{"role": "assistant", "content": "other branch"}]),
        },
    )
    session = SimpleNamespace(
        session_id="session",
        observation=SimpleNamespace(calls=calls),
        session_sampling_params={"temperature": 0.8},
        forest=forest,
        group=SimpleNamespace(rollout_mode="train", group_id="group", result_cells={"session": None}),
    )
    observation = AgentTrace(
        args, session, SimpleNamespace(input_payload={"messages": [{"role": "user", "content": "task"}]})
    )
    identity = observation.span.context()
    observation.assign(0)
    observation.assign(1)
    observation.execution_end(None)
    observation.capture_states(forest)
    observation.capture_exports(
        [
            ("a", {"name": "first", "sample_payload": {"sample_export_id": 11, "response_length": 3, "reward": 0.0}}),
            ("c", {"name": "second", "sample_payload": {"sample_export_id": 12, "response_length": 3}}),
        ],
        1.0,
    )
    observation.finish()
    record = spans(root)[0]
    assert identity["span_id"] == record["spanId"]
    assert "rl.step.id" not in attrs(record)
    assert attrs(record)["rl.session.turn_count"] == "1"
    assert attrs(record)["rl.session.export_count"] == "2"
    data = artifact(record, "rl.trajectory.data")
    assert len(data["states"]) == 3
    assert len(data["requests"][0]["attempts"]) == 2
    assert data["tools"][0]["return_message"]["content"] == "final tool result"
    assert data["requests"][0]["start_time"] == 1794182400.0
    assert {item["export_id"] for item in data["exports"]} == {"11", "12"}


@pytest.mark.parametrize("masked_value", [4.0, float("nan")])
@pytest.mark.parametrize("full_advantage", [False, True])
def test_training_rpc_combines_cp_facts_without_duplicate_stage(run, monkeypatch, masked_value, full_advantage):
    import time

    import torch

    from relax.utils import rl_train_trace as train

    args, root = run
    args.advantage_estimator = "grpo"
    args.use_agentic_rollout = True
    args.qkv_format = "thd"
    args.agentic_obs_train_generation_id = "generation"
    cp = {"rank": 0}
    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **kwargs: True,
        get_data_parallel_rank=lambda **kwargs: 0,
        get_context_parallel_rank=lambda: cp["rank"],
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))

    def slice_cp(mask, *args, **kwargs):
        return mask[cp["rank"] * 2 : (cp["rank"] + 1) * 2]

    monkeypatch.setitem(
        sys.modules, "relax.backends.megatron.cp_utils", SimpleNamespace(slice_log_prob_with_cp=slice_cp)
    )

    @train.training_rpc
    def execute(owner, rollout_id):
        data = {
            "sample_export_ids": [23],
            "total_lengths": [6],
            "response_lengths": [4],
            "loss_masks": [torch.ones(4)],
            "advantages": [
                torch.tensor([2.0, masked_value, 6.0, masked_value])
                if full_advantage
                else torch.tensor([2.0 if cp["rank"] == 0 else 6.0, masked_value])
            ],
        }
        with train.window(args, "actor", rollout_id, data=data):
            with train.update(args, rollout_id, 0, "actor"):
                train.capture(args, data, masks=[torch.tensor([1.0, 0.0, 1.0, 0.0])])
                train.optimizer_result(False, "zero_effective_tokens")
        time.sleep(0.01)  # checkpoint/offload outside the observed window

    reports = []
    for rank in (0, 1):
        cp["rank"] = rank
        reports.append(execute(SimpleNamespace(args=args), 0))
    before_export = time.time_ns()
    train.emit_windows(args, reports)
    record = spans(root)[0]
    assert len(spans(root)) == 1
    assert attrs(record)["rl.batch.size"] == "1"
    assert int(record["endTimeUnixNano"]) < before_export
    result = artifact(record, "rl.train.results")
    assert result["updates"] == [
        {"update_id": "generation:actor:0:0", "status": "skipped", "reason": "zero_effective_tokens"}
    ]
    assert result["uses"] == [
        {
            "export_id": "23",
            "effective_tokens": 2,
            "update_id": "generation:actor:0:0",
            "used": True,
            "advantage_mean": 4.0,
        }
    ]
    assert record.get("status", {}).get("code", 0) == 0


def test_incomplete_rank_results_are_not_published_as_complete(run):
    from relax.utils.rl_train_trace import emit_windows

    args, root = run
    args.use_agentic_rollout = True
    rank = {"dp": 0, "cp": 0, "tp": 0, "pp": 0}
    item = {
        "name": "rl.actor.train",
        "sequence": 0,
        "attributes": {},
        "owner": True,
        "rank": rank,
        "members": ["export:1"],
        "start_time": 100,
        "end_time": 200,
        "updates": [],
        "records": [],
    }
    reports = [
        {"rl_trace_producer": {"owner": True, "rank": rank}, "rl_trace_windows": [item]},
        {"rl_trace_producer": {"owner": True, "rank": {**rank, "cp": 1}}, "rl_trace_windows": []},
    ]
    emit_windows(args, reports)
    record = spans(root)[0]
    assert "rl.train.results_ref" not in attrs(record)
    assert "missing observation owner" in artifact(record, "relax.observation.partial")["errors"]
    assert record.get("status", {}).get("code", 0) == 0


def test_checkpoint_failure_preserves_finished_training_without_changing_exception(run, monkeypatch):
    from relax.utils import rl_train_trace as train

    args, root = run
    args.advantage_estimator = "grpo"
    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda **kwargs: True,
        get_data_parallel_rank=lambda **kwargs: 0,
        get_context_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    original = OSError("checkpoint failed")

    @train.training_rpc
    def execute(owner, rollout_id):
        with train.window(args, "actor", rollout_id, data={"sample_indices": [1]}):
            with train.update(args, rollout_id, 0, "actor"):
                train.optimizer_result(True)
        raise original

    with pytest.raises(OSError) as caught:
        execute(SimpleNamespace(args=args), 0)
    assert caught.value is original
    record = spans(root)[0]
    assert record.get("status", {}).get("code", 0) == 0
    assert artifact(record, "relax.observation.partial")["updates"][0]["status"] == "committed"

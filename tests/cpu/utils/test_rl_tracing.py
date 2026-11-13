# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
import random
import re
import subprocess
import sys
from argparse import Namespace
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest

from relax.utils import rl_trace_hooks as hooks
from relax.utils import rl_tracing as tracing


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_OTEL_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("RELAX_OTEL_RUN_ID", "")
    args = Namespace(hf_checkpoint="models/policy", rm_type="math", custom_rm_path=None, reward_key=None)
    tracing.initialize_run(args)
    return args, tmp_path / args.rl_trace_run_id


def spans(root, name=None):
    result = []
    for path in root.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            for resource in json.loads(line)["resourceSpans"]:
                for scope in resource["scopeSpans"]:
                    assert scope["scope"] == {"name": "relax.rl", "version": "0.21.0"}
                    result.extend(scope["spans"])
    return [span for span in result if name is None or span["name"] == name]


def attrs(span):
    return {entry["key"]: next(iter(entry["value"].values())) for entry in span.get("attributes", [])}


def artifact(span, key):
    assert key + "_ref" not in attrs(span)
    return json.loads(attrs(span)[key])


def sample():
    return SimpleNamespace(metadata={}, sample_export_id=-1)


def test_otlp_roots_links_resources_and_full_inline_data(run, monkeypatch):
    args, root = run
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "32")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "32")
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
    assert not (root / "artifacts").exists()


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


def test_trace_ids_are_independent_of_the_training_random_seed(run):
    args, _ = run
    previous = random.getstate()
    roots, children = [], []
    try:
        with tracing.stage(args, "rl.actor.train_step") as dispatch:
            for _ in range(2):
                random.seed(42)
                initial = random.getstate()
                with tracing.stage(args, "rl.rollout.generate") as root:
                    roots.append(root.span.get_span_context().trace_id)
                with tracing.stage(args, "rl.actor.train_rank", parent=dispatch) as child:
                    children.append(child.span.get_span_context().span_id)
                assert random.getstate() == initial
    finally:
        random.setstate(previous)
    assert len(set(roots)) == len(set(children)) == 2


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
    (generation,) = spans(root, "rl.rollout.sample")
    (reward_span,) = spans(root, "rl.reward.score")
    assert int(generation["endTimeUnixNano"]) <= int(reward_span["startTimeUnixNano"])
    assert reward_span["links"][0]["spanId"] == generation["spanId"]


def test_generation_batch_records_128_inputs_without_waiting_for_rewards(run):
    args, root = run

    async def execute():
        release_reward = asyncio.Event()
        all_generated = asyncio.Event()
        generated = 0

        @hooks.trace_generation
        async def generate(state, args, value):
            nonlocal generated
            value.weight_versions = ["snapshot-1"]
            generated += 1
            if generated == 128:
                all_generated.set()
            return value

        @hooks.trace_reward
        async def reward(args, value):
            await release_reward.wait()
            return 0.0

        async def pipeline(value):
            value = await generate(None, args, value)
            return await reward(args, value)

        with tracing.iteration_context(0), hooks.generation_batch(args):
            tasks = []
            for _ in range(128):
                value = sample()
                tasks.append(hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value]))
        await all_generated.wait()
        (generation,) = spans(root, "rl.rollout.generate")
        assert generation["name"] == "rl.rollout.generate"
        assert attrs(generation)["rl.batch.size"] == "128"
        assert attrs(generation)["rl.model.version"] == "snapshot-1"
        assert not any(task.done() for task in tasks)
        release_reward.set()
        assert await asyncio.gather(*tasks) == [0.0] * 128
        return generation

    generation = asyncio.run(execute())
    records = spans(root)
    assert len(records) == 258
    assert all(s["name"] != "rl.agent.run" for s in records)
    rewards = [s for s in records if s["name"] == "rl.reward.score"]
    samples = spans(root, "rl.rollout.sample")
    assert len(samples) == 128
    assert all(s["parentSpanId"] == generation["spanId"] and s["traceId"] == generation["traceId"] for s in samples)
    (reward_batch,) = spans(root, "rl.reward.batch")
    assert (reward_batch["traceId"], reward_batch["parentSpanId"]) == (generation["traceId"], generation["spanId"])
    assert attrs(reward_batch)["rl.batch.size"] == "128"
    assert all(
        s["parentSpanId"] == reward_batch["spanId"] and s["traceId"] == reward_batch["traceId"] for s in rewards
    )
    assert {s["links"][0]["spanId"] for s in rewards} == {s["spanId"] for s in samples}
    # Streaming scoring can begin before the entire generation batch closes.
    assert int(rewards[0]["startTimeUnixNano"]) < int(generation["endTimeUnixNano"])


@pytest.mark.parametrize("failure", ["before_dispatch", "during_generation", "during_reward", "cancel_before_start"])
def test_generation_batch_failure_keeps_running_siblings_and_native_end(run, failure):
    args, root = run
    original = ValueError(failure)

    async def execute():
        release = asyncio.Event()
        started = asyncio.Event()
        values = [sample(), sample()]
        children = []

        @hooks.trace_generation
        async def generate(state, args, value):
            if value is values[0]:
                if failure == "during_generation":
                    raise original
            else:
                started.set()
                await release.wait()
            return value

        async def pipeline(value):
            result = await generate(None, args, value)
            if value is values[0] and failure == "during_reward":
                raise original
            return result

        async def group():
            if failure == "before_dispatch":
                raise original
            for value in values:
                children.append(hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value]))
            return await asyncio.gather(*children)

        with hooks.generation_batch(args):
            task = hooks.track_generation_task(asyncio.create_task(group()), values)
        if failure == "cancel_before_start":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if failure == "cancel_before_start" else ValueError):
            await task
        if children:
            await started.wait()
            assert not spans(root, "rl.rollout.generate"), "A failed group must not end a still-generating sibling"
            release.set()
            await asyncio.gather(*children, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(execute())
    (record,) = spans(root, "rl.rollout.generate")
    assert attrs(record)["rl.batch.size"] == "2"
    assert record.get("status", {}).get("code", 0) == (0 if failure == "during_reward" else 2)


def test_generation_batches_keep_topups_eval_and_cached_members_distinct(run):
    args, root = run
    cached = sample()
    cached.status = SimpleNamespace(name="COMPLETED")
    cached.metadata["_rl_source"] = {"original": True}

    @hooks.trace_generation
    async def generate(state, args, value):
        await asyncio.sleep(0)
        value.weight_versions = ["snapshot-1"]
        return value

    async def pipeline(value):
        if value is cached:
            return value
        return await generate(None, args, value)

    @hooks.trace_generation_window
    async def evaluate(args):
        tasks = []
        for _ in range(3):
            value = sample()
            tasks.append(hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value]))
        return await asyncio.gather(*tasks)

    async def execute():
        tasks = []
        with tracing.iteration_context(4):
            # Two real admissions in one iteration, plus a cached-only batch.
            for size in [2, 1, 0]:
                with hooks.generation_batch(args):
                    for value in [cached] + [sample() for _ in range(size)]:
                        tasks.append(hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value]))
            tasks.append(asyncio.create_task(evaluate(args)))
        await asyncio.gather(*tasks)

    asyncio.run(execute())
    records = spans(root, "rl.rollout.generate")
    assert len(records) == 3
    assert len({attrs(s)["rl.batch.id"] for s in records}) == 3
    assert {(attrs(s)["rl.purpose"], attrs(s)["rl.batch.size"]) for s in records} == {
        ("training", "1"),
        ("training", "2"),
        ("evaluation", "3"),
    }
    assert cached.metadata["_rl_source"] == {"original": True}


@pytest.mark.parametrize("mixed", [False, True])
def test_batched_reward_parents_fanout_spans_and_preserves_sources(run, mixed):
    args, root = run
    values = [sample(), sample()]

    @hooks.trace_reward
    async def reward(args, value):
        hooks.record_reward_route(value, "verifier" if mixed and value is values[1] else "math", "rule")
        await asyncio.sleep(0)
        return 0.0 if value is values[0] else 1.0

    @hooks.trace_batched_reward
    async def batch(args, samples):
        return await asyncio.gather(*(reward(args, value) for value in samples))

    with tracing.stage(args, "rl.rollout.generate") as generation:
        hooks._publish_source(values, generation)
    assert asyncio.run(batch(args, values)) == [0.0, 1.0]
    records = spans(root)
    assert len(records) == 4
    scored = next(s for s in records if s["name"] == "rl.reward.batch")
    assert len(scored["links"]) == 1
    assert attrs(scored)["rl.reward.name"] == ("router" if mixed else "math")
    results = artifact(scored, "rl.reward.results")
    assert [r["source"] for r in results] == ["math", "verifier" if mixed else "math"]
    assert [r["value"] for r in results] == [0.0, 1.0]
    assert len({r["export_id"] for r in results}) == 2
    # Context reset: a later standalone score still emits its own span.
    assert asyncio.run(reward(args, sample())) == 1.0
    assert len(spans(root)) == 6


def test_batched_reward_preserves_success_and_running_sibling_after_failure(run):
    args, root = run
    original = ValueError("scorer failed")

    async def execute():
        release = asyncio.Event()
        values = [sample() for _ in range(3)]
        children = []

        @hooks.trace_reward
        async def reward(args, value):
            hooks.record_reward_route(value, "math", "rule")
            if value is values[1]:
                raise original
            if value is values[2]:
                await release.wait()
            return 1.0

        @hooks.trace_batched_reward
        async def batch(args, samples):
            children.extend(asyncio.create_task(reward(args, value)) for value in samples)
            return await asyncio.gather(*children)

        with pytest.raises(ValueError) as error:
            await batch(args, values)
        assert error.value is original
        assert not spans(root, "rl.reward.batch"), (
            "Do not publish incomplete results while a real scoring child is running"
        )
        release.set()
        await asyncio.gather(*children, return_exceptions=True)

    asyncio.run(execute())
    (record,) = spans(root, "rl.reward.batch")
    assert record["status"]["code"] == 2
    results = artifact(record, "rl.reward.results")
    assert [item["status"] for item in results] == ["valid", "error", "valid"]
    assert results[0]["value"] == results[2]["value"] == 1.0
    assert "scorer failed" in results[1]["reason"]


def test_eval_generation_is_exported_while_rewards_are_pending(run):
    args, root = run

    async def execute():
        generated = asyncio.Event()
        release = asyncio.Event()

        @hooks.trace_generation
        async def generate(state, args, value):
            generated.set()
            return value

        async def pipeline(value):
            await generate(None, args, value)
            await release.wait()

        @hooks.trace_generation_window
        async def evaluate(args):
            value = sample()
            task = hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value])
            hooks.seal_generation_batch()
            return await task

        evaluation = asyncio.create_task(evaluate(args))
        await generated.wait()
        (record,) = spans(root, "rl.rollout.generate")
        assert attrs(record)["rl.purpose"] == "evaluation"
        assert not evaluation.done()
        release.set()
        await evaluation

    asyncio.run(execute())


def test_sglang_submission_emits_one_generation_batch_with_streaming_rewards(run, monkeypatch):
    rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)
    from relax.utils.types import Sample

    args, root = run
    args.custom_generate_function_path = None
    args.group_rm = args.partial_rollout = False

    async def execute():
        release = asyncio.Event()
        generated = asyncio.Event()
        count = 0
        state = object.__new__(rollout.GenerateState)
        state.args = args
        state.sampling_params = {}
        state.semaphore = asyncio.Semaphore(128)
        state.dp_rank_context = nullcontext
        state.reset()

        async def generate(args, sample, params, evaluation=False):
            nonlocal count
            sample.status = Sample.Status.COMPLETED
            sample.weight_versions = ["1"]
            count += 1
            if count == 128:
                generated.set()
            return sample

        @hooks.trace_reward
        async def reward(args, sample):
            await release.wait()
            return 0.0

        monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(rollout, "generate", generate)
        monkeypatch.setattr(rollout, "async_rm", reward)
        monkeypatch.setattr(rollout.opd, "is_opd_enabled", lambda *a, **kw: False)
        state.submit_generate_tasks([[Sample() for _ in range(8)] for _ in range(16)])
        await generated.wait()
        (record,) = spans(root, "rl.rollout.generate")
        assert attrs(record)["rl.batch.size"] == "128"
        assert record["name"] == "rl.rollout.generate"
        release.set()
        await asyncio.gather(*state.pendings)

    asyncio.run(execute())
    assert len([s for s in spans(root) if s["name"] == "rl.rollout.generate"]) == 1


def test_real_batched_reward_entrypoint_has_one_parent(run, monkeypatch):
    pytest.importorskip("ray")
    from relax.engine import rewards

    args, root = run

    async def execute(args, sample, **kwargs):
        hooks.record_reward_route(sample, "math", "rule")
        return 1.0

    monkeypatch.setattr(rewards, "_get_reward_executor", lambda args: SimpleNamespace(execute=execute))
    assert asyncio.run(rewards.batched_async_rm(args, [sample(), sample()])) == [1.0, 1.0]
    (record,) = spans(root, "rl.reward.batch")
    assert not record.get("parentSpanId")
    assert len(artifact(record, "rl.reward.results")) == 2


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
    assert not spans(root, "rl.data.filter.sample")
    for record in spans(root, "rl.data.filter"):
        assert record.get("status", {}).get("code", 0) == 0
        results = artifact(record, "rl.filter.results")
        assert len({r["export_id"] for r in results}) == 2
        assert all(r["decision"] == "drop" for r in results)


def test_transfer_backfill_keeps_source_iteration_and_real_ack(run):
    args, root = run

    @hooks.trace_transfer
    async def transfer(args, samples, batch_count, rollout_id, client):
        return "ack"

    member = sample()
    with tracing.iteration_context(4):
        assert asyncio.run(transfer(args, [[member]], 0, 3, None)) == "ack"
    record = spans(root, "rl.data.transfer")[0]
    assert attrs(record)["rl.step.id"] == "iteration:4"
    assert attrs(record)["rl.transfer.target.step_id"] == "iteration:3"
    assert artifact(record, "rl.transfer.data")["receive_ack_time"] > 0
    assert not spans(root, "rl.data.transfer.sample")
    assert member.metadata["_rl_source"]["span_id"] == record["spanId"]
    assert artifact(record, "rl.transfer.data")["members"] == [{"export_id": hooks.export_id(member)}]


def test_failed_transfer_does_not_invent_ack(run):
    args, root = run

    @hooks.trace_transfer
    async def transfer(args, samples, batch_count, rollout_id, client):
        raise TimeoutError("no ACK")

    with pytest.raises(TimeoutError):
        asyncio.run(transfer(args, [[sample()]], 0, 1, None))
    assert "receive_ack_time" not in artifact(spans(root, "rl.data.transfer")[0], "rl.transfer.data")


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
    from relax.agentic.pipeline.runtime import _build_session_specs
    from relax.agentic.runner.ipc import SessionInput
    from relax.utils.types import Sample

    (spec,) = _build_session_specs(
        [Sample(prompt="task", prompt_id="question-A")], session_ids=["session"], include_input_payload=True
    )
    session_input = SessionInput(
        "session", "train", "group", input_payload=spec.input_payload, prompt_id=spec.prompt_id
    )
    assert session_input.to_agent_payload() == {"messages": [{"role": "user", "content": "task"}]}
    observation = AgentTrace(args, session, session_input)
    identity = observation.span.context()
    observation.assign(0)
    observation.assign(1)
    observation.execution_end(None)
    observation.capture_states(forest)
    exports = [
        (
            "a",
            {"name": "first", "sample_payload": Sample(sample_export_id=11, response_length=3, reward=0.0).to_dict()},
        ),
        ("c", {"name": "second", "sample_payload": Sample(sample_export_id=12, response_length=3).to_dict()}),
    ]
    observation.capture_exports(exports, 1.0)
    observation.finish()

    @hooks.trace_reward
    async def score(args, sample):
        return sample.sample_export_id / 10

    for _, payload in exports:
        asyncio.run(score(args, Sample.from_dict(payload["sample_payload"])))
    (detail,) = spans(root, "rl.rollout.sample")
    scores = spans(root, "rl.reward.score")
    assert attrs(detail)["rl.prompt.id"] == spec.prompt_id == "question-A"
    assert {(attrs(s)["rl.prompt.id"], attrs(s)["rl.sample.id"], attrs(s)["rl.export.id"]) for s in scores} == {
        ("question-A", "session", "11"),
        ("question-A", "session", "12"),
    }
    rows = [row for b in spans(root, "rl.reward.batch") for row in artifact(b, "rl.reward.results")]
    assert {(r["prompt_id"], r["sample_id"], r["export_id"], r["value"]) for r in rows} == {
        ("question-A", "session", "11", 1.1),
        ("question-A", "session", "12", 1.2),
    }
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


@pytest.mark.parametrize("identity_kind", ["export", "index"])
@pytest.mark.parametrize("masked_value", [4.0, float("nan")])
@pytest.mark.parametrize("full_advantage", [False, True])
def test_training_rpc_retains_each_cp_shard_on_its_worker(
    run, monkeypatch, masked_value, full_advantage, identity_kind
):
    import time

    import torch

    from relax.utils import rl_train_trace as train

    args, root = run
    args.advantage_estimator = "grpo"
    args.use_agentic_rollout = identity_kind == "export"
    identity_field = "sample_export_ids" if identity_kind == "export" else "sample_indices"
    result_key = "export_id" if identity_kind == "export" else "sample_index"
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
            identity_field: [23],
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
                train.prepare_batch(data)
                # Device tensors cannot be read on the host; use the CPU snapshot
                # retained before the existing batch-to-device conversion.
                data[identity_field] = torch.tensor([23], device="meta")
                train.capture(args, data, masks=[torch.tensor([1.0, 0.0, 1.0, 0.0])])
                train.optimizer_result(False, "zero_effective_tokens")
        time.sleep(0.01)  # checkpoint/offload outside the observed window
        cp["rank"] += 10  # live group queries are no longer valid after offload

    with tracing.stage(args, "rl.actor.train_step", iteration=0, attributes={"rl.rank.world_size": 2}) as dispatch:
        for rank in (0, 1):
            cp["rank"] = rank
            worker = SimpleNamespace(args=args, role="actor", _rank=rank, _world_size=2)
            assert execute(worker, 0, trace_headers=dispatch.headers()) is None
    rank_spans = {r["spanId"]: r for r in spans(root, "rl.actor.train_rank")}
    stages = spans(root, "rl.actor.train")
    assert len(stages) == len(rank_spans) == 2
    assert not spans(root, "rl.actor.train.sample")
    for record in stages:
        rank = int(attrs(record)["rl.rank.global"])
        assert attrs(record)["rl.timing.rank.cp"] == str(rank)
        assert record["traceId"] == dispatch.context()["trace_id"]
        assert attrs(rank_spans[record["parentSpanId"]])["rl.rank.global"] == str(rank)
        assert attrs(record)["rl.batch.size"] == "1"
        result = artifact(record, "rl.train.results")
        assert result["updates"] == [
            {"update_id": "generation:actor:0:0", "status": "skipped", "reason": "zero_effective_tokens"}
        ]
        assert result["uses"] == [
            {
                result_key: "23",
                "effective_tokens": 1,
                "update_id": "generation:actor:0:0",
                "used": True,
                "advantage_mean": 2.0 if rank == 0 else 6.0,
            }
        ]
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

    with tracing.stage(args, "rl.actor.train_step", iteration=0) as dispatch:
        with pytest.raises(OSError) as caught:
            execute(
                SimpleNamespace(args=args, role="actor", _rank=0, _world_size=1), 0, trace_headers=dispatch.headers()
            )
    assert caught.value is original
    (record,) = spans(root, "rl.actor.train")
    (worker,) = spans(root, "rl.actor.train_rank")
    assert record["parentSpanId"] == worker["spanId"]
    assert worker["status"]["code"] == 2
    assert record.get("status", {}).get("code", 0) == 0
    assert artifact(record, "rl.train.results")["updates"][0]["status"] == "committed"


@pytest.mark.parametrize(
    "versions, expected", [(["3", "3"], "3"), (["3", "4"], None), ([], None), ([None, "3"], None), (None, None)]
)
@pytest.mark.parametrize("batched", [False, True])
def test_generation_reports_only_confirmed_unique_weight_version(run, versions, expected, batched):
    args, root = run
    value = sample()
    value.weight_versions = versions

    @hooks.trace_generation
    async def generate(state, args, value):
        return value

    async def execute():
        if batched:
            with hooks.generation_batch(args):
                task = hooks.track_generation_task(asyncio.create_task(generate(None, args, value)), [value])
            return await task
        return await generate(None, args, value)

    assert asyncio.run(execute()) is value
    assert attrs(spans(root)[0]).get("rl.model.version") == expected


def test_complete_sample_continues_across_windows_and_keeps_input(run):
    import copy

    args, root = run
    value = sample()
    value.prompt_id = "question-before-resume"
    value.prompt = [{"role": "user", "content": "题目" * 10000}]
    value.response = "partial"
    value.status = SimpleNamespace(name="ABORTED")

    headers = []

    @hooks.trace_generation
    async def generate(state, args, value):
        headers.append(tracing.inference_headers())
        return value

    async def window(value, iteration):
        with tracing.iteration_context(iteration), hooks.generation_batch(args):
            task = hooks.track_generation_task(asyncio.create_task(generate(None, args, value)), [value])
        await task

    asyncio.run(window(value, 0))
    assert not spans(root, "rl.rollout.sample")
    (first,) = spans(root, "rl.rollout.generate")
    resumed = copy.deepcopy(value)  # A queue may deserialize into a different Python object.
    resumed.status = SimpleNamespace(name="PENDING")

    @hooks.trace_generation
    async def finish(state, args, value):
        headers.append(tracing.inference_headers())
        value.response = "完整答案" * 10000
        value.status = SimpleNamespace(name="COMPLETED")
        return value

    async def second():
        with tracing.iteration_context(1), hooks.generation_batch(args):
            task = hooks.track_generation_task(asyncio.create_task(finish(None, args, resumed)), [resumed])
        await task

    asyncio.run(second())
    (detail,) = spans(root, "rl.rollout.sample")
    assert detail["parentSpanId"] == first["spanId"]
    assert detail["traceId"] == first["traceId"]
    assert "rl.step.id" not in attrs(detail)
    assert len(artifact(detail, "rl.sample.memberships")) == 2
    assert artifact(detail, "rl.sample.input") == value.prompt
    assert attrs(detail)["rl.prompt.id"] == resumed.prompt_id == "question-before-resume"
    assert artifact(detail, "rl.sample.response")[0]["text"] == resumed.response
    assert len(detail["links"]) == 1

    assert len(headers) == 2 and headers[0]["traceparent"] == headers[1]["traceparent"]


def test_custom_batch_reward_keeps_results_without_synthetic_children(run):
    args, root = run
    values = [sample(), sample()]
    values[0].metadata["prompt_id"] = 0

    @hooks.trace_batched_reward
    async def reward(args, values):
        return [0.0, 1.0]

    assert asyncio.run(reward(args, values)) == [0.0, 1.0]
    (batch,) = spans(root, "rl.reward.batch")
    assert not spans(root, "rl.reward.score")
    assert attrs(batch)["rl.batch.size"] == "2"
    rows = artifact(batch, "rl.reward.results")
    assert [row["value"] for row in rows] == [0.0, 1.0]
    assert rows[0]["prompt_id"] == "0"
    assert "prompt_id" not in rows[1]
    assert all(value.metadata["_rl_source"]["span_id"] == batch["spanId"] for value in values)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_agent_group_owns_samples_and_their_agent_runs(run, startup_failure):
    from relax.utils.rl_agent_trace import AgentTrace, trace_agentic_group

    args, root = run
    args.agent_command = "agent.py"
    traces = []
    original = RuntimeError("launcher failed")

    @trace_agentic_group
    async def start(owner, mode, group_id, specs):
        for index, spec in enumerate(specs):
            if startup_failure and index == 1:
                traces[0].finish(original)  # start_group awaits existing cleanup before propagating.
                raise original
            session = SimpleNamespace(
                session_id=spec.session_id,
                observation=SimpleNamespace(calls={}),
                session_sampling_params={},
                forest=None,
                group=SimpleNamespace(
                    rollout_mode=mode, group_id=group_id, result_cells={s.session_id: None for s in specs}
                ),
            )
            traces.append(
                AgentTrace(
                    args, session, SimpleNamespace(input_payload={"messages": [{"role": "user", "content": "task"}]})
                )
            )

    specs = [SimpleNamespace(session_id="s1"), SimpleNamespace(session_id="s2")]
    if startup_failure:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(start(SimpleNamespace(args=args), "train", "g", specs))
        assert caught.value is original
    else:
        asyncio.run(start(SimpleNamespace(args=args), "train", "g", specs))
        traces[0].assign(0)
        traces[0].assign(1)
        traces[0].finish()
        assert not spans(root, "rl.rollout.generate")
        traces[1].assign(1)
        traces[1].finish()
        traces[1].finish()  # Idempotent terminal notification.
    (batch,) = spans(root, "rl.rollout.generate")
    details = spans(root, "rl.rollout.sample")
    agents = spans(root, "rl.agent.run")
    assert len(details) == len(agents) == (1 if startup_failure else 2)
    assert attrs(batch)["rl.batch.size"] == "2"
    assert batch.get("status", {}).get("code", 0) == (2 if startup_failure else 0)
    assert all(d["parentSpanId"] == batch["spanId"] and d["traceId"] == batch["traceId"] for d in details)
    assert all(a["traceId"] == batch["traceId"] for a in agents)
    assert {a["parentSpanId"] for a in agents} == {d["spanId"] for d in details}


def test_bad_custom_scorer_shape_does_not_replace_business_return(run):
    args, root = run
    value = [1.0]

    @hooks.trace_batched_reward
    async def score(args, samples):
        return value

    assert asyncio.run(score(args, [sample(), sample()])) is value
    (batch,) = spans(root, "rl.reward.batch")
    assert "relax.observation.partial" in attrs(batch)


def test_discarded_partial_sample_preserves_known_response(run):
    from relax.utils.rl_sample_trace import discard_samples

    args, root = run
    value = sample()
    value.prompt, value.response = "question", "unfinished answer"
    value.status = SimpleNamespace(name="ABORTED")

    @hooks.trace_generation
    async def generate(state, args, value):
        return value

    asyncio.run(generate(None, args, value))
    assert not spans(root, "rl.rollout.sample")
    discard_samples(args, [value])
    (detail,) = spans(root, "rl.rollout.sample")
    assert attrs(detail)["rl.sample.cancelled"] is True
    assert artifact(detail, "rl.sample.response")[0]["text"] == value.response
    assert artifact(detail, "rl.sample.response")[0]["status"] == "aborted"


@pytest.mark.parametrize("mutable", [False, True])
def test_custom_batch_reward_keeps_final_values_and_real_inner_calls(run, mutable):
    args, root = run
    values = [sample(), sample()]

    @hooks.trace_reward
    async def inner(args, value):
        return {"score": 1.0} if mutable else 1.0

    @hooks.trace_batched_reward
    async def outer(args, values):
        results = await asyncio.gather(*(inner(args, value) for value in values))
        if mutable:
            for result in results:
                result["score"] *= 2
            return results
        return [2 * result for result in results]

    expected = [{"score": 2.0}] * 2 if mutable else [2.0, 2.0]
    assert asyncio.run(outer(args, values)) == expected
    (batch,) = spans(root, "rl.reward.batch")
    assert [row["value"] for row in artifact(batch, "rl.reward.results")] == expected
    children = spans(root, "rl.reward.score")
    assert len(children) == 2
    intermediate = {"score": 1.0} if mutable else 1.0
    assert all(artifact(child, "rl.reward.results")[0]["value"] == intermediate for child in children)
    assert all(attrs(child)["rl.sample.timing"] == "operation" for child in children)
    assert all(value.metadata["_rl_source"]["span_id"] == batch["spanId"] for value in values)


@pytest.mark.parametrize("role", ["actor", "teacher"])
@pytest.mark.parametrize("success", [True, False])
def test_opd_forward_uses_one_request_span_and_reports_result(run, role, success):
    args, root = run
    value = sample()
    value.response_length = 2
    owner = SimpleNamespace(args=args, topk_worker=None)
    source = tracing.Stage(args, "rl.rollout.generate", attributes={"rl.batch.id": "source"})
    value.metadata["_rl_source"] = source.context()
    source.end()

    @hooks.trace_opd_forward(role)
    async def forward(owner, value):
        return success

    assert asyncio.run(forward(owner, value)) is success
    (record,) = spans(root, f"rl.{role}.forward")
    assert not spans(root, f"rl.{role}.forward.sample")
    assert attrs(record)["rl.execution.side"] == "rollout"
    assert not record.get("parentSpanId")
    assert attrs(record)["rl.batch.size"] == "1"
    assert attrs(record)["rl.export.id"] == hooks.export_id(value)
    assert record["links"][0]["spanId"] == source.context()["span_id"]
    assert artifact(record, "rl.forward.results")[0]["status"] == ("valid" if success else "invalid")
    assert record.get("status", {}).get("code", 0) == (0 if success else 2)


def test_validator_rejects_synthetic_result_spans_in_new_scope(run):
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    parent = tracing.Stage(args, "rl.data.filter", attributes={"rl.batch.id": "batch"})
    parent.result_detail("rl.filter.results", {"sample_index": "1", "decision": "keep"})
    parent.end()
    with pytest.raises(ValueError, match="Result rows must be stored on the batch"):
        validate(root)


def test_validator_rejects_reward_batch_parented_to_sample(run):
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    with tracing.stage(args, "rl.rollout.generate", attributes={"rl.batch.id": "rollout"}) as rollout:
        with tracing.stage(
            args,
            "rl.rollout.sample",
            parent=rollout,
            attributes={"rl.batch.id": "rollout", "rl.sample.timing": "operation"},
        ) as detail:
            with tracing.stage(args, "rl.reward.batch", parent=detail, attributes={"rl.batch.id": "reward"}):
                pass
    with pytest.raises(ValueError, match="direct child of rl.rollout.generate"):
        validate(root)


def test_inference_headers_use_explicit_parent_without_mutating_routing(run, monkeypatch):
    from opentelemetry.trace import get_current_span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    args, _ = run
    headers = {"X-SMG-Routing-Key": "session-1", "TraceParent": "stale", "tracestate": "stale=state"}
    assert tracing.inference_headers(headers) is headers
    parent = tracing.Stage(args, "rl.rollout.sample")
    try:
        with tracing.inference_context(parent):
            carrier = tracing.inference_headers(headers)
            context = get_current_span(TraceContextTextMapPropagator().extract(carrier)).get_span_context()
            assert context.trace_id == parent.span.get_span_context().trace_id
            assert context.span_id == parent.span.get_span_context().span_id
            assert carrier["X-SMG-Routing-Key"] == "session-1"
            assert "TraceParent" not in carrier and "tracestate" not in carrier
            assert headers["TraceParent"] == "stale"
            with tracing.inference_context(None):
                assert tracing.inference_headers() is None
            assert tracing.inference_headers()["traceparent"] == carrier["traceparent"]
            monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
            assert tracing.inference_headers(headers) is headers
            monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
        assert tracing.inference_headers() is None
    finally:
        parent.end()
    with tracing.inference_context(parent):
        assert tracing.inference_headers() is None


@pytest.mark.parametrize("batched", [False, True])
def test_concurrent_generation_requests_inherit_their_own_sample(run, batched):
    args, root = run
    carriers = {}

    async def execute():
        ready = asyncio.Event()
        entered = 0

        @hooks.trace_generation
        async def generate(state, args, value):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            carriers[value.index] = tracing.inference_headers({"X-SMG-Routing-Key": str(value.index)})
            await asyncio.sleep(0)
            assert tracing.inference_headers()["traceparent"] == carriers[value.index]["traceparent"]
            return value

        values = [sample(), sample()]
        for index, value in enumerate(values):
            value.index = index
        if batched:
            with hooks.generation_batch(args):
                tasks = [
                    hooks.track_generation_task(asyncio.create_task(generate(None, args, value)), [value])
                    for value in values
                ]
        else:
            tasks = [asyncio.create_task(generate(None, args, value)) for value in values]
        await asyncio.gather(*tasks)
        assert tracing.inference_headers() is None

    asyncio.run(execute())
    records = spans(root, "rl.rollout.sample")
    assert len(records) == 2
    for record in records:
        index = int(attrs(record)["rl.sample.index"])
        assert carriers[index]["traceparent"].split("-")[1:3] == [record["traceId"], record["spanId"]]
        assert carriers[index]["X-SMG-Routing-Key"] == str(index)


def test_generation_context_resets_after_failure(run):
    args, root = run

    @hooks.trace_generation
    async def generate(state, args, value):
        assert tracing.inference_headers()["traceparent"]
        raise RuntimeError("generation failed")

    async def execute():
        with pytest.raises(RuntimeError, match="generation failed"):
            await generate(None, args, sample())
        assert tracing.inference_headers() is None

    asyncio.run(execute())
    assert spans(root, "rl.rollout.sample")[0]["status"]["code"] == 2


def test_dispatch_preserves_async_refs_errors_and_backend_arguments(run, monkeypatch):
    from concurrent.futures import Future

    from relax.utils.rl_train_trace import dispatch_training

    args, root = run
    futures = [Future(), Future()]
    refs = [SimpleNamespace(future=lambda future=future: future) for future in futures]
    calls = []

    def submit(index, step, **kwargs):
        calls.append((step, kwargs))
        return refs[index]

    group = SimpleNamespace(
        args=args,
        role="critic",
        _actor_handlers=[SimpleNamespace(train=SimpleNamespace(remote=partial(submit, i))) for i in range(2)],
    )
    assert dispatch_training(group, "train", 3) == refs
    assert calls[0] == calls[1] and "traceparent" in calls[0][1]["trace_headers"]
    futures[0].set_exception(ValueError("rank 0 failed"))
    assert not spans(root)
    futures[1].set_result(None)
    (parent,) = spans(root, "rl.critic.train_step")
    assert parent["status"]["code"] == 2
    assert "rank 0 failed" in parent["status"]["message"]
    args.train_backend = "fsdp"
    assert dispatch_training(group, "train", 4) == refs
    assert calls[-2:] == [(4, {}), (4, {})]
    args.train_backend = "megatron"
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "0")
    assert dispatch_training(group, "train", 5) == refs
    assert calls[-2:] == [(5, {}), (5, {})]
    assert len(spans(root)) == 1


@pytest.mark.parametrize("delegated", [False, True], ids=["direct", "hybrid-delegation"])
def test_rank_weight_sync_preserves_sender_link_and_independent_startup(run, monkeypatch, delegated):
    import torch.distributed as dist

    from relax.utils import rl_train_trace as train
    from relax.utils import rl_weights_trace as weights
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    monkeypatch.setitem(sys.modules, "relax.utils.distributed_utils", SimpleNamespace(get_gloo_group=lambda: object()))
    rank = 0
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)

    @weights.trace_weight_call
    def publish(worker):
        weights.begin_weights(worker)
        weights.weights_delivered(worker)

    @weights.trace_weight_sync
    def update_weights_fully_async(worker, rollout_id, rollout_only=False):
        publish(worker)

    @weights.trace_weight_sync
    def sync(worker):
        if delegated:
            update_weights_fully_async(worker, 0, rollout_only=True)
        else:
            publish(worker)

    @train.training_rpc
    @hooks.trace_iteration
    def execute(worker, rollout_id):
        sync(worker)

    worker = SimpleNamespace(
        args=args,
        role="actor",
        _rank=0,
        _world_size=2,
        weight_version=4,
        rollout_engines=[SimpleNamespace(_actor_id=b"target")],
    )
    sync(worker)
    with tracing.stage(args, "rl.actor.train_step", iteration=1, attributes={"rl.rank.world_size": 2}) as dispatch:
        for rank in range(2):
            worker._rank = rank
            execute(worker, 1, trace_headers=dispatch.headers())
    workers = {attrs(s)["rl.rank.global"]: s for s in spans(root, "rl.actor.train_rank")}
    standalone, first, second = spans(root, "rl.weight.sync")
    assert not standalone.get("parentSpanId")
    assert first["parentSpanId"] == workers["0"]["spanId"]
    assert second["parentSpanId"] == workers["1"]["spanId"]
    deliveries, loads = spans(root, "rl.actor.publish_weights"), spans(root, "rl.rollout.load_weights")
    assert len(deliveries) == len(loads) == 2
    for stage, delivery, activation in zip((standalone, first), deliveries, loads, strict=True):
        assert stage["spanId"] == delivery["parentSpanId"] == activation["parentSpanId"]
        assert stage["traceId"] == delivery["traceId"] == activation["traceId"]
        assert activation["links"][0]["spanId"] == delivery["spanId"]
        assert attrs(delivery)["rl.weight.transfer.id"] == attrs(activation)["rl.weight.transfer.id"]
    validate(root)


def test_advantage_excludes_pipeline_noop_without_changing_call(run, monkeypatch):
    from relax.utils import rl_train_trace as train

    args, root = run
    args.advantage_estimator = "grpo"
    rank = {"pp": 0}
    mpu = SimpleNamespace(
        is_pipeline_last_stage=lambda **kw: rank["pp"] == 3,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda **kw: 0,
        get_pipeline_model_parallel_rank=lambda: rank["pp"],
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    calls = []

    @hooks.trace_advantage
    def compute(args, data):
        calls.append(rank["pp"])
        return "done" if rank["pp"] == 3 else None

    @train.training_rpc
    def execute(owner, rollout_id):
        compute(args, {"rollout_log_probs": [1.0]})

    with tracing.stage(args, "rl.actor.train_step", iteration=0) as dispatch:
        for i in (0, 3):
            rank["pp"] = i
            execute(
                SimpleNamespace(args=args, role="actor", _rank=i, _world_size=4), 0, trace_headers=dispatch.headers()
            )
    workers = spans(root, "rl.actor.train_rank")
    assert len(workers) == 2
    (advantage,) = spans(root, "rl.advantage.compute")
    assert attrs(advantage)["rl.rank.global"] == "3"
    assert advantage["parentSpanId"] == workers[1]["spanId"]
    assert calls == [0, 3]


def test_async_training_starts_at_train_call_not_rpc_entry(run, monkeypatch):
    from relax.utils import rl_train_trace as train

    args, root = run
    args.advantage_estimator = "grpo"
    clock = {"now": 100}
    monkeypatch.setattr(train.time, "time_ns", lambda: clock["now"])
    mpu = SimpleNamespace(
        is_pipeline_last_stage=lambda **kw: True,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda **kw: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))

    @train.training_rpc
    def train_async(owner, rollout_id):
        clock["now"] = 500  # data preparation before entering actual train
        with train.window(args, "actor", rollout_id):
            clock["now"] = 600
        clock["now"] = 900  # checkpoint/export after train

    with tracing.stage(args, "rl.actor.train_step", iteration=0) as dispatch:
        train_async(
            SimpleNamespace(args=args, role="actor", _rank=0, _world_size=1), 0, trace_headers=dispatch.headers()
        )
    (operation,) = spans(root, "rl.actor.train")
    assert int(operation["startTimeUnixNano"]) == 500
    assert int(operation["endTimeUnixNano"]) == 600
    (worker,) = spans(root, "rl.actor.train_rank")
    assert int(worker["endTimeUnixNano"]) == 900


def test_filter_excludes_result_serialization_time(run, monkeypatch):
    args, root = run
    clock = {"now": 100}
    monkeypatch.setattr(tracing.time, "time_ns", lambda: clock["now"])
    original = hooks._record_filter_results

    def record(*args):
        clock["now"] = 900
        return original(*args)

    monkeypatch.setattr(hooks, "_record_filter_results", record)

    @hooks.trace_filter
    def apply(fn, args, values):
        clock["now"] = 200
        return SimpleNamespace(keep=True, reason=None)

    apply(lambda: None, args, [sample()])
    (record,) = spans(root)
    assert int(record["endTimeUnixNano"]) == 200
    assert len(artifact(record, "rl.filter.results")) == 1


def test_custom_reward_batch_times_postprocessing_and_excludes_result_serialization(run, monkeypatch):
    args, root = run
    args.custom_rm_path = "custom.scorer"
    clock = {"now": 100}
    monkeypatch.setattr(tracing.time, "time_ns", lambda: clock["now"])
    original = hooks._record_batched_values

    def record(*args):
        clock["now"] = 900
        return original(*args)

    monkeypatch.setattr(hooks, "_record_batched_values", record)

    @hooks.trace_reward
    async def one(args, value):
        clock["now"] = 200
        return 1.0

    @hooks.trace_batched_reward
    async def batch(args, values):
        result = [await one(args, v) for v in values]
        clock["now"] = 400  # real batch work after individual scoring
        return result  # unchanged values still require the later boundary

    assert asyncio.run(batch(args, [sample()])) == [1.0]
    (record,) = spans(root, "rl.reward.batch")
    assert int(record["endTimeUnixNano"]) == 400
    assert int(spans(root, "rl.reward.score")[0]["endTimeUnixNano"]) == 200


@pytest.mark.parametrize("evaluation", [False, True])
def test_framework_rollout_owns_multiple_submissions(run, evaluation):
    import copy

    from relax.engine.rollout.base_types import call_rollout_fn
    from relax.utils.data.data import Dataset
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    path = root.parent / "questions.jsonl"
    path.write_text('{"text":"What is 2+2?"}\n{"text":"What is 3+3?"}\n')
    dataset = Dataset(
        str(path), None, None, None, record_prompt_id=True, source_split="eval" if evaluation else "train"
    )
    prompt_ids = {s.prompt_id for s in dataset.samples}
    assert len(prompt_ids) == 2 and all(re.fullmatch(r"prompt_[0-9a-f]{64}", p) for p in prompt_ids)
    values = [copy.deepcopy(s) for _ in range(2) for s in dataset.samples]
    for i, value in enumerate(values):
        value.group_index = i % 2
        value.response = ("4", "6", "5", "7")[i]
        value.weight_versions = ["actor-7"]

    @hooks.trace_generation
    async def generate(state, config, value):
        await asyncio.sleep(0)
        return value

    @hooks.trace_reward
    async def score(args, value):
        return float(value.response in ("4", "6"))

    async def pipeline(value):
        await score(args, await generate(None, args, value))

    async def collect():
        for offset in range(0, 4, 2):
            with hooks.generation_batch(args, evaluation=evaluation):
                tasks = [
                    hooks.track_generation_task(asyncio.create_task(pipeline(value)), [value])
                    for value in values[offset : offset + 2]
                ]
            await asyncio.gather(*tasks)
        assert not spans(root, "rl.rollout.generate")
        return values

    call_rollout_fn(lambda *a, **k: asyncio.run(collect()), args, 3, evaluation=evaluation)
    (batch,) = spans(root, "rl.rollout.generate")
    details = spans(root, "rl.rollout.sample")
    assert attrs(batch)["rl.batch.size"] == "4"
    assert attrs(batch)["rl.model.version"] == "actor-7"
    assert attrs(batch)["rl.step.id"] == "iteration:3"
    assert attrs(batch)["rl.purpose"] == ("evaluation" if evaluation else "training")
    assert sorted(attrs(s)["rl.group.id"] for s in details) == ["0", "0", "1", "1"]
    assert all(s["parentSpanId"] == batch["spanId"] and s["traceId"] == batch["traceId"] for s in details)
    assert all(int(s["endTimeUnixNano"]) <= int(batch["endTimeUnixNano"]) for s in details)

    rewards = spans(root, "rl.reward.batch")
    assert len(rewards) == 2
    assert len({attrs(item)["rl.batch.id"] for item in [batch, *rewards]}) == 3
    scores = spans(root, "rl.reward.score")
    assert len(scores) == 4
    answers = {attrs(s)["rl.sample.id"]: attrs(s)["rl.prompt.id"] for s in details}
    assert len(answers) == 4
    assert all(list(answers.values()).count(prompt_id) == 2 for prompt_id in prompt_ids)
    assert all(attrs(s)["rl.prompt.id"] == answers[attrs(s)["rl.sample.id"]] for s in scores)
    for reward in rewards:
        rows = artifact(reward, "rl.reward.results")
        assert {row["prompt_id"] for row in rows} == prompt_ids
        assert all(row["prompt_id"] == answers[row["sample_id"]] for row in rows)
        assert "rl.prompt.id" not in attrs(reward)
        assert (reward["traceId"], reward["parentSpanId"]) == (batch["traceId"], batch["spanId"])
        assert attrs(reward)["rl.step.id"] == "iteration:3"
        assert attrs(reward)["rl.purpose"] == attrs(batch)["rl.purpose"]
        assert attrs(reward)["rl.batch.size"] == "2"
        assert sum(s["parentSpanId"] == reward["spanId"] for s in scores) == 2
    assert {s["links"][0]["spanId"] for s in scores} == {s["spanId"] for s in details}
    validate(root)


@pytest.mark.parametrize("agentic", [False, True], ids=["ordinary", "agentic-resume"])
def test_rollout_operations_use_the_invocation_at_start(run, agentic):
    from relax.engine.filters.base_types import call_dynamic_filter
    from relax.engine.rollout.base_types import call_rollout_fn
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    args.use_agentic_rollout = agentic
    value = sample()
    value.response_length, value.custom_advantage, value.loss_mask = 1, 1.0, [1]
    owner = SimpleNamespace(args=args, custom_advantage_func=sum, topk_worker=None)
    group = SimpleNamespace(samples=[value])

    def accept(*args):
        return SimpleNamespace(keep=True, reason=None)

    @hooks.trace_custom_advantage
    async def advantage(owner, group):
        return True

    async def prefill(owner, sample):
        return True

    @hooks.trace_transfer
    async def ordinary_put(args, samples, count, partition):
        return "ack"

    @hooks.trace_agentic_transfer
    async def agentic_put(**kwargs):
        return "ack"

    async def execute():
        started, resume = asyncio.Event(), asyncio.Event()

        @hooks.trace_opd_forward("teacher")
        async def teacher(owner, sample):
            started.set()
            await resume.wait()
            return True

        async def work():
            await teacher(owner, value)
            assert call_dynamic_filter(accept, [value]).keep
            if agentic:
                assert await advantage(owner, group)
            assert await hooks.trace_opd_forward("actor")(prefill)(owner, value)
            if agentic:
                assert await agentic_put(args=args, batch_samples=[[value]], partition_id=0) == "ack"
            else:
                assert await ordinary_put(args, [[value]], 1, 0) == "ack"

        if not agentic:
            resume.set()
            await work()
            return []
        with tracing.rollout_call(args, 0, False) as first:
            live = first.span
            with hooks.resident_rollout_parent(lambda: live):
                task = asyncio.create_task(work())
            await started.wait()
        with tracing.rollout_call(args, 1, False) as second:
            live = second.span
            resume.set()
            await task

    if agentic:
        asyncio.run(execute())
    else:
        call_rollout_fn(lambda *a, **kw: asyncio.run(execute()), args, 1, evaluation=False)
    roots = {attrs(s)["rl.step.id"]: s for s in spans(root, "rl.rollout.generate")}
    operations = [s for s in spans(root) if s["name"] != "rl.rollout.generate"]
    expected = {"rl.data.filter", "rl.data.transfer", "rl.actor.forward", "rl.teacher.forward"}
    if agentic:
        expected.add("rl.advantage.compute")
    assert sorted(s["name"] for s in operations) == sorted(expected)
    for record in operations:
        name = record["name"]
        step = "iteration:0" if agentic and name == "rl.teacher.forward" else "iteration:1"
        assert (record["traceId"], record["parentSpanId"]) == (roots[step]["traceId"], roots[step]["spanId"])
        assert attrs(record)["rl.step.id"] == step
        assert attrs(record)["rl.execution.side"] == "rollout"
        assert attrs(record)["rl.batch.id"] != attrs(roots[step])["rl.batch.id"]
        if name == "rl.data.transfer":
            assert attrs(record)["rl.transfer.target.step_id"] == "iteration:0"
    validate(root)


def test_megatron_and_sglang_student_forward_keep_distinct_origins(run, monkeypatch):
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    args.use_opd = True
    owner = SimpleNamespace(args=args)
    mpu = SimpleNamespace(
        is_pipeline_last_stage=lambda **kwargs: True,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda **kwargs: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    value = sample()
    value.response_length = 1

    @hooks.trace_forward
    def megatron_forward(owner, data_iterator, num_microbatches, **kwargs):
        return True

    @hooks.trace_opd_forward("actor")
    async def sglang_prefill(owner, sample):
        return True

    with tracing.rollout_call(args, 0, False):
        assert megatron_forward(owner, [SimpleNamespace(rollout_data={"sample_indices": [0]})], 1)
        assert asyncio.run(sglang_prefill(owner, value))
    (rollout,) = spans(root, "rl.rollout.generate")
    training, prefill = spans(root, "rl.actor.forward")
    assert attrs(training)["rl.forward.purpose"] == attrs(prefill)["rl.forward.purpose"] == "distillation"
    assert attrs(training)["rl.execution.side"] == "training" and not training.get("parentSpanId")
    assert attrs(prefill)["rl.execution.side"] == "rollout"
    assert (prefill["traceId"], prefill["parentSpanId"]) == (rollout["traceId"], rollout["spanId"])
    validate(root)


def test_resident_filter_without_active_step_or_in_nested_rollout(run):
    from relax.engine.filters.base_types import call_dynamic_filter

    args, root = run

    async def execute():
        released = asyncio.Event()

        async def later():
            await released.wait()
            with tracing.rollout_call(args, 7, True):
                call_dynamic_filter(lambda values: SimpleNamespace(keep=True, reason=None), [sample()])
            call_dynamic_filter(lambda values: SimpleNamespace(keep=True, reason=None), [sample()])

        with tracing.iteration_context(2), tracing.rollout_call(args, 2, False) as first:
            live = first.span
            with hooks.resident_rollout_parent(lambda: live):
                task = asyncio.create_task(later())
        live = None
        released.set()
        await task

    asyncio.run(execute())
    nested, standalone = spans(root, "rl.data.filter")
    nested_root = next(s for s in spans(root, "rl.rollout.generate") if attrs(s)["rl.step.id"] == "iteration:7")
    assert (nested["traceId"], nested["parentSpanId"]) == (nested_root["traceId"], nested_root["spanId"])
    assert attrs(nested)["rl.purpose"] == "evaluation"
    assert not standalone.get("parentSpanId")
    assert "rl.step.id" not in attrs(standalone) and "rl.execution.side" not in attrs(standalone)


def test_agentic_resident_scoring_belongs_to_rollout_at_submission(run, monkeypatch):
    from relax.agentic.pipeline.reward import RewardDomain
    from relax.engine import rewards
    from scripts.tools.validate_rl_traces import validate

    args, root = run
    args.use_agentic_rollout = True
    args.group_rm = False
    args.agentic_custom_advantage_path = None
    domain = RewardDomain(args, "train", None)
    values = [sample() for _ in range(3)]
    for value in values:
        value.reward = None

    async def execute():
        started, resume = asyncio.Event(), asyncio.Event()

        async def score(args, value):
            if value is values[0]:
                started.set()
                await resume.wait()
            return 1.0

        monkeypatch.setattr(rewards, "_get_reward_executor", lambda args: SimpleNamespace(execute=score))

        async def score_exports():
            await domain._score_sample(values[0])
            await domain.score_group(SimpleNamespace(samples=values[1:]))

        with tracing.rollout_call(args, 0, False) as first:
            owner = first.span
            with hooks.resident_reward_parent(lambda: owner):
                task = asyncio.create_task(score_exports())
            await started.wait()
        with tracing.rollout_call(args, 1, False) as second:
            owner = second.span
            resume.set()
            await task
            await domain._score_sample(values[0])  # An export with an existing reward skips the scorer.

    asyncio.run(execute())
    assert [value.reward for value in values] == [1.0, 1.0, 1.0]
    roots = {attrs(s)["rl.step.id"]: s for s in spans(root, "rl.rollout.generate")}
    batches = spans(root, "rl.reward.batch")
    assert len(batches) == 2
    for size, name in [("1", "iteration:0"), ("2", "iteration:1")]:
        (batch,) = [s for s in batches if attrs(s)["rl.batch.size"] == size]
        assert attrs(batch)["rl.step.id"] == name
        assert (batch["traceId"], batch["parentSpanId"]) == (roots[name]["traceId"], roots[name]["spanId"])
        assert sum(s["parentSpanId"] == batch["spanId"] for s in spans(root, "rl.reward.score")) == int(size)
    validate(root)


def test_agent_callback_preserves_its_call_or_falls_back_to_its_session(run):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    from relax.engine.rollout.base_types import call_rollout_fn
    from relax.utils.rl_agent_trace import AgentTrace

    args, root = run
    args.use_agentic_rollout = True
    args.agent_command = "example"
    external = TracerProvider()
    exported = InMemorySpanExporter()
    external.add_span_processor(SimpleSpanProcessor(exported))
    tracer = external.get_tracer("test.external")
    propagator = TraceContextTextMapPropagator()

    async def simulate_session(sid):
        rollout = tracing.current_rollout()
        rollout.admit([sid])
        session = SimpleNamespace(
            session_id=sid,
            observation=SimpleNamespace(calls={}),
            session_sampling_params={},
            forest=None,
            group=SimpleNamespace(rollout_mode="train", group_id="prompt", result_cells={"A": None, "B": None}),
        )
        # Ray crosses a serialization boundary, including when preparation is prefetched.
        origin = json.loads(json.dumps(rollout.carrier()))
        agent = AgentTrace(args, session, SimpleNamespace(input_payload={}, rollout_trace=origin))
        launch = agent.span.headers()
        with tracer.start_as_current_span("llm.call", context=propagator.extract(launch)) as app:
            callback = {}
            propagator.inject(callback)
            callback["tracestate"] = "vendor=local"
            foreign = dict(traceparent="00-" + "1" * 32 + "-" + "2" * 16 + "-01", tracestate="wrong=foreign")
            for incoming, forwarded in [
                (callback, True),
                (None, False),
                (foreign, False),
                ({"traceparent": "bad", "tracestate": "wrong=bad"}, False),
            ]:
                with tracing.inference_context(agent.span, incoming):
                    await asyncio.sleep(0)
                    carrier = tracing.inference_headers({"X-SMG-Routing-Key": sid})
                    assert carrier["X-SMG-Routing-Key"] == sid
                    assert carrier.get("tracestate") == ("vendor=local" if forwarded else None)
                    with tracer.start_as_current_span("Req", context=propagator.extract(carrier)) as req:
                        expected = app if forwarded else agent.span.span
                        assert req.parent.span_id == expected.get_span_context().span_id
                        assert req.parent.trace_id == rollout.span.span.get_span_context().trace_id
            app.set_attribute("session", sid)
        agent.assign(0)
        agent.finish()

    async def collect():
        await asyncio.gather(simulate_session("A"), simulate_session("B"))
        return []

    try:
        call_rollout_fn(lambda *a, **k: asyncio.run(collect()), args, 0, evaluation=False)
        (batch,) = spans(root, "rl.rollout.generate")
        agents = {attrs(s)["rl.session.id"]: s for s in spans(root, "rl.agent.run")}
        details = {attrs(s)["rl.session.id"]: s for s in spans(root, "rl.rollout.sample")}
        calls = [s for s in exported.get_finished_spans() if s.name == "llm.call"]
        assert len(calls) == 2 and attrs(batch)["rl.batch.size"] == "2"
        for call in calls:
            sid = call.attributes["session"]
            assert details[sid]["parentSpanId"] == batch["spanId"]
            assert agents[sid]["parentSpanId"] == details[sid]["spanId"]
            assert call.parent.span_id == int(agents[sid]["spanId"], 16)
    finally:
        external.shutdown()


def test_prefetched_agent_keeps_first_parent_and_links_consumer(run):
    from relax.utils.rl_agent_trace import AgentTrace

    args, root = run
    args.agent_command = "agent.py"
    args.use_agentic_rollout = True
    with tracing.rollout_call(args, 0, False):
        origin = tracing.current_rollout()
        origin.admit(["prefetched"])
        session = SimpleNamespace(
            session_id="prefetched",
            session_sampling_params={},
            observation=SimpleNamespace(calls={}),
            forest=None,
            group=SimpleNamespace(group_id="g", result_cells={}, rollout_mode="train"),
        )
        origin_carrier = origin.carrier()
    agent = AgentTrace(args, session, SimpleNamespace(input_payload={}, rollout_trace=origin_carrier))
    with tracing.rollout_call(args, 1, False):
        consumer = tracing.current_rollout()
        consumer.admit(["prefetched"])
        agent.assign(1, consumer.carrier())
        agent.assign(1, consumer.carrier())
        agent.finish()
    roots = {attrs(s)["rl.step.id"]: s for s in spans(root, "rl.rollout.generate")}
    (detail,) = spans(root, "rl.rollout.sample")
    assert detail["parentSpanId"] == roots["iteration:0"]["spanId"]
    assert len(detail["links"]) == 1 and detail["links"][0]["spanId"] == roots["iteration:1"]["spanId"]
    memberships = artifact(detail, "rl.sample.memberships")
    assert memberships == [
        {"batch_id": attrs(roots["iteration:0"])["rl.batch.id"]},
        {"batch_id": attrs(roots["iteration:1"])["rl.batch.id"], "step_id": "iteration:1"},
    ]


def test_rollout_prefetch_counts_late_admission_without_extending_duration(run, monkeypatch):
    args, root = run
    now = {"ns": 100}
    monkeypatch.setattr(tracing.time, "time_ns", lambda: now["ns"])

    async def exercise():
        loop = asyncio.get_running_loop()
        fetch, decision, cleanup = [loop.create_future() for _ in range(3)]
        with tracing.rollout_call(args, 0, False) as origin:
            origin.track_admission(fetch)
            now["ns"] = 200
        assert not spans(root, "rl.rollout.generate")
        now["ns"] = 300
        origin.track_admission(decision, cleanup)
        fetch.set_result(None)
        await asyncio.sleep(0)
        assert not spans(root, "rl.rollout.generate")
        decision.set_result(True)
        origin.admit(["late"])
        await asyncio.sleep(0)
        cleanup.cancel()
        await asyncio.sleep(0)

    asyncio.run(exercise())
    (record,) = spans(root, "rl.rollout.generate")
    assert int(record["endTimeUnixNano"]) == 200
    assert attrs(record)["rl.batch.size"] == "1"


def test_resumed_group_attributes_model_version_to_consuming_rollout(run):
    from relax.engine.rollout.base_types import RolloutFnTrainOutput, call_rollout_fn

    args, root = run
    args.use_agentic_rollout = True
    member = sample()
    member.metadata = {"_rl_sample_id": "resumed", "_rl_session_model_version": "actor-4"}
    with tracing.rollout_call(args, 0, False) as origin:
        origin.admit(["resumed"])

    def consume(*args, **kwargs):
        tracing.current_rollout().admit(["resumed"])
        return RolloutFnTrainOutput(samples=[[member]])

    call_rollout_fn(consume, args, 1, evaluation=False)
    roots = {attrs(s)["rl.step.id"]: attrs(s) for s in spans(root, "rl.rollout.generate")}
    assert "rl.model.version" not in roots["iteration:0"]
    assert roots["iteration:1"]["rl.model.version"] == "actor-4"


def test_role_startup_across_threads_and_replacement_processes(run):
    from concurrent.futures import ThreadPoolExecutor

    from relax.utils import rl_service_trace as lifecycle

    args, root = run
    args.resource = {"actor": (1, 4), "rollout": (1, 4)}
    with tracing.iteration_context(1), tracing.stage(args, "rl.actor.train_step"):
        with lifecycle.operation(args, "rl.service.startup"):

            def allocate(role):
                return role, None, "no placement group" if role == "rollout" else None

            with ThreadPoolExecutor(max_workers=2) as pool:
                prepare = lifecycle.phase_task(args, "allocate resources", allocate)
                assert list(pool.map(prepare, args.resource))[-1][2] == "no placement group"
                start = lifecycle.phase_task(
                    args, "deploy", lambda role: (role, lifecycle.deployment_config(args, role), None)
                )
                _, bound, _ = pool.submit(start, "actor").result()

    # Serve retains the bound arguments across constructor retries and process replacements.
    program = """
import json, os, sys
from argparse import Namespace
from relax.utils.rl_service_trace import trace_replica_init

class Replica:
    @trace_replica_init
    def __init__(self, config):
        if sys.argv[2] == "kill":
            os._exit(9)
        if sys.argv[2] == "raise":
            raise RuntimeError("constructor failed")

Replica(Namespace(**json.loads(sys.argv[1])))
"""
    for mode, code in (("kill", 9), ("raise", 1), ("ready", 0)):
        result = subprocess.run([sys.executable, "-c", program, json.dumps(vars(bound)), mode], capture_output=True)
        assert result.returncode == code, result.stderr
    records = spans(root)
    (train,) = [s for s in records if s["name"] == "rl.actor.train_step"]
    (startup,) = [s for s in records if s["name"] == "start services"]
    (start,) = [s for s in records if s["name"] == "start actor"]
    prepare = [s for s in records if s["name"] == "rl.service.prepare"]
    assert not startup.get("parentSpanId") and train["traceId"] != startup["traceId"]
    assert all(
        s["parentSpanId"] == startup["spanId"] and s["traceId"] == startup["traceId"] for s in [*prepare, start]
    )
    assert next(s for s in prepare if attrs(s)["rl.service.role"] == "rollout")["status"]["code"] == 2
    assert attrs(start)["rl.producer.role"] == "driver" and attrs(start)["rl.service.gpus"] == "4"
    enters = [s for s in records if s["name"] == "enter actor_service"]
    inits = [s for s in records if s["name"] == "init actor_service"]
    assert len(enters) == len({s["traceId"] for s in enters}) == 3 and len(inits) == 2
    assert all(s["startTimeUnixNano"] == s["endTimeUnixNano"] for s in enters)
    assert {s.get("status", {}).get("code", 0) for s in inits} == {0, 2}
    assert {s["parentSpanId"] for s in enters} >= {s["spanId"] for s in inits}
    assert all(attrs(s)["rl.producer.role"] == "actor_service" for s in enters + inits)
    assert all(attrs(s)["rl.service.deployment.id"] == start["spanId"] for s in enters + inits)
    assert all(s["links"][0]["spanId"] == start["spanId"] for s in inits)
    assert all("rl.step.id" not in attrs(s) for s in records if s is not train)
    assert not hasattr(args, "_rl_service_deployment")


def test_native_megatron_step_inherits_rank_local_training_window(run, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from relax.backends.megatron import telemetry
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
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("native-test")
    # Exercise the production hook with a real independent SDK provider, without installing Megatron/Lens.
    monkeypatch.setattr(telemetry, "_handle", SimpleNamespace(is_exporting=True))
    monkeypatch.setattr(telemetry, "span", lambda group, name, **attrs: tracer.start_as_current_span(name))
    monkeypatch.setattr(telemetry, "flush", lambda: None)
    previous = trace.get_current_span()

    @train.training_rpc
    def execute(owner, rollout_id):
        with train.window(args, "actor", rollout_id):
            with telemetry.observe_step(rollout_id, 0, "actor"):
                with tracer.start_as_current_span("megatron.microbatch.forward"):
                    raise RuntimeError("failed forward")

    with tracing.stage(args, "rl.actor.train_step") as dispatch:
        owner = SimpleNamespace(args=args, role="actor", _rank=0, _world_size=1)
        with pytest.raises(RuntimeError, match="failed forward"):
            execute(owner, 2, trace_headers=dispatch.headers())

    (rank,) = spans(root, "rl.actor.train_rank")
    (operation,) = spans(root, "rl.actor.train")
    native = {s.name: s for s in exporter.get_finished_spans()}
    step = native["relax.train.step"]
    assert operation["parentSpanId"] == rank["spanId"]
    assert operation["traceId"] == f"{step.context.trace_id:032x}"
    assert operation["spanId"] == f"{step.parent.span_id:016x}"
    assert native["megatron.microbatch.forward"].parent == step.context
    assert trace.get_current_span() is previous
    assert train.current_window_span() is None
    with telemetry.observe_step(3, 0, "actor"):
        pass
    expected_parent = previous.get_span_context() if previous.get_span_context().is_valid else None
    assert exporter.get_finished_spans()[-1].parent == expected_parent
    provider.shutdown()


@pytest.mark.parametrize("relax_enabled", [True, False])
def test_native_megatron_resource_preserves_lens_identity(run, monkeypatch, relax_enabled):
    from relax.backends.megatron import telemetry

    args, _ = run
    args.rank, args.world_size = 3, 4
    args.tensor_model_parallel_size = args.pipeline_model_parallel_size = 1
    args.context_parallel_size = args.expert_model_parallel_size = 1
    monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
    monkeypatch.setenv("RELAX_OTEL_ENABLED", str(int(relax_enabled)))
    monkeypatch.setattr(tracing, "_PROCESS_ROLE", (os.getpid(), "actor_rank"))
    captured = {}
    lens_config = SimpleNamespace(run_id="explicit-lens-id", exporter="console", span_groups="step")

    def setup(config, **kwargs):
        assert config is lens_config
        captured.update(kwargs)
        return SimpleNamespace(is_exporting=False, shutdown=lambda: None)

    monkeypatch.setitem(sys.modules, "megatron.core.telemetry.span_groups", SimpleNamespace(MegatronSpanGroup=object))
    monkeypatch.setitem(sys.modules, "megatron.training", SimpleNamespace(global_vars=SimpleNamespace()))
    monkeypatch.setitem(
        sys.modules,
        "nemo.lens",
        SimpleNamespace(NemoLensConfig=SimpleNamespace(from_env=lambda **kwargs: lens_config), setup_telemetry=setup),
    )
    monkeypatch.setattr(telemetry, "atexit", SimpleNamespace(register=lambda callback: None))
    monkeypatch.setattr(telemetry, "_handle", None)
    telemetry.initialize(args)

    resource = captured["resource_attributes"]
    assert (captured["rank"], captured["world_size"]) == (3, 4)
    assert resource["relax.run.id"] == "explicit-lens-id"
    assert ("rl.run.id" in resource) is relax_enabled
    assert ("rl.producer.role" in resource) is relax_enabled
    if relax_enabled:
        assert resource["rl.run.id"] == args.rl_trace_run_id
        assert resource["rl.producer.role"] == "actor_rank"

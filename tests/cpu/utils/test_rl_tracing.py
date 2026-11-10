# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
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
    monkeypatch.delenv("RELAX_OTEL_RUN_ID", raising=False)
    args = Namespace(hf_checkpoint="models/policy", rm_type="math", custom_rm_path=None, reward_key=None)
    tracing.initialize_run(args)
    return args, tmp_path / args.rl_trace_run_id


def spans(root, name=None):
    result = []
    for path in root.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            for resource in json.loads(line)["resourceSpans"]:
                for scope in resource["scopeSpans"]:
                    assert scope["scope"] == {"name": "relax.rl", "version": "0.11.0"}
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
    pytest.importorskip("ray")
    pytest.importorskip("sglang_router")
    from relax.engine.rollout import sglang_rollout as rollout
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
    assert record["name"] == "rl.reward.batch"
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


@pytest.mark.parametrize("identity_kind", ["export", "index"])
@pytest.mark.parametrize("masked_value", [4.0, float("nan")])
@pytest.mark.parametrize("full_advantage", [False, True])
def test_training_rpc_combines_cp_facts_without_duplicate_stage(
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

    reports = []
    for rank in (0, 1):
        cp["rank"] = rank
        reports.append(execute(SimpleNamespace(args=args), 0))
    before_export = time.time_ns()
    train.emit_windows(args, reports)
    record = spans(root, "rl.actor.train")[0]
    assert len(spans(root, "rl.actor.train")) == 1
    assert not spans(root, "rl.actor.train.sample")
    assert attrs(record)["rl.batch.size"] == "1"
    assert int(record["endTimeUnixNano"]) < before_export
    result = artifact(record, "rl.train.results")
    assert result["updates"] == [
        {"update_id": "generation:actor:0:0", "status": "skipped", "reason": "zero_effective_tokens"}
    ]
    assert result["uses"] == [
        {
            result_key: "23",
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
    assert "rl.train.results" not in attrs(record)
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
    value.prompt = [{"role": "user", "content": "题目" * 10000}]
    value.response = "partial"
    value.status = SimpleNamespace(name="ABORTED")

    @hooks.trace_generation
    async def generate(state, args, value):
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
    assert artifact(detail, "rl.sample.response")[0]["text"] == resumed.response
    assert len(detail["links"]) == 1


def test_custom_batch_reward_keeps_results_without_synthetic_children(run):
    args, root = run
    values = [sample(), sample()]

    @hooks.trace_batched_reward
    async def reward(args, values):
        return [0.0, 1.0]

    assert asyncio.run(reward(args, values)) == [0.0, 1.0]
    (batch,) = spans(root, "rl.reward.batch")
    assert not spans(root, "rl.reward.score")
    assert attrs(batch)["rl.batch.size"] == "2"
    assert [row["value"] for row in artifact(batch, "rl.reward.results")] == [0.0, 1.0]
    assert all(value.metadata["_rl_source"]["span_id"] == batch["spanId"] for value in values)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_agent_group_owns_samples_and_keeps_agents_independent(run, startup_failure):
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
    assert all(a["traceId"] != batch["traceId"] and not a.get("parentSpanId") for a in agents)
    assert {a["links"][0]["spanId"] for a in agents} == {d["spanId"] for d in details}


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

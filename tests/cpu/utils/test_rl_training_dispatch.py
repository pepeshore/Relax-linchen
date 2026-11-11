# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils import rl_train_trace as train
from relax.utils.rl_trace_hooks import trace_forward, trace_iteration
from relax.utils.rl_tracing import initialize_run, set_process_role
from scripts.tools.validate_rl_traces import attributes, validate


class _CPUWorker:
    def __init__(self, config, rank):
        self._rank, self._world_size = rank, 2
        self.args = config
        self.role = "actor"
        set_process_role("actor_rank")

    @train.training_rpc
    @trace_iteration
    def train(self, rollout_id, *, trace_headers=None):
        import sys

        # The RPC, OTLP exporter and local Ray processes are real. These
        # CPU workers provide only the parallel coordinates used by Megatron.
        mpu = SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: 0,
            is_pipeline_last_stage=lambda **kw: self._rank == 1,
            get_data_parallel_rank=lambda **kw: 0,
            get_context_parallel_rank=lambda: 0,
            get_pipeline_model_parallel_rank=lambda: self._rank,
        )
        sys.modules["megatron.core"] = SimpleNamespace(mpu=mpu)

        @trace_forward
        def log_probs(owner, data_iterator, num_microbatches, **kwargs):
            pass

        data = [SimpleNamespace(rollout_data={"sample_indices": [7]})]
        for prefix in ("ref_", ""):
            log_probs(self, data, 1, store_prefix=prefix)
        if self._rank == 1:
            with train.window(self.args, "advantage", rollout_id, estimator="grpo"):
                pass
        with train.window(self.args, "actor", rollout_id, data={"sample_indices": [7]}):
            if rollout_id == 3 and self._rank == 1:
                raise ValueError("failed on rank 1")
        return os.getpid()

    train_hybrid = train
    train_async = train


def test_ray_dispatch_exports_one_child_per_worker_process(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    from relax.distributed.ray.actor_group import RayTrainGroup

    monkeypatch.setenv("RELAX_OTEL_DIR", str(tmp_path))
    monkeypatch.setenv("RELAX_OTEL_ENABLED", "1")
    monkeypatch.delenv("RELAX_OTEL_RUN_ID", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    args = Namespace(train_backend="megatron", hf_checkpoint="models/policy", advantage_estimator="grpo")
    initialize_run(args)
    root = tmp_path / args.rl_trace_run_id

    def read():
        return [
            (span, attributes(resource["resource"]["attributes"]))
            for file in root.glob("*.jsonl")
            for line in file.read_text().splitlines(keepends=True)
            if line.endswith("\n")
            for resource in json.loads(line)["resourceSpans"]
            for scope in resource["scopeSpans"]
            for span in scope["spans"]
        ]

    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])
    )
    Worker = ray.remote(num_cpus=1)(_CPUWorker)
    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        group = object.__new__(RayTrainGroup)
        group.role, group.args = "actor", args
        group._actor_handlers = [Worker.remote(args, i) for i in range(2)]
        for step, dispatch in enumerate(
            (group.async_train, group.train_hybrid, group.train_fully_async, group.async_train)
        ):
            refs = dispatch(step)
            if step == 3:
                with pytest.raises(ray.exceptions.RayTaskError, match="failed on rank 1"):
                    ray.get(refs)
            else:
                pids = ray.get(refs)
                assert len(set(pids)) == 2 and os.getpid() not in pids
            deadline = time.monotonic() + 10
            while True:
                rows = [(s, r) for s, r in read() if attributes(s["attributes"])["rl.step.id"] == f"iteration:{step}"]
                parents = [s for s, _ in rows if s["name"] == "rl.actor.train_step"]
                if parents:
                    break
                assert time.monotonic() < deadline, "The dispatch completion span did not reach disk"
                time.sleep(0.02)
            (parent,) = parents
            assert attributes(parent["attributes"])["rl.producer.role"] == "actor_service"
            assert {attributes(s["attributes"])["rl.producer.role"] for s, _ in rows if s is not parent} == {
                "actor_rank"
            }
            workers = {s["spanId"]: (s, r) for s, r in rows if s["name"] == "rl.actor.train_rank"}
            assert len(workers) == 2
            assert {int(attributes(s["attributes"])["rl.rank.global"]) for s, _ in workers.values()} == {0, 1}
            assert {int(r["process.pid"]) for _, r in workers.values()} == set(pids)
            assert all(
                (s["traceId"], s["parentSpanId"]) == (parent["traceId"], parent["spanId"]) for s, _ in workers.values()
            )
            stages = [
                (s, r)
                for s, r in rows
                if s["name"] in ("rl.actor.train", "rl.advantage.compute", "rl.actor.forward", "rl.reference.forward")
            ]
            assert sorted(s["name"] for s, _ in stages) == sorted(
                ["rl.actor.train", "rl.actor.forward", "rl.reference.forward"] * 2 + ["rl.advantage.compute"]
            )
            for s, r in stages:
                rank_span, rank_res = workers[s["parentSpanId"]]
                assert s["traceId"] == rank_span["traceId"] and r["process.pid"] == rank_res["process.pid"]
                assert "rl.timing.ranks" not in attributes(s["attributes"])
            if step == 3:
                failed = [s["name"] for s, _ in rows if s.get("status", {}).get("code") == 2]
                failed_stage = next(s for s, _ in stages if s.get("status", {}).get("code") == 2)
                fields = attributes(failed_stage["attributes"])
                assert "rl.train.results" not in fields
                assert "failed on rank 1" in fields["relax.observation.partial"]
                assert sorted(failed) == ["rl.actor.train", "rl.actor.train_rank", "rl.actor.train_step"]
            else:
                validate(root, min_train_steps=step + 1)
    finally:
        ray.shutdown()

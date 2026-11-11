#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate local RL OTLP JSONL, inline results and batch/member
relationships."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse


def attributes(items: list[dict]) -> dict:
    return {item["key"]: next(iter(item["value"].values())) for item in items}


def _validate_timing(span: dict, attrs: dict) -> None:
    if attrs.get("rl.timing.clock") != "host_wall":
        raise ValueError("Missing host timing semantics")
    scope = attrs.get("rl.timing.scope")
    if scope in ("operation", "source_sync", "rank_local"):
        return
    if scope != "representative_rank":
        raise ValueError("Unknown timing scope")
    rows = json.loads(attrs["rl.timing.ranks"])
    if len(rows) != int(attrs["rl.timing.rank_count"]):
        raise ValueError("Mismatched timing rank count")
    selected = [row for row in rows if row["representative"]]
    if len(selected) != 1:
        raise ValueError("Missing unique representative rank")
    row = selected[0]
    identities = {tuple(sorted(item["rank"].items())) for item in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate timing rank")
    if row["rank"]["pp"] != max(item["rank"]["pp"] for item in rows):
        raise ValueError("Representative is not the final pipeline stage")
    if any(row["rank"][key] != 0 for key in ("dp", "cp", "tp")):
        raise ValueError("Incorrect representative rank")
    for key, value in row["rank"].items():
        if int(attrs[f"rl.timing.rank.{key}"]) != value:
            raise ValueError("Mismatched representative rank")
    if (int(span["startTimeUnixNano"]), int(span["endTimeUnixNano"])) != (
        int(row["start_time_unix_nano"]),
        int(row["end_time_unix_nano"]),
    ):
        raise ValueError("Span does not retain representative timing")
    for row in rows:
        if int(row["end_time_unix_nano"]) < int(row["start_time_unix_nano"]):
            raise ValueError("Negative rank duration")
        if not row["host_name"] or not row["process_pid"]:
            raise ValueError("Missing timing producer identity")


def validate(root: Path, min_train_steps: int = 0) -> dict:
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No trace files in {root}")
    counts, statuses, updates = Counter(), Counter(), Counter()
    runs, steps, references = set(), set(), set()
    records = {}
    processes = {}
    rank_children = {}
    inline_values = 0
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                for resource in json.loads(line)["resourceSpans"]:
                    resource_attrs = attributes(resource["resource"]["attributes"])
                    for key in ("service.name", "service.instance.id", "host.name", "process.pid"):
                        if key not in resource_attrs:
                            raise ValueError(f"Missing Resource {key}")
                    for scope in resource["scopeSpans"]:
                        version = scope["scope"].get("version")
                        if scope["scope"].get("name") != "relax.rl" or version not in (
                            "0.9.0",
                            "0.10.0",
                            "0.11.0",
                            "0.12.0",
                            "0.13.0",
                            "0.14.0",
                            "0.15.0",
                            "0.16.0",
                            "0.17.0",
                            "0.18.0",
                        ):
                            raise ValueError("Unexpected instrumentation scope")
                        for span in scope["spans"]:
                            for key, length in (("traceId", 32), ("spanId", 16)):
                                if not re.fullmatch(f"[0-9a-f]{{{length}}}", span[key]):
                                    raise ValueError(f"Invalid OTLP {key}")
                            identity = (span["traceId"], span["spanId"])
                            if identity in records:
                                raise ValueError("Duplicate exported span")
                            if int(span["endTimeUnixNano"]) < int(span["startTimeUnixNano"]):
                                raise ValueError("Negative stage duration")
                            attrs = attributes(span["attributes"])
                            processes[identity] = (resource_attrs["host.name"], resource_attrs["process.pid"])
                            records[identity] = (span, attrs, version)
                            runs.add(attrs["rl.run.id"])
                            counts[span["name"]] += 1
                            statuses[str(span.get("status", {}).get("code", 0))] += 1
                            if span["name"].endswith(".train"):
                                steps.add(attrs["rl.step.id"])
                                for key in ("rl.batch.id", "rl.batch.size", "rl.model.name", "rl.backend.name"):
                                    if (
                                        version == "0.18.0"
                                        and key == "rl.batch.size"
                                        and attrs.get("rl.timing.scope") == "rank_local"
                                    ):
                                        continue
                                    if key not in attrs:
                                        raise ValueError(f"Missing train attribute {key}")
                            for key, value in attrs.items():
                                if key.endswith("_ref"):
                                    if version != "0.9.0":
                                        raise ValueError(f"New results must be inline: {key}")
                                    parsed = urlparse(value)
                                    if parsed.scheme != "file" or not Path(unquote(parsed.path)).is_absolute():
                                        raise ValueError(f"Invalid artifact URI {value}")
                                    content = json.loads(Path(unquote(parsed.path)).read_text())
                                    references.add(value)
                                    key = key[:-4]
                                elif key.endswith((".results", ".data")) or key in (
                                    "relax.observation.partial",
                                    "rl.sample.input",
                                    "rl.sample.response",
                                    "rl.sample.memberships",
                                ):
                                    content = json.loads(value)
                                    inline_values += 1
                                else:
                                    continue
                                if key == "relax.observation.partial":
                                    raise ValueError("Incomplete rank observations")
                                if key == "rl.train.results" and span["name"].endswith(".train"):
                                    updates.update(row["status"] for row in content["updates"])
            except (KeyError, ValueError, OSError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    prompts_by_sample = {
        (attrs["rl.run.id"], attrs["rl.sample.id"]): attrs["rl.prompt.id"]
        for span, attrs, version in records.values()
        if version in ("0.16.0", "0.17.0", "0.18.0")
        and span["name"] == "rl.rollout.sample"
        and "rl.prompt.id" in attrs
    }
    children = Counter()
    timing = Counter()
    for (trace_id, span_id), (span, attrs, version) in records.items():
        if version == "0.9.0":
            continue
        name = span["name"]
        if version == "0.18.0":
            scope = attrs.get("rl.timing.scope")
            if scope == "representative_rank" or "rl.timing.ranks" in attrs:
                raise ValueError("New training spans must be emitted by individual ranks")
            if name in ("rl.actor.train", "rl.critic.train") and scope != "rank_local":
                raise ValueError("Megatron training stage is missing rank-local timing")
            parent_key = (trace_id, span.get("parentSpanId"))
            parent = records.get(parent_key)
            if name in ("rl.actor.train_step", "rl.critic.train_step"):
                if int(attrs.get("rl.rank.world_size", -1)) < 0 or span.get("parentSpanId"):
                    raise ValueError("Invalid training dispatch")
            if name in ("rl.actor.train_rank", "rl.critic.train_rank"):
                if parent is None or parent[0]["name"] != name.replace("train_rank", "train_step"):
                    raise ValueError("Missing training dispatch parent")
                size = int(parent[1]["rl.rank.world_size"])
                rank = int(attrs["rl.rank.global"])
                if (
                    not 0 <= rank < size
                    or int(attrs["rl.rank.world_size"]) != size
                    or scope != "rank_local"
                    or attrs["rl.step.id"] != parent[1]["rl.step.id"]
                ):
                    raise ValueError("Invalid training rank identity")
                rank_children.setdefault(parent_key, set()).add(rank)
            elif scope == "rank_local" or (name == "rl.weight.sync" and span.get("parentSpanId")):
                if (
                    parent is None
                    or parent[0]["name"] not in ("rl.actor.train_rank", "rl.critic.train_rank")
                    or attrs["rl.rank.global"] != parent[1]["rl.rank.global"]
                    or attrs["rl.step.id"] != parent[1]["rl.step.id"]
                    or processes[(trace_id, span_id)] != processes[parent_key]
                ):
                    raise ValueError("Rank-local stage belongs to the wrong worker or parent")
            elif name in ("rl.actor.publish_weights", "rl.rollout.load_weights") and span.get("parentSpanId"):
                if parent is None or parent[0]["name"] != "rl.weight.sync":
                    raise ValueError("Incorrect weight sync parent")
        if version in ("0.16.0", "0.17.0", "0.18.0") and name in ("rl.reward.score", "rl.reward.batch"):
            expected = prompts_by_sample.get((attrs["rl.run.id"], attrs.get("rl.sample.id")))
            prompt = attrs.get("rl.prompt.id")
            if name == "rl.reward.score" and expected is not None and prompt != expected:
                raise ValueError("Score prompt_id differs from its rollout sample")
            for row in json.loads(attrs.get("rl.reward.results", "[]")):
                expected = prompts_by_sample.get((attrs["rl.run.id"], row.get("sample_id")), prompt)
                if expected is not None and row.get("prompt_id") != expected:
                    raise ValueError("Reward result prompt_id differs from its source")
        if version in ("0.13.0", "0.14.0", "0.15.0", "0.16.0", "0.17.0", "0.18.0"):
            _validate_timing(span, attrs)
        if version in ("0.11.0", "0.12.0", "0.13.0", "0.14.0", "0.15.0", "0.16.0", "0.17.0", "0.18.0"):
            if attrs.get("rl.sample.timing") == "result":
                raise ValueError("Result rows must be stored on the batch, not emitted as spans")
            if name.endswith(".sample") and name != "rl.rollout.sample":
                raise ValueError(f"Redundant member span in {version}: {name}")
        if version in ("0.17.0", "0.18.0") and name in (
            "rl.data.filter",
            "rl.data.transfer",
            "rl.actor.train",
            "rl.critic.train",
            "rl.advantage.compute",
            "rl.actor.forward",
            "rl.teacher.forward",
            "rl.reference.forward",
            "rl.critic.forward",
        ):
            side = attrs.get("rl.execution.side")
            parent = records.get((trace_id, span.get("parentSpanId")))
            rollout_parent = parent is not None and parent[0]["name"] == "rl.rollout.generate"
            if side not in (None, "training", "rollout") or (rollout_parent and side != "rollout"):
                raise ValueError(f"Incorrect execution side for {name}")
            if side == "rollout" and span.get("parentSpanId") and not rollout_parent:
                raise ValueError(f"Incorrect rollout parent for {name}")
            if rollout_parent:
                for key in ("rl.run.id", "rl.step.id", "rl.iteration.index", "rl.purpose"):
                    if attrs.get(key) != parent[1].get(key):
                        raise ValueError(f"Rollout child {name} disagrees with its parent on {key}")
            if attrs.get("rl.timing.scope") == "representative_rank" or attrs.get("rl.backend.name") == "megatron":
                if side != "training":
                    raise ValueError(f"Missing training execution side for {name}")
            if attrs.get("rl.backend.name") == "sglang" and name in ("rl.actor.forward", "rl.teacher.forward"):
                if side != "rollout":
                    raise ValueError(f"Missing rollout execution side for {name}")
        if (
            name == "rl.reward.batch"
            and version in ("0.15.0", "0.16.0", "0.17.0", "0.18.0")
            and span.get("parentSpanId")
        ):
            parent = records.get((trace_id, span["parentSpanId"]))
            if parent is None or parent[0]["name"] != "rl.rollout.generate":
                raise ValueError("A rollout-owned reward batch must be a direct child of rl.rollout.generate")
            if attrs["rl.batch.id"] == parent[1]["rl.batch.id"]:
                raise ValueError("A scoring batch must have its own batch identity")
        if name == "rl.agent.run" and version in ("0.14.0", "0.15.0", "0.16.0", "0.17.0", "0.18.0"):
            parent = records.get((trace_id, span.get("parentSpanId")))
            if parent is None or parent[0]["name"] != "rl.rollout.sample":
                raise ValueError("Agent run must be a child of its sample")
            if not attrs.get("rl.session.id") or attrs["rl.session.id"] != parent[1].get("rl.session.id"):
                raise ValueError("Agent run and sample must belong to the same Session")
        parent_name = {"rl.rollout.sample": "rl.rollout.generate", "rl.reward.score": "rl.reward.batch"}.get(name)
        if parent_name is None and name.endswith(".sample"):
            parent_name = name[:-7]
        if parent_name is not None:
            parent = records.get((trace_id, span.get("parentSpanId")))
            if parent is None or parent[0]["name"] != parent_name:
                raise ValueError(f"Missing or incorrect parent for {name}")
            if attrs.get("rl.batch.id") != parent[1].get("rl.batch.id"):
                raise ValueError(f"Mismatched batch identity for {name}")
            kind = attrs.get("rl.sample.timing")
            if kind not in ("operation", "result"):
                raise ValueError(f"Missing member timing semantics for {name}")
            if kind == "result" and span["startTimeUnixNano"] != span["endTimeUnixNano"]:
                raise ValueError(f"Fabricated member duration for {name}")
            if "rl.batch.size" in attrs:
                raise ValueError("Batch size duplicated on a member")
            timing[kind] += 1
            children[parent_name] += 1
    for key, (span, attrs, version) in records.items():
        if version == "0.18.0" and span["name"] in ("rl.actor.train_step", "rl.critic.train_step"):
            if rank_children.get(key, set()) != set(range(int(attrs["rl.rank.world_size"]))):
                raise ValueError("Training dispatch is missing rank spans")
    if len(runs) != 1 or len(steps) < min_train_steps:
        raise ValueError(f"Expected one run and >= {min_train_steps} train steps; got {runs}, {steps}")
    if statuses.get("2"):
        raise ValueError(f"Found {statuses['2']} failed spans")
    return {
        "run_id": next(iter(runs)),
        "files": len(files),
        "spans": dict(sorted(counts.items())),
        "status_codes": dict(statuses),
        "train_steps": sorted(steps),
        "update_statuses": dict(updates),
        "artifacts": len(references),
        "inline_values": inline_values,
        "children_by_stage": dict(sorted(children.items())),
        "member_timing": dict(timing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--min-train-steps", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_directory, args.min_train_steps), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

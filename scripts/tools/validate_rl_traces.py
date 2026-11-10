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
    if scope in ("operation", "source_sync"):
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
                            records[identity] = (span, attrs, version)
                            runs.add(attrs["rl.run.id"])
                            counts[span["name"]] += 1
                            statuses[str(span.get("status", {}).get("code", 0))] += 1
                            if span["name"].endswith(".train"):
                                steps.add(attrs["rl.step.id"])
                                for key in ("rl.batch.id", "rl.batch.size", "rl.model.name", "rl.backend.name"):
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
    children = Counter()
    timing = Counter()
    for (trace_id, _), (span, attrs, version) in records.items():
        if version == "0.9.0":
            continue
        name = span["name"]
        if version == "0.13.0":
            _validate_timing(span, attrs)
        if version in ("0.11.0", "0.12.0", "0.13.0"):
            if attrs.get("rl.sample.timing") == "result":
                raise ValueError("Result rows must be stored on the batch, not emitted as spans")
            if name.endswith(".sample") and name != "rl.rollout.sample":
                raise ValueError(f"Redundant member span in {version}: {name}")
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

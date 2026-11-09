#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate local RL OTLP JSONL and referenced artifacts without an SDK."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse


def attributes(items: list[dict]) -> dict:
    return {item["key"]: next(iter(item["value"].values())) for item in items}


def validate(root: Path, min_train_steps: int = 0) -> dict:
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No trace files in {root}")
    counts, statuses, updates = Counter(), Counter(), Counter()
    runs, identities, steps, references = set(), set(), set(), set()
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                payload = json.loads(line)
                for resource in payload["resourceSpans"]:
                    resource_attrs = attributes(resource["resource"]["attributes"])
                    for key in ("service.name", "service.instance.id", "host.name", "process.pid"):
                        if key not in resource_attrs:
                            raise ValueError(f"Missing Resource {key}")
                    for scope in resource["scopeSpans"]:
                        if scope["scope"] != {"name": "relax.rl", "version": "0.9.0"}:
                            raise ValueError("Unexpected instrumentation scope")
                        for span in scope["spans"]:
                            for key, length in (("traceId", 32), ("spanId", 16)):
                                if not re.fullmatch(f"[0-9a-f]{{{length}}}", span[key]):
                                    raise ValueError(f"Invalid OTLP {key}")
                            identity = (span["traceId"], span["spanId"])
                            if identity in identities:
                                raise ValueError("Duplicate exported span")
                            identities.add(identity)
                            if int(span["endTimeUnixNano"]) < int(span["startTimeUnixNano"]):
                                raise ValueError("Negative stage duration")
                            attrs = attributes(span["attributes"])
                            runs.add(attrs["rl.run.id"])
                            counts[span["name"]] += 1
                            statuses[str(span.get("status", {}).get("code", 0))] += 1
                            if span["name"].endswith(".train"):
                                steps.add(attrs["rl.step.id"])
                                for key in ("rl.batch.id", "rl.batch.size", "rl.model.name", "rl.backend.name"):
                                    if key not in attrs:
                                        raise ValueError(f"Missing train attribute {key}")
                            for key, uri in attrs.items():
                                if not key.endswith("_ref"):
                                    continue
                                parsed = urlparse(uri)
                                if parsed.scheme != "file" or not Path(unquote(parsed.path)).is_absolute():
                                    raise ValueError(f"Invalid artifact URI {uri}")
                                content = json.loads(Path(unquote(parsed.path)).read_text())
                                references.add(uri)
                                if key == "relax.observation.partial_ref":
                                    raise ValueError("Incomplete rank observations")
                                if key == "rl.train.results_ref":
                                    updates.update(row["status"] for row in content["updates"])
            except (KeyError, ValueError, OSError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--min-train-steps", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_directory, args.min_train_steps), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


grpc = pytest.importorskip("grpc")


def test_otlp_receiver_preserves_parent_identity_and_acknowledges_disk_write(tmp_path):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub

    repo = Path(__file__).resolve().parents[3]
    ready, output = tmp_path / "ready.json", tmp_path / "backend.jsonl"
    env = {**os.environ, "PYTHONPATH": str(repo)}
    process = subprocess.Popen(
        [
            sys.executable,
            "scripts/tools/collect_otlp_traces.py",
            "--listen",
            "127.0.0.1:0",
            "--output",
            str(output),
            "--ready-file",
            str(ready),
        ],
        cwd=repo,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert process.poll() is None, process.stderr.read().decode()
            assert time.monotonic() < deadline, "receiver did not start"
            time.sleep(0.05)
        port = json.loads(ready.read_text())["port"]
        request = ExportTraceServiceRequest()
        resource = request.resource_spans.add()
        scope = resource.scope_spans.add()
        scope.scope.name = "sglang server"
        span = scope.spans.add(
            name="request",
            trace_id=bytes.fromhex("12" * 16),
            span_id=bytes.fromhex("34" * 8),
            parent_span_id=bytes.fromhex("56" * 8),
            start_time_unix_nano=10,
            end_time_unix_nano=20,
        )
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            TraceServiceStub(channel).Export(request, timeout=10)
        record = json.loads(output.read_text())["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert record["traceId"] == span.trace_id.hex()
        assert record["spanId"] == span.span_id.hex()
        assert record["parentSpanId"] == span.parent_span_id.hex()
    finally:
        process.terminate()
        process.wait(timeout=15)

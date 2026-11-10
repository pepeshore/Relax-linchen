#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Receive OTLP/gRPC traces into an append-only OTLP JSONL file."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.rl_tracing import _hex_trace_ids


logger = get_logger(__name__)


def serve(listen: str, output: Path, ready_file: Path | None = None) -> None:
    # Use the dependencies already supplied by SGLang's tracing extra.
    import grpc
    from google.protobuf.json_format import MessageToDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
        TraceServiceServicer,
        add_TraceServiceServicer_to_server,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Receiver(TraceServiceServicer):
        def Export(self, request: Any, context: Any) -> Any:
            try:
                payload = MessageToDict(request, use_integers_for_enums=True)
                _hex_trace_ids(payload)
                line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                with lock, output.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except Exception:
                logger.exception("Could not persist OTLP trace batch")
                context.abort(grpc.StatusCode.INTERNAL, "Could not persist trace batch")
            return ExportTraceServiceResponse()

    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    with ThreadPoolExecutor(max_workers=8) as executor:
        server = grpc.server(executor, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
        add_TraceServiceServicer_to_server(Receiver(), server)
        port = server.add_insecure_port(listen)
        if not port:
            raise RuntimeError(f"Cannot listen on {listen}")
        server.start()
        try:
            if ready_file is not None:
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.write_text(json.dumps({"listen": listen, "port": port, "output": str(output)}))
            logger.info("OTLP/gRPC receiver listening on %s (port %s), writing %s", listen, port, output)
            stopped.wait()
        finally:
            server.stop(grace=10).wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True, help="Bind host:port; port 0 selects an unused port")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    serve(args.listen, args.output, args.ready_file)


if __name__ == "__main__":
    main()

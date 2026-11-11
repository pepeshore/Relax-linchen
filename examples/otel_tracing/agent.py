# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-turn local Agentic rollout for tracing the real inference path."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path

import httpx


@contextmanager
def model_request_headers():
    """Use the Session's parent for a real application span around each HTTP
    call."""
    headers = {"Authorization": "Bearer " + os.environ["RELAX_SESSION_ID"]}
    traceparent = os.getenv("RELAX_TRACEPARENT")
    if not traceparent:
        yield headers
        return

    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    propagator = TraceContextTextMapPropagator()
    carrier = {"traceparent": traceparent}
    if tracestate := os.getenv("RELAX_TRACESTATE"):
        carrier["tracestate"] = tracestate
    with trace.get_tracer("relax.example.agent").start_as_current_span(
        "llm.call", context=propagator.extract(carrier), kind=trace.SpanKind.CLIENT
    ):
        propagator.inject(headers)
        yield headers


def configure_tracer():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint or not os.getenv("RELAX_TRACEPARENT"):
        return None
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "relax-example-agent"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


async def main() -> None:
    session_input = json.loads(Path(os.environ["RELAX_INPUT_JSON"]).read_text(encoding="utf-8"))
    messages = list(session_input["messages"])
    url = os.environ["RELAX_BASE_URL"].rstrip("/") + "/v1/chat/completions"
    timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for turn in range(2):
            if turn:
                messages.append(
                    {"role": "user", "content": "Check your solution and provide the final answer in \\boxed{}."}
                )
            with model_request_headers() as headers:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"model": "model", "messages": messages, "stream": False},
                )
                response.raise_for_status()
            # Keep the full message, including reasoning, for SessionForest lineage.
            messages.append(response.json()["choices"][0]["message"])
    # The configured DAPO reward function scores the real implicit export.
    output = Path(os.environ["RELAX_OUTPUT_JSON"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("{}", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    tracer_provider = configure_tracer()
    try:
        asyncio.run(main())
    finally:
        if tracer_provider is not None:
            tracer_provider.shutdown()

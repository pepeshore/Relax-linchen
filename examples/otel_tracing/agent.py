# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-turn local Agentic rollout for tracing the real inference path."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx


@contextmanager
def agent_span(name: str, attributes: dict[str, str], *, entry: bool = False, client: bool = False) -> Iterator[bool]:
    """Attach the invocation to Relax and each operation to its active
    application parent."""
    traceparent = os.getenv("RELAX_TRACEPARENT")
    if not traceparent:
        yield False
        return

    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier = {"traceparent": traceparent}
    if tracestate := os.getenv("RELAX_TRACESTATE"):
        carrier["tracestate"] = tracestate
    parent = TraceContextTextMapPropagator().extract(carrier, context=Context())
    session = trace.get_current_span(parent).get_span_context()
    if not session.is_valid:
        yield False
        return
    if not entry and trace.get_current_span().get_span_context().trace_id == session.trace_id:
        parent = None  # Let the SDK use the active operation within this invocation.

    kind = trace.SpanKind.CLIENT if client else trace.SpanKind.INTERNAL
    with trace.get_tracer("relax.example.agent").start_as_current_span(
        name, context=parent, kind=kind, attributes={**attributes, "rl.producer.role": "agent"}
    ):
        yield True


@contextmanager
def model_request_headers() -> Iterator[dict[str, str]]:
    headers = {"Authorization": "Bearer " + os.environ["RELAX_SESSION_ID"]}
    with agent_span(
        "llm.call", {"gen_ai.operation.name": "chat", "gen_ai.request.model": "model"}, client=True
    ) as active:
        if active:
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

            TraceContextTextMapPropagator().inject(headers)
        yield headers


async def call_tool(name: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Execute an actual tool and keep its span open until its result is
    available."""
    with agent_span("tool.call", {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}):
        result = function(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


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
    with agent_span(
        "agent.invoke", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "two-turn-math"}, entry=True
    ):
        await run_session()


async def run_session() -> None:
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

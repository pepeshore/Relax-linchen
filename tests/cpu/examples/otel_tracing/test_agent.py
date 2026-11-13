# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json

import httpx
import pytest

from examples.otel_tracing import agent


@pytest.fixture
def sdk(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    try:
        yield provider, exporter
    finally:
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("traced", [False, True])
async def test_two_turn_agent_preserves_reasoning_and_uses_real_reward(tmp_path, monkeypatch, sdk, traced):
    from opentelemetry.context import Context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    provider, exported = sdk
    tracer = provider.get_tracer("test.inference")
    trace_id, run_id = "1" * 32, "2" * 16
    monkeypatch.delenv("RELAX_TRACEPARENT", raising=False)
    if traced:
        monkeypatch.setenv("RELAX_TRACEPARENT", f"00-{trace_id}-{run_id}-01")
        monkeypatch.setenv("RELAX_TRACESTATE", "vendor=session")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps({"messages": messages}))
    for key, value in {
        "RELAX_INPUT_JSON": str(source),
        "RELAX_OUTPUT_JSON": str(output),
        "RELAX_BASE_URL": "http://session.test",
        "RELAX_SESSION_ID": "session-token",
    }.items():
        monkeypatch.setenv(key, value)
    assistant = {"role": "assistant", "content": "4", "reasoning_content": "Adding two and two gives four."}
    requests = []

    async def respond(request):
        assert str(request.url) == "http://session.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer session-token"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        requests.append(payload)
        if traced:
            assert request.headers["tracestate"] == "vendor=session"
            parent = TraceContextTextMapPropagator().extract(request.headers, context=Context())
            with tracer.start_as_current_span("Req", context=parent):
                pass
        else:
            assert "traceparent" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": assistant}]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        agent.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
    )
    await agent.main()
    assert len(requests) == 2
    assert requests[0]["messages"] == messages
    assert requests[1]["messages"][:2] == messages + [assistant]
    assert requests[1]["messages"][2]["role"] == "user"
    assert json.loads(output.read_text()) == {}
    assert not output.with_suffix(".json.tmp").exists()
    spans = exported.get_finished_spans()
    if traced:
        assert all(span.attributes["rl.producer.role"] == "agent" for span in spans if span.name != "Req")
        assert [span.name for span in spans] == ["Req", "llm.call", "Req", "llm.call", "agent.invoke"]
        first_req, first_llm, second_req, second_llm, invocation = spans
        assert invocation.parent.span_id == int(run_id, 16)
        assert first_llm.parent.span_id == second_llm.parent.span_id == invocation.context.span_id
        assert first_llm.context.span_id != second_llm.context.span_id
        assert [req.parent.span_id for req in (first_req, second_req)] == [
            first_llm.context.span_id,
            second_llm.context.span_id,
        ]
        assert all(s.context.trace_id == int(trace_id, 16) for s in spans)
    else:
        assert not spans


def test_model_calls_inject_distinct_children_of_the_session_agent_run(monkeypatch, sdk):
    _, exporter = sdk
    monkeypatch.setenv("RELAX_SESSION_ID", "session")
    trace_id, run_id = "1" * 32, "2" * 16
    monkeypatch.setenv("RELAX_TRACEPARENT", f"00-{trace_id}-{run_id}-01")
    monkeypatch.setenv("RELAX_TRACESTATE", "vendor=session")
    sent = []
    for _ in range(2):
        with agent.model_request_headers() as headers:
            sent.append(headers["traceparent"])
            assert headers["tracestate"] == "vendor=session"
    calls = exporter.get_finished_spans()
    assert len(calls) == len(set(sent)) == 2
    for call, header in zip(calls, sent, strict=True):
        assert call.name == "llm.call"
        assert call.parent.span_id == int(run_id, 16)
        assert header == f"00-{trace_id}-{call.context.span_id:016x}-01"


@pytest.mark.asyncio
async def test_tool_spans_cover_actual_execution_including_async_failure(monkeypatch, sdk):
    from opentelemetry import trace

    _, exported = sdk
    monkeypatch.setenv("RELAX_TRACEPARENT", "00-" + "1" * 32 + "-" + "2" * 16 + "-01")
    seen = []

    def add(a, b):
        seen.append(trace.get_current_span().get_span_context())
        return a + b

    async def fail_later():
        await asyncio.sleep(0)
        seen.append(trace.get_current_span().get_span_context())
        raise ValueError("tool failed")

    async def session_with_tools():
        assert await agent.call_tool("add", add, 3, 7) == 10
        await agent.call_tool("failing", fail_later)

    monkeypatch.setattr(agent, "run_session", session_with_tools)
    with pytest.raises(ValueError, match="tool failed"):
        await agent.main()
    first, second, invocation = exported.get_finished_spans()
    assert [s.name for s in (first, second, invocation)] == ["tool.call", "tool.call", "agent.invoke"]
    assert seen == [first.context, second.context]
    assert [s.attributes["gen_ai.tool.name"] for s in (first, second)] == ["add", "failing"]
    assert all(s.parent.span_id == invocation.context.span_id for s in (first, second))
    assert second.status.status_code == invocation.status.status_code == trace.StatusCode.ERROR

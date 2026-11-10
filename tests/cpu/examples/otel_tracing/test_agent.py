# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import httpx
import pytest

from examples.otel_tracing import agent


@pytest.mark.asyncio
async def test_two_turn_agent_preserves_reasoning_and_uses_real_reward(tmp_path, monkeypatch):
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

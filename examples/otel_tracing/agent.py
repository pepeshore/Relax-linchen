# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-turn local Agentic rollout for tracing the real inference path."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx


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
            response = await client.post(
                url,
                headers={"Authorization": "Bearer " + os.environ["RELAX_SESSION_ID"]},
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
    asyncio.run(main())

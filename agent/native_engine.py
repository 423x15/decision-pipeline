"""
Native OpenAI tool-use engine — Claude Code style agentic loop.

The LLM decides which tools to call and when to stop.
No JEV, no skill bundles, no iteration counter.
Yields PipelineEvent in the same format as pipeline.run().
"""
from __future__ import annotations
import json
import time
from typing import Generator

from agent.pipeline import PipelineEvent
from agent.tools import TOOL_REGISTRY

MAX_NATIVE_ITERATIONS = 10


def _build_openai_tools(tools_config: dict[str, dict]) -> list[dict]:
    result = []
    for name, meta in tools_config.items():
        if name not in TOOL_REGISTRY:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": meta.get("input_hint", "query en langage naturel"),
                        }
                    },
                    "required": ["input"],
                },
            },
        })
    return result


def run(
    query: str,
    openai_client,
    tools_config: dict[str, dict],
    model: str = "gpt-4o-mini",
) -> Generator[PipelineEvent, None, None]:
    openai_tools = _build_openai_tools(tools_config)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant qui utilise des tools pour répondre précisément. "
                "Appelle les tools nécessaires, puis formule une réponse concise."
            ),
        },
        {"role": "user", "content": query},
    ]
    history: list[dict] = []
    iteration = 0

    while iteration < MAX_NATIVE_ITERATIONS:
        t0 = time.perf_counter()
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        messages.append(msg.model_dump(exclude_unset=False))

        if finish_reason == "stop" or not msg.tool_calls:
            yield PipelineEvent(
                type="final",
                data={
                    "response": msg.content or "",
                    "tool_history": history,
                    "llm_prompt": query,
                },
                jev_confidence=None,
                iteration=iteration,
                latency_ms=latency_ms,
            )
            return

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
                tool_input = args.get("input", query)
            except (json.JSONDecodeError, KeyError):
                tool_input = query

            t1 = time.perf_counter()
            tool_fn = TOOL_REGISTRY.get(name)
            result = tool_fn(tool_input, history) if tool_fn else f"Tool '{name}' introuvable."
            tool_lat = (time.perf_counter() - t1) * 1000

            history.append({"tool": name, "input": tool_input, "result": result})
            yield PipelineEvent(
                type="tool_called",
                data={"tool": name, "input": tool_input, "result": result},
                jev_confidence=None,
                iteration=iteration,
                latency_ms=tool_lat,
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

        iteration += 1

    # MAX_NATIVE_ITERATIONS reached — force final synthesis
    t0 = time.perf_counter()
    fallback = openai_client.chat.completions.create(
        model=model,
        messages=messages + [{"role": "user", "content": "Formule maintenant la réponse finale."}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    yield PipelineEvent(
        type="final",
        data={
            "response": fallback.choices[0].message.content or "",
            "tool_history": history,
            "llm_prompt": query,
        },
        jev_confidence=None,
        iteration=iteration,
        latency_ms=latency_ms,
    )

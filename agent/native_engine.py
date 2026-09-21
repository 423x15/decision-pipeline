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
from agent.llm_provider import LLMProvider

_DEFAULT_MAX_ITERATIONS = 10


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
    provider: LLMProvider,
    tools_config: dict[str, dict],
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
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

    while iteration < max_iterations:
        t0 = time.perf_counter()
        resp = provider.chat(messages, tools=openai_tools)
        latency_ms = (time.perf_counter() - t0) * 1000

        messages.append(provider.assistant_message(resp))

        if resp.finish_reason == "stop" or not resp.tool_calls:
            yield PipelineEvent(
                type="final",
                data={"response": resp.content or "", "tool_history": history, "llm_prompt": query},
                jev_confidence=None,
                iteration=iteration,
                latency_ms=latency_ms,
            )
            return

        for tc in resp.tool_calls:
            try:
                args = json.loads(tc.arguments)
                tool_input = args.get("input", query)
            except (json.JSONDecodeError, KeyError):
                tool_input = query

            t1 = time.perf_counter()
            tool_fn = TOOL_REGISTRY.get(tc.name)
            result = tool_fn(tool_input, history) if tool_fn else f"Tool '{tc.name}' introuvable."
            tool_lat = (time.perf_counter() - t1) * 1000

            history.append({"tool": tc.name, "input": tool_input, "result": result})
            yield PipelineEvent(
                type="tool_called",
                data={"tool": tc.name, "input": tool_input, "result": result},
                jev_confidence=None,
                iteration=iteration,
                latency_ms=tool_lat,
            )
            messages.append(provider.tool_result_message(tc.id, str(result)))

        iteration += 1

    # MAX_NATIVE_ITERATIONS reached — force final synthesis
    t0 = time.perf_counter()
    fallback = provider.chat(
        messages + [{"role": "user", "content": "Formule maintenant la réponse finale."}]
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    yield PipelineEvent(
        type="final",
        data={"response": fallback.content or "", "tool_history": history, "llm_prompt": query},
        jev_confidence=None,
        iteration=iteration,
        latency_ms=latency_ms,
    )

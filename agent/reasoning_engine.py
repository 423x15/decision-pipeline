"""
Reasoning engine — transparent ReAct agent with explicit planning.

Architecture:
  1. Planning phase  — LLM produces a todo list (no tools in context)
  2. Execution phase — for each step, load only its skill's tools
                       LLM reasons + calls tools (visible chain of thought)
  3. Synthesis       — final answer from full context

Every decision and reasoning trace is emitted as PipelineEvent.
"""
from __future__ import annotations
import json
import time
from typing import Generator

from agent.pipeline import PipelineEvent
from agent.tools import TOOL_REGISTRY
from agent.llm_provider import LLMProvider

_DEFAULT_MAX_STEPS = 6
_DEFAULT_MAX_TOOL_CALLS_PER_STEP = 3


def _skill_tools(skill_name: str, skills_config: list[dict],
                 tools_config: dict[str, dict]) -> list[dict]:
    """Return OpenAI tool definitions for the given skill bundle only."""
    skill = next((s for s in skills_config if s["name"] == skill_name), None)
    tool_names = skill["tools"] if skill else []
    result = []
    for name in tool_names:
        if name not in TOOL_REGISTRY or name not in tools_config:
            continue
        meta = tools_config[name]
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
                            "description": meta.get("input_hint", "query"),
                        }
                    },
                    "required": ["input"],
                },
            },
        })
    return result


def _plan(query: str, skills_config: list[dict],
          provider: LLMProvider) -> tuple[list[dict], float]:
    """Ask the LLM to produce a step-by-step plan. Returns steps + latency_ms."""
    skill_names = [s["name"] for s in skills_config]
    skill_desc = "\n".join(f"  - {s['name']}: {s['description']}" for s in skills_config)

    t0 = time.perf_counter()
    response = provider.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a planning agent. "
                    "Break down the request into concrete steps.\n"
                    "For each step, specify which skill to use.\n\n"
                    f"Available skills:\n{skill_desc}\n\n"
                    "Reply ONLY in JSON:\n"
                    '{"steps": [{"goal": "...", "skill": "<skill_name>", '
                    '"reasoning": "why this skill"}, ...]}\n\n'
                    f"Valid skill names: {skill_names}"
                ),
            },
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    raw = response.content or "{}"
    try:
        data = json.loads(raw)
        steps = data.get("steps", [])
        valid = {s["name"] for s in skills_config}
        for step in steps:
            if step.get("skill") not in valid:
                step["skill"] = "general_skills"
        return steps, latency_ms
    except (json.JSONDecodeError, KeyError):
        return [{"goal": query, "skill": "general_skills", "reasoning": "fallback"}], latency_ms


def run(
    query: str,
    provider: LLMProvider,
    skills_config: list[dict],
    tools_config: dict[str, dict],
    max_steps: int = _DEFAULT_MAX_STEPS,
    max_tool_calls_per_step: int = _DEFAULT_MAX_TOOL_CALLS_PER_STEP,
) -> Generator[PipelineEvent, None, None]:

    # ── Phase 1 : Planning ──────────────────────────────────────────────────
    steps, plan_lat = _plan(query, skills_config, provider)
    yield PipelineEvent(
        type="plan",
        data={"steps": steps, "query": query},
        jev_confidence=None,
        iteration=0,
        latency_ms=plan_lat,
    )

    all_history: list[dict] = []
    step_results: list[dict] = []

    # ── Phase 2 : Execute each step ─────────────────────────────────────────
    for step_idx, step in enumerate(steps[:max_steps]):
        goal = step.get("goal", query)
        skill_name = step.get("skill", "general_skills")
        reasoning = step.get("reasoning", "")

        openai_tools = _skill_tools(skill_name, skills_config, tools_config)

        yield PipelineEvent(
            type="skill_loaded",
            data={
                "skill": skill_name,
                "tools": [t["function"]["name"] for t in openai_tools],
                "step_goal": goal,
                "reasoning": reasoning,
            },
            jev_confidence=None,
            iteration=step_idx,
            latency_ms=0.0,
        )

        prior = ""
        if step_results:
            prior = "\n\nPrevious step results:\n" + "\n".join(
                f"  [{r['step'][:40]}] {r['tool']}: {str(r['result'])[:120]}"
                for r in step_results
            )

        step_messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    f"You are working on step {step_idx + 1}/{len(steps)}: {goal}\n"
                    f"Original query: {query}{prior}\n\n"
                    "Think aloud if needed, then call the appropriate tool. "
                    "If no tool is needed, answer directly."
                ),
            },
            {"role": "user", "content": goal},
        ]

        for _ in range(max_tool_calls_per_step):
            t0 = time.perf_counter()
            resp = provider.chat(step_messages, tools=openai_tools or None)
            call_lat = (time.perf_counter() - t0) * 1000

            step_messages.append(provider.assistant_message(resp))

            if resp.content:
                yield PipelineEvent(
                    type="reasoning",
                    data={"text": resp.content, "step": step_idx},
                    jev_confidence=None,
                    iteration=step_idx,
                    latency_ms=call_lat,
                )

            if not resp.tool_calls or resp.finish_reason == "stop":
                step_results.append({"step": goal, "tool": "—", "result": resp.content or ""})
                break

            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc.arguments)
                    tool_input = args.get("input", goal)
                except (json.JSONDecodeError, KeyError):
                    tool_input = goal

                t1 = time.perf_counter()
                tool_fn = TOOL_REGISTRY.get(tc.name)
                result = tool_fn(tool_input, all_history) if tool_fn else f"Tool '{tc.name}' not found."
                tool_lat = (time.perf_counter() - t1) * 1000

                all_history.append({"tool": tc.name, "input": tool_input, "result": result})
                step_results.append({"step": goal, "tool": tc.name, "result": result})

                yield PipelineEvent(
                    type="tool_called",
                    data={"tool": tc.name, "input": tool_input, "result": result},
                    jev_confidence=None,
                    iteration=step_idx,
                    latency_ms=tool_lat,
                )

                step_messages.append(provider.tool_result_message(tc.id, str(result)))

        yield PipelineEvent(
            type="skill_unloaded",
            data={"skill": skill_name, "step_done": goal},
            jev_confidence=None,
            iteration=step_idx,
            latency_ms=0.0,
        )

    # ── Phase 3 : Final synthesis ───────────────────────────────────────────
    context = "\n".join(
        f"  - {r['tool']}({r['step'][:40]}) → {str(r['result'])[:120]}"
        for r in step_results
    ) or "No tools were called."
    llm_prompt = f"Question: {query}\n\nCollected information:\n{context}\n\nAnswer concisely."

    t0 = time.perf_counter()
    final_resp = provider.chat([{"role": "user", "content": llm_prompt}])
    synth_lat = (time.perf_counter() - t0) * 1000

    yield PipelineEvent(
        type="final",
        data={
            "response": final_resp.content or "",
            "tool_history": all_history,
            "llm_prompt": llm_prompt,
        },
        jev_confidence=None,
        iteration=len(steps),
        latency_ms=synth_lat,
    )

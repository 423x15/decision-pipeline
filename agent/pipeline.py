from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generator

from agent.skills_loader import select_skill
from agent.tool_selector import select_tool
from agent.decision_engine import decide, MAX_ITERATIONS
from agent.tools import TOOL_REGISTRY
from agent.mistral_client import build_synthesis_prompt


@dataclass
class PipelineEvent:
    type: str
    data: dict = field(default_factory=dict)
    jev_confidence: float | None = None
    iteration: int = 0
    latency_ms: float = 0.0


CONFIDENCE_THRESHOLD: float = 0.80

_FALLBACK_SKILL = "general_skills"


def _below_threshold(conf: float, threshold: float) -> bool:
    return conf < threshold


def run(
    query: str,
    jev_client,
    mistral_client,
    skills_config: list[dict],
    tools_config: dict[str, dict],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Generator[PipelineEvent, None, None]:
    history: list[dict] = []

    skill_name, active_tools, skill_conf, skill_lat = select_skill(query, skills_config, jev_client)
    if _below_threshold(skill_conf, confidence_threshold):
        fallback = next((s for s in skills_config if s["name"] == _FALLBACK_SKILL), skills_config[0])
        skill_name, active_tools = fallback["name"], fallback["tools"]
        yield PipelineEvent(
            type="low_confidence",
            data={"step": "skill_loaded", "original_choice": skill_name, "fallback": _FALLBACK_SKILL},
            jev_confidence=skill_conf,
            iteration=0,
            latency_ms=skill_lat,
        )
    yield PipelineEvent(
        type="skill_loaded",
        data={"skill": skill_name, "tools": active_tools},
        jev_confidence=skill_conf,
        iteration=0,
        latency_ms=skill_lat,
    )

    for iteration in range(MAX_ITERATIONS):
        tool_name, tool_conf, tool_sel_lat = select_tool(
            query, active_tools, tools_config, history, jev_client
        )
        if tool_name == "__skip__":
            break
        if _below_threshold(tool_conf, confidence_threshold):
            yield PipelineEvent(
                type="low_confidence",
                data={"step": "tool_selector", "original_choice": tool_name},
                jev_confidence=tool_conf,
                iteration=iteration,
                latency_ms=tool_sel_lat,
            )
            break  # confiance trop faible → passe directement à la synthèse

        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = tool_fn(query, history) if tool_fn else f"Tool '{tool_name}' non trouvé."

        history.append({"tool": tool_name, "input": query, "result": tool_result})
        yield PipelineEvent(
            type="tool_called",
            data={"tool": tool_name, "input": query, "result": tool_result},
            jev_confidence=tool_conf,
            iteration=iteration,
            latency_ms=tool_sel_lat,
        )

        decision, dec_conf, dec_lat = decide(query, history, iteration, jev_client)
        if _below_threshold(dec_conf, confidence_threshold):
            yield PipelineEvent(
                type="low_confidence",
                data={"step": "decision", "original_choice": decision},
                jev_confidence=dec_conf,
                iteration=iteration,
                latency_ms=dec_lat,
            )
            decision = "final_answer"  # fallback safe
        yield PipelineEvent(
            type="decision",
            data={"decision": decision},
            jev_confidence=dec_conf,
            iteration=iteration,
            latency_ms=dec_lat,
        )
        if decision == "final_answer":
            break
        if decision == "switch_skill":
            skill_name, active_tools, skill_conf, skill_lat = select_skill(
                query, skills_config, jev_client, history=history
            )
            if _below_threshold(skill_conf, confidence_threshold):
                yield PipelineEvent(
                    type="low_confidence",
                    data={"step": "switch_skill", "original_choice": skill_name},
                    jev_confidence=skill_conf,
                    iteration=iteration,
                    latency_ms=skill_lat,
                )
                break
            yield PipelineEvent(
                type="skill_loaded",
                data={"skill": skill_name, "tools": active_tools, "switched": True},
                jev_confidence=skill_conf,
                iteration=iteration,
                latency_ms=skill_lat,
            )

    llm_prompt = build_synthesis_prompt(query, history)
    final_response = mistral_client.synthesize(query, history)
    yield PipelineEvent(
        type="final",
        data={"response": final_response, "tool_history": history, "llm_prompt": llm_prompt},
        jev_confidence=None,
        iteration=len(history),
        latency_ms=0.0,
    )

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generator

from agent.skills_loader import select_skill
from agent.tool_selector import select_tool
from agent.decision_engine import decide, MAX_ITERATIONS
from agent.tools import TOOL_REGISTRY


@dataclass
class PipelineEvent:
    type: str
    data: dict = field(default_factory=dict)
    jev_confidence: float | None = None
    iteration: int = 0
    latency_ms: float = 0.0


def run(
    query: str,
    jev_client,
    mistral_client,
    skills_config: list[dict],
    tools_config: dict[str, dict],
) -> Generator[PipelineEvent, None, None]:
    history: list[dict] = []

    skill_name, active_tools, skill_conf, skill_lat = select_skill(query, skills_config, jev_client)
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
        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = tool_fn(query, history) if tool_fn else f"Tool '{tool_name}' non trouvé."

        history.append({"tool": tool_name, "input": query, "result": tool_result})
        yield PipelineEvent(
            type="tool_called",
            data={"tool": tool_name, "result": tool_result},
            jev_confidence=tool_conf,
            iteration=iteration,
            latency_ms=tool_sel_lat,
        )

        decision, dec_conf, dec_lat = decide(query, history, iteration, jev_client)
        yield PipelineEvent(
            type="decision",
            data={"decision": decision},
            jev_confidence=dec_conf,
            iteration=iteration,
            latency_ms=dec_lat,
        )
        if decision == "final_answer":
            break

    final_response = mistral_client.synthesize(query, history)
    yield PipelineEvent(
        type="final",
        data={"response": final_response, "tool_history": history},
        jev_confidence=None,
        iteration=len(history),
        latency_ms=0.0,
    )

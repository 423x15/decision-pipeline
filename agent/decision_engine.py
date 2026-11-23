from __future__ import annotations
import time
from typesafe_sdk import Choice, Noul

MAX_ITERATIONS: int = 5


def decide(
    query: str,
    history: list[dict],
    iteration: int,
    jev_client,
) -> tuple[str, float, float]:
    if iteration >= MAX_ITERATIONS:
        return "final_answer", 1.0, 0.0

    results_summary = "\n".join(
        f"  [{i+1}] {h['tool']}({h.get('input', '')[:40]}) → {str(h['result'])[:120]}"
        for i, h in enumerate(history)
    )
    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=(
            f"Query: {query}\n\n"
            f"Results ({len(history)} tool(s)):\n{results_summary or '  (none)'}\n\n"
            f"Iteration: {iteration + 1}/{MAX_ITERATIONS}"
        ),
        questions={
            "next": Choice(
                instructions="What is the next step to fully answer the query?",
                criteria={
                    "call_another_tool": "Information is missing; the current skill has more useful tools",
                    "switch_skill": "Information is missing but requires a different skill domain",
                    "final_answer": "Collected information is sufficient to answer",
                },
            ),
            "sufficient_info": Noul(
                instructions="Do the results allow a complete and precise answer?"
            ),
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    sufficient = response.nouls["sufficient_info"].noul > 0.5
    if sufficient:
        return "final_answer", response.choices["next"].confidence, latency_ms
    return response.choices["next"].choice, response.choices["next"].confidence, latency_ms

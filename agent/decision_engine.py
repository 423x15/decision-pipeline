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
            f"Requête: {query}\n\n"
            f"Résultats ({len(history)} tool(s)):\n{results_summary or '  (aucun)'}\n\n"
            f"Itération: {iteration + 1}/{MAX_ITERATIONS}"
        ),
        questions={
            "next": Choice(
                instructions="Faut-il appeler un autre tool ou a-t-on assez d'informations ?",
                criteria={
                    "call_another_tool": "Des informations manquent encore pour répondre complètement",
                    "final_answer": "Les informations collectées sont suffisantes pour répondre",
                },
            ),
            "sufficient_info": Noul(
                instructions="Les résultats permettent-ils une réponse complète et précise ?"
            ),
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices["next"].choice, response.choices["next"].confidence, latency_ms

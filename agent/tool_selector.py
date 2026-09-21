# agent/tool_selector.py
from __future__ import annotations
import time
from pathlib import Path
import yaml
from typesafe_sdk import Choice


def load_tools_config(config_path: Path) -> dict[str, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tools"]


def select_tool(
    query: str,
    active_tool_names: list[str],
    tools_config: dict[str, dict],
    history: list[dict],
    jev_client,
) -> tuple[str, float, float]:
    if not active_tool_names:
        return "__skip__", 1.0, 0.0

    criteria = {
        name: tools_config[name]["description"]
        for name in active_tool_names
        if name in tools_config
    }
    already_called = list(dict.fromkeys(h["tool"] for h in history))
    history_summary = ""
    if history:
        lines = [f"  - {h['tool']}: {str(h['result'])[:80]}" for h in history[-3:]]
        history_summary = (
            "\nTools déjà appelés (éviter de répéter sauf si résultat insuffisant): "
            + ", ".join(already_called)
            + "\nHistorique récent:\n" + "\n".join(lines)
        )

    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=(
            f"Requête: {query}"
            f"{history_summary}\n"
            f"Tools disponibles: {', '.join(criteria)}"
        ),
        questions={
            "tool": Choice(
                instructions="Quel tool appeler pour progresser vers la réponse ?",
                criteria=criteria,
            )
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return response.choices["tool"].choice, response.choices["tool"].confidence, latency_ms

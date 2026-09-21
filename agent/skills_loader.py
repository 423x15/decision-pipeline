# agent/skills_loader.py
from __future__ import annotations
import time
from pathlib import Path
import yaml
from typesafe_sdk import Choice


def load_skills_config(config_path: Path) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["skills"]


def select_skill(
    query: str,
    skills: list[dict],
    jev_client,
) -> tuple[str, list[str], float, float]:
    criteria = {s["name"]: s["description"] for s in skills}
    t0 = time.perf_counter()
    response = jev_client.system_one(
        state=f"Requête utilisateur: {query}\nSkills disponibles: {', '.join(criteria)}",
        questions={
            "skill": Choice(
                instructions="Quel skill bundle correspond le mieux à cette requête ?",
                criteria=criteria,
            )
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    chosen = response.choices["skill"].choice
    confidence = response.choices["skill"].confidence
    active_tools = next((s["tools"] for s in skills if s["name"] == chosen), [])
    return chosen, active_tools, confidence, latency_ms

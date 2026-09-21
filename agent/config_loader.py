"""
Central config loader. Reads config/agent_config.yaml once and exposes
typed accessors. All magic numbers in the codebase route through here.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "agent_config.yaml"


@dataclass
class AgentConfig:
    # Pipeline (JEV mode)
    pipeline_max_iterations: int
    pipeline_confidence_threshold: float
    # Native engine
    native_max_iterations: int
    # ReAct / reasoning engine
    reasoning_max_steps: int
    reasoning_max_tool_calls_per_step: int
    # LLM provider (env vars take precedence)
    llm_provider: str
    llm_model: str


def load_config(path: Path = CONFIG_PATH) -> AgentConfig:
    raw: dict = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    p = raw.get("pipeline", {})
    n = raw.get("native_engine", {})
    r = raw.get("reasoning_engine", {})
    llm = raw.get("llm", {})

    return AgentConfig(
        pipeline_max_iterations=int(p.get("max_iterations", 8)),
        pipeline_confidence_threshold=float(p.get("confidence_threshold", 0.80)),
        native_max_iterations=int(n.get("max_iterations", 10)),
        reasoning_max_steps=int(r.get("max_steps", 6)),
        reasoning_max_tool_calls_per_step=int(r.get("max_tool_calls_per_step", 3)),
        # env vars always win over yaml
        llm_provider=os.environ.get("LLM_PROVIDER") or llm.get("provider", "openai"),
        llm_model=os.environ.get("LLM_MODEL") or llm.get("model", "gpt-4o-mini"),
    )

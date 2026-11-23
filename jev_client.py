from __future__ import annotations

import os
import time
from typing import Any

import streamlit as st

ENV_KEY = "TYPESAFE_API_KEY"


class JevError(RuntimeError):
    pass


def _to_dict(obj: Any) -> dict:
    """Best-effort conversion of a typesafe_sdk answer object to a plain dict,
    so the shared ui.py renderer (which reads dicts) works unchanged."""
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def _build_question(q: dict):
    from typesafe_sdk import Choice, Noul, Score

    qtype = q.get("type")
    instructions = q.get("instructions", "")
    if qtype == "choice":
        return Choice(instructions=instructions, criteria=q.get("criteria") or {})
    if qtype == "score":
        return Score(instructions=instructions, criteria=q.get("criteria") or [])
    if qtype == "noul":
        return Noul(instructions=instructions)
    raise JevError(f"Unsupported question type: {qtype!r}")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError as exc:
        raise JevError(
            "Package 'typesafe-sdk' is not installed. Run: pip install typesafe-sdk"
        ) from exc

    if not api_key:
        raise JevError(
            f"No TypeSafe API key. Set it in the sidebar or "
            f"export the {ENV_KEY} environment variable (create one at https://console.typesafe.ai/)."
        )

    try:
        return TypeSafeClient(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise JevError(f"Failed to initialize TypeSafe client: {exc}") from exc


def default_api_key() -> str:
    return os.environ.get(ENV_KEY, "")


def run_jev_predict(client, state: Any, questions: dict[str, Any]) -> tuple[dict, float]:
    built_questions = {key: _build_question(q) for key, q in questions.items()}

    t0 = time.perf_counter()
    try:
        response = client.system_one(state=state, questions=built_questions)
    except Exception as exc:  # noqa: BLE001
        raise JevError(f"TypeSafe API call failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    buckets = {"choice": "choices", "score": "scores", "noul": "nouls"}
    answers: dict[str, Any] = {}
    for key, q in questions.items():
        bucket = getattr(response, buckets.get(q.get("type"), ""), None)
        try:
            raw = bucket[key] if bucket is not None else None
        except Exception:
            raw = None
        answers[key] = _to_dict(raw) if raw is not None else {}

    return {"answers": answers, "provider": "jev"}, latency_ms

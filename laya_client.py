from __future__ import annotations

import hashlib
import random
import time
from typing import Any

import streamlit as st

MODEL_ID = "convaiinnovations/laya"


class LayaLoadError(RuntimeError):
    pass


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@st.cache_resource(show_spinner=False)
def load_agent(model_id: str = MODEL_ID):
    """Downloads (first run only) and loads the real Laya model.

    Cached per model_id for the lifetime of the Streamlit process, so the
    ~421M-parameter model is loaded once, not on every click.
    """
    try:
        import laya
    except ImportError as exc:
        raise LayaLoadError(
            "Le package 'laya' n'est pas installé. Lancez : pip install -r requirements.txt"
        ) from exc

    try:
        return laya.load(model_id)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
        raise LayaLoadError(
            "Impossible de charger le modèle Laya. Vérifiez votre connexion internet "
            f"(premier chargement = téléchargement des poids depuis Hugging Face) : {exc}"
        ) from exc


def run_real_predict(agent, state: Any, questions: dict[str, Any]) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = agent.predict(state, questions)
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def _seeded_random(*parts: str) -> random.Random:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def run_mock_predict(state: Any, questions: dict[str, Any]) -> tuple[dict, float]:
    """Deterministic, plausible-looking fake inference for exploring the UI
    without downloading the model or needing a GPU."""
    state_key = str(state)
    answers: dict[str, Any] = {}

    for key, question in questions.items():
        rng = _seeded_random(state_key, key, question.get("instructions", ""))
        qtype = question.get("type")

        if qtype == "choice":
            criteria = question.get("criteria") or {}
            labels = list(criteria.keys()) or ["option_a", "option_b"]
            raw = [rng.random() ** 2 for _ in labels]
            total = sum(raw) or 1.0
            probs = {label: v / total for label, v in zip(labels, raw)}
            winner = max(probs, key=probs.get)
            answers[key] = {
                "choice": winner,
                "confidence": probs[winner],
                "probabilities": probs,
            }

        elif qtype == "score":
            levels = question.get("criteria") or ["bas", "moyen", "haut"]
            n = len(levels)
            raw = [rng.random() ** 1.5 for _ in range(n)]
            total = sum(raw) or 1.0
            dist = {lvl: v / total for lvl, v in zip(levels, raw)}
            expected = sum(i * v for i, v in enumerate(dist.values())) / max(n - 1, 1)
            best_level = max(dist, key=dist.get)
            answers[key] = {
                "score": expected,
                "level": best_level,
                "distribution": dist,
            }

        elif qtype == "noul":
            p = min(max(rng.betavariate(2, 2), 0.01), 0.99)
            answers[key] = {"probability": p}

        else:
            answers[key] = {"raw": None}

    latency_ms = 30 + _seeded_random(state_key, "latency").random() * 12
    return {"answers": answers, "mode": "mock"}, latency_ms

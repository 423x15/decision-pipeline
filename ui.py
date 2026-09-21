from __future__ import annotations

import html
from typing import Any

import streamlit as st

# Categorical slots (dark-mode steps), fixed order, never cycled per-question.
CATEGORICAL = [
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SEQUENTIAL = "#3987e5"


def _get_first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _flat(html_str: str) -> str:
    """Strip per-line leading whitespace so Streamlit's markdown parser never
    mistakes an indented HTML line for a fenced code block."""
    return "\n".join(line.strip() for line in html_str.strip("\n").splitlines())


def _md(html_str: str) -> None:
    st.markdown(_flat(html_str), unsafe_allow_html=True)


MODE_BADGES = {
    "mock": ('laya-mode-mock', 'Démo simulée'),
    "laya": ('laya-mode-real', 'Laya (local)'),
    "jev": ('laya-mode-real', 'Jev (API TypeSafe)'),
}


def render_latency(latency_ms: float, mode: str) -> None:
    css_class, label = MODE_BADGES.get(mode, ('laya-mode-real', mode))
    badge = f'<span class="laya-mode-badge {css_class}">{html.escape(label)}</span>'
    _md(f"""
        <div class="laya-hero">
            <div class="laya-hero-value">{latency_ms:.1f}<span class="laya-hero-unit">ms</span></div>
            <div class="laya-hero-sub">latence de la passe unique {badge}</div>
        </div>
        """)


def _bar_row(label: str, prob: float, color: str, winner: bool) -> str:
    pct = max(0.0, min(1.0, prob)) * 100
    check = "✓ " if winner else ""
    weight = "700" if winner else "500"
    return _flat(f"""
    <div class="laya-bar-row">
        <div class="laya-bar-label" style="font-weight:{weight};">{check}{html.escape(str(label))}</div>
        <div class="laya-bar-track">
            <div class="laya-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
        <div class="laya-bar-pct">{pct:.1f}%</div>
    </div>
    """)


def render_choice(key: str, question: dict, answer: dict) -> None:
    criteria = question.get("criteria") or {}
    labels = list(criteria.keys())
    probs = _get_first(answer, ["probabilities", "scores", "distribution"], {})
    winner = _get_first(answer, ["choice", "label", "answer"], None)

    if not probs and winner is not None:
        probs = {winner: _get_first(answer, ["confidence", "probability", "score"], 1.0)}
    if not labels:
        labels = list(probs.keys())

    rows = "".join(
        _bar_row(lbl, probs.get(lbl, 0.0), CATEGORICAL[i % len(CATEGORICAL)], lbl == winner)
        for i, lbl in enumerate(labels)
    )
    desc = criteria.get(winner, "") if isinstance(criteria, dict) else ""
    _md(f"""
        <div class="laya-card">
            <div class="laya-card-head">
                <span class="laya-type-badge laya-type-choice">▤ choice</span>
                <span class="laya-card-title">{html.escape(question.get('instructions', key))}</span>
            </div>
            <div class="laya-winner">Réponse : <b>{html.escape(str(winner))}</b>
                <span class="laya-winner-desc">{html.escape(str(desc))}</span>
            </div>
            {rows}
        </div>
        """)


def render_score(key: str, question: dict, answer: dict) -> None:
    levels = question.get("criteria") or []
    n = max(len(levels), 2)
    expected = _get_first(answer, ["score", "value", "expected"], 0.0)
    try:
        expected = float(expected)
    except (TypeError, ValueError):
        expected = 0.0
    if expected > 1.0 and n > 1:
        # looks like it's already on a 0..n-1 scale rather than normalized
        expected_norm = expected / (n - 1)
    else:
        expected_norm = expected
    expected_norm = max(0.0, min(1.0, expected_norm))
    level_label = _get_first(answer, ["level", "label"], None)
    if level_label is None and levels:
        idx = round(expected_norm * (n - 1))
        level_label = levels[idx] if idx < len(levels) else levels[-1]

    ticks = "".join(
        f'<div class="laya-score-tick{" laya-score-tick-active" if lvl == level_label else ""}">{html.escape(str(lvl))}</div>'
        for lvl in levels
    )
    _md(f"""
        <div class="laya-card">
            <div class="laya-card-head">
                <span class="laya-type-badge laya-type-score">↕ score</span>
                <span class="laya-card-title">{html.escape(question.get('instructions', key))}</span>
            </div>
            <div class="laya-winner">Niveau : <b>{html.escape(str(level_label))}</b></div>
            <div class="laya-score-track">
                <div class="laya-score-fill" style="width:{expected_norm*100:.1f}%;background:{SEQUENTIAL};"></div>
                <div class="laya-score-marker" style="left:{expected_norm*100:.1f}%;"></div>
            </div>
            <div class="laya-score-ticks">{ticks}</div>
        </div>
        """)


def render_noul(key: str, question: dict, answer: dict) -> None:
    p = _get_first(answer, ["probability", "p", "noul", "score", "value"], 0.0)
    try:
        p = float(p)
    except (TypeError, ValueError):
        p = 0.0
    p = max(0.0, min(1.0, p))

    if p < 0.33:
        color, icon, word = STATUS["good"], "✓", "faible"
    elif p < 0.5:
        color, icon, word = STATUS["warning"], "⚠", "modérée"
    elif p < 0.75:
        color, icon, word = STATUS["serious"], "⚠", "élevée"
    else:
        color, icon, word = STATUS["critical"], "✕", "très élevée"

    _md(f"""
        <div class="laya-card">
            <div class="laya-card-head">
                <span class="laya-type-badge laya-type-noul">% noul</span>
                <span class="laya-card-title">{html.escape(question.get('instructions', key))}</span>
            </div>
            <div class="laya-winner">
                <span style="color:{color};font-weight:700;">{icon} Probabilité {word}</span>
                — {p*100:.1f}%
            </div>
            <div class="laya-bar-track laya-noul-track">
                <div class="laya-bar-fill" style="width:{p*100:.1f}%;background:{color};"></div>
            </div>
        </div>
        """)


def render_answer(key: str, question: dict, answer: dict) -> None:
    qtype = question.get("type")
    if qtype == "choice":
        render_choice(key, question, answer)
    elif qtype == "score":
        render_score(key, question, answer)
    elif qtype == "noul":
        render_noul(key, question, answer)
    else:
        st.json(answer)


def render_results(result: dict, questions: dict[str, Any], latency_ms: float, mode: str) -> None:
    render_latency(latency_ms, mode)
    answers = result.get("answers", result)
    for key, question in questions.items():
        answer = answers.get(key, {})
        render_answer(key, question, answer)

    with st.expander("Réponse brute (JSON)"):
        st.json(result)

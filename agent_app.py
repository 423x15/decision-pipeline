# agent_app.py
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.db import Database
from agent.mistral_client import MistralClient
from agent.pipeline import PipelineEvent, run as pipeline_run
from agent.native_engine import run as native_run
from agent.skills_loader import load_skills_config
from agent.tool_selector import load_tools_config
from jev_client import get_client as get_jev_client, default_api_key

DB_PATH = Path("db/conversations.db")
SKILLS_YAML = Path("config/skills.yaml")
TOOLS_YAML = Path("config/tools.yaml")


def _fmt_event(event: PipelineEvent) -> str:
    conf = f"[conf: {event.jev_confidence:.2f}]" if event.jev_confidence is not None else ""
    lat = f"· {event.latency_ms:.0f}ms"
    if event.type == "skill_loaded":
        tools = ", ".join(event.data.get("tools", [])) or "—"
        prefix = "🔀 JEV switch →" if event.data.get("switched") else "🧠 JEV →"
        return f"{prefix} {event.data['skill']} {conf} {lat}  _(tools: {tools})_"
    if event.type == "tool_called":
        preview = str(event.data.get("result", ""))[:60]
        return f"🔧 JEV → {event.data['tool']} {conf} {lat}  → `{preview}`"
    if event.type == "decision":
        icon = "✅" if event.data["decision"] == "final_answer" else "🔄"
        return f"{icon} JEV → {event.data['decision']} {conf} {lat}"
    if event.type == "final":
        return f"📝 OpenAI synthesis {lat}"
    return f"• {event.type} {lat}"


@st.cache_resource
def get_db() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_configs():
    return load_skills_config(SKILLS_YAML), load_tools_config(TOOLS_YAML)


def render_chat(
    db: Database,
    jev_client,
    mistral_client,
    skills_config,
    tools_config,
    use_native: bool = False,
    tools_config_raw: dict | None = None,
):
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("➕ Nouvelle", width="stretch"):
            st.session_state["active_conv_id"] = None
            st.rerun()
        for conv in db.list_conversations():
            cols = st.columns([5, 1])
            label = (conv["title"] or conv["id"][:8])[:40]
            if cols[0].button(label, key=f"c_{conv['id']}", width="stretch"):
                st.session_state["active_conv_id"] = conv["id"]
                st.rerun()
            if cols[1].button("🗑", key=f"d_{conv['id']}"):
                db.delete_conversation(conv["id"])
                if st.session_state.get("active_conv_id") == conv["id"]:
                    st.session_state["active_conv_id"] = None
                st.rerun()

    conv_id = st.session_state.get("active_conv_id")
    if conv_id:
        for msg in db.load_messages(conv_id):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("tool_trace"):
                    with st.expander("🔍 Trace JEV", expanded=False):
                        for item in msg["tool_trace"]:
                            e = PipelineEvent(**item)
                            st.markdown(f"`{_fmt_event(e)}`")
                        for item in msg["tool_trace"]:
                            e = PipelineEvent(**item)
                            if e.type == "tool_called":
                                with st.expander(f"📥 {e.data.get('tool')} — I/O complet", expanded=False):
                                    st.markdown("**Input (query)**")
                                    st.code(e.data.get("input", ""), language="text")
                                    st.markdown("**Résultat**")
                                    st.code(str(e.data.get("result", "")), language="text")
                            if e.type == "final" and e.data.get("llm_prompt"):
                                with st.expander("📤 Prompt envoyé au LLM", expanded=False):
                                    st.code(e.data["llm_prompt"], language="text")

    user_input = st.chat_input("Posez votre question...")
    if not user_input:
        return
    if not use_native and not jev_client:
        st.error("TYPESAFE_API_KEY manquante dans .env")
        return
    if not mistral_client:
        st.error("OPENAI_API_KEY manquante dans .env")
        return

    if not conv_id:
        conv_id = db.create_conversation(user_input[:50])
        st.session_state["active_conv_id"] = conv_id

    db.save_message(conv_id, "user", user_input, None)
    with st.chat_message("user"):
        st.markdown(user_input)

    events: list[PipelineEvent] = []
    final_response = ""
    with st.chat_message("assistant"):
        trace_ph = st.empty()
        resp_ph = st.empty()
        trace_lines: list[str] = []

        event_source = (
            native_run(user_input, mistral_client._client, tools_config_raw or tools_config)
            if use_native
            else pipeline_run(user_input, jev_client, mistral_client, skills_config, tools_config)
        )
        for event in event_source:
            events.append(event)
            if event.type in ("tool_called", "decision"):
                st.session_state["session_decisions"].append({
                    "type": event.type,
                    "tool": event.data.get("tool", event.data.get("decision", "")),
                    "confidence": event.jev_confidence,
                    "latency_ms": round(event.latency_ms, 1),
                    "iteration": event.iteration,
                })
            if event.type != "final":
                trace_lines.append(_fmt_event(event))
                trace_ph.markdown("\n\n".join(f"`{l}`" for l in trace_lines))
            else:
                final_response = event.data.get("response", "")
                resp_ph.markdown(final_response)
                trace_ph.empty()

        with st.expander("🔍 Trace JEV", expanded=False):
            for line in trace_lines:
                st.markdown(f"`{line}`")
            for e in events:
                if e.type == "tool_called":
                    with st.expander(f"📥 {e.data.get('tool')} — I/O complet", expanded=False):
                        st.markdown("**Input (query)**")
                        st.code(e.data.get("input", ""), language="text")
                        st.markdown("**Résultat**")
                        st.code(str(e.data.get("result", "")), language="text")
                if e.type == "final" and e.data.get("llm_prompt"):
                    with st.expander("📤 Prompt envoyé au LLM", expanded=False):
                        st.code(e.data["llm_prompt"], language="text")

    trace_dicts = [
        {"type": e.type, "data": e.data, "jev_confidence": e.jev_confidence,
         "iteration": e.iteration, "latency_ms": e.latency_ms}
        for e in events
    ]
    db.save_message(conv_id, "assistant", final_response, trace_dicts)
    st.rerun()


def render_skills_tools(skills_config, tools_config):
    st.subheader("Skills actifs")
    st.dataframe(pd.DataFrame([
        {"Skill": s["name"], "Description": s["description"],
         "Tools": ", ".join(s["tools"]) if s["tools"] else "—"}
        for s in skills_config
    ]), hide_index=True)

    st.divider()
    st.subheader("Tools disponibles")
    st.dataframe(pd.DataFrame([
        {"Tool": name, "Description": meta["description"], "Input hint": meta.get("input_hint", "")}
        for name, meta in tools_config.items()
    ]), hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("skills.yaml")
        content = SKILLS_YAML.read_text() if SKILLS_YAML.exists() else ""
        new_content = st.text_area("Éditer", value=content, height=250, key="skills_editor")
        if st.button("💾 Sauvegarder skills.yaml"):
            SKILLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()
    with col2:
        st.subheader("tools.yaml")
        content = TOOLS_YAML.read_text() if TOOLS_YAML.exists() else ""
        new_content = st.text_area("Éditer", value=content, height=250, key="tools_editor")
        if st.button("💾 Sauvegarder tools.yaml"):
            TOOLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()


def render_debug():
    st.subheader("Variables d'environnement")
    for key in ["TYPESAFE_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(key, "")
        status = "✅ définie" if val else "❌ manquante"
        masked = (val[:4] + "****" + val[-4:]) if len(val) > 8 else "****"
        st.text(f"{key}: {status}  {masked if val else ''}")

    st.divider()
    decisions = st.session_state.get("session_decisions", [])
    if not decisions:
        st.info("Lance une requête depuis Chat pour voir les décisions ici.")
        return
    df = pd.DataFrame(decisions)
    st.subheader("Décisions JEV — session")
    st.dataframe(df, hide_index=True)
    if "tool" in df.columns:
        counts = df[df["type"] == "tool_called"]["tool"].value_counts()
        if not counts.empty:
            st.bar_chart(counts)


def render_guide():
    st.subheader("Créer un nouveau Skill + ses Tools")
    st.markdown("""
Un **skill** est un bundle de tools qu'JEV active selon la nature de la requête.
Un **tool** est une fonction Python `run(query, history) -> str` enregistrée dans le pipeline.

---

### Exemple complet : Skill Finance
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. `config/skills.yaml` — déclarer le skill")
        st.code("""skills:
  # ... skills existants ...
  - name: finance_skills
    description: >
      Analyse financière, valorisation d'entreprise,
      calcul DCF, ratios boursiers, rendement dividendes
    tools:
      - dcf_valuation
      - financial_ratios
      - dividend_yield""", language="yaml")

        st.markdown("#### 2. `config/tools.yaml` — décrire les tools")
        st.code("""tools:
  # ... tools existants ...
  dcf_valuation:
    description: >
      Calcule la valeur intrinsèque d'une action par DCF
      (flux de trésorerie actualisés). Entrée : données financières.
    input_hint: "free cash flow, taux croissance, taux actualisation"

  financial_ratios:
    description: >
      Calcule P/E, P/B, EV/EBITDA et autres ratios boursiers.
    input_hint: "question sur les ratios d'une action ou secteur"

  dividend_yield:
    description: >
      Calcule le rendement dividende et le taux de distribution.
    input_hint: "dividende annuel et prix de l'action" """, language="yaml")

    with col2:
        st.markdown("#### 3. `agent/tools/dcf_valuation.py`")
        st.code("""def run(query: str, history: list[dict]) -> str:
    # Extraire les paramètres de la query
    fcf = _extract_number(query, "FCF", default=100)
    growth = _extract_number(query, "croissance", default=0.05)
    discount = _extract_number(query, "actualisation", default=0.10)
    terminal_growth = 0.02

    # Calcul DCF sur 5 ans
    years = 5
    total_pv = 0
    for year in range(1, years + 1):
        cf = fcf * (1 + growth) ** year
        pv = cf / (1 + discount) ** year
        total_pv += pv

    # Valeur terminale
    terminal = (fcf * (1 + growth)**years * (1 + terminal_growth)
                / (discount - terminal_growth))
    terminal_pv = terminal / (1 + discount)**years

    intrinsic = total_pv + terminal_pv
    return (
        f"Valeur intrinsèque (DCF) : {intrinsic:.0f}M€\\n"
        f"  FCF base: {fcf}M€ | Croissance: {growth*100:.1f}%"
        f" | Taux: {discount*100:.1f}%\\n"
        f"  VAN flux 5 ans: {total_pv:.0f}M€"
        f" | Valeur terminale: {terminal_pv:.0f}M€"
    )

def _extract_number(text: str, keyword: str,
                    default: float) -> float:
    import re
    pattern = rf"{keyword}[^0-9]*([0-9]+(?:[.,][0-9]+)?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", "."))
    return default""", language="python")

        st.markdown("#### 4. `agent/tools/__init__.py` — enregistrer")
        st.code("""from agent.tools import (
    calculator, datetime_tool, web_search,
    text_utils,
    dcf_valuation,        # ← ajouter
    financial_ratios,     # ← ajouter
    dividend_yield,       # ← ajouter
)

TOOL_REGISTRY = {
    ...
    "dcf_valuation":    dcf_valuation.run,
    "financial_ratios": financial_ratios.run,
    "dividend_yield":   dividend_yield.run,
}""", language="python")

    st.divider()
    st.markdown("#### Checklist")
    st.markdown("""
| Étape | Fichier | Action |
|-------|---------|--------|
| 1 | `config/skills.yaml` | Ajouter le skill + liste des tools |
| 2 | `config/tools.yaml` | Décrire chaque tool (description précise pour JEV) |
| 3 | `agent/tools/mon_tool.py` | Implémenter `run(query, history) -> str` |
| 4 | `agent/tools/__init__.py` | Import + entrée dans `TOOL_REGISTRY` |
| 5 | `tests/test_mon_tool.py` | Au moins `test_run_returns_string` |
| 6 | UI → 🛠 Skills & Tools | Vérifier que le skill apparaît dans la liste |
| 7 | UI → 💬 Chat | Poser une question, vérifier la trace 🔍 |
""")

    st.divider()
    st.info(
        "💡 **Tip JEV** — La `description` dans `skills.yaml` et `tools.yaml` est le seul signal "
        "que JEV utilise pour router. Plus elle est précise et distinctive par rapport aux autres "
        "skills/tools, meilleur est le routing. Inclure des exemples de données typiques aide."
    )


def main():
    st.set_page_config(page_title="JEV Agent", page_icon="🤖", layout="wide")

    for key, val in [("active_conv_id", None), ("session_decisions", [])]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("🤖 Agent Pipeline")

    engine_mode = st.sidebar.radio(
        "Moteur de décision",
        ["🧠 JEV (TypeSafe)", "⚡ OpenAI natif (tool use)"],
        help=(
            "JEV : classificateur rapide (50–150ms) + OpenAI pour la synthèse finale.\n\n"
            "OpenAI natif : le LLM choisit lui-même ses tools et s'arrête quand c'est fini "
            "(style Claude Code — raisonnement + tool calls en boucle)."
        ),
    )
    use_native = engine_mode.startswith("⚡")
    st.sidebar.caption(
        "⚡ OpenAI décide tout" if use_native else "🧠 JEV route · OpenAI synthétise"
    )

    jev_client = None
    mistral_client = None
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            mistral_client = MistralClient(api_key=openai_key)
        except Exception:
            pass
    if not use_native:
        api_key = default_api_key()
        if api_key:
            try:
                jev_client = get_jev_client(api_key)
            except Exception:
                pass

    db = get_db()
    skills_config, tools_config = get_configs()

    tab_chat, tab_skills, tab_debug, tab_guide = st.tabs(
        ["💬 Chat", "🛠 Skills & Tools", "⚙️ Config & Debug", "📖 Guide"]
    )
    with tab_chat:
        render_chat(
            db, jev_client, mistral_client, skills_config, tools_config,
            use_native=use_native, tools_config_raw=tools_config,
        )
    with tab_skills:
        render_skills_tools(skills_config, tools_config)
    with tab_debug:
        render_debug()
    with tab_guide:
        render_guide()


if __name__ == "__main__":
    main()

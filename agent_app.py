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
        return f"🧠 JEV → {event.data['skill']} {conf} {lat}  _(tools: {tools})_"
    if event.type == "tool_called":
        preview = str(event.data.get("result", ""))[:60]
        return f"🔧 JEV → {event.data['tool']} {conf} {lat}  → `{preview}`"
    if event.type == "decision":
        icon = "✅" if event.data["decision"] == "final_answer" else "🔄"
        return f"{icon} JEV → {event.data['decision']} {conf} {lat}"
    if event.type == "final":
        return f"📝 Mistral synthesis {lat}"
    return f"• {event.type} {lat}"


@st.cache_resource
def get_db() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_configs():
    return load_skills_config(SKILLS_YAML), load_tools_config(TOOLS_YAML)


def render_chat(db: Database, jev_client, mistral_client, skills_config, tools_config):
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
                            st.markdown(f"`{_fmt_event(PipelineEvent(**item))}`")

    user_input = st.chat_input("Posez votre question...")
    if not user_input:
        return
    if not jev_client:
        st.error("TYPESAFE_API_KEY manquante dans .env")
        return
    if not mistral_client:
        st.error("MISTRAL_API_KEY manquante dans .env")
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

        for event in pipeline_run(user_input, jev_client, mistral_client, skills_config, tools_config):
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
    for key in ["TYPESAFE_API_KEY", "MISTRAL_API_KEY"]:
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


def main():
    st.set_page_config(page_title="JEV Agent", page_icon="🤖", layout="wide")

    for key, val in [("active_conv_id", None), ("session_decisions", [])]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("🤖 JEV Agent Pipeline")
    st.caption("JEV orchestre · Mistral synthétise · Trace inline")

    jev_client = None
    mistral_client = None
    api_key = default_api_key()
    if api_key:
        try:
            jev_client = get_jev_client(api_key)
        except Exception:
            pass
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if mistral_key:
        try:
            mistral_client = MistralClient(api_key=mistral_key)
        except Exception:
            pass

    db = get_db()
    skills_config, tools_config = get_configs()

    tab_chat, tab_skills, tab_debug = st.tabs(["💬 Chat", "🛠 Skills & Tools", "⚙️ Config & Debug"])
    with tab_chat:
        render_chat(db, jev_client, mistral_client, skills_config, tools_config)
    with tab_skills:
        render_skills_tools(skills_config, tools_config)
    with tab_debug:
        render_debug()


if __name__ == "__main__":
    main()

# agent_app.py — Streamlit UI for the Laya/JEV agent demo
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.config_loader import load_config, CONFIG_PATH
from agent.db import Database
from agent.mistral_client import MistralClient
from agent.pipeline import PipelineEvent, run as pipeline_run
from agent.native_engine import run as native_run
from agent.reasoning_engine import run as reasoning_run
from agent.llm_provider import create_provider
from agent.skills_loader import load_skills_config
from agent.tool_selector import load_tools_config
from jev_client import get_client as get_jev_client, default_api_key

DB_PATH = Path("db/conversations.db")
SKILLS_YAML = Path("config/skills.yaml")
TOOLS_YAML = Path("config/tools.yaml")


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_event(event: PipelineEvent) -> str:
    conf = f"[conf: {event.jev_confidence:.2f}]" if event.jev_confidence is not None else ""
    lat = f"· {event.latency_ms:.0f}ms"
    if event.type == "plan":
        steps = event.data.get("steps", [])
        goals = " → ".join(s.get("goal", "?")[:30] for s in steps)
        return f"📋 Plan ({len(steps)} steps) {lat}  _{goals}_"
    if event.type == "skill_loaded":
        tools = ", ".join(event.data.get("tools", [])) or "—"
        prefix = "🔀 switch →" if event.data.get("switched") else "📦 load →"
        goal = event.data.get("step_goal", "")
        goal_str = f"  _{goal[:40]}_" if goal else ""
        return f"{prefix} {event.data['skill']} {conf} {lat}  (tools: {tools}){goal_str}"
    if event.type == "skill_unloaded":
        return f"📤 unload → {event.data.get('skill')} {lat}"
    if event.type == "reasoning":
        text = event.data.get("text", "")[:120]
        return f"💭 Reasoning  → _{text}_"
    if event.type == "tool_called":
        preview = str(event.data.get("result", ""))[:60]
        return f"🔧 {event.data['tool']} {conf} {lat}  → `{preview}`"
    if event.type == "decision":
        icon = "✅" if event.data["decision"] == "final_answer" else "🔄"
        return f"{icon} JEV → {event.data['decision']} {conf} {lat}"
    if event.type == "low_confidence":
        step = event.data.get("step", "")
        choice = event.data.get("original_choice", "?")
        return f"⚠️ Low confidence {conf} → fallback at step {step} (skipped: {choice})"
    if event.type == "final":
        return f"📝 LLM synthesis {lat}"
    return f"• {event.type} {lat}"


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_db() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_configs():
    return load_skills_config(SKILLS_YAML), load_tools_config(TOOLS_YAML)


@st.cache_resource
def get_agent_config():
    return load_config()


# ── Tab: Chat ─────────────────────────────────────────────────────────────────

def render_chat(
    db: Database,
    jev_client,
    provider,
    mistral_client,
    skills_config: list[dict],
    tools_config: dict,
    use_native: bool,
    use_reasoning: bool,
    confidence_threshold: float,
    cfg,
):
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("➕ New", width="stretch"):
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
                    with st.expander("🔍 Trace", expanded=False):
                        for item in msg["tool_trace"]:
                            e = PipelineEvent(**item)
                            st.markdown(f"`{_fmt_event(e)}`")
                        for item in msg["tool_trace"]:
                            e = PipelineEvent(**item)
                            if e.type == "tool_called":
                                with st.expander(f"📥 {e.data.get('tool')} — full I/O", expanded=False):
                                    st.markdown("**Input**")
                                    st.code(e.data.get("input", ""), language="text")
                                    st.markdown("**Result**")
                                    st.code(str(e.data.get("result", "")), language="text")
                            if e.type == "final" and e.data.get("llm_prompt"):
                                with st.expander("📤 Prompt sent to LLM", expanded=False):
                                    st.code(e.data["llm_prompt"], language="text")

    user_input = st.chat_input("Ask a question...")
    if not user_input:
        return
    if not use_native and not use_reasoning and not jev_client:
        st.error("TYPESAFE_API_KEY missing — add it to .env")
        return
    if not provider:
        st.error("MISTRAL_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) missing — add it to .env")
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

        if use_reasoning:
            event_source = reasoning_run(
                user_input, provider, skills_config, tools_config,
                max_steps=cfg.reasoning_max_steps,
                max_tool_calls_per_step=cfg.reasoning_max_tool_calls_per_step,
            )
        elif use_native:
            event_source = native_run(
                user_input, provider, tools_config,
                max_iterations=cfg.native_max_iterations,
            )
        else:
            event_source = pipeline_run(
                user_input, jev_client, mistral_client, skills_config, tools_config,
                confidence_threshold=confidence_threshold,
                max_iterations=cfg.pipeline_max_iterations,
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

        with st.expander("🔍 Trace", expanded=False):
            for line in trace_lines:
                st.markdown(f"`{line}`")
            for e in events:
                if e.type == "tool_called":
                    with st.expander(f"📥 {e.data.get('tool')} — full I/O", expanded=False):
                        st.markdown("**Input**")
                        st.code(e.data.get("input", ""), language="text")
                        st.markdown("**Result**")
                        st.code(str(e.data.get("result", "")), language="text")
                if e.type == "final" and e.data.get("llm_prompt"):
                    with st.expander("📤 Prompt sent to LLM", expanded=False):
                        st.code(e.data["llm_prompt"], language="text")

    trace_dicts = [
        {"type": e.type, "data": e.data, "jev_confidence": e.jev_confidence,
         "iteration": e.iteration, "latency_ms": e.latency_ms}
        for e in events
    ]
    db.save_message(conv_id, "assistant", final_response, trace_dicts)
    st.rerun()


# ── Tab: Skills & Tools ───────────────────────────────────────────────────────

def render_skills_tools(skills_config, tools_config):
    st.subheader("Active Skills")
    st.dataframe(pd.DataFrame([
        {"Skill": s["name"], "Description": s["description"],
         "Tools": ", ".join(s["tools"]) if s["tools"] else "—"}
        for s in skills_config
    ]), hide_index=True)

    st.divider()
    st.subheader("Available Tools")
    st.dataframe(pd.DataFrame([
        {"Tool": name, "Description": meta["description"], "Input hint": meta.get("input_hint", "")}
        for name, meta in tools_config.items()
    ]), hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("skills.yaml")
        content = SKILLS_YAML.read_text() if SKILLS_YAML.exists() else ""
        new_content = st.text_area("Edit", value=content, height=250, key="skills_editor")
        if st.button("💾 Save skills.yaml"):
            SKILLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()
    with col2:
        st.subheader("tools.yaml")
        content = TOOLS_YAML.read_text() if TOOLS_YAML.exists() else ""
        new_content = st.text_area("Edit", value=content, height=250, key="tools_editor")
        if st.button("💾 Save tools.yaml"):
            TOOLS_YAML.write_text(new_content)
            st.cache_resource.clear()
            st.rerun()


# ── Tab: Config & Debug ───────────────────────────────────────────────────────

def render_debug():
    st.subheader("Environment Variables")
    for key in ["MISTRAL_API_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "LLM_PROVIDER", "LLM_MODEL"]:
        val = os.environ.get(key, "")
        status = "✅ set" if val else "❌ missing"
        masked = (val[:4] + "****" + val[-4:]) if len(val) > 8 else "****"
        st.text(f"{key}: {status}  {masked if val else ''}")

    st.divider()
    st.subheader("agent_config.yaml — agent parameters")
    content = CONFIG_PATH.read_text() if CONFIG_PATH.exists() else ""
    new_content = st.text_area("Edit", value=content, height=300, key="agent_config_editor")
    if st.button("💾 Save agent_config.yaml"):
        CONFIG_PATH.write_text(new_content)
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    decisions = st.session_state.get("session_decisions", [])
    if not decisions:
        st.info("Run a query from Chat to see JEV decisions here.")
        return
    df = pd.DataFrame(decisions)
    st.subheader("JEV Decisions — session")
    st.dataframe(df, hide_index=True)
    if "tool" in df.columns:
        counts = df[df["type"] == "tool_called"]["tool"].value_counts()
        if not counts.empty:
            st.bar_chart(counts)


# ── Tab: Guide ────────────────────────────────────────────────────────────────

def render_guide():
    st.subheader("Adding a new Skill + Tools")
    st.markdown("""
A **skill** is a bundle of tools that JEV activates based on the query type.
A **tool** is a Python function `run(query, history) -> str` registered in the pipeline.

---

### Full example: Finance Skill
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. `config/skills.yaml` — declare the skill")
        st.code("""skills:
  # ... existing skills ...
  - name: finance_skills
    description: >
      Financial analysis, company valuation,
      DCF calculation, stock ratios, dividend yield
    tools:
      - dcf_valuation
      - financial_ratios
      - dividend_yield""", language="yaml")

        st.markdown("#### 2. `config/tools.yaml` — describe the tools")
        st.code("""tools:
  dcf_valuation:
    description: >
      Computes intrinsic stock value via DCF
      (discounted cash flows).
    input_hint: "free cash flow, growth rate, discount rate"

  financial_ratios:
    description: Computes P/E, P/B, EV/EBITDA and other stock ratios.
    input_hint: "question about ratios for a stock or sector"

  dividend_yield:
    description: Computes dividend yield and payout ratio.
    input_hint: "annual dividend and stock price" """, language="yaml")

    with col2:
        st.markdown("#### 3. `agent/tools/dcf_valuation.py`")
        st.code("""def run(query: str, history: list[dict]) -> str:
    fcf = _extract_number(query, "FCF", default=100)
    growth = _extract_number(query, "growth", default=0.05)
    discount = _extract_number(query, "discount", default=0.10)
    terminal_growth = 0.02
    years = 5
    total_pv = sum(
        fcf * (1 + growth)**y / (1 + discount)**y
        for y in range(1, years + 1)
    )
    terminal_pv = (
        fcf * (1 + growth)**years * (1 + terminal_growth)
        / (discount - terminal_growth)
        / (1 + discount)**years
    )
    return (
        f"Intrinsic value (DCF): {total_pv + terminal_pv:.0f}M\\n"
        f"  Base FCF: {fcf}M | Growth: {growth*100:.1f}%"
    )

def _extract_number(text, keyword, default):
    import re
    m = re.search(rf"{keyword}[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
                  text, re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else default""",
                language="python")

        st.markdown("#### 4. `agent/tools/__init__.py` — register")
        st.code("""from agent.tools import (
    calculator, datetime_tool, web_search,
    text_utils,
    dcf_valuation,        # ← add
    financial_ratios,     # ← add
    dividend_yield,       # ← add
)

TOOL_REGISTRY = {
    ...
    "dcf_valuation":    dcf_valuation.run,
    "financial_ratios": financial_ratios.run,
    "dividend_yield":   dividend_yield.run,
}""", language="python")

    st.divider()
    st.markdown("""
| Step | File | Action |
|------|------|--------|
| 1 | `config/skills.yaml` | Add skill + tool list |
| 2 | `config/tools.yaml` | Describe each tool (precise description for JEV routing) |
| 3 | `agent/tools/my_tool.py` | Implement `run(query, history) -> str` |
| 4 | `agent/tools/__init__.py` | Import + entry in `TOOL_REGISTRY` |
| 5 | `tests/test_my_tool.py` | At least `test_run_returns_string` |
| 6 | UI → 🛠 Skills & Tools | Verify the skill appears |
| 7 | UI → 💬 Chat | Test with a real query |
""")
    st.info(
        "💡 **JEV Tip** — The `description` in `skills.yaml` and `tools.yaml` is the only signal "
        "JEV uses for routing. The more precise and distinctive it is, the better the routing."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="JEV Agent", page_icon="🤖", layout="wide")

    for key, val in [("active_conv_id", None), ("session_decisions", [])]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("🤖 Agent Pipeline — Mistral × JEV")

    cfg = get_agent_config()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    confidence_threshold = st.sidebar.slider(
        "JEV confidence threshold",
        min_value=0.0, max_value=1.0,
        value=cfg.pipeline_confidence_threshold, step=0.05,
        help="If JEV is less confident than this threshold, its decision is ignored (safe fallback).",
    )
    engine_mode = st.sidebar.radio(
        "Decision engine",
        ["🧠 JEV (TypeSafe)", "⚡ LLM native (tool use)", "🔬 ReAct Planning (transparent)"],
        help=(
            "**JEV**: fast classifier (50–150ms), skill bundles, calibrated confidence.\n\n"
            "**LLM native**: LLM chooses its own tools, full context, opaque.\n\n"
            "**ReAct Planning**: LLM plans first (visible todo list), "
            "loads/unloads skills per step, reasoning visible."
        ),
    )
    use_native = engine_mode.startswith("⚡")
    use_reasoning = engine_mode.startswith("🔬")
    st.sidebar.caption(
        "🔬 Plan → isolated skills → visible reasoning"
        if use_reasoning else
        ("⚡ LLM decides everything" if use_native else "🧠 JEV routes · Mistral synthesizes")
    )

    # ── Client init ───────────────────────────────────────────────────────────
    jev_client = None
    provider = None
    mistral_client = None  # kept for JEV synthesis path

    try:
        provider = create_provider(
            provider_name=cfg.llm_provider,
            model=cfg.llm_model,
        )
    except ValueError:
        pass  # missing API key — shown as error in chat

    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if mistral_key:
        try:
            mistral_client = MistralClient(api_key=mistral_key)
        except Exception:
            pass

    if not use_native and not use_reasoning:
        api_key = default_api_key()
        if api_key:
            try:
                jev_client = get_jev_client(api_key)
            except Exception:
                pass

    db = get_db()
    skills_config, tools_config = get_configs()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_chat, tab_skills, tab_debug, tab_guide = st.tabs(
        ["💬 Chat", "🛠 Skills & Tools", "⚙️ Config & Debug", "📖 How to extend"]
    )
    with tab_chat:
        render_chat(
            db, jev_client, provider, mistral_client,
            skills_config, tools_config,
            use_native=use_native, use_reasoning=use_reasoning,
            confidence_threshold=confidence_threshold,
            cfg=cfg,
        )
    with tab_skills:
        render_skills_tools(skills_config, tools_config)
    with tab_debug:
        render_debug()
    with tab_guide:
        render_guide()


if __name__ == "__main__":
    main()

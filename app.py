from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from jev_client import JevError, default_api_key, get_client as get_jev_client, run_jev_predict
from laya_client import LayaLoadError, detect_device, load_agent, run_mock_predict, run_real_predict
from scenarios import SCENARIO_ORDER, SCENARIOS
from ui import render_results

PROVIDERS = ["Simulated Demo", "Laya (local model)", "Jev (TypeSafe API)"]


def load_css() -> None:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def parse_custom_questions(df: pd.DataFrame) -> dict:
    questions: dict = {}
    for _, row in df.iterrows():
        key = str(row.get("key") or "").strip()
        qtype = str(row.get("type") or "").strip()
        instructions = str(row.get("instructions") or "").strip()
        criteria_raw = str(row.get("criteria") or "").strip()
        if not key or qtype not in {"choice", "score", "noul"}:
            continue

        question: dict = {"type": qtype, "instructions": instructions or key}

        if qtype == "choice":
            criteria = {}
            for item in criteria_raw.split(","):
                item = item.strip()
                if not item:
                    continue
                if ":" in item:
                    label, desc = item.split(":", 1)
                else:
                    label, desc = item, ""
                criteria[label.strip()] = desc.strip()
            question["criteria"] = criteria or {"option_a": "", "option_b": ""}

        elif qtype == "score":
            levels = [x.strip() for x in criteria_raw.split(",") if x.strip()]
            question["criteria"] = levels or ["low", "medium", "high"]

        questions[key] = question

    return questions


def questions_to_editor_df(questions: dict) -> pd.DataFrame:
    rows = []
    for key, q in questions.items():
        criteria = q.get("criteria")
        if isinstance(criteria, dict):
            criteria_str = ", ".join(f"{k}:{v}" for k, v in criteria.items())
        elif isinstance(criteria, list):
            criteria_str = ", ".join(criteria)
        else:
            criteria_str = ""
        rows.append(
            {
                "key": key,
                "type": q.get("type", "choice"),
                "instructions": q.get("instructions", ""),
                "criteria": criteria_str,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Laya Playground",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    header_html = """
        <div class="laya-header">
            <h1>🧠 Laya Playground</h1>
        </div>
        <div class="laya-header-sub">
            Typed decision engine — Laya (local, convaiinnovations/laya) or Jev (TypeSafe API)
        </div>
        <div class="laya-facts">
            <span class="laya-fact">421M parameters</span>
            <span class="laya-fact">~33ms / single pass</span>
            <span class="laya-fact">100+ languages</span>
            <span class="laya-fact">Apache 2.0</span>
            <span class="laya-fact">choice · score · noul</span>
        </div>
        """
    st.markdown(
        "\n".join(line.strip() for line in header_html.strip("\n").splitlines()),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Provider")
        provider = st.radio(
            "Answer source",
            PROVIDERS,
            index=0,
            help=(
                "Simulated Demo: fake responses, to explore the UI without installing anything. "
                "Laya: open-weights model downloaded and run 100% locally. "
                "Jev: API hosted by TypeSafe — data leaves your machine and goes to their cloud."
            ),
        )

        agent = None
        jev_client = None

        if provider == "Laya (local model)":
            device = detect_device()
            st.caption(f"Detected device: `{device}`")
            if st.button("Load Laya model", use_container_width=True):
                with st.spinner("Loading model (may download ~421M parameters on first run)..."):
                    try:
                        st.session_state["agent"] = load_agent()
                    except LayaLoadError as exc:
                        st.session_state["agent"] = None
                        st.session_state["agent_error"] = str(exc)
                    else:
                        st.session_state["agent_error"] = None

            agent = st.session_state.get("agent")
            load_error = st.session_state.get("agent_error")
            if agent is not None:
                st.success("Model loaded.")
            elif load_error:
                st.error(load_error)
            else:
                st.info("Click 'Load Laya model' before running a decision.")

        elif provider == "Jev (TypeSafe API)":
            st.caption("⚠️ Network call to TypeSafe's hosted API — not local.")
            api_key = st.text_input(
                "TypeSafe API key",
                value=default_api_key(),
                type="password",
                help="Created at console.typesafe.ai, or set as TYPESAFE_API_KEY environment variable.",
            )
            if api_key:
                try:
                    jev_client = get_jev_client(api_key)
                    st.success("Jev client ready.")
                except JevError as exc:
                    jev_client = None
                    st.error(str(exc))
            else:
                st.info("Enter your API key to enable Jev.")

        st.divider()
        st.subheader("Scenario")
        scenario_name = st.selectbox("Choose a scenario", SCENARIO_ORDER)
        st.caption(SCENARIOS[scenario_name]["description"])

        st.divider()
        if st.session_state["history"]:
            st.subheader("History")
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("Clear history", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

    scenario = SCENARIOS[scenario_name]

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### State to evaluate")
        state_text = st.text_area(
            "state",
            value=scenario["state"],
            height=180,
            label_visibility="collapsed",
        )

        st.markdown("#### Typed questions")
        if scenario_name == "Custom":
            editor_df = questions_to_editor_df(scenario["questions"])
            edited_df = st.data_editor(
                editor_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "type": st.column_config.SelectboxColumn(
                        "type", options=["choice", "score", "noul"], required=True
                    ),
                    "criteria": st.column_config.TextColumn(
                        "criteria",
                        help="choice: label:description, label:description | score: level1, level2, level3",
                    ),
                },
            )
            questions = parse_custom_questions(edited_df)
        else:
            questions = scenario["questions"]
            for key, q in questions.items():
                criteria = q.get("criteria")
                extra = f" — {criteria}" if criteria else ""
                st.markdown(f"- **{key}** (`{q['type']}`) : {q['instructions']}{extra}")

        run_clicked = st.button("▶ Run decision", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Result")
        if run_clicked:
            if not questions:
                st.warning("Add at least one typed question before running the decision.")
                run_result = None
            elif provider == "Simulated Demo":
                run_result = ("mock", *run_mock_predict(state_text, questions))
            elif provider == "Laya (local model)":
                if agent is None:
                    st.warning("Load the Laya model from the sidebar first.")
                    run_result = None
                else:
                    try:
                        result, latency_ms = run_real_predict(agent, state_text, questions)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Inference error: {exc}")
                        run_result = None
                    else:
                        run_result = ("laya", result, latency_ms)
            else:  # Jev (TypeSafe API)
                if jev_client is None:
                    st.warning("Enter a valid API key in the sidebar.")
                    run_result = None
                else:
                    try:
                        result, latency_ms = run_jev_predict(jev_client, state_text, questions)
                    except JevError as exc:
                        st.error(str(exc))
                        run_result = None
                    else:
                        run_result = ("jev", result, latency_ms)

            if run_result is not None:
                mode, result, latency_ms = run_result
                render_results(result, questions, latency_ms, mode=mode)
                st.session_state["history"].insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "scenario": scenario_name,
                        "provider": mode,
                        "latency_ms": round(latency_ms, 1),
                    },
                )
        else:
            st.info("Configure the state and questions on the left, then run the decision.")


if __name__ == "__main__":
    main()

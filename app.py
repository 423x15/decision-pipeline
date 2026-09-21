from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from laya_client import LayaLoadError, detect_device, load_agent, run_mock_predict, run_real_predict
from scenarios import SCENARIO_ORDER, SCENARIOS
from ui import render_results


def load_css() -> None:
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def parse_custom_questions(df: pd.DataFrame) -> dict:
    questions: dict = {}
    for _, row in df.iterrows():
        key = str(row.get("clé") or "").strip()
        qtype = str(row.get("type") or "").strip()
        instructions = str(row.get("instructions") or "").strip()
        criteria_raw = str(row.get("critères") or "").strip()
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
            question["criteria"] = levels or ["bas", "moyen", "haut"]

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
                "clé": key,
                "type": q.get("type", "choice"),
                "instructions": q.get("instructions", ""),
                "critères": criteria_str,
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
            Moteur de décisions typées non-autorégressif — convaiinnovations/laya
        </div>
        <div class="laya-facts">
            <span class="laya-fact">421M paramètres</span>
            <span class="laya-fact">~33ms / passe unique</span>
            <span class="laya-fact">100+ langues</span>
            <span class="laya-fact">Apache 2.0</span>
            <span class="laya-fact">choice · score · noul</span>
        </div>
        """
    st.markdown(
        "\n".join(line.strip() for line in header_html.strip("\n").splitlines()),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Mode")
        mode = st.radio(
            "Source des réponses",
            ["Démo simulée", "Modèle réel (local)"],
            index=0,
            help=(
                "Démo simulée : réponses factices déterministes, pour explorer l'interface "
                "instantanément sans rien télécharger. Modèle réel : télécharge et charge "
                "convaiinnovations/laya via le package pip 'laya' (premier chargement = "
                "téléchargement des poids, nécessite une connexion internet)."
            ),
        )
        mock_mode = mode == "Démo simulée"

        agent = None
        load_error = None
        if not mock_mode:
            device = detect_device()
            st.caption(f"Device détecté : `{device}`")
            if st.button("Charger le modèle Laya", use_container_width=True):
                with st.spinner("Chargement du modèle (peut télécharger ~421M paramètres au premier lancement)..."):
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
                st.success("Modèle chargé.")
            elif load_error:
                st.error(load_error)
            else:
                st.info("Cliquez sur 'Charger le modèle Laya' avant de lancer une décision.")

        st.divider()
        st.subheader("Scénario")
        scenario_name = st.selectbox("Choisir un scénario", SCENARIO_ORDER)
        st.caption(SCENARIOS[scenario_name]["description"])

        st.divider()
        if st.session_state["history"]:
            st.subheader("Historique")
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("Vider l'historique", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

    scenario = SCENARIOS[scenario_name]

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### État à évaluer")
        state_text = st.text_area(
            "state",
            value=scenario["state"],
            height=180,
            label_visibility="collapsed",
        )

        st.markdown("#### Questions typées")
        if scenario_name == "Personnalisé":
            editor_df = questions_to_editor_df(scenario["questions"])
            edited_df = st.data_editor(
                editor_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "type": st.column_config.SelectboxColumn(
                        "type", options=["choice", "score", "noul"], required=True
                    ),
                    "critères": st.column_config.TextColumn(
                        "critères",
                        help="choice: label:description, label:description | score: niveau1, niveau2, niveau3",
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

        run_clicked = st.button("▶ Lancer la décision", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Résultat")
        if run_clicked:
            if not questions:
                st.warning("Ajoutez au moins une question typée avant de lancer la décision.")
            elif mock_mode:
                result, latency_ms = run_mock_predict(state_text, questions)
                render_results(result, questions, latency_ms, mode="mock")
                st.session_state["history"].insert(
                    0,
                    {
                        "heure": datetime.now().strftime("%H:%M:%S"),
                        "scénario": scenario_name,
                        "mode": "démo",
                        "latence_ms": round(latency_ms, 1),
                    },
                )
            elif agent is None:
                st.warning("Chargez d'abord le modèle réel depuis la barre latérale.")
            else:
                try:
                    result, latency_ms = run_real_predict(agent, state_text, questions)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erreur pendant l'inférence : {exc}")
                else:
                    render_results(result, questions, latency_ms, mode="real")
                    st.session_state["history"].insert(
                        0,
                        {
                            "heure": datetime.now().strftime("%H:%M:%S"),
                            "scénario": scenario_name,
                            "mode": "réel",
                            "latence_ms": round(latency_ms, 1),
                        },
                    )
        else:
            st.info("Configurez l'état et les questions à gauche, puis lancez la décision.")


if __name__ == "__main__":
    main()

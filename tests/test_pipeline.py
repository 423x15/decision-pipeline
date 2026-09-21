from pathlib import Path
from unittest.mock import MagicMock
from agent.pipeline import PipelineEvent, run
from agent.skills_loader import load_skills_config
from agent.tool_selector import load_tools_config


def _make_jev(skill="math_skills", tool="calculator", decision="final_answer"):
    client = MagicMock()
    call_count = [0]
    def side_effect(state, questions):
        call_count[0] += 1
        resp = MagicMock()
        keys = list(questions.keys())
        if "skill" in keys:
            resp.choices = {"skill": MagicMock(choice=skill, confidence=0.9)}
        elif "tool" in keys:
            resp.choices = {"tool": MagicMock(choice=tool, confidence=0.9)}
        else:
            resp.choices = {"next": MagicMock(choice=decision, confidence=0.9)}
            resp.nouls = {"sufficient_info": MagicMock(noul=0.8)}
        return resp
    client.system_one.side_effect = side_effect
    return client


def test_pipeline_event_dataclass():
    e = PipelineEvent(type="skill_loaded", data={"skill": "math_skills"},
                      jev_confidence=0.9, iteration=0, latency_ms=45.0)
    assert e.type == "skill_loaded"
    assert e.jev_confidence == 0.9


def test_run_yields_all_event_types():
    skills = load_skills_config(Path("config/skills.yaml"))
    tools_cfg = load_tools_config(Path("config/tools.yaml"))
    jev = _make_jev()
    mistral = MagicMock()
    mistral.synthesize.return_value = "La réponse est 42."

    events = list(run("combien font 6*7", jev, mistral, skills, tools_cfg))
    types = {e.type for e in events}
    assert "skill_loaded" in types
    assert "tool_called" in types
    assert "decision" in types
    assert "final" in types


def test_run_final_event_has_response():
    skills = load_skills_config(Path("config/skills.yaml"))
    tools_cfg = load_tools_config(Path("config/tools.yaml"))
    jev = _make_jev()
    mistral = MagicMock()
    mistral.synthesize.return_value = "Réponse finale."

    events = list(run("test", jev, mistral, skills, tools_cfg))
    final = next(e for e in events if e.type == "final")
    assert final.data["response"] == "Réponse finale."

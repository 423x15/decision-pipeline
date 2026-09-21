# tests/test_skills_loader.py
from pathlib import Path
from unittest.mock import MagicMock
from agent.skills_loader import load_skills_config, select_skill

SKILLS_YAML = Path("config/skills.yaml")

def test_load_skills_config_returns_list():
    skills = load_skills_config(SKILLS_YAML)
    assert isinstance(skills, list) and len(skills) > 0

def test_load_skills_config_has_required_fields():
    for s in load_skills_config(SKILLS_YAML):
        assert "name" in s and "description" in s and "tools" in s

def test_select_skill_returns_tuple():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"skill": MagicMock(choice="math_skills", confidence=0.9)}
    )
    skill_name, tools, confidence, latency = select_skill(
        "combien font 3 * 7", load_skills_config(SKILLS_YAML), mock_client
    )
    assert skill_name == "math_skills"
    assert isinstance(tools, list)
    assert 0.0 <= confidence <= 1.0
    assert latency >= 0.0

def test_select_skill_passes_query_in_state():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"skill": MagicMock(choice="search_skills", confidence=0.85)}
    )
    select_skill("cherche des infos sur python", load_skills_config(SKILLS_YAML), mock_client)
    state_arg = str(mock_client.system_one.call_args)
    assert "cherche des infos sur python" in state_arg

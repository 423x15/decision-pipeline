import pytest
from agent.tools import TOOL_REGISTRY


def test_registry_has_all_tools():
    for name in ["calculator", "datetime_tool", "web_search", "text_utils"]:
        assert name in TOOL_REGISTRY


def test_calculator_basic():
    result = TOOL_REGISTRY["calculator"]("combien font 6 * 7", [])
    assert "42" in result


def test_calculator_expression_in_query():
    result = TOOL_REGISTRY["calculator"]("calcule 1500 * 0.197", [])
    assert "295.5" in result or "295,5" in result


def test_calculator_no_expression():
    result = TOOL_REGISTRY["calculator"]("quelle heure est-il", [])
    assert "expression" in result.lower() or "trouv" in result.lower()


def test_datetime_tool_returns_date():
    result = TOOL_REGISTRY["datetime_tool"]("quelle est la date aujourd'hui", [])
    assert "2026" in result or "septembre" in result.lower() or "09" in result


def test_web_search_returns_stub():
    result = TOOL_REGISTRY["web_search"]("prix bitcoin", [])
    assert len(result) > 10


def test_text_utils_word_count():
    result = TOOL_REGISTRY["text_utils"]("compte les mots dans: bonjour monde comment vas-tu", [])
    assert "mot" in result.lower() or any(c.isdigit() for c in result)

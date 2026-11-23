import pytest
from agent.tools import TOOL_REGISTRY


def test_registry_has_all_tools():
    for name in ["calculator", "datetime_tool", "web_search", "text_utils", "code_runner"]:
        assert name in TOOL_REGISTRY


def test_calculator_basic():
    result = TOOL_REGISTRY["calculator"]("6 * 7", [])
    assert "42" in result


def test_calculator_expression_in_query():
    result = TOOL_REGISTRY["calculator"]("compute 1500 * 0.197", [])
    assert "295.5" in result or "295,5" in result


def test_calculator_no_expression():
    result = TOOL_REGISTRY["calculator"]("what time is it", [])
    assert "No mathematical expression" in result


def test_datetime_tool_returns_date():
    result = TOOL_REGISTRY["datetime_tool"]("what is today's date", [])
    assert "2026" in result or any(m in result for m in ["January","February","March","April","May","June","July","August","September","October","November","December"])


def test_web_search_returns_something():
    result = TOOL_REGISTRY["web_search"]("Mistral AI", [])
    assert len(result) > 10


def test_text_utils_word_count():
    result = TOOL_REGISTRY["text_utils"]("analyze: hello world how are you", [])
    assert any(c.isdigit() for c in result)


def test_code_runner_print():
    result = TOOL_REGISTRY["code_runner"]("print('hello world')", [])
    assert "hello world" in result


def test_code_runner_variable():
    result = TOOL_REGISTRY["code_runner"]("x = 2 + 2", [])
    assert "4" in result


def test_code_runner_forbidden_import():
    result = TOOL_REGISTRY["code_runner"]("import os", [])
    assert "Refused" in result or "Forbidden" in result


def test_code_runner_forbidden_call():
    result = TOOL_REGISTRY["code_runner"]("open('file.txt')", [])
    assert "Refused" in result or "Forbidden" in result


def test_code_runner_syntax_error():
    result = TOOL_REGISTRY["code_runner"]("def foo(:", [])
    assert "Syntax error" in result or "error" in result.lower()

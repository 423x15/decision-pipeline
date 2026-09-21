# tests/test_tool_selector.py
from pathlib import Path
from unittest.mock import MagicMock
from agent.tool_selector import load_tools_config, select_tool

TOOLS_YAML = Path("config/tools.yaml")

def test_load_tools_config_returns_dict():
    tools = load_tools_config(TOOLS_YAML)
    assert isinstance(tools, dict) and "calculator" in tools

def test_load_tools_config_has_description():
    for name, meta in load_tools_config(TOOLS_YAML).items():
        assert "description" in meta

def test_select_tool_empty_tools_returns_skip():
    tool_name, confidence, latency = select_tool(
        query="bonjour comment vas-tu",
        active_tool_names=[],
        tools_config={},
        history=[],
        jev_client=None,
    )
    assert tool_name == "__skip__"
    assert confidence == 1.0
    assert latency == 0.0


def test_select_tool_returns_tuple():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"tool": MagicMock(choice="calculator", confidence=0.95)}
    )
    tool_name, confidence, latency = select_tool(
        query="combien font 6 * 7",
        active_tool_names=["calculator", "datetime_tool"],
        tools_config=load_tools_config(TOOLS_YAML),
        history=[],
        jev_client=mock_client,
    )
    assert tool_name == "calculator"
    assert 0.0 <= confidence <= 1.0
    assert latency >= 0.0

def test_select_tool_includes_history_in_state():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"tool": MagicMock(choice="web_search", confidence=0.8)}
    )
    select_tool(
        query="test",
        active_tool_names=["web_search"],
        tools_config=load_tools_config(TOOLS_YAML),
        history=[{"tool": "calculator", "result": "42", "input": "6*7"}],
        jev_client=mock_client,
    )
    state_str = str(mock_client.system_one.call_args)
    assert "calculator" in state_str or "42" in state_str

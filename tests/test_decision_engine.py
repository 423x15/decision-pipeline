from unittest.mock import MagicMock
from agent.decision_engine import decide, MAX_ITERATIONS

def test_max_iterations_is_5():
    assert MAX_ITERATIONS == 5

def test_decide_forces_final_at_max_iterations():
    mock_client = MagicMock()
    decision, confidence, latency = decide("test", [{"tool": "x", "result": "y"}], 5, mock_client)
    assert decision == "final_answer"
    mock_client.system_one.assert_not_called()

def test_decide_returns_jev_choice_before_max():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"next": MagicMock(choice="final_answer", confidence=0.92)},
        nouls={"sufficient_info": MagicMock(probability=0.88)},
    )
    decision, confidence, latency = decide(
        "quelle est la date?",
        [{"tool": "datetime_tool", "result": "2026-09-21", "input": "date"}],
        1, mock_client,
    )
    assert decision == "final_answer"
    assert confidence == 0.92

def test_decide_call_another_tool():
    mock_client = MagicMock()
    mock_client.system_one.return_value = MagicMock(
        choices={"next": MagicMock(choice="call_another_tool", confidence=0.75)},
        nouls={"sufficient_info": MagicMock(probability=0.3)},
    )
    decision, _, _ = decide(
        "combien font 3*7 et quelle heure",
        [{"tool": "calculator", "result": "21", "input": "3*7"}],
        1, mock_client,
    )
    assert decision == "call_another_tool"

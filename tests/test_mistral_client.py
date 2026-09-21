from unittest.mock import MagicMock
from agent.mistral_client import MistralClient, build_synthesis_prompt


def test_build_synthesis_prompt_contains_query():
    prompt = build_synthesis_prompt(
        "combien font 6 * 7",
        [{"tool": "calculator", "input": "6 * 7", "result": "6 * 7 = 42"}],
    )
    assert "6 * 7" in prompt and "42" in prompt


def test_build_synthesis_prompt_empty_history():
    prompt = build_synthesis_prompt("quelle heure", [])
    assert "quelle heure" in prompt
    assert "Aucun" in prompt or "aucun" in prompt


def test_mistral_client_synthesize_calls_api():
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="La réponse est 42."))]
    )
    client = MistralClient.__new__(MistralClient)
    client._client = mock_mistral
    client.model = "mistral-small-latest"

    result = client.synthesize(
        "combien font 6 * 7",
        [{"tool": "calculator", "input": "6*7", "result": "42"}],
    )
    assert result == "La réponse est 42."
    mock_mistral.chat.complete.assert_called_once()

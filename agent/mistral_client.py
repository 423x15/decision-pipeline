from __future__ import annotations
import os
import time
from mistralai.client.sdk import Mistral
from mistralai.client import errors as mistral_errors

DEFAULT_MODEL = "mistral-small-latest"


def build_synthesis_prompt(query: str, history: list[dict]) -> str:
    if history:
        context = "\n".join(
            f"- {h['tool']}({h.get('input', '')[:60]}) → {h['result']}"
            for h in history
        )
    else:
        context = "Aucun outil n'a été appelé."
    return (
        f"Question: {query}\n\n"
        f"Informations collectées:\n{context}\n\n"
        f"Réponds de manière concise en te basant uniquement sur ces informations."
    )


class MistralClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY manquant. Ajoute-le dans .env")
        self._client = Mistral(api_key=key)
        self.model = model

    def synthesize(self, query: str, history: list[dict]) -> str:
        delays = [5, 15]
        for attempt, delay in enumerate(delays + [None]):
            try:
                response = self._client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": build_synthesis_prompt(query, history)}],
                )
                return response.choices[0].message.content
            except mistral_errors.SDKError as exc:
                if "429" in str(exc) and delay is not None:
                    time.sleep(delay)
                    continue
                raise

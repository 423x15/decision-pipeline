from __future__ import annotations
import os
from mistralai.client import Mistral

DEFAULT_MODEL = "mistral-small-latest"


def build_synthesis_prompt(query: str, history: list[dict]) -> str:
    if history:
        context = "\n".join(
            f"- {h['tool']}({h.get('input', '')[:60]}) → {h['result']}"
            for h in history
        )
    else:
        context = "No tools were called."
    return (
        f"Question: {query}\n\n"
        f"Collected information:\n{context}\n\n"
        f"Answer concisely based only on this information."
    )


class MistralClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY missing — add it to .env")
        self._client = Mistral(api_key=key)
        self.model = model

    def synthesize(self, query: str, history: list[dict]) -> str:
        response = self._client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": build_synthesis_prompt(query, history)}],
        )
        return response.choices[0].message.content

from __future__ import annotations
import os
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"


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
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY manquant. Ajoute-le dans .env")
        self._client = OpenAI(api_key=key)
        self.model = model

    def synthesize(self, query: str, history: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_synthesis_prompt(query, history)}],
        )
        return response.choices[0].message.content

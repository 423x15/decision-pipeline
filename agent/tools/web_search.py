# agent/tools/web_search.py
from __future__ import annotations
import hashlib

_STUBS = {
    "bitcoin": "Bitcoin (BTC) : cours simulé ~65 000 USD. Capitalisation ~1.3T USD. Variation 24h: +2.1%.",
    "tesla": "Tesla (TSLA) : cours simulé ~248 USD. P/E: 65. Variation 24h: -0.8%.",
    "météo": "Météo simulée : 22°C, partiellement nuageux, vent 15 km/h.",
    "france": "France : 68 millions d'habitants. Capitale: Paris. PIB: ~2 800 Mds EUR.",
    "python": "Python 3.12 : langage interprété créé par Guido van Rossum.",
    "ia": "Intelligence artificielle : domaine simulant l'intelligence humaine. LLMs, vision, robotique.",
}

def run(query: str, history: list[dict]) -> str:
    q = query.lower()
    for key, result in _STUBS.items():
        if key in q:
            return f"[Web Search Stub] {result}"
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    views = 1000 + (seed % 50000)
    return f"[Web Search Stub] {views:,} résultats simulés pour '{query}'. Informations générales disponibles."

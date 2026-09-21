# agent/tools/datetime_tool.py
from __future__ import annotations
from datetime import datetime

_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

def run(query: str, history: list[dict]) -> str:
    now = datetime.now()
    q = query.lower()
    if any(w in q for w in ["heure", "time", "maintenant", "now"]):
        return f"Il est {now.strftime('%H:%M:%S')} le {now.day} {_MONTHS_FR[now.month-1]} {now.year}."
    if any(w in q for w in ["jour", "date", "aujourd", "today"]):
        return f"Nous sommes le {now.day} {_MONTHS_FR[now.month-1]} {now.year}."
    if any(w in q for w in ["année", "annee", "year"]):
        return f"Nous sommes en {now.year}."
    if any(w in q for w in ["mois", "month"]):
        return f"Nous sommes en {_MONTHS_FR[now.month-1]} {now.year}."
    return f"Date et heure : {now.strftime('%d/%m/%Y %H:%M:%S')}."

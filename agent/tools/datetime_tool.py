# agent/tools/datetime_tool.py
from __future__ import annotations
from datetime import datetime

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def run(query: str, history: list[dict]) -> str:
    now = datetime.now()
    q = query.lower()
    if any(w in q for w in ["heure", "time", "maintenant", "now"]):
        return f"It is {now.strftime('%H:%M:%S')} on {_MONTHS[now.month-1]} {now.day}, {now.year}."
    if any(w in q for w in ["jour", "date", "aujourd", "today"]):
        return f"Today is {_MONTHS[now.month-1]} {now.day}, {now.year}."
    if any(w in q for w in ["année", "annee", "year"]):
        return f"The current year is {now.year}."
    if any(w in q for w in ["mois", "month"]):
        return f"The current month is {_MONTHS[now.month-1]} {now.year}."
    return f"Date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}."

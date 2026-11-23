# agent/tools/text_utils.py
from __future__ import annotations
import re
from collections import Counter

_STOP = {"the","a","an","of","in","on","at","to","for","is","are","was","were",
         "and","or","but","not","with","by","from","as","it","its","this","that",
         "be","been","being","have","has","had","do","does","did","will","would",
         # French stop words (kept for bilingual support)
         "le","la","les","de","du","des","un","une","et","en","au","aux",
         "est","je","tu","il","elle","nous","vous","ils","elles","que","qui",
         "ne","pas","se","ce","sa","son","dans","sur","par","pour","avec"}

def _extract_text(query: str) -> str:
    for sep in ["dans:", "dans :", "texte:", "texte :", "analyze:", "analyse:", "text:"]:
        if sep in query.lower():
            return query[query.lower().index(sep)+len(sep):].strip()
    return query

def run(query: str, history: list[dict]) -> str:
    text = _extract_text(query)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    keywords = [w for w in words if w not in _STOP and len(w) > 2]
    top = [w for w, _ in Counter(keywords).most_common(5)]
    return (
        f"Text analysis ({len(text)} chars):\n"
        f"- Words: {len(words)}\n"
        f"- Sentences: {len(sentences)}\n"
        f"- Keywords: {', '.join(top) if top else 'none'}"
    )

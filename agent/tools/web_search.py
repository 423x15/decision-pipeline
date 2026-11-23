# agent/tools/web_search.py
from __future__ import annotations


def run(query: str, history: list[dict]) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return f"[Web Search] No results found for '{query}'."
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")[:200]
            lines.append(f"• {title}: {body}")
        return f"[Web Search] Results for '{query}':\n" + "\n".join(lines)
    except ImportError:
        return "[Web Search] ddgs not installed. Run: pip install ddgs"
    except Exception as exc:
        return f"[Web Search] Error: {exc}"

# agent/tools/__init__.py
from __future__ import annotations
from typing import Callable
from agent.tools import calculator, datetime_tool, web_search, text_utils

TOOL_REGISTRY: dict[str, Callable[[str, list[dict]], str]] = {
    "calculator": calculator.run,
    "datetime_tool": datetime_tool.run,
    "web_search": web_search.run,
    "text_utils": text_utils.run,
}

"""
LLM provider abstraction — swap OpenAI / Mistral / Anthropic without touching engine code.

Engines always work with OpenAI-format message dicts internally.
Adapters handle conversion at the API boundary.
"""
from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_use"


class LLMProvider(ABC):
    """Minimal interface all engines use."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ChatResponse: ...

    @abstractmethod
    def assistant_message(self, response: ChatResponse) -> dict:
        """Return provider-native dict to append to messages for assistant turn."""
        ...

    @abstractmethod
    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """Return provider-native dict to append to messages for a tool result."""
        ...


# ── OpenAI adapter ────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages, tools=None, response_format=None) -> ChatResponse:
        kwargs: dict = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )

    def assistant_message(self, response: ChatResponse) -> dict:
        msg: dict = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
        return msg

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ── Mistral adapter ───────────────────────────────────────────────────────────

class MistralProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        from mistralai.client import Mistral
        self._client = Mistral(api_key=api_key)
        self.model = model

    def chat(self, messages, tools=None, response_format=None) -> ChatResponse:
        kwargs: dict = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.complete(**kwargs)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        return ChatResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )

    def assistant_message(self, response: ChatResponse) -> dict:
        msg: dict = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
        return msg

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ── Anthropic adapter ─────────────────────────────────────────────────────────

def _oai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _oai_messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split system prompt + convert tool/assistant messages to Anthropic format."""
    system_parts: list[str] = []
    converted: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_parts.append(msg.get("content", ""))
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg.get("content", ""),
            }
            if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            content: list = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn.get("name", ""),
                    "input": inp,
                })
            converted.append({"role": "assistant", "content": content or msg.get("content", "")})
            continue
        converted.append(msg)

    return "\n\n".join(system_parts), converted


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages, tools=None, response_format=None) -> ChatResponse:
        system, converted = _oai_messages_to_anthropic(messages)
        kwargs: dict = dict(model=self.model, max_tokens=2048, messages=converted)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _oai_tools_to_anthropic(tools)
        if response_format and response_format.get("type") == "json_object":
            kwargs["system"] = (kwargs.get("system", "") + "\nReply ONLY with valid JSON.").strip()

        resp = self._client.messages.create(**kwargs)

        content_text: str | None = None
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input),
                ))
        return ChatResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else "stop",
        )

    def assistant_message(self, response: ChatResponse) -> dict:
        content: list = []
        if response.content:
            content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            try:
                inp = json.loads(tc.arguments)
            except json.JSONDecodeError:
                inp = {}
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": inp})
        return {"role": "assistant", "content": content or response.content or ""}

    def tool_result_message(self, tool_call_id: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content}],
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def create_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """
    provider_name : "mistral" | "openai" | "anthropic"  (default: LLM_PROVIDER env var or "mistral")
    api_key       : override — otherwise reads MISTRAL_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    model         : override — otherwise reads LLM_MODEL env var or provider default
    """
    name = (provider_name or os.environ.get("LLM_PROVIDER", "mistral")).lower()
    if name == "mistral":
        key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise ValueError("MISTRAL_API_KEY missing — add it to .env")
        m = model or os.environ.get("LLM_MODEL", "mistral-small-latest")
        return MistralProvider(api_key=key, model=m)
    if name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY missing — add it to .env")
        m = model or os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
        return AnthropicProvider(api_key=key, model=m)
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY missing — add it to .env")
    m = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return OpenAIProvider(api_key=key, model=m)

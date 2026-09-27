import json
from functools import cached_property
import uuid
from typing import List, Optional, Type, TYPE_CHECKING

from openai import AsyncOpenAI, pydantic_function_tool

from llm.i_llm_client import ILLMClient, LLMResponse, LLMResponseError
from llm.tool_schema_builder import tools_to_openai_format
from llm.messages import Message, ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel
    from tool_framework.i_tool import ITool


_OPENAI_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-3.5-turbo-16k": 16_385,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
}
# Model families matched by prefix; the longest matching prefix wins.
_OPENAI_CONTEXT_WINDOW_PREFIXES: dict[str, int] = {
    "gpt-5": 400_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "o4": 200_000,
    # OpenRouter also serves other vendors through this client.
    "claude-": 200_000,
    "gemini-": 1_048_576,
}
_OPENAI_FALLBACK_CONTEXT_WINDOW = 128_000


def _openai_context_window(model: str) -> int:
    """Estimate a model's context window from exact names, then family prefixes.

    OpenRouter-style ``vendor/model`` names and ``:variant`` suffixes are
    reduced to the bare model name before lookup; unknown modern models get a
    generous default rather than a legacy 8k window.
    """
    name = model.strip().lower().rsplit("/", 1)[-1].split(":", 1)[0]
    if name in _OPENAI_CONTEXT_WINDOWS:
        return _OPENAI_CONTEXT_WINDOWS[name]
    matches = [prefix for prefix in _OPENAI_CONTEXT_WINDOW_PREFIXES if name.startswith(prefix)]
    if matches:
        return _OPENAI_CONTEXT_WINDOW_PREFIXES[max(matches, key=len)]
    return _OPENAI_FALLBACK_CONTEXT_WINDOW


class OpenAICompatibleClient(ILLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, OpenRouter, Ollama, LM Studio, etc.).

    Uses the official `openai` SDK with configurable base_url for compatibility.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1",
        api_key: str = "not-needed",
        context_window: Optional[int] = None,
    ):
        """Configure asynchronous access to an OpenAI-compatible model endpoint."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._closed = False
        self._context_window_override = context_window

    @cached_property
    def client(self):
        """Allocate the SDK connection pool only when a request needs it."""
        if self._closed:
            raise RuntimeError("Model client is closed")
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120.0)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self.__dict__.pop("client", None)
        if client is not None:
            await client.close()

    @property
    def context_window(self) -> int:
        """Return the explicit context capacity or the locally configured model estimate and fallback."""
        if self._context_window_override is not None:
            return self._context_window_override
        return _openai_context_window(self.model)

    async def chat(
        self,
        messages: List["Message"],
        system: str,
        tools: Optional[List["ITool"]] = None,
        max_tokens: int = 4096,
        response_schema: Optional[Type["BaseModel"]] = None,
    ) -> LLMResponse:
        """Send a chat-completion request and normalize the response for the agent runtime.

        Include tools and strict JSON Schema output settings when requested;
        unparseable JSON leaves structured_data unset.
        """
        openai_messages = self._convert_messages(messages, system)

        kwargs: dict = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools_to_openai_format(tools)

        if response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    # The public SDK helper normalizes nested objects and
                    # defaulted fields to the provider's strict schema subset.
                    "schema": pydantic_function_tool(response_schema)["function"][
                        "parameters"
                    ],
                    "strict": True,
                },
            }

        response = await self.client.chat.completions.create(**kwargs)

        result = self._parse_response(response)
        if response_schema and result.message.text:
            try:
                result.structured_data = json.loads(result.message.text)
            except (json.JSONDecodeError, ValueError):
                result.structured_data = None
        return result

    def _convert_messages(self, messages: List[Message], system: str) -> List[dict]:
        """Adapt neutral conversation data immediately before a completion call."""
        converted = [{"role": "system", "content": system}]
        for message in messages:
            if message.role == "tool":
                converted.append({
                    "role": "tool", "tool_call_id": message.tool_call_id,
                    "content": f"[Tool error]\n{message.text}" if message.is_error else message.text,
                })
                continue
            item = {"role": message.role, "content": message.text}
            if message.images:
                item["content"] = ([{"type": "text", "text": message.text}] if message.text else []) + [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image.media_type};base64,{image.data}"
                    }} for image in message.images
                ]
            if message.tool_calls:
                extras = message.provider_data.get(f"openai:{self.base_url}:{self.model}", {})
                item["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": json.dumps(call.arguments)
                    }, **({"extra_content": extras[call.id]} if call.id in extras else {})}
                    for call in message.tool_calls
                ]
                if not message.text:
                    item["content"] = None
            converted.append(item)
        return converted

    # ── Response parsing ────────────────────────────────────────────

    def _parse_response(self, response) -> LLMResponse:
        """Normalize the first chat-completion choice into agent content, stop reason, and usage.

        Reject unusable responses before any tool calls reach the executor.
        Generate missing tool-call identifiers for later result matching.
        """
        if not response.choices:
            raise LLMResponseError("OpenAI", "empty_response")
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"
        if getattr(message, "refusal", None) or finish_reason == "content_filter":
            raise LLMResponseError("OpenAI", "refusal")

        assistant = Message(role="assistant", text=message.content or "")
        extras = {}

        if message.tool_calls:
            for tool_call in message.tool_calls:
                func = tool_call.function
                try:
                    args = json.loads(func.arguments)
                except (json.JSONDecodeError, TypeError):
                    raise LLMResponseError("OpenAI", "invalid_tool_arguments") from None
                if not isinstance(args, dict):
                    raise LLMResponseError("OpenAI", "invalid_tool_arguments")

                call_id = tool_call.id or str(uuid.uuid4())
                extra = getattr(tool_call, "extra_content", None)
                if extra:
                    extras[call_id] = extra
                assistant.tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=func.name,
                        arguments=args,
                    )
                )

        if extras:
            assistant.provider_data[f"openai:{self.base_url}:{self.model}"] = extras

        stop_reason = "end_turn"
        if finish_reason == "tool_calls":
            stop_reason = "tool_calls"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        if (
            not assistant.text.strip()
            and not assistant.tool_calls
            and stop_reason != "max_tokens"
        ):
            raise LLMResponseError("OpenAI", "empty_response")

        usage_data = response.usage
        usage = {
            "input_tokens": getattr(usage_data, "prompt_tokens", 0) if usage_data else 0,
            "output_tokens": getattr(usage_data, "completion_tokens", 0) if usage_data else 0,
        }

        return LLMResponse(
            message=assistant,
            stop_reason=stop_reason,
            usage=usage,
        )

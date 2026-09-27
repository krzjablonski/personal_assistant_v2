import json
from functools import cached_property
from copy import deepcopy
import anthropic
from anthropic.types import (
    Message as AnthropicMessage,
    MessageParam as AnthropicMessageParam,
    OutputConfigParam,
    TextBlockParam,
    ToolUnionParam,
)
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from typing import List, Optional, Type, TYPE_CHECKING, cast

from llm.i_llm_client import ILLMClient, LLMResponse, LLMResponseError
from llm.tool_schema_builder import tools_to_anthropic_format
from llm.messages import Message, ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel
    from tool_framework.i_tool import ITool


_ANTHROPIC_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-2.0": 100_000,
}
# Model families matched by prefix; the longest matching prefix wins.
_ANTHROPIC_CONTEXT_WINDOW_PREFIXES: dict[str, int] = {
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3": 200_000,
    "claude-2.1": 200_000,
}
_ANTHROPIC_FALLBACK_CONTEXT_WINDOW = 200_000


def _anthropic_context_window(model: str) -> int:
    """Estimate a Claude model's context window from exact names, then family prefixes."""
    name = model.strip().lower()
    if name in _ANTHROPIC_CONTEXT_WINDOWS:
        return _ANTHROPIC_CONTEXT_WINDOWS[name]
    matches = [prefix for prefix in _ANTHROPIC_CONTEXT_WINDOW_PREFIXES if name.startswith(prefix)]
    if matches:
        return _ANTHROPIC_CONTEXT_WINDOW_PREFIXES[max(matches, key=len)]
    return _ANTHROPIC_FALLBACK_CONTEXT_WINDOW


class AnthropicClient(ILLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        context_window: Optional[int] = None,
        *,
        api_key: str,
    ):
        """Configure an asynchronous Claude client with a model, credentials, and optional context override."""
        self.api_key = api_key
        self._closed = False
        self.model = model
        self._context_window_override = context_window

    @cached_property
    def client(self):
        """Allocate the SDK connection pool only when a request needs it."""
        if self._closed:
            raise RuntimeError("Model client is closed")
        return anthropic.AsyncAnthropic(api_key=self.api_key)

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
        return _anthropic_context_window(self.model)

    async def chat(
        self,
        messages: List["Message"],
        system: str,
        tools: Optional[List["ITool"]] = None,
        max_tokens: int = 4096,
        response_schema: Optional[Type["BaseModel"]] = None,
    ) -> LLMResponse:
        """Send a Claude request and normalize its content, stop reason, and token usage.

        Include tools and a requested JSON output schema when supplied; invalid
        JSON output leaves structured_data unset.
        """
        msg_dicts = self._convert_messages(messages)

        kwargs: MessageCreateParamsNonStreaming = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": False,
            "system": cast(
                list[TextBlockParam],
                [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            ),
            "messages": msg_dicts,
        }

        if tools:
            converted = cast(
                list[ToolUnionParam], tools_to_anthropic_format(tools)
            )
            if converted:
                converted[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = converted

        if response_schema:
            output_config: OutputConfigParam = {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        **response_schema.model_json_schema(),
                        "additionalProperties": False,
                    },
                }
            }
            kwargs["output_config"] = output_config

        response: AnthropicMessage = await self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise LLMResponseError("Anthropic", "refusal")
        assistant = self._parse_message(response.content)
        if (
            not assistant.text.strip()
            and not assistant.tool_calls
            and response.stop_reason != "max_tokens"
        ):
            raise LLMResponseError("Anthropic", "empty_response")

        structured_data = None
        if response_schema:
            raw = assistant.text
            try:
                structured_data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                structured_data = None

        usage: dict[str, int] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        cache_creation_input_tokens = response.usage.cache_creation_input_tokens
        if cache_creation_input_tokens is not None:
            usage["cache_creation_input_tokens"] = cache_creation_input_tokens

        cache_read_input_tokens = response.usage.cache_read_input_tokens
        if cache_read_input_tokens is not None:
            usage["cache_read_input_tokens"] = cache_read_input_tokens

        stop_reason = response.stop_reason or "end_turn"
        if stop_reason == "tool_use":
            stop_reason = "tool_calls"

        return LLMResponse(
            message=assistant,
            stop_reason=stop_reason,
            usage=usage,
            structured_data=structured_data,
        )

    def _convert_messages(self, messages: List[Message]) -> list[AnthropicMessageParam]:
        """Build Claude blocks here; consecutive tool replies share one user turn."""
        converted: list[dict] = []
        for message in messages:
            if message.role == "tool":
                result = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.text,
                }
                if message.is_error:
                    result["is_error"] = True
                if converted and converted[-1]["role"] == "user":
                    content = converted[-1]["content"]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                        converted[-1]["content"] = content
                    content.insert(sum(b["type"] == "tool_result" for b in content), result)
                else:
                    converted.append({"role": "user", "content": [result]})
                continue
            replay = message.provider_data.get("anthropic")
            if message.role == "assistant" and replay and replay.get("model", self.model) == self.model:
                current = message.to_dict()
                current.pop("provider_data", None)
                if current != replay["message"]:
                    raise ValueError(
                        "Cannot edit a signed Anthropic message without clearing its provider_data"
                    )
                # Copy so assembling later requests never mutates stored history.
                content = deepcopy(replay["content"])
            else:
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.media_type,
                            "data": image.data,
                        },
                    }
                    for image in message.images
                ]
                if message.text:
                    content.append({"type": "text", "text": message.text})
                content.extend(
                    {"type": "tool_use", "id": call.id,
                     "name": call.name, "input": call.arguments}
                    for call in message.tool_calls
                )
                if not message.images and not message.tool_calls:
                    content = message.text
            converted.append({"role": message.role, "content": content})
        return cast(list[AnthropicMessageParam], converted)

    def _parse_message(self, sdk_content) -> Message:
        """Extract application fields and retain signed Claude content for replay."""
        message = Message(role="assistant")
        texts, reasoning, raw = [], [], []
        signed = False
        for block in sdk_content:
            data = (
                block.model_dump(mode="json", exclude_none=True)
                if hasattr(block, "model_dump") else vars(block).copy()
            )
            raw.append(data)
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                if not isinstance(block.input, dict):
                    raise LLMResponseError("Anthropic", "invalid_tool_arguments")
                message.tool_calls.append(ToolCall(block.id, block.name, block.input))
            elif block.type == "thinking":
                reasoning.append(block.thinking)
                signed = True
            elif block.type == "redacted_thinking":
                signed = True
            else:
                raise ValueError(f"Unsupported Anthropic response content: {block.type}")
        message.text = "\n".join(texts)
        message.reasoning = "\n".join(reasoning)
        if signed:
            message.provider_data["anthropic"] = {
                "model": self.model, "content": raw, "message": message.to_dict(),
            }
        return message

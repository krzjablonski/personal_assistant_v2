import json
from functools import cached_property
import base64
import uuid
from typing import List, Optional, Type, TYPE_CHECKING

from google import genai
from google.genai import types as genai_types

from llm.i_llm_client import ILLMClient, LLMResponse, LLMResponseError
from llm.tool_schema_builder import build_parameters_schema
from llm.messages import Message, ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel
    from tool_framework.i_tool import ITool

_GEMINI_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-pro": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    "gemini-2.5-flash-preview": 1_048_576,
    "gemini-2.5-pro-preview": 1_048_576,
}
_GEMINI_FALLBACK_CONTEXT_WINDOW = 1_048_576
_GEMINI_HTTP_TIMEOUT_MS = 120_000


class GeminiClient(ILLMClient):
    """Client for Google Gemini API using the native google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        context_window: Optional[int] = None,
    ):
        """Configure a Gemini client for the selected model and context budget.

        Credentials are resolved by the caller; reject a missing key.
        """
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.model = model
        self.api_key = api_key
        self._closed = False
        self._context_window_override = context_window

    @cached_property
    def client(self):
        """Allocate the SDK connection pool only when a request needs it."""
        if self._closed:
            raise RuntimeError("Model client is closed")
        return genai.Client(
            api_key=self.api_key,
            http_options=genai_types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self.__dict__.pop("client", None)
        if client is not None:
            try:
                await client.aio.aclose()
            finally:
                client.close()

    @property
    def context_window(self) -> int:
        """Return the explicit context capacity or the locally configured Gemini estimate and fallback."""
        if self._context_window_override is not None:
            return self._context_window_override
        return _GEMINI_CONTEXT_WINDOWS.get(self.model, _GEMINI_FALLBACK_CONTEXT_WINDOW)

    async def chat(
        self,
        messages: List["Message"],
        system: str,
        tools: Optional[List["ITool"]] = None,
        max_tokens: int = 4096,
        response_schema: Optional[Type["BaseModel"]] = None,
    ) -> LLMResponse:
        """Send a Gemini request and return normalized content, usage, and optional parsed JSON.

        Expose requested tool calls for the agent to execute instead of enabling
        SDK automatic function execution.
        """
        contents = self._convert_messages(messages, tools_active=bool(tools))

        config_kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "thinking_config": genai_types.ThinkingConfig(include_thoughts=True),
        }

        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
            config_kwargs["automatic_function_calling"] = (
                genai_types.AutomaticFunctionCallingConfig(
                    disable=True,
                )
            )

        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema.model_json_schema()

        config = genai_types.GenerateContentConfig(**config_kwargs)

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        result = self._parse_response(response)

        if response_schema and result.message.text:
            try:
                result.structured_data = json.loads(result.message.text)
            except (json.JSONDecodeError, ValueError):
                result.structured_data = None

        return result

    def _convert_messages(
        self, messages: List[Message], tools_active: bool = True
    ) -> List[genai_types.Content]:
        """Adapt neutral messages and group tool replies into Gemini user turns."""
        names = {call.id: call.name for message in messages for call in message.tool_calls}
        contents = []
        previous_was_tool = False
        for message in messages:
            role = "model" if message.role == "assistant" else "user"
            parts = self._message_parts(message, names, tools_active)
            if not parts:
                continue
            if message.role == "tool" and previous_was_tool:
                contents[-1].parts.extend(parts)
            else:
                contents.append(genai_types.Content(role=role, parts=parts))
            previous_was_tool = message.role == "tool"
        return contents

    def _message_parts(self, message: Message, names: dict, tools_active: bool) -> list:
        if message.role == "tool":
            name = message.tool_name or names.get(message.tool_call_id, "unknown")
            if tools_active:
                return [genai_types.Part(function_response=genai_types.FunctionResponse(
                    id=message.tool_call_id, name=name,
                    response={"error" if message.is_error else "result": message.text},
                ))]
            label = "Tool Error" if message.is_error else "Tool Result"
            return [genai_types.Part(text=f"[{label}: {name}]\n{message.text}")]

        replay = message.provider_data.get("gemini")
        if (message.role == "assistant" and replay and replay["model"] == self.model
                and (tools_active or not message.tool_calls)):
            current = message.to_dict()
            current.pop("provider_data", None)
            if current != replay["message"]:
                raise ValueError("Cannot edit a signed Gemini message without clearing its provider_data")
            return [genai_types.Part.model_validate(part) for part in replay["parts"]]

        parts = [genai_types.Part(text=message.text)] if message.text else []
        parts.extend(genai_types.Part(inline_data=genai_types.Blob(
            mime_type=image.media_type, data=base64.b64decode(image.data),
        )) for image in message.images)
        for call in message.tool_calls:
            if tools_active:
                # Google's documented marker for tool history from another model
                # or deterministic runtime calls that have no Gemini signature.
                parts.append(genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=call.id, name=call.name, args=call.arguments,
                    ),
                    thought_signature=b"skip_thought_signature_validator",
                ))
            else:
                parts.append(genai_types.Part(
                    text=f"[Tool Call: {call.name}]\nArguments: {json.dumps(call.arguments, indent=2)}"
                ))
        return parts

    # ── Tool conversion ─────────────────────────────────────────────

    def _convert_tools(self, tools: List["ITool"]) -> List[genai_types.Tool]:
        """Describe available tools as Gemini function declarations using the adapter's schema subset."""
        declarations = []
        for tool in tools:
            schema = build_parameters_schema(tool)
            schema = self._sanitize_schema_for_gemini(schema)
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=schema,
                )
            )
        return [genai_types.Tool(function_declarations=declarations)]

    def _sanitize_schema_for_gemini(
        self, schema: dict, *, in_properties: bool = False
    ) -> dict:
        """Reduce a JSON Schema to the subset emitted by this Gemini adapter.

        Drop selected metadata and composition fields, flatten a single non-null
        anyOf branch, and retain property names that match filtered metadata keys.
        This conversion can weaken the original validation constraints.
        """
        _UNSUPPORTED_KEYS = {
            "additionalProperties",
            "additional_properties",
            "$defs",
            "$ref",
            "$schema",
            "allOf",
            "oneOf",
            "title",
            "default",
            "examples",
            "const",
        }

        if not isinstance(schema, dict):
            return schema

        result = {}
        for key, value in schema.items():
            # Inside `properties`, dict keys are user-defined property names,
            # so they must not be filtered as JSON Schema metadata.
            if not in_properties and key in _UNSUPPORTED_KEYS:
                continue

            if key == "anyOf" and isinstance(value, list):
                # Flatten simple Optional patterns: anyOf[{type: X}, {type: null}]
                non_null = [
                    v
                    for v in value
                    if not (isinstance(v, dict) and v.get("type") == "null")
                ]
                if len(non_null) == 1:
                    result.update(self._sanitize_schema_for_gemini(non_null[0]))
                # else: drop anyOf entirely — Gemini can't handle it
                continue

            if key == "properties" and isinstance(value, dict):
                result[key] = self._sanitize_schema_for_gemini(
                    value, in_properties=True
                )
                continue

            if isinstance(value, dict):
                result[key] = self._sanitize_schema_for_gemini(
                    value, in_properties=False
                )
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._sanitize_schema_for_gemini(item, in_properties=False)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                result[key] = value

        return result

    # ── Response parsing ────────────────────────────────────────────

    def _parse_response(self, response) -> LLMResponse:
        """Normalize the first Gemini candidate into text, reasoning, tool calls, and usage data.

        Retain tool-call thought signatures and supply identifiers when absent
        so later tool results can be associated with their calls.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if (
            block_reason
            and getattr(block_reason, "value", block_reason) != "BLOCKED_REASON_UNSPECIFIED"
        ):
            raise LLMResponseError("Gemini", "refusal")
        if not response.candidates:
            raise LLMResponseError("Gemini", "empty_response")
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason = getattr(finish_reason, "value", finish_reason)
        if finish_reason in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
            raise LLMResponseError("Gemini", "invalid_tool_arguments")
        if finish_reason in {
            "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
            "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
        }:
            raise LLMResponseError("Gemini", "refusal")
        if finish_reason not in {None, "FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS"}:
            raise LLMResponseError("Gemini", "unsupported_response")
        message = Message(role="assistant")
        texts, reasoning, raw = [], [], []
        signed = False
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                signature = getattr(part, "thought_signature", None)
                signed = signed or bool(signature)
                # Store SDK parts as JSON values, never SDK objects or bytes.
                if hasattr(part, "model_dump"):
                    data = part.model_dump(mode="json", exclude_none=True)
                else:
                    data = {}
                    if part.text is not None:
                        data["text"] = part.text
                    if part.thought:
                        data["thought"] = True
                    if signature:
                        data["thought_signature"] = base64.b64encode(signature).decode() if isinstance(signature, bytes) else signature
                    if part.function_call:
                        data["function_call"] = vars(part.function_call).copy()
                if part.function_call:
                    fc = part.function_call
                    if fc.args is not None and not isinstance(fc.args, dict):
                        raise LLMResponseError("Gemini", "invalid_tool_arguments")
                    call_id = getattr(fc, "id", None) or str(uuid.uuid4())
                    message.tool_calls.append(ToolCall(call_id, fc.name, dict(fc.args or {})))
                    # Keep generated IDs consistent in the replay and tool replies.
                    data["function_call"]["id"] = call_id
                elif part.text:
                    (reasoning if part.thought else texts).append(part.text)
                raw.append(data)
        message.text = "\n".join(texts)
        message.reasoning = "\n".join(reasoning)
        if signed:
            message.provider_data["gemini"] = {
                "model": self.model, "parts": raw, "message": message.to_dict(),
            }

        # Determine stop reason
        if finish_reason == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif message.tool_calls:
            stop_reason = "tool_calls"
        else:
            stop_reason = "end_turn"

        if (
            not message.text.strip()
            and not message.tool_calls
            and stop_reason != "max_tokens"
        ):
            raise LLMResponseError("Gemini", "empty_response")

        # Usage
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": (
                getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
            ),
            "output_tokens": (
                getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
            ),
        }
        if usage_meta and hasattr(usage_meta, "thoughts_token_count"):
            usage["thinking_tokens"] = usage_meta.thoughts_token_count

        return LLMResponse(
            message=message,
            stop_reason=stop_reason,
            usage=usage,
        )

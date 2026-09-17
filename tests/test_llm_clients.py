from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

try:
    from google import genai as _google_genai  # type: ignore[import-not-found]
except ImportError:
    google_module = sys.modules.setdefault("google", types.ModuleType("google"))
    genai_module = types.ModuleType("google.genai")
    genai_types_module = types.ModuleType("google.genai.types")

    class _BaseStub:
        def __init__(self, **kwargs):
            """Provide SDK-style attribute containers when the optional Gemini SDK is unavailable."""
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _StubClient:
        def __init__(self, api_key: str):
            """Provide a mockable asynchronous Gemini client shape without installing the SDK."""
            self.api_key = api_key
            self.aio = SimpleNamespace(
                models=SimpleNamespace(generate_content=AsyncMock())
            )

    for class_name in (
        "ThinkingConfig",
        "AutomaticFunctionCallingConfig",
        "GenerateContentConfig",
        "Content",
        "Part",
        "Blob",
        "FunctionCall",
        "FunctionResponse",
        "FunctionDeclaration",
        "Tool",
    ):
        setattr(genai_types_module, class_name, type(class_name, (_BaseStub,), {}))

    genai_module.Client = _StubClient
    genai_module.types = genai_types_module
    google_module.genai = genai_module
    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.types"] = genai_types_module

from llm.messages import Message, ToolCall
from llm.anthropic_client import AnthropicClient
from llm.gemini_client import GeminiClient
from llm.i_llm_client import ILLMClient, LLMResponse
from llm.langfuse_llm_client import LangfuseTrackedLLMClient
from llm.openai_compatible_client import OpenAICompatibleClient


class StructuredReply(BaseModel):
    answer: str


class NestedReply(BaseModel):
    item: StructuredReply
    alternatives: list[StructuredReply] = []
    detail: str | None = None


class LLMResponseAssertionsMixin:
    def assert_response_contract(
        self,
        response: LLMResponse,
        *,
        stop_reason: str,
        usage: dict,
        text_content: str,
        tool_use_count: int = 0,
        structured_data: dict | None = None,
    ) -> None:
        """Check the common response fields every provider adapter must expose to the agent."""
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.stop_reason, stop_reason)
        self.assertEqual(response.usage, usage)
        self.assertEqual(response.message.text, text_content)
        self.assertEqual(len(response.message.tool_calls), tool_use_count)
        self.assertEqual(response.has_tool_calls, stop_reason == "tool_calls")
        self.assertEqual(response.structured_data, structured_data)


class TestAnthropicClient(
    unittest.IsolatedAsyncioTestCase, LLMResponseAssertionsMixin
):
    async def test_success_criterion_11_anthropic_retains_normalized_contract(self) -> None:
        """Verify Anthropic responses retain thinking, text, tools, usage, and parsed structured data."""
        sdk_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="thinking",
                    thinking="internal reasoning",
                    signature="sig-1",
                ),
                SimpleNamespace(type="text", text='{"answer":"ok"}'),
                SimpleNamespace(
                    type="tool_use",
                    id="tool-1",
                    name="weather_lookup",
                    input={"city": "Warsaw"},
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=2,
            ),
            stop_reason="tool_use",
        )

        client = AnthropicClient(model="claude-test", api_key="test-key")

        client.client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=sdk_response))
        )

        response = await client.chat(
            messages=[Message(role="user", text="hello")],
            system="system prompt",
            response_schema=StructuredReply,
        )

        self.assert_response_contract(
            response,
            stop_reason="tool_calls",
            usage={
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
            },
            text_content='{"answer":"ok"}',
            tool_use_count=1,
            structured_data={"answer": "ok"},
        )
        self.assertEqual(response.message.reasoning, "internal reasoning")
        self.assertIsInstance(response.message.tool_calls[0], ToolCall)


class TestGeminiClient(unittest.IsolatedAsyncioTestCase, LLMResponseAssertionsMixin):
    async def test_chat_returns_tool_use_llm_response(self) -> None:
        """Verify Gemini tool responses preserve visible text, thinking usage, and thought signatures."""
        sdk_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                thought=True,
                                text="internal reasoning",
                                function_call=None,
                            ),
                            SimpleNamespace(
                                thought=False,
                                text="visible answer",
                                function_call=None,
                            ),
                            SimpleNamespace(
                                thought=False,
                                text=None,
                                function_call=SimpleNamespace(
                                    id="tool-1",
                                    name="weather_lookup",
                                    args={"city": "Warsaw"},
                                ),
                                thought_signature="sig-2",
                            ),
                        ]
                    ),
                    finish_reason=None,
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=19,
                candidates_token_count=5,
                thoughts_token_count=2,
            ),
        )

        client = GeminiClient(model="gemini-test", api_key="test-key")
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=sdk_response)
                )
            )
        )

        response = await client.chat(
            messages=[Message(role="user", text="hello")],
            system="system prompt",
        )

        self.assert_response_contract(
            response,
            stop_reason="tool_calls",
            usage={
                "input_tokens": 19,
                "output_tokens": 5,
                "thinking_tokens": 2,
            },
            text_content="visible answer",
            tool_use_count=1,
        )
        self.assertEqual(response.message.reasoning, "internal reasoning")
        self.assertIsInstance(response.message.tool_calls[0], ToolCall)
        self.assertIn("gemini", response.message.provider_data)

    async def test_success_criterion_11_gemini_retains_structured_contract(self) -> None:
        """Verify Gemini JSON replies expose parsed data through the common completed-response contract."""
        sdk_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                thought=False,
                                text='{"answer":"ok"}',
                                function_call=None,
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=9,
                candidates_token_count=4,
            ),
        )

        client = GeminiClient(model="gemini-test", api_key="test-key")
        client.client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(return_value=sdk_response)
                )
            )
        )

        response = await client.chat(
            messages=[Message(role="user", text="hello")],
            system="system prompt",
            response_schema=StructuredReply,
        )

        self.assert_response_contract(
            response,
            stop_reason="end_turn",
            usage={"input_tokens": 9, "output_tokens": 4},
            text_content='{"answer":"ok"}',
            structured_data={"answer": "ok"},
        )


class TestOpenAICompatibleClient(
    unittest.IsolatedAsyncioTestCase, LLMResponseAssertionsMixin
):

    async def test_requests_emit_strict_nested_schemas(self) -> None:
        """Validate the outgoing schema, including optional and nested fields."""
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"answer":"ok"}', tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )
        create = AsyncMock(return_value=sdk_response)
        client = OpenAICompatibleClient(api_key="test-key")
        await client.client.close()
        client.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ))

        def assert_strict(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(value.get("additionalProperties"), False)
                    self.assertEqual(set(value["required"]), set(value["properties"]))
                for child in value.values():
                    assert_strict(child)
            elif isinstance(value, list):
                for child in value:
                    assert_strict(child)

        tool_schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        tools = [SimpleNamespace(name="weather_lookup", description="Look up weather",
                                 input_schema=tool_schema)]
        for schema in (StructuredReply, NestedReply):
            with self.subTest(schema=schema.__name__):
                before = schema.model_json_schema()
                await client.chat([Message("user", "hello")], "instructions",
                                  tools=tools, response_schema=schema)
                output_format = create.call_args.kwargs["response_format"]["json_schema"]
                self.assertTrue(output_format["strict"])
                assert_strict(output_format["schema"])
                self.assertEqual(schema.model_json_schema(), before)
                tool = create.call_args.kwargs["tools"][0]["function"]
                self.assertEqual(tool["name"], "weather_lookup")
                self.assertEqual(tool["parameters"], tool_schema)

    async def test_chat_returns_tool_use_llm_response(self) -> None:
        """Verify compatible-provider tool calls normalize into the agent’s text and tool response types."""
        sdk_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="visible answer",
                        tool_calls=[
                            SimpleNamespace(
                                id="tool-1",
                                function=SimpleNamespace(
                                    name="weather_lookup",
                                    arguments='{"city":"Warsaw"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=8),
        )

        client = OpenAICompatibleClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
            api_key="test-key",
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=sdk_response))
            )
        )

        response = await client.chat(
            messages=[Message(role="user", text="hello")],
            system="system prompt",
        )

        self.assert_response_contract(
            response,
            stop_reason="tool_calls",
            usage={"input_tokens": 13, "output_tokens": 8},
            text_content="visible answer",
            tool_use_count=1,
        )
        self.assertIsInstance(response.message.tool_calls[0], ToolCall)

    async def test_success_criterion_11_openai_retains_structured_contract(self) -> None:
        """Verify compatible-provider JSON replies expose parsed data and normalized completion metadata."""
        sdk_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"answer":"ok"}',
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )

        client = OpenAICompatibleClient(
            base_url="http://localhost:1234/v1",
            model="test-model",
            api_key="test-key",
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=sdk_response))
            )
        )

        response = await client.chat(
            messages=[Message(role="user", text="hello")],
            system="system prompt",
            response_schema=StructuredReply,
        )

        self.assert_response_contract(
            response,
            stop_reason="end_turn",
            usage={"input_tokens": 7, "output_tokens": 3},
            text_content='{"answer":"ok"}',
            structured_data={"answer": "ok"},
        )


class FakeInnerLLMClient(ILLMClient):
    def __init__(self, response: LLMResponse):
        """Store the response and call history used to test the tracing wrapper."""
        self.response = response
        self.calls: list[dict] = []

    @property
    def context_window(self) -> int:
        """Provide a fixed inner-client context budget for wrapper tests."""
        return 1234

    async def chat(
        self,
        messages,
        system,
        tools=None,
        max_tokens: int = 4096,
        response_schema=None,
    ) -> LLMResponse:
        """Record forwarded chat arguments and return the configured response unchanged."""
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
                "response_schema": response_schema,
            }
        )
        return self.response


class FakeGeneration:
    def __init__(self):
        """Initialize an empty record of tracing-generation updates."""
        self.updated_with: dict | None = None
        self.ended = False

    def end(self):
        self.ended = True

    def update(self, **kwargs) -> None:
        """Capture the latest generation output and usage update for assertions."""
        self.updated_with = kwargs


class FakeLangfuseClient:
    def __init__(self):
        """Prepare a fake observation recorder and generation context for tracing tests."""
        self.observation_kwargs: dict | None = None
        self.generation = FakeGeneration()

    def start_observation(self, **kwargs):
        """Record observation metadata and supply the fake generation context."""
        self.observation_kwargs = kwargs
        return self.generation


class TestLangfuseTrackedLLMClient(
    unittest.IsolatedAsyncioTestCase, LLMResponseAssertionsMixin
):
    async def test_success_criterion_11_langfuse_retains_normalized_contract(self) -> None:
        """Ensure tracing preserves the inner response and schema forwarding while recording input, output, and usage."""
        inner_response = LLMResponse(
            message=Message(role="assistant", text='visible answer'),
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
        )
        inner_client = FakeInnerLLMClient(inner_response)
        fake_langfuse = FakeLangfuseClient()
        messages = [Message(role="user", text="hello", provider_data={"vendor": {"signature": "private"}})]

        with patch(
            "llm.langfuse_llm_client.get_langfuse",
            return_value=fake_langfuse,
            create=True,
        ):
            client = LangfuseTrackedLLMClient(inner=inner_client, model_name="test", content_mode="redacted")
            response = await client.chat(
                messages=messages,
                system="system prompt",
                response_schema=StructuredReply,
            )

        self.assertIs(response, inner_response)
        self.assert_response_contract(
            response,
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
            text_content="visible answer",
        )
        self.assertEqual(len(inner_client.calls), 1)
        self.assertIs(inner_client.calls[0]["response_schema"], StructuredReply)
        recorded = fake_langfuse.observation_kwargs
        self.assertEqual(recorded["as_type"], "generation")
        self.assertEqual(recorded["model"], "test")
        self.assertEqual(recorded["input"]["system"], "system prompt")
        self.assertEqual(recorded["input"]["messages"][0]["text"], "hello")
        self.assertNotIn("provider_data", recorded["input"]["messages"][0])
        self.assertNotIn("private", repr(recorded))
        self.assertEqual(recorded["metadata"]["trace_content"], "redacted")
        self.assertTrue(fake_langfuse.generation.ended)
        self.assertEqual(
            fake_langfuse.generation.updated_with,
            {
                "output": {
                    "role": "assistant", "text": "visible answer", "reasoning": "", "images": [],
                    "tool_calls": [], "tool_name": None, "is_error": False,
                },
                "usage_details": {"input": 5, "output": 2},
                "metadata": {"outcome": "completed"},
            },
        )

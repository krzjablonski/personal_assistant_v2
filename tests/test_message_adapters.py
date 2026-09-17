from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock, ToolUseBlock
from google.genai import types

from llm.anthropic_client import AnthropicClient
from llm.gemini_client import GeminiClient
from llm.messages import Image, Message, ToolCall
from llm.openai_compatible_client import OpenAICompatibleClient


class TestMessageAdapters(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.anthropic = AnthropicClient(api_key="test-key")
        self.openai = OpenAICompatibleClient(api_key="test-key")
        self.gemini = GeminiClient(api_key="test-key")
        self.history = [
            Message("user", "Look", images=[Image("aGVsbG8=", "image/png")]),
            Message("assistant", "Checking", tool_calls=[
                ToolCall("c1", "lookup", {"city": "Warsaw"}),
                ToolCall("c2", "lookup", {"city": "Paris"}),
            ]),
            Message("tool", "Sunny", tool_call_id="c1", tool_name="lookup"),
            Message("tool", "Unavailable", tool_call_id="c2", tool_name="lookup", is_error=True),
            Message("assistant", "Warsaw is sunny."),
        ]

    async def asyncTearDown(self):
        for client in (self.anthropic.client, self.openai.client):
            if hasattr(client, "close"):
                await client.close()
        if hasattr(self.gemini.client, "aio") and hasattr(self.gemini.client.aio, "aclose"):
            await self.gemini.client.aio.aclose()
        if hasattr(self.gemini.client, "close"):
            self.gemini.client.close()

    def assert_provider_error(self, code, callback):
        with self.assertRaises(ValueError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn("private-response", str(caught.exception))

    def test_openai_invalid_arguments_never_become_default_empty_calls(self):
        for arguments in ('{"private-response":', "[]", "null", "42", None):
            with self.subTest(arguments=arguments):
                response = SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="tool_calls", message=SimpleNamespace(
                        content=None, tool_calls=[SimpleNamespace(
                            id="c1", function=SimpleNamespace(name="mutate", arguments=arguments),
                        )],
                    ),
                )], usage=None)
                self.assert_provider_error("invalid_tool_arguments", lambda: self.openai._parse_response(response))

    def test_openai_refusal_and_empty_results_are_explicit_errors(self):
        for response, code in (
            (SimpleNamespace(choices=[]), "empty_response"),
            (SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content=None, tool_calls=None, refusal="private-response",
            ))], usage=None), "refusal"),
            (SimpleNamespace(choices=[SimpleNamespace(finish_reason="content_filter", message=SimpleNamespace(
                content="private-response", tool_calls=None,
            ))], usage=None), "refusal"),
            (SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="", tool_calls=None,
            ))], usage=None), "empty_response"),
        ):
            with self.subTest(code=code):
                self.assert_provider_error(code, lambda: self.openai._parse_response(response))

    def test_gemini_blocked_and_empty_results_are_explicit_errors(self):
        responses = (
            (types.GenerateContentResponse(), "empty_response"),
            (types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY,
                block_reason_message="private-response",
            )), "refusal"),
            (types.GenerateContentResponse(candidates=[types.Candidate(
                finish_reason=types.FinishReason.SAFETY,
                content=types.Content(parts=[types.Part(text="private-response")]),
            )]), "refusal"),
            (types.GenerateContentResponse(candidates=[types.Candidate(
                finish_reason=types.FinishReason.STOP, content=types.Content(parts=[]),
            )]), "empty_response"),
        )
        for response, code in responses:
            with self.subTest(code=code):
                self.assert_provider_error(code, lambda: self.gemini._parse_response(response))

    def test_gemini_nonobject_tool_arguments_are_rejected(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(
            finish_reason=None, content=SimpleNamespace(parts=[SimpleNamespace(
                text=None, thought=False, function_call=SimpleNamespace(
                    id="c1", name="mutate", args=[("private-response", "value")],
                ),
            )]),
        )], usage_metadata=None)
        self.assert_provider_error("invalid_tool_arguments", lambda: self.gemini._parse_response(response))

    def test_gemini_truncated_tool_response_does_not_request_execution(self):
        response = self.gemini._parse_response(types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason=types.FinishReason.MAX_TOKENS,
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                name="mutate", args={"value": "partial"},
            ))]),
        )]))
        self.assertEqual(response.stop_reason, "max_tokens")
        self.assertFalse(response.has_tool_calls)

    def test_gemini_malformed_function_stop_reason_rejects_apparent_tool_calls(self):
        response = types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                name="mutate", args={},
            ))]),
        )])
        self.assert_provider_error("invalid_tool_arguments", lambda: self.gemini._parse_response(response))

    def test_anthropic_nonobject_tool_arguments_are_rejected(self):
        self.assert_provider_error("invalid_tool_arguments", lambda: self.anthropic._parse_message([
            SimpleNamespace(type="tool_use", id="c1", name="mutate", input=["private-response"]),
        ]))

    async def test_anthropic_refusal_and_empty_results_are_explicit_errors(self):
        await self.anthropic.client.close()
        for stop_reason, content, code in (
            ("refusal", [TextBlock(type="text", text="private-response")], "refusal"),
            ("end_turn", [], "empty_response"),
        ):
            with self.subTest(code=code):
                self.anthropic.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
                    return_value=SimpleNamespace(stop_reason=stop_reason, content=content,
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                            cache_creation_input_tokens=None, cache_read_input_tokens=None)),
                )))
                with self.assertRaises(ValueError) as caught:
                    await self.anthropic.chat([Message("user", "hello")], "instructions")
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("private-response", str(caught.exception))

    async def test_same_neutral_history_reaches_all_three_provider_boundaries(self):
        before = deepcopy(self.history)
        self.anthropic.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(content=[TextBlock(type="text", text="Done")], stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                    cache_creation_input_tokens=None, cache_read_input_tokens=None)))))
        self.openai.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content="Done", tool_calls=None))], usage=None)))))
        self.gemini.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(
            return_value=types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text="Done")]))])))))

        for client in (self.anthropic, self.openai, self.gemini):
            response = await client.chat(self.history, "instructions")
            self.assertEqual(response.message, Message("assistant", "Done"))
        a = self.anthropic.client.messages.create.call_args.kwargs["messages"]
        o = self.openai.client.chat.completions.create.call_args.kwargs["messages"]
        # Gemini chat without active tools renders historical calls as text.
        g = self.gemini._convert_messages(self.history, tools_active=True)
        self.assertEqual(a[0]["content"][0], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "aGVsbG8="}})
        self.assertEqual(a[1]["content"][1]["input"], {"city": "Warsaw"})
        self.assertEqual(a[2], {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "Sunny"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "Unavailable", "is_error": True},
        ]})
        self.assertEqual(o[0], {"role": "system", "content": "instructions"})
        self.assertEqual(o[1]["content"][1]["image_url"]["url"], "data:image/png;base64,aGVsbG8=")
        self.assertEqual(json.loads(o[2]["tool_calls"][0]["function"]["arguments"]), {"city": "Warsaw"})
        self.assertEqual(o[3:5], [
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
            {"role": "tool", "tool_call_id": "c2", "content": "[Tool error]\nUnavailable"},
        ])
        self.assertEqual(g[0].parts[1].inline_data.data, b"hello")
        self.assertEqual(g[1].role, "model")
        self.assertEqual(g[1].parts[1].function_call.args, {"city": "Warsaw"})
        self.assertEqual(g[2].role, "user")
        self.assertEqual([p.function_response.id for p in g[2].parts], ["c1", "c2"])
        self.assertEqual(g[2].parts[1].function_response.response, {"error": "Unavailable"})
        self.assertEqual(self.history, before)

    def test_anthropic_signed_content_round_trips_and_stays_private_to_client(self):
        blocks = [
            ThinkingBlock(type="thinking", thinking="Plan", signature="sig"),
            TextBlock(type="text", text="Checking"),
            ToolUseBlock(type="tool_use", id="c1", name="lookup", input={}),
            RedactedThinkingBlock(type="redacted_thinking", data="opaque"),
        ]
        message = self.anthropic._parse_message(blocks)
        restored = Message.from_dict(json.loads(json.dumps(message.to_dict())))
        self.assertEqual(self.anthropic._convert_messages([restored])[0]["content"],
                         [b.model_dump(mode="json", exclude_none=True) for b in blocks])
        self.assertEqual(restored.reasoning, "Plan")
        other = self.openai._convert_messages([restored], "instructions")
        self.assertNotIn("signature", json.dumps(other))
        self.assertNotIn("opaque", json.dumps(other))
        self.assertEqual(other[1]["tool_calls"][0]["id"], "c1")
        restored.text = "Edited"
        with self.assertRaises(ValueError):
            self.anthropic._convert_messages([restored])

    def test_gemini_replays_signed_parts_after_json_storage_including_non_utf8_bytes(self):
        parts = [
            types.Part(text="Plan", thought=True, thought_signature=b"\xff\x00\xaa"),
            types.Part(text="Checking", thought_signature=b"text-signature"),
            types.Part(function_call=types.FunctionCall(name="lookup", args={}),
                       thought_signature=b"call-signature"),
        ]
        response = self.gemini._parse_response(types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=parts))]))
        restored = Message.from_dict(json.loads(json.dumps(response.message.to_dict())))
        replay = self.gemini._convert_messages([restored])[0].parts
        self.assertEqual(replay[0].thought_signature, b"\xff\x00\xaa")
        self.assertEqual(replay[1].thought_signature, b"text-signature")
        self.assertEqual(replay[2].thought_signature, b"call-signature")
        self.assertEqual(replay[2].function_call.id, restored.tool_calls[0].id)
        self.assertTrue(restored.tool_calls[0].id)
        other = self.anthropic._convert_messages([restored])
        self.assertNotIn("signature", json.dumps(other))
        restored.tool_calls[0].arguments["new"] = 1
        with self.assertRaises(ValueError):
            self.gemini._convert_messages([restored])

    def test_gemini_preserves_signed_text_when_tools_are_disabled(self):
        response = self.gemini._parse_response(types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text="Answer", thought_signature=b"sig")]))]))
        replay = self.gemini._convert_messages([response.message], tools_active=False)
        self.assertEqual(replay[0].parts[0].thought_signature, b"sig")

    def test_gemini_uses_documented_signature_for_foreign_tool_history(self):
        converted = self.gemini._convert_messages(self.history)
        self.assertEqual(converted[1].parts[1].thought_signature, b"skip_thought_signature_validator")

    def test_tools_disabled_keeps_calls_and_errors_as_context(self):
        converted = self.gemini._convert_messages(self.history, tools_active=False)
        text = "\n".join(part.text for content in converted for part in content.parts if part.text)
        self.assertIn('[Tool Call: lookup]', text)
        self.assertIn('[Tool Error: lookup]\nUnavailable', text)
        self.assertFalse(any(part.function_call or part.function_response for content in converted for part in content.parts))

    def test_openai_extra_metadata_only_returns_to_same_endpoint(self):
        sdk = SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="lookup", arguments="{}"),
                                                     extra_content={"vendor": {"signature": "sig"}})]))], usage=None)
        response = self.openai._parse_response(sdk)
        restored = Message.from_dict(json.loads(json.dumps(response.message.to_dict())))
        same = self.openai._convert_messages([restored], "")
        self.assertEqual(same[1]["tool_calls"][0]["extra_content"], {"vendor": {"signature": "sig"}})
        other = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="test")
        self.assertNotIn("extra_content", other._convert_messages([restored], "")[1]["tool_calls"][0])
        other_model = OpenAICompatibleClient(model="different-model", api_key="test")
        self.assertNotIn("extra_content", other_model._convert_messages([restored], "")[1]["tool_calls"][0])

    def test_native_provider_signatures_do_not_transfer_between_models(self):
        anthropic_message = self.anthropic._parse_message([
            ThinkingBlock(type="thinking", thinking="Plan", signature="sig"),
            TextBlock(type="text", text="Checking"),
        ])
        self.anthropic.model = "different-model"
        self.assertEqual(self.anthropic._convert_messages([anthropic_message]),
                         [{"role": "assistant", "content": "Checking"}])
        gemini_response = self.gemini._parse_response(types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text="Answer", thought_signature=b"sig")]))]))
        self.gemini.model = "different-model"
        parts = self.gemini._convert_messages([gemini_response.message])[0].parts
        self.assertEqual(parts[0].text, "Answer")
        self.assertIsNone(parts[0].thought_signature)

    def test_gemini_token_limit_maps_to_neutral_stop_reason(self):
        response = self.gemini._parse_response(types.GenerateContentResponse(candidates=[types.Candidate(
            finish_reason=types.FinishReason.MAX_TOKENS, content=types.Content(parts=[types.Part(text="partial")]))]))
        self.assertEqual(response.stop_reason, "max_tokens")

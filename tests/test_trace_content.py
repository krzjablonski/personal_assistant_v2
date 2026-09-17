"""Tracing is explicit, content-limited, and cannot change execution."""

import asyncio
import copy
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llm.i_llm_client import LLMResponse
from llm.langfuse_llm_client import LangfuseTrackedLLMClient, trace_agent_run
from llm.messages import Image, Message, ToolCall
from message_logger.redaction import redact_text


class Observation:
    def __init__(self, sdk, fields):
        self.sdk = sdk
        self.fields = fields
        self.updates = []
        self.ended = False
        self.parent = None

    def update(self, **fields):
        self.updates.append(fields)
        if self.sdk.failure == "update":
            raise RuntimeError("private tracing failure")

    def end(self):
        self.ended = True
        if self.sdk.failure == "end":
            raise RuntimeError("private tracing failure")

    def start_observation(self, **fields):
        child = self.sdk.start_observation(**fields)
        child.parent = self
        return child


class TraceSDK:
    def __init__(self, failure=None):
        self.records = []
        self.auto_errors = []
        self.failure = failure

    def start_observation(self, **fields):
        if self.failure == "start":
            raise RuntimeError("private tracing failure")
        observation = Observation(self, fields)
        self.records.append(observation)
        return observation

    @contextmanager
    def start_as_current_observation(self, **fields):
        observation = self.start_observation(**fields)
        try:
            yield observation
        except BaseException as error:
            self.auto_errors.append(str(error))
            raise
        finally:
            observation.end()


class TestTraceContent(unittest.IsolatedAsyncioTestCase):
    def fixture(self, usage=None):
        result = LLMResponse(
            Message("assistant", "reply token=response-private", provider_data={"hidden": "opaque-response"}),
            "end_turn", usage if usage is not None else {"input_tokens": 7, "cache_read_input_tokens": 3},
        )
        inner = SimpleNamespace(context_window=100_000, chat=AsyncMock(return_value=result), aclose=AsyncMock())
        messages = [Message("user", 'normal request GOOGLE_OAUTH_TOKEN_JSON={"refresh_token":"oauth-private"}',
                            images=[Image("private-base64-image", "image/png")],
                            tool_calls=[ToolCall("call", "lookup", {"api_key": "argument-private", "query": "normal"})],
                            provider_data={"vendor": "opaque-request"})]
        return inner, result, messages

    async def test_default_trace_contains_metadata_without_content_and_preserves_missing_usage(self):
        inner, result, messages = self.fixture()
        sdk = TraceSDK()
        with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
            actual = await LangfuseTrackedLLMClient(inner, "test-model").chat(messages, "private system")
        self.assertIs(actual, result)
        serialized = repr([(record.fields, record.updates) for record in sdk.records])
        for private in ("normal request", "private system", "oauth-private", "opaque-request", "opaque-response", "private-base64-image", "response-private", "argument-private"):
            self.assertNotIn(private, serialized)
        self.assertEqual(sdk.records[0].updates[-1]["usage_details"], {"input": 7, "cache_read_input_tokens": 3})
        self.assertNotIn("input", sdk.records[0].fields)
        self.assertNotIn("output", sdk.records[0].updates[-1])

    async def test_redacted_content_omits_media_and_opaque_state_without_mutating_messages(self):
        inner, result, messages = self.fixture()
        original = copy.deepcopy(messages)
        sdk = TraceSDK()
        with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
            await LangfuseTrackedLLMClient(inner, "test-model", content_mode="redacted").chat(messages, "system api_key=system-private")
        serialized = repr([(record.fields, record.updates) for record in sdk.records])
        self.assertIn("normal request", serialized)
        self.assertIn("image/png", serialized)
        for private in ("oauth-private", "system-private", "argument-private", "response-private", "opaque-request", "opaque-response", "private-base64-image"):
            self.assertNotIn(private, serialized)
        self.assertEqual(messages, original)
        self.assertIn("response-private", result.message.text)

    async def test_redacted_content_masks_spaced_and_escaped_credential_json_in_prose(self):
        inner, _, messages = self.fixture()
        messages[0].text = 'Keep request GOOGLE_OAUTH_TOKEN_JSON={"token": "json-private", "refresh_token": "refresh-private"} and after.'
        system = 'config client_secret="escaped-\\\"private\\\"" should disappear'
        sdk = TraceSDK()
        with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
            await LangfuseTrackedLLMClient(inner, "model", content_mode="redacted").chat(messages, system)
        payload = sdk.records[0].fields["input"]
        self.assertIn("Keep request", payload["messages"][0]["text"])
        self.assertIn("and after", payload["messages"][0]["text"])
        self.assertNotIn("json-private", repr(payload))
        self.assertNotIn("refresh-private", repr(payload))
        self.assertNotIn("private", payload["system"])
        self.assertEqual(redact_text(payload["system"]), payload["system"])

    async def test_tracing_failures_do_not_replace_the_provider_result_or_error(self):
        for failure in ("start", "update", "end"):
            with self.subTest(failure=failure):
                inner, result, messages = self.fixture()
                sdk = TraceSDK(failure)
                with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
                    actual = await LangfuseTrackedLLMClient(inner, "model").chat(messages, "system")
                self.assertIs(actual, result)
                inner.chat.assert_awaited_once()

    async def test_provider_failures_and_cancellation_never_enter_sdk_auto_exception_capture(self):
        for error in (RuntimeError("private provider failure"), asyncio.CancelledError("private cancellation")):
            with self.subTest(error=type(error)):
                inner, _, messages = self.fixture()
                inner.chat.side_effect = error
                sdk = TraceSDK()
                with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
                    with self.assertRaises(type(error)) as raised:
                        await LangfuseTrackedLLMClient(inner, "model").chat(messages, "system")
                self.assertIs(raised.exception, error)
                self.assertEqual(sdk.auto_errors, [])
                self.assertTrue(sdk.records[0].ended)
                self.assertNotIn(str(error), repr(sdk.records[0].updates))

    async def test_wrapper_closes_only_its_inner_provider(self):
        inner, _, _ = self.fixture()
        sdk = TraceSDK()
        with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk) as get_sdk:
            await LangfuseTrackedLLMClient(inner, "model").aclose()
        inner.aclose.assert_awaited_once()
        get_sdk.assert_not_called()

    async def test_absent_or_broken_sdk_still_delegates_the_request(self):
        for error in (ImportError("optional dependency absent"), RuntimeError("SDK setup failed")):
            with self.subTest(error=type(error)):
                inner, result, messages = self.fixture()
                with patch("llm.langfuse_llm_client.get_langfuse", side_effect=error):
                    actual = await LangfuseTrackedLLMClient(inner, "model").chat(messages, "system")
                self.assertIs(actual, result)
                inner.chat.assert_awaited_once()

    async def test_empty_usage_stays_unknown_and_observed_cache_counts_survive(self):
        for usage, expected in (
            ({}, {}),
            ({"output_tokens": 0, "cache_creation_input_tokens": 4, "cache_read_input_tokens": 2},
             {"output": 0, "cache_creation_input_tokens": 4, "cache_read_input_tokens": 2}),
        ):
            with self.subTest(usage=usage):
                inner, _, messages = self.fixture(usage)
                sdk = TraceSDK()
                with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk):
                    await LangfuseTrackedLLMClient(inner, "model").chat(messages, "system")
                self.assertEqual(sdk.records[0].updates[-1]["usage_details"], expected)

    async def test_parent_scope_groups_calls_without_recording_errors_or_leaking_context(self):
        inner, _, messages = self.fixture()
        sdk = TraceSDK()
        client = LangfuseTrackedLLMClient(inner, "model")
        exits = []

        class Attributes:
            def __enter__(self):
                pass

            def __exit__(self, *error):
                exits.append(error)

        error = asyncio.CancelledError("private run cancellation")
        with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk), \
                patch("llm.langfuse_llm_client.propagate_attributes", return_value=Attributes()):
            with self.assertRaises(asyncio.CancelledError) as raised:
                with trace_agent_run(client, "session", "private label"):
                    await client.chat(messages, "system")
                    raise error
            await client.chat(messages, "system")
        self.assertIs(raised.exception, error)
        self.assertEqual(exits, [(None, None, None)])
        self.assertIs(sdk.records[1].parent, sdk.records[0])
        self.assertIsNone(sdk.records[2].parent)
        self.assertTrue(all(record.ended for record in sdk.records))
        self.assertEqual(sdk.auto_errors, [])
        self.assertNotIn("private", repr([(record.fields, record.updates) for record in sdk.records]))

    async def test_parent_sdk_failures_cannot_replace_provider_failure_or_cancellation(self):
        for failure in ("start", "update", "end"):
            for error in (RuntimeError("private provider failure"), asyncio.CancelledError("private cancellation")):
                with self.subTest(failure=failure, error=type(error)):
                    inner, _, messages = self.fixture()
                    inner.chat.side_effect = error
                    client = LangfuseTrackedLLMClient(inner, "model")
                    sdk = TraceSDK(failure)
                    with patch("llm.langfuse_llm_client.get_langfuse", return_value=sdk), \
                            patch("llm.langfuse_llm_client.propagate_attributes", return_value=nullcontext()):
                        with self.assertRaises(type(error)) as raised:
                            with trace_agent_run(client, "session"):
                                await client.chat(messages, "system")
                    self.assertIs(raised.exception, error)
                    self.assertEqual(sdk.auto_errors, [])

    async def test_cli_rebuild_keeps_explicit_trace_policy_and_all_client_references(self):
        from agent.simple_agent.simple_agent import SimpleAgent
        from personal_assistant.cli_runtime import CliRuntime
        from personal_assistant.cli_settings import RuntimeSettings

        for mode in (None, "metadata", "redacted"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                providers = []

                def build(*args, **kwargs):
                    inner, _, _ = self.fixture()
                    providers.append(inner)
                    return SimpleAgent(llm_client=inner)

                with patch("personal_assistant.cli_runtime.build_agent", side_effect=build), \
                        patch("llm.langfuse_llm_client.get_langfuse") as get_sdk:
                    runtime = CliRuntime(
                        RuntimeSettings(), memory_provider=lambda: object(),
                         trace_content=mode,
                        config=SimpleNamespace(DB_PATH=Path(folder) / "config.db"),
                    )
                    try:
                        for iteration in range(2):
                            client = runtime.agent.llm_client
                            self.assertEqual(isinstance(client, LangfuseTrackedLLMClient), mode is not None)
                            if mode is not None:
                                self.assertEqual(client.content_mode, mode)
                            self.assertIs(runtime.agent.short_term_memory._llm_client, client)
                            if iteration == 0:
                                await runtime.rebuild(runtime.settings)
                    finally:
                        await runtime.aclose()
                    self.assertTrue(all(provider.aclose.await_count == 1 for provider in providers))
                    get_sdk.assert_not_called()


class TestTraceEntryPoint(unittest.TestCase):
    def test_cli_import_never_imports_optional_tracing_sdk(self):
        script = '''
import importlib.abc, sys
class RejectTraceImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'langfuse':
            raise AssertionError('Tracing SDK imported without opt-in')
sys.meta_path.insert(0, RejectTraceImport())
from personal_assistant.cli import build_parser
assert build_parser().parse_args([]).trace is None
assert build_parser().parse_args(['--trace', 'metadata']).trace == 'metadata'
assert build_parser().parse_args(['--trace', 'redacted']).trace == 'redacted'
'''
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

from __future__ import annotations

from llm.messages import Message

import io
import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from config_service.config_service import ConfigService
from personal_assistant.cli import (
    _next_agent_prompt,
    _run,
    _run_one_shot,
    build_parser,
)
from personal_assistant.cli_commands import (
    HELP_TEXT,
    _handle_command,
    _settings_menu,
)
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.services.agent_builder import build_skill_env_provider
from personal_assistant.cli_settings import (
    RuntimeSettings,
    ensure_provider_credentials,
    guided_setup,
    resolve_settings,
    status_rows,
)
from tool_framework.approval import ToolApprovalStore


class FakeConfig:
    def __init__(
        self,
        values: dict[str, str] | None = None,
        *,
        has_master_password: bool = False,
    ) -> None:
        """Create a locked in-memory settings store with configurable password presence and a write log."""
        self.values = dict(values or {})
        self.master_password_exists = has_master_password
        self.locked = True
        self.set_calls: list[tuple[str, str]] = []

    def get(self, key: str) -> str | None:
        """Return the fake stored setting or None when absent."""
        return self.values.get(key)

    def contains(self, key: str) -> bool:
        return key in self.values

    def set(self, key: str, value: str) -> None:
        """Store settings while enforcing the fake lock for provider and Google secrets."""
        if key.startswith(("llm.", "google.")) and self.locked:
            raise ValueError("Secret settings require an unlocked config store")
        self.values[key] = value
        self.set_calls.append((key, value))

    def set_many(self, values: dict[str, str]) -> None:
        """Apply one validated batch, mirroring the configuration store's transaction."""
        if self.locked and any(key.startswith(("llm.", "google.")) for key in values):
            raise ValueError("Secret settings require an unlocked config store")
        self.values.update(values)
        self.set_calls.extend(values.items())

    def has_master_password(self) -> bool:
        """Report whether the fake store has a master password configured."""
        return self.master_password_exists

    def is_locked(self) -> bool:
        """Report the fake store’s current lock state."""
        return self.locked

    def set_master_password(self, password: str) -> None:
        """Simulate password setup by marking the fake store configured and unlocked."""
        self.master_password_exists = True
        self.locked = False

    def unlock(self, password: str) -> bool:
        """Unlock the fake store only for the fixed test password."""
        if password != "master-password":
            return False
        self.locked = False
        return True


def fake_agent_build(request, **kwargs):
    """Supply a lightweight agent double with settings and lifecycle hooks needed by CLI tests."""
    messages = []
    return SimpleNamespace(
        session_id="cli-test",
        system_prompt="system prompt",
        name="test",
        subscribe=MagicMock(),
        unsubscribe=MagicMock(),
        llm_client=SimpleNamespace(context_window=1000, aclose=AsyncMock()),
        short_term_memory=SimpleNamespace(),
        clear_session=AsyncMock(side_effect=messages.clear),
        cancel_active_run=AsyncMock(),
        close_resources=AsyncMock(),
        add_message=messages.append,
        get_messages=lambda: list(messages),
        status=SimpleNamespace(value="running"),
        config=SimpleNamespace(
            max_iterations=request.max_iterations,

        ),
    )


def passthrough_ui() -> MagicMock:
    """Provide a mock console whose tracking method actually awaits the submitted operation."""
    ui = MagicMock()

    async def track(_agent_name, operation, _subscriber, **_kwargs):
        """Await the tracked operation without rendering console activity."""
        return await operation

    ui.track = AsyncMock(side_effect=track)
    return ui


class TestCli(unittest.TestCase):
    def test_global_work_mode_is_not_a_cli_setting(self) -> None:
        settings = resolve_settings(
            build_parser().parse_args([]), FakeConfig({"cli.mode": "execute"})
        )
        self.assertFalse(hasattr(settings, "mode"))
        self.assertNotIn("Mode", dict(status_rows(settings, FakeConfig())))
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--mode", "execute"])

    def test_email_recipient_policy_is_passed_to_skill_environment(self) -> None:
        for value in ("friend@example.com,other@example.com", ""):
            with self.subTest(value=value), patch.dict("os.environ", {"ALLOWED_EMAIL_RECIPIENTS": value}):
                environment = build_skill_env_provider(FakeConfig({"email.to": "default@example.com"}))(("ALLOWED_EMAIL_RECIPIENTS",))
                self.assertIn("ALLOWED_EMAIL_RECIPIENTS", environment)
                self.assertEqual(environment["ALLOWED_EMAIL_RECIPIENTS"], value)
        with patch.dict("os.environ", {}, clear=True):
            self.assertNotIn("ALLOWED_EMAIL_RECIPIENTS", build_skill_env_provider(FakeConfig())(("ALLOWED_EMAIL_RECIPIENTS",)))

    def setUp(self) -> None:
        """Redirect each CLI test’s session logs to an automatically cleaned temporary directory."""
        log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(log_dir.cleanup)
        patcher = patch("personal_assistant.cli_runtime.RUN_LOG_DIR", Path(log_dir.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failed_live_setting_save_preserves_agent_and_session(self) -> None:
        """Ensure failed persistence leaves live loop settings, message history untouched."""
        for field, value, agent_field in (
            ("max_iterations", 20, "max_iterations"),

        ):
            with self.subTest(field=field), patch(
                "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
            ):
                config = FakeConfig()
                runtime = CliRuntime(
                    RuntimeSettings(), memory_provider=lambda: object(),
                    config=config,
                )
                self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
                previous = runtime.settings
                runtime.agent.add_message(Message(role="user", text="keep"))
                with patch.object(config, "set_many", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        runtime.update_loop_setting(field, value)

                self.assertEqual(getattr(runtime.agent.config, agent_field), getattr(previous, field))
                self.assertEqual(runtime.settings, previous)
                self.assertEqual(runtime.agent.get_messages()[0].text, "keep")

    def test_failed_rebuild_save_keeps_active_agent_and_approvals(self) -> None:
        """Preserve the active agent, settings, history, and approvals when rebuild persistence fails."""
        with patch("personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build):
            config = FakeConfig()
            runtime = CliRuntime(
                RuntimeSettings(), memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
            old_agent = runtime.agent
            old_settings = runtime.settings
            runtime.agent.add_message(Message(role="user", text="keep"))
            pending = runtime.approvals.request("tool", {}, "reason")
            with patch.object(config, "set_many", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    asyncio.run(runtime.rebuild(replace(old_settings, model="new"), persist_fields={"model"}))

        self.assertIs(runtime.agent, old_agent)
        self.assertEqual(runtime.settings, old_settings)
        self.assertEqual(runtime.agent.get_messages()[0].text, "keep")
        self.assertEqual(runtime.approvals.pending(), [pending])

    def test_runtime_uses_one_config_for_client_and_skill_environment(self) -> None:
        """Ensure injected configuration supplies both provider credentials and skill environment values."""
        config = FakeConfig({
            "llm.openai_api_key": "test-provider-key",
            "google.oauth_token_json": "test-google-token",
        })
        with patch("llm.openai_compatible_client.AsyncOpenAI") as sdk:
            runtime = CliRuntime(
                RuntimeSettings(), memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
            sdk.assert_not_called()
            self.assertIs(runtime.agent.llm_client.client, sdk.return_value)
            sdk.return_value.close = AsyncMock()
            self.assertEqual(sdk.call_args.kwargs["api_key"], "test-provider-key")
        self.assertEqual(
            runtime.agent.skill_runtime.env_provider(("GOOGLE_OAUTH_TOKEN_JSON",))["GOOGLE_OAUTH_TOKEN_JSON"],
            "test-google-token",
        )

    def test_next_agent_prompt_returns_interactive_input(self) -> None:
        """Verify ordinary interactive input is returned as the next agent prompt."""
        ui = MagicMock()

        with patch("builtins.input", return_value="hello"):
            prompt = asyncio.run(_next_agent_prompt(SimpleNamespace(), ui))

        self.assertEqual(prompt, "hello")


    def test_eof_and_interrupt_end_interactive_prompt_cleanly(self) -> None:
        """Treat EOF and keyboard interruption as a clean end to prompt collection."""
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                ui = MagicMock()
                with patch("builtins.input", side_effect=error), patch(
                    "sys.stdout", io.StringIO()
                ):
                    prompt = asyncio.run(_next_agent_prompt(SimpleNamespace(), ui))

                self.assertIsNone(prompt)
                ui.end_prompt.assert_called_once()

    def test_slash_opens_commands_palette(self) -> None:
        """Verify a bare slash is handled by showing the command table."""
        runtime = SimpleNamespace(approvals=ToolApprovalStore())
        ui = MagicMock()

        handled = asyncio.run(_handle_command("/", runtime, ui))

        self.assertTrue(handled)
        self.assertEqual(ui.table.call_count, 1)

    def test_defaults_to_openai_session(self) -> None:
        """Protect the default OpenAI session settings."""
        args = build_parser().parse_args(["hello"])
        settings = resolve_settings(args, FakeConfig())

        self.assertEqual(args.prompt, ["hello"])
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.model, "gpt-5.4")

    def test_removed_approval_commands_are_unknown_and_plain_text_never_authorizes(self):
        store = ToolApprovalStore()
        request = store.request("fixture_command", {"command": "fixture"}, "reason")
        runtime = SimpleNamespace(approvals=store)
        for command in ("/approvals", "/approve", "/approve-always", "/deny"):
            with self.subTest(command=command), patch("sys.stdout", io.StringIO()) as output:
                self.assertTrue(asyncio.run(_handle_command(f"{command} {request.approval_id}", runtime, MagicMock())))
                self.assertIn("Unknown command", output.getvalue())
                self.assertNotIn(command, HELP_TEXT)
        for text in ("approved", "continue"):
            self.assertFalse(asyncio.run(_handle_command(text, runtime, MagicMock())))
        self.assertEqual(store.get(request.approval_id).status, "pending")

    def test_accepts_provider_alias_and_openrouter(self) -> None:
        """Verify the provider flag alias accepts OpenRouter."""
        args = build_parser().parse_args(["--provider", "openrouter"])

        self.assertEqual(args.client, "openrouter")


    def test_accepts_google_connect_with_optional_saved_client(self) -> None:
        """Support Google connection with an explicit client file or the saved-client sentinel."""
        with_path = build_parser().parse_args(
            ["--connect-google", "/tmp/google-client.json"]
        )
        saved_client = build_parser().parse_args(["--connect-google"])

        self.assertEqual(
            str(with_path.connect_google), "/tmp/google-client.json"
        )
        self.assertEqual(saved_client.connect_google, "")

    def test_saved_settings_fill_omitted_flags(self) -> None:
        """Use saved runtime settings for omitted flags without writing them back."""
        config = FakeConfig(
            {
                "cli.provider": "anthropic",
                "cli.model": "claude-custom",
                "cli.mode": "execute",
                "cli.max_iterations": "18",
                "cli.planning_enabled": "false",
                "cli.reflection_enabled": "true",
            }
        )

        settings = resolve_settings(build_parser().parse_args([]), config)

        self.assertEqual(settings.provider, "anthropic")
        self.assertEqual(settings.model, "claude-custom")
        self.assertEqual(settings.max_iterations, 18)
        self.assertFalse(hasattr(settings, "profile"))
        self.assertFalse(hasattr(settings, "reflection_enabled"))
        self.assertFalse(hasattr(settings, "planning_enabled"))
        self.assertEqual(config.set_calls, [])


    def test_planning_is_automatic_and_has_no_disable_flag(self) -> None:
        """Expose automatic planning and reject the removed CLI override."""
        settings = resolve_settings(build_parser().parse_args([]), FakeConfig({
            "cli.planning_enabled": "false",
        }))
        self.assertFalse(hasattr(settings, "planning_enabled"))
        self.assertNotIn("Planning", dict(status_rows(settings, FakeConfig())))
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--no-planning"])


    def test_flags_override_saved_settings_without_persisting(self) -> None:
        """Give explicit CLI flags precedence over saved defaults without persisting the overrides."""
        config = FakeConfig(
            {
                "cli.provider": "anthropic",
                "cli.model": "claude-custom",
                "cli.profile": "personal_ops",
            }
        )
        args = build_parser().parse_args(
            [
                "--provider",
                "gemini",
                "--model",
                "gemini-custom",
                "--max-iterations",
                "25",
            ]
        )

        settings = resolve_settings(args, config)

        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.model, "gemini-custom")
        self.assertEqual(settings.max_iterations, 25)
        self.assertFalse(hasattr(settings, "planning_enabled"))
        self.assertEqual(config.set_calls, [])

    def test_runtime_settings_validate_resource_bounds(self) -> None:
        """Reject excessive iteration limits, nonpositive context windows, and non-HTTP local endpoints."""
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            RuntimeSettings(max_iterations=101)
        with self.assertRaisesRegex(ValueError, "[Cc]ontext window"):
            RuntimeSettings(provider="local", context_window=0)
        with self.assertRaisesRegex(ValueError, "HTTP"):
            RuntimeSettings(provider="local", base_url="file:///tmp/model")

    def test_missing_credential_fails_without_interactive_input(self) -> None:
        """Fail with credential guidance when a noninteractive session lacks its provider key."""
        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaisesRegex(ValueError, "OPEN_AI_API_KEY"):
                ensure_provider_credentials(RuntimeSettings(), FakeConfig())

    def test_missing_credential_creates_encrypted_store(self) -> None:
        """Verify interactive first-time credential setup unlocks the store and saves the provider key."""
        config = FakeConfig()
        secret_input = iter(
            ["master-password", "master-password", "provider-secret"]
        )

        with patch("sys.stdin.isatty", return_value=True), patch(
            "personal_assistant.cli_settings.getpass", side_effect=lambda _: next(secret_input)
        ):
            ensure_provider_credentials(RuntimeSettings(), config)

        self.assertFalse(config.locked)
        self.assertEqual(
            config.set_calls,
            [("llm.openai_api_key", "provider-secret")],
        )

    def test_saved_openrouter_credential_is_reused_after_restart(self) -> None:
        """Verify a restarted encrypted store needs only its master password to reuse an OpenRouter key."""
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ConfigService, "DB_PATH", Path(tmp) / "agent_config.db"
        ), patch.object(ConfigService, "_PBKDF2_ITERATIONS", 1), patch.dict(
            "os.environ", {}, clear=True
        ):
            setup = ConfigService()
            setup.set_master_password("master-password")
            setup.set("llm.openrouter_api_key", "saved-openrouter-key")
            setup._conn.close()

            restarted = ConfigService()
            try:
                with patch("sys.stdin.isatty", return_value=True), patch(
                    "personal_assistant.cli_settings.getpass", return_value="master-password"
                ) as password_input:
                    ensure_provider_credentials(
                        RuntimeSettings(
                            provider="openrouter", model="openai/gpt-5.4"
                        ),
                        restarted,
                    )

                self.assertEqual(
                    restarted.get("llm.openrouter_api_key"), "saved-openrouter-key"
                )
                password_input.assert_called_once_with(
                    "Master password (blank to cancel): "
                )
            finally:
                restarted._conn.close()

    def test_existing_store_retries_unlock_without_exposing_key(self) -> None:
        """Allow retry after an incorrect master password while keeping the subsequently saved key out of output."""
        config = FakeConfig(has_master_password=True)
        secret_input = iter(["wrong-password", "master-password", "provider-secret"])

        with patch("sys.stdin.isatty", return_value=True), patch(
            "personal_assistant.cli_settings.getpass", side_effect=lambda _: next(secret_input)
        ), patch("sys.stdout", io.StringIO()) as output:
            ensure_provider_credentials(RuntimeSettings(), config)

        self.assertIn("Incorrect master password", output.getvalue())
        self.assertNotIn("provider-secret", output.getvalue())
        self.assertEqual(
            config.set_calls,
            [("llm.openai_api_key", "provider-secret")],
        )

    def test_new_store_retries_password_confirmation(self) -> None:
        """Verify setup retries short or mismatched passwords before saving a provider credential."""
        config = FakeConfig()
        secret_input = iter(
            [
                "short",
                "master-password",
                "mismatch",
                "master-password",
                "master-password",
                "provider-secret",
            ]
        )

        with patch("sys.stdin.isatty", return_value=True), patch(
            "personal_assistant.cli_settings.getpass", side_effect=lambda _: next(secret_input)
        ), patch("sys.stdout", io.StringIO()) as output:
            ensure_provider_credentials(RuntimeSettings(), config)

        self.assertIn("at least 8 characters", output.getvalue())
        self.assertIn("do not match", output.getvalue())
        self.assertEqual(
            config.set_calls,
            [("llm.openai_api_key", "provider-secret")],
        )

    def test_credential_setup_can_be_cancelled(self) -> None:
        """Allow blank password input to cancel interactive credential setup."""
        config = FakeConfig(has_master_password=True)

        with patch("sys.stdin.isatty", return_value=True), patch(
            "personal_assistant.cli_settings.getpass", return_value=""
        ), self.assertRaisesRegex(ValueError, "cancelled"):
            ensure_provider_credentials(RuntimeSettings(), config)

    def test_existing_environment_credential_does_not_prompt_or_persist(self) -> None:
        """Reuse an already available provider credential without prompting or rewriting it."""
        config = FakeConfig({"llm.openai_api_key": "environment-secret"})

        with patch("personal_assistant.cli_settings.getpass") as password_input:
            ensure_provider_credentials(RuntimeSettings(), config)

        password_input.assert_not_called()
        self.assertEqual(config.set_calls, [])

    def test_status_never_prints_secret_values(self) -> None:
        """Verify status reports credential availability rather than the supplied secret values."""
        config = FakeConfig(
            {
                "llm.openai_api_key": "do-not-print-this",
                "google.oauth_token_json": "also-do-not-print-this",
            }
        )

        status = dict(status_rows(RuntimeSettings(), config))

        self.assertEqual(status["Credentials"], "configured")
        self.assertIn("first use", status["Google"])
        self.assertNotIn("do-not-print-this", status.values())
        self.assertNotIn("also-do-not-print-this", status.values())

    def test_session_status_does_not_unlock_or_read_integration_secrets(self):
        requested = []
        config = SimpleNamespace(get=lambda key: requested.append(key), contains=lambda key: False)
        status_rows(RuntimeSettings(provider="local"), config)
        self.assertEqual(requested, ['web.search_provider', 'web.extract_provider', 'browser.headless'])

    def test_first_run_guides_provider_then_persists_choices(self) -> None:
        """Verify guided first-run setup saves selected provider and local-model defaults."""
        args = build_parser().parse_args([])
        config = FakeConfig()

        with patch("sys.stdin.isatty", return_value=True), patch(
            "builtins.input",
            side_effect=["5", "", "", ""],
        ), patch("sys.stdout", io.StringIO()):
            settings = guided_setup(
                args, resolve_settings(args, config), config
            )

        self.assertEqual(settings.provider, "local")
        self.assertEqual(
            dict(config.set_calls),
            {
                "cli.provider": "local",
                "cli.model": "local-model",
                "cli.local_base_url": "http://localhost:1234/v1",
                "cli.local_context_window": "",
            },
        )

    def test_rebuild_clears_conversation_and_pending_approvals(self) -> None:
        """Verify a successful rebuild replaces agent logging and clears the previous conversation and approvals."""
        config = FakeConfig()

        with patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ):
            runtime = CliRuntime(
                RuntimeSettings(provider="local", model="local-model"),
                memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
            old_agent = runtime.agent
            old_logger = runtime.session_logger
            old_agent.discard_checkpoint = MagicMock(return_value=True)
            runtime.agent.add_message(Message(role="user", text="old"))
            runtime.approvals.request("send_email", {}, "approval needed")

            asyncio.run(runtime.rebuild(
                replace(runtime.settings, model="new-local-model"),
                persist_fields={"model"},
            ))

        self.assertIsNot(runtime.agent, old_agent)
        self.assertIsNot(runtime.session_logger, old_logger)
        old_agent.unsubscribe.assert_called_once_with(old_logger)
        runtime.agent.subscribe.assert_called_once_with(runtime.session_logger)
        old_agent.clear_session.assert_awaited_once()
        self.assertEqual(runtime.agent.get_messages(), [])
        self.assertEqual(runtime.approvals.pending(), [])
        self.assertEqual(config.set_calls, [("cli.model", "new-local-model")])

    def test_clear_discards_conversation_and_approvals(self) -> None:
        """Ensure the clear command resets conversation memory and in-memory approvals."""
        with patch("personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build):
            runtime = CliRuntime(
                RuntimeSettings(), memory_provider=lambda: object(),
                config=FakeConfig(),
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
        store = runtime.approvals
        store.request("send_email", {}, "approval needed")
        agent = runtime.agent
        runtime.agent.add_message(Message(role="user", text="old"))

        with patch("sys.stdout", io.StringIO()):
            handled = asyncio.run(_handle_command("/clear", runtime, MagicMock()))

        self.assertTrue(handled)
        self.assertEqual(runtime.agent.get_messages(), [])
        self.assertEqual(store.all(), [])
        agent.clear_session.assert_awaited_once()

    def test_runtime_does_not_implicitly_mount_cli_launch_directory(self) -> None:
        """Let agent construction select managed storage when no workspace was supplied."""
        config = FakeConfig()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "personal_assistant.cli_runtime.Path.cwd", return_value=Path(tmp)
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ) as build_mock:
            runtime = CliRuntime(
                RuntimeSettings(provider="local", model="local-model"),
                memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))

        self.assertIsNone(runtime.working_directory)
        self.assertIsNone(build_mock.call_args.kwargs["workspace_root"])

    def test_interactive_runtime_waits_for_approval_decisions(self) -> None:
        """Configure a positive approval wait timeout for interactive sessions."""
        with patch("sys.stdin.isatty", return_value=True), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ):
            runtime = CliRuntime(
                RuntimeSettings(provider="local", model="local-model"),
                memory_provider=lambda: object(),
                config=FakeConfig(),
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))

        self.assertGreater(runtime.approvals.decision_timeout_seconds, 0)

    def test_failed_rebuild_preserves_active_session_and_saved_defaults(self) -> None:
        """Keep the current agent and history when candidate construction fails, without saving new defaults."""
        config = FakeConfig()
        initial_agent = fake_agent_build(
            SimpleNamespace(
                max_iterations=10,

            )
        )

        with patch(
            "personal_assistant.cli_runtime.build_agent",
            side_effect=[initial_agent, RuntimeError("candidate failed")],
        ):
            runtime = CliRuntime(
                RuntimeSettings(provider="local", model="local-model"),
                memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
            old_agent = runtime.agent
            runtime.agent.add_message(Message(role="user", text="keep"))

            with self.assertRaisesRegex(RuntimeError, "candidate failed"):
                asyncio.run(runtime.rebuild(
                    replace(runtime.settings, model="broken-model"),
                    persist_fields={"model"},
                ))

        self.assertIs(runtime.agent, old_agent)
        self.assertEqual(runtime.agent.get_messages()[0].text, "keep")
        self.assertEqual(config.set_calls, [])

    def test_loop_setting_change_preserves_agent_and_conversation(self) -> None:
        """Update and persist a loop setting without replacing the agent, conversation."""
        config = FakeConfig()

        with patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ):
            runtime = CliRuntime(
                RuntimeSettings(provider="local", model="local-model"),
                memory_provider=lambda: object(),
                config=config,
            )
            self.addCleanup(lambda runtime=runtime: asyncio.run(runtime.aclose()))
            agent = runtime.agent
            runtime.agent.add_message(Message(role="user", text="keep"))

            runtime.update_loop_setting("max_iterations", 20)

        self.assertIs(runtime.agent, agent)
        self.assertEqual(runtime.agent.get_messages()[0].text, "keep")
        self.assertEqual(runtime.agent.config.max_iterations, 20)
        self.assertEqual(config.set_calls, [("cli.max_iterations", "20")])

    def test_unknown_slash_command_is_not_sent_to_the_model(self) -> None:
        """Consume unknown slash commands locally and show a diagnostic."""
        runtime = SimpleNamespace(
            settings=RuntimeSettings(provider="local", model="local-model"),
            config=FakeConfig(),
            approvals=SimpleNamespace(),
        )
        output = io.StringIO()

        with patch("sys.stdout", output):
            handled = asyncio.run(_handle_command("/does-not-exist", runtime, MagicMock()))

        self.assertTrue(handled)
        self.assertIn("Unknown command", output.getvalue())

    def test_settings_menu_has_no_global_work_mode(self) -> None:
        output = io.StringIO()
        runtime = SimpleNamespace(settings=RuntimeSettings(), update_loop_setting=MagicMock())
        with patch("builtins.input", side_effect=["2", "20", "0"]), patch("sys.stdout", output):
            asyncio.run(_settings_menu(runtime))
        self.assertNotIn("Work mode", output.getvalue())
        self.assertNotIn("EXECUTE", output.getvalue())
        runtime.update_loop_setting.assert_called_once_with("max_iterations", 20)


class TestCliRun(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Keep asynchronous CLI tests’ session logs in an automatically cleaned temporary directory."""
        log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(log_dir.cleanup)
        patcher = patch("personal_assistant.cli_runtime.RUN_LOG_DIR", Path(log_dir.name))
        patcher.start()
        self.addCleanup(patcher.stop)


    async def test_one_shot_help_is_handled_without_agent_prompt(self) -> None:
        """Handle one-shot help locally with successful status and command guidance."""
        output = io.StringIO()

        with patch("sys.stdout", output):
            exit_code = await _run_one_shot(
                SimpleNamespace(approvals=ToolApprovalStore()),
                MagicMock(),
                "/help",
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Commands:", output.getvalue())


    async def test_one_shot_uses_structured_messages_and_live_activity(self) -> None:
        """Verify one-shot sessions render both messages and share the agent’s approval store with activity tracking."""
        args = build_parser().parse_args(
            ["--provider", "local", "hello"]
        )
        ui = passthrough_ui()
        result = SimpleNamespace(response="Hello back", status="completed")

        with patch(
            "personal_assistant.cli.ConsoleUI", return_value=ui
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ) as build_agent, patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(return_value=result),
        ):
            code = await _run(args)

        self.assertEqual(code, 0)
        ui.header.assert_called_once_with("Personal Assistant")
        ui.summary.assert_called_once()
        ui.message.assert_any_call("user", "hello")
        ui.message.assert_any_call("assistant", "Hello back")
        ui.track.assert_awaited_once()
        self.assertIsInstance(
            ui.track.await_args.kwargs["approval_store"],
            ToolApprovalStore,
        )
        self.assertIs(
            ui.track.await_args.kwargs["approval_store"],
            build_agent.call_args.kwargs["approval_store"],
        )

    async def test_response_serialization_error_is_not_reported_as_provider_failure(
        self,
    ) -> None:
        """Keep response serialization errors distinct from provider request failures."""
        args = build_parser().parse_args(
            ["--provider", "local", "hello"]
        )
        ui = passthrough_ui()
        result = SimpleNamespace(response={"invalid": object()}, usage=None)
        output = io.StringIO()

        with patch(
            "personal_assistant.cli.ConsoleUI", return_value=ui
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(return_value=result),
        ), patch("sys.stdout", output):
            with self.assertRaises(TypeError):
                await _run(args)

        self.assertNotIn("Provider request failed", output.getvalue())

    async def test_response_rendering_error_is_not_reported_as_provider_failure(
        self,
    ) -> None:
        """Let console-rendering failures propagate without mislabeling them as provider errors."""
        args = build_parser().parse_args(
            ["--provider", "local", "hello"]
        )
        ui = passthrough_ui()

        def show_message(role, _content):
            """Fail only assistant-message rendering to isolate post-response console errors."""
            if role == "assistant":
                raise RuntimeError("render failed")

        ui.message.side_effect = show_message
        output = io.StringIO()

        with patch(
            "personal_assistant.cli.ConsoleUI", return_value=ui
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(return_value=SimpleNamespace(response="Done", status="completed")),
        ), patch("sys.stdout", output):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                await _run(args)

        self.assertNotIn("Provider request failed", output.getvalue())

    async def test_one_shot_blocker_is_printed_and_exits_incomplete(self) -> None:
        """Show a blocked agent’s question once and signal incomplete work to the caller."""
        args = build_parser().parse_args(
            ["--provider", "local", "send"]
        )
        ui = passthrough_ui()
        result = SimpleNamespace(
            response="Which account should I use?",
            status="blocked",
        )

        with patch(
            "personal_assistant.cli.ConsoleUI", return_value=ui
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(return_value=result),
        ):
            code = await _run(args)

        self.assertEqual(code, 1)
        self.assertEqual(
            ui.message.call_args_list.count(
                unittest.mock.call("assistant", "Which account should I use?")
            ),
            1,
        )

    async def test_one_shot_incomplete_terminal_results_exit_nonzero(self) -> None:
        """Keep failed, cancelled and exhausted work distinct from successful completion."""
        args = build_parser().parse_args(
            ["--provider", "local", "work"]
        )

        for status in ("budget_exhausted", "failed", "cancelled"):
            with self.subTest(status=status):
                ui = passthrough_ui()
                result = SimpleNamespace(response=f"Result: {status}", status=status)
                with patch(
                    "personal_assistant.cli.ConsoleUI", return_value=ui
                ), patch(
                    "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
                ), patch(
                    "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
                ), patch(
                    "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
                    new=AsyncMock(return_value=result),
                ):
                    code = await _run(args)

                self.assertEqual(code, 1)

    async def test_google_connect_exits_without_building_an_agent(self) -> None:
        """Complete the Google connection command independently of agent construction."""
        args = build_parser().parse_args(
            ["--connect-google", "/tmp/google-client.json"]
        )
        output = io.StringIO()

        with patch(
            "personal_assistant.cli.ensure_config_unlocked"
        ) as unlock, patch(
            "personal_assistant.cli.connect_google",
            return_value="me@example.com",
        ) as connect, patch(
            "personal_assistant.cli_runtime.build_agent"
        ) as build_agent, patch("sys.stdout", output):
            code = await _run(args)

        self.assertEqual(code, 0)
        unlock.assert_called_once()
        connect.assert_called_once()
        build_agent.assert_not_called()
        self.assertIn("me@example.com", output.getvalue())

    async def test_interactive_provider_failure_is_sanitized_and_recoverable(self) -> None:
        """Keep interactive sessions usable after a provider error while hiding its private detail."""
        args = build_parser().parse_args(
            ["--provider", "local"]
        )

        output = io.StringIO()
        with patch("personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(side_effect=RuntimeError("provider-secret-detail")),
        ), patch(
            "builtins.input", side_effect=["hello", "/exit"]
        ), patch(
            "sys.stdout", output
        ):
            code = await _run(args)

        self.assertEqual(code, 0)
        self.assertIn("Provider request failed", output.getvalue())
        self.assertNotIn("provider-secret-detail", output.getvalue())

    async def test_interactive_provider_failure_retains_request_before_retry(self) -> None:
        """Preserve submitted work instead of deleting potentially partial execution history."""
        args = build_parser().parse_args(
            ["--provider", "local"]
        )
        ui = passthrough_ui()
        seen_messages = []

        async def run_turn(runtime, _subscriber, new_input):
            """Record submitted histories, fail the first turn, and provide a recovery response afterward."""
            runtime.agent.add_message(Message("user", new_input))
            seen_messages.append([message.to_dict() for message in runtime.agent.get_messages()])
            if len(seen_messages) == 1:
                raise RuntimeError("first request failed")
            return SimpleNamespace(response="Recovered", status="completed")

        with patch(
            "personal_assistant.cli.ConsoleUI", return_value=ui
        ), patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            autospec=True, side_effect=run_turn
        ), patch(
            "builtins.input", side_effect=["first", "second", "/exit"]
        ), patch("sys.stdout", io.StringIO()):
            code = await _run(args)

        self.assertEqual(code, 0)
        self.assertEqual(
            seen_messages,
            [
                [{"role": "user", "text": "first"}],
                [{"role": "user", "text": "first"}, {"role": "user", "text": "second"}],
            ],
        )


    async def test_one_shot_provider_failure_returns_nonzero_without_details(self) -> None:
        """Report one-shot provider failures with a nonzero exit and a sanitized message."""
        args = build_parser().parse_args(
            ["--provider", "local", "hello"]
        )
        output = io.StringIO()

        with patch(
            "personal_assistant.cli_runtime.build_agent", side_effect=fake_agent_build
        ), patch(
            "personal_assistant.cli.LongTermMemory", return_value=SimpleNamespace(close=MagicMock())
        ), patch(
            "personal_assistant.cli_runtime.CliRuntime.run_agent_turn",
            new=AsyncMock(side_effect=RuntimeError("private provider detail")),
        ), patch("sys.stdout", output):
            code = await _run(args)

        self.assertEqual(code, 1)
        self.assertIn("Provider request failed", output.getvalue())
        self.assertNotIn("private provider detail", output.getvalue())


if __name__ == "__main__":
    unittest.main()

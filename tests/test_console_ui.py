from __future__ import annotations

import asyncio
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent.agent_event import AgentEvent, AgentEventType
from personal_assistant.console_ui import ConsoleUI, _activity_text, _event_line
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from tool_framework.approval import ToolApprovalStore


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        """Advertise terminal support so buffered tests exercise interactive console behavior."""
        return True


class TestConsoleUI(unittest.TestCase):
    def test_user_and_assistant_messages_use_distinct_labeled_panels(self) -> None:
        """Verify user and assistant messages have distinct labels, colors, and panel borders."""
        output = TtyBuffer()
        ui = ConsoleUI(output, width=48, color=True)

        ui.message("user", "Please check my calendar")
        ui.message("assistant", "You are free after 15:00")

        rendered = output.getvalue()
        self.assertIn("YOU", rendered)
        self.assertIn("ASSISTANT", rendered)
        self.assertIn("\033[36m", rendered)
        self.assertIn("\033[32m", rendered)
        self.assertIn("╭", rendered)
        self.assertIn("╯", rendered)

    def test_assistant_message_renders_markdown_instead_of_showing_syntax(self) -> None:
        """Ensure assistant markdown appears as readable formatted content."""
        output = io.StringIO()
        ui = ConsoleUI(output, width=60)

        ui.message(
            "assistant",
            "# Recommendation\n\n**Best option**\n\n"
            "| Shop | Price |\n| --- | --- |\n| Ceneo | 2,199 zł |",
        )

        rendered = output.getvalue()
        self.assertIn("Recommendation", rendered)
        self.assertIn("Best option", rendered)
        self.assertIn("Ceneo", rendered)
        self.assertNotIn("# Recommendation", rendered)
        self.assertNotIn("**Best option**", rendered)
        self.assertNotIn("| Shop | Price |", rendered)

    def test_panels_strip_terminal_control_sequences_and_fit_width(self) -> None:
        """Verify the panel’s rendered text fits the configured width after color codes are removed."""
        output = TtyBuffer()
        ui = ConsoleUI(output, width=36, color=True)

        ui.message("assistant", "safe\033[31m unsafe " + "word " * 20)

        rendered = output.getvalue()
        plain = re.sub(r"\033\[[0-9;]*m", "", rendered)
        self.assertNotIn("\033[31m", plain)
        self.assertTrue(all(len(line) <= 36 for line in plain.splitlines()))

    def test_table_renders_rows_and_wraps_long_values(self) -> None:
        """Keep table content visible within the configured console width."""
        output = io.StringIO()
        ui = ConsoleUI(output, width=42)

        ui.table("Session", (("Provider", "OpenAI (a-very-long-model-name)"),))

        rendered = output.getvalue()
        self.assertIn("SESSION", rendered)
        self.assertIn("Provider", rendered)
        self.assertIn("OpenAI", rendered)
        self.assertIn("┬", rendered)
        self.assertTrue(all(len(line) <= 42 for line in rendered.splitlines()))

    def test_tables_treat_rich_markup_in_values_as_literal_text(self) -> None:
        """Prevent table and summary values from being interpreted as Rich markup."""
        output = io.StringIO()
        ui = ConsoleUI(output, width=60)

        ui.table("Approvals", (("[red]ID[/]", "[bold]reason[/]"),))
        ui.summary("Session", (("Agent", "[blue]General[/]"),))

        rendered = output.getvalue()
        self.assertIn("[red]ID[/]", rendered)
        self.assertIn("[bold]reason[/]", rendered)
        self.assertIn("[blue]General[/]", rendered)

    def test_summary_places_labels_above_a_single_row_of_values(self) -> None:
        """Verify summary labels share a header row above their corresponding values."""
        output = io.StringIO()
        ui = ConsoleUI(output, width=72)

        ui.summary(
            "Session",
            (("Provider", "OpenAI"), ("Agent", "General"), ("Mode", "draft")),
        )

        lines = output.getvalue().splitlines()
        header = next(line for line in lines if "Provider" in line)
        values = next(line for line in lines if "OpenAI" in line)
        self.assertIn("Agent", header)
        self.assertIn("Mode", header)
        self.assertIn("General", values)
        self.assertIn("draft", values)

    def test_full_session_summary_is_a_clear_three_column_grid(self) -> None:
        """Keep the full session summary grouped into a compact three-column layout."""
        output = io.StringIO()
        ui = ConsoleUI(output, width=88)

        ui.summary(
            "Session",
            (
                ("Provider", "OpenRouter (openai/gpt-5.6-luna)"),
                ("Agent", "General"),
                ("Mode", "draft"),
                ("Planning", "Auto"),
                ("Context reserve", "2048"),
                ("Max iterations", "10"),
                ("MCP", "enabled"),
                ("Credentials", "configured"),
                ("Google", "connected"),
            ),
        )

        lines = output.getvalue().splitlines()
        first_group = next(line for line in lines if "Provider" in line)
        second_group = next(line for line in lines if "Planning" in line)
        third_group = next(line for line in lines if "MCP" in line)
        self.assertIn("Agent", first_group)
        self.assertIn("Mode", first_group)
        self.assertNotIn("Planning", first_group)
        self.assertIn("Context reserve", second_group)
        self.assertIn("Max iterations", second_group)
        self.assertIn("Credentials", third_group)
        self.assertIn("Google", third_group)
        self.assertTrue(any("├" in line for line in lines))
        self.assertTrue(
            any("OpenRouter (openai/gpt-5.6-luna)" in line for line in lines)
        )
        self.assertLessEqual(len(lines), 13)

    def test_activity_text_describes_action_without_exposing_arguments(self) -> None:
        """Ensure tool activity names the action without displaying its arguments."""
        event = AgentEvent(
            event_type=AgentEventType.TOOL_CALL,
            session_id="session",
            message="Calling tool",
            iteration=1,
            data={"tool_name": "calendar", "args": {"token": "secret"}},
        )

        text = _activity_text(event)

        self.assertEqual(text, "Using calendar")
        self.assertNotIn("secret", text)

    def test_activity_text_cannot_add_terminal_lines(self) -> None:
        """Keep tool names with newlines from introducing extra activity lines."""
        event = AgentEvent(
            event_type=AgentEventType.TOOL_CALL,
            session_id="session",
            message="Calling tool",
            iteration=1,
            data={"tool_name": "calendar\nFAKE STATUS"},
        )

        self.assertEqual(_activity_text(event), "Using calendar FAKE STATUS")

    def test_activity_text_covers_protocol_without_exposing_event_data(self) -> None:
        """Verify protocol events use fixed activity labels rather than private event payloads."""
        expected = {
            AgentEventType.RESULT_CLASSIFIED: "Result classified",
            AgentEventType.RETRY_SCHEDULED: "Retry scheduled",
            AgentEventType.TERMINAL_STATE: "Finishing",
        }

        for event_type, text in expected.items():
            with self.subTest(event_type=event_type):
                event = AgentEvent(
                    event_type=event_type,
                    session_id="session",
                    message="secret message",
                    iteration=1,
                    data={"question": "secret question", "evidence": "secret"},
                )
                self.assertEqual(_activity_text(event), text)

    def test_event_line_keeps_tool_activity_compact_and_hides_arguments(self) -> None:
        """Verify event lines combine the agent name and readable tool activity without arguments."""
        event = AgentEvent(
            event_type=AgentEventType.TOOL_CALL,
            session_id="session",
            message="Calling tool",
            iteration=1,
            data={"tool_name": "web_search", "args": {"token": "secret"}},
        )

        line = _event_line(event, "Personal Ops")

        self.assertEqual(line, "│  ● Personal Ops · Using web search")
        self.assertNotIn("secret", line)

    def test_event_line_skips_internal_llm_events(self) -> None:
        """Keep internal LLM-response details out of visible activity lines."""
        event = AgentEvent(
            event_type=AgentEventType.LLM_RESPONSE,
            session_id="session",
            message="raw response",
            iteration=1,
            data={"usage": {"input_tokens": 10}},
        )

        self.assertIsNone(_event_line(event, "General"))


class TestConsoleActivity(unittest.IsolatedAsyncioTestCase):
    async def test_enhanced_approval_shows_exact_unredacted_scope(self):
        output = TtyBuffer()
        store = ToolApprovalStore()
        request = store.request("run_command", {"argv": ["python", "-"], "stdin": "print(42)",
                                "cwd": "/workspace", "token": "private-secret", "policy": {"network": "none"}}, "Review code")
        with patch("sys.stdout", output), patch("personal_assistant.console_ui.supports_interactive_input", return_value=True):
            ui = ConsoleUI(color=False)
            with patch("personal_assistant.console_ui.ApprovalInput") as dialog, patch.object(ui, "_read_input", AsyncMock()) as legacy:
                dialog.return_value.read = AsyncMock(return_value=True)
                await ui.approval_prompt(request, store)
                summary, details = dialog.call_args.args
                self.assertIn("python -", summary)
                self.assertIn("print(42)", summary)
                self.assertIn("/workspace", summary)
                self.assertIn('"network": "none"', details)
                # Approval never masks model-authored scope; logs redact separately.
                self.assertIn("private-secret", details)
                legacy.assert_not_called()
        self.assertEqual(store.get(request.approval_id).status, "approved")

    async def test_enhanced_approval_does_not_report_expired_request_as_allowed(self):
        output = TtyBuffer()
        store = ToolApprovalStore()
        request = store.request("run_command", {"argv": ["pwd"]}, "Review", expires_at=0)
        with patch("sys.stdout", output), patch("personal_assistant.console_ui.supports_interactive_input", return_value=True):
            ui = ConsoleUI(color=False)
            with patch("personal_assistant.console_ui.ApprovalInput") as dialog:
                dialog.return_value.read = AsyncMock(return_value=True)
                await ui.approval_prompt(request, store)
        self.assertEqual(store.get(request.approval_id).status, "unavailable")
        self.assertNotIn("Allowed once:", output.getvalue())
        self.assertIn("Request no longer available:", output.getvalue())

    async def test_plain_mode_keeps_line_based_approval(self):
        output = TtyBuffer()
        store = ToolApprovalStore()
        request = store.request("run_command", {"argv": ["pwd"]}, "Review")
        with patch("sys.stdout", output), patch("personal_assistant.console_ui.supports_interactive_input", return_value=True):
            ui = ConsoleUI(color=False, plain=True)
            with patch("personal_assistant.console_ui.ApprovalInput") as dialog, patch.object(ui, "_read_input", AsyncMock(return_value="2")):
                await ui.approval_prompt(request, store)
                dialog.assert_not_called()
        self.assertEqual(store.get(request.approval_id).status, "denied")
        self.assertIn("Resolved action:", output.getvalue())

    async def test_track_prompts_for_and_grants_one_time_approval(self) -> None:
        """Verify interactive tracking presents a command and grants the selected one-time approval."""
        output = TtyBuffer()
        ui = ConsoleUI(output, width=72, color=False)
        subscriber = EventBufferSubscriber()
        store = ToolApprovalStore(decision_timeout_seconds=1)

        async def work() -> str:
            """Request a workspace listing and wait for the console’s approval decision."""
            request = store.request(
                "run_command",
                {"argv": ["ls", "-la"]},
                "List workspace files",
            )
            return await store.wait_for_decision(request.approval_id)

        with patch("sys.stdin.isatty", return_value=True), patch.object(
            ui, "_read_input", new=AsyncMock(return_value="1")
        ):
            result = await ui.track(
                "General",
                work(),
                subscriber,
                approval_store=store,
                refresh_interval=0.001,
            )

        self.assertEqual(result, "approved")
        self.assertIn("Allow once", output.getvalue())
        self.assertNotIn("Allow always", output.getvalue())
        self.assertIn("ls -la", output.getvalue())

    async def test_track_denies_without_persistent_approval_choice(self):
        output = TtyBuffer()
        ui = ConsoleUI(output, color=False)
        store = ToolApprovalStore(decision_timeout_seconds=1)
        async def work():
            request = store.request("run_command", {"argv": ["pwd"]}, "Show workspace path")
            return await store.wait_for_decision(request.approval_id)
        with patch("sys.stdin.isatty", return_value=True), patch.object(ui, "_read_input", AsyncMock(return_value="2")):
            result = await ui.track("General", work(), EventBufferSubscriber(), approval_store=store, refresh_interval=.001)
        self.assertEqual(result, "denied")
        self.assertNotIn("Allow always", output.getvalue())

    async def test_track_displays_live_agent_activity(self) -> None:
        """Verify tracking shows incoming tool activity and a completion message."""
        output = TtyBuffer()
        ui = ConsoleUI(output, width=60, color=True)
        subscriber = EventBufferSubscriber()

        async def work() -> str:
            """Publish a tool-call event and remain active briefly so the display can refresh."""
            subscriber.on_event(
                AgentEvent(
                    event_type=AgentEventType.TOOL_CALL,
                    session_id="session",
                    message="Calling tool",
                    iteration=1,
                    data={"tool_name": "calendar"},
                )
            )
            await asyncio.sleep(0.02)
            return "done"

        result = await ui.track(
            "Personal Ops", work(), subscriber, refresh_interval=0.001
        )

        self.assertEqual(result, "done")
        self.assertIn("Personal Ops · Using calendar", output.getvalue())
        self.assertIn("Completed", output.getvalue())

    async def test_no_color_still_refreshes_status_in_place(self) -> None:
        """Preserve in-place terminal status updates when colors are disabled."""
        output = TtyBuffer()
        ui = ConsoleUI(output, width=60, color=False)
        subscriber = EventBufferSubscriber()

        async def work() -> str:
            """Keep a fake task active briefly to exercise colorless status refresh."""
            await asyncio.sleep(0.01)
            return "done"

        await ui.track("General", work(), subscriber, refresh_interval=0.001)

        self.assertIn("\r\033[2K", output.getvalue())


if __name__ == "__main__":
    unittest.main()

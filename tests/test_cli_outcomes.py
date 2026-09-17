from __future__ import annotations

import asyncio
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from llm.messages import Message

from message_logger.event_buffer_subscriber import EventBufferSubscriber
from personal_assistant.cli import _run_one_shot, _execute_turn
from agent.agent_event import AgentEvent, AgentEventType
from personal_assistant.console_ui import ConsoleUI
from tool_framework.approval import ToolApprovalStore


class TestCliOutcomes(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_displays_retained_report_and_still_propagates(self):
        async def cancelled_turn(subscriber, prompt):
            subscriber.on_event(AgentEvent(
                AgentEventType.ASSISTANT_MESSAGE, "test", "Interrupted run report", 1,
                data={"text": "Cancelled. Execution budget: 1/10. Run log: /synthetic/run.log", "interrupted": True},
            ))
            raise asyncio.CancelledError()
        output = io.StringIO()
        runtime = SimpleNamespace(approvals=ToolApprovalStore(), run_agent_turn=cancelled_turn)
        with self.assertRaises(asyncio.CancelledError):
            await _execute_turn(runtime, ConsoleUI(output), "Inspect")
        self.assertIn("Cancelled.", output.getvalue())
        self.assertIn("/synthetic/run.log", output.getvalue())
        self.assertNotIn("Provider request failed", output.getvalue())

    async def test_finished_operation_cancels_idle_approval_input(self):
        store = ToolApprovalStore()
        store.request("test", {}, "Needs approval")
        reading = asyncio.Event()
        closed = asyncio.Event()

        async def read():
            reading.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def work():
            await reading.wait()
            return SimpleNamespace(status="blocked")

        ui = ConsoleUI(io.StringIO())
        with patch("sys.stdin.isatty", return_value=True), patch.object(ui, "_read_input", read):
            result = await asyncio.wait_for(ui.track(
                "Test", work(), EventBufferSubscriber(), approval_store=store,
            ), timeout=0.5)
        self.assertEqual(result.status, "blocked")
        self.assertTrue(closed.is_set())

    async def test_input_pipe_handles_partial_line_eof_and_cancellation(self):
        for ending in ("newline", "eof", "cancel"):
            with self.subTest(ending=ending):
                read_fd, write_fd = os.pipe()
                loop = asyncio.get_running_loop()
                try:
                    with os.fdopen(read_fd) as source, patch("sys.stdin", source):
                        read = asyncio.create_task(ConsoleUI(io.StringIO())._read_input())
                        await asyncio.sleep(0)
                        os.write(write_fd, b"1")
                        # A readable fragment must not block the event loop.
                        await asyncio.sleep(0.01)
                        self.assertFalse(read.done())
                        if ending == "newline":
                            os.write(write_fd, b"\n")
                            self.assertEqual(await asyncio.wait_for(read, 0.5), "1\n")
                        elif ending == "eof":
                            os.close(write_fd)
                            write_fd = None
                            with self.assertRaises(EOFError):
                                await asyncio.wait_for(read, 0.5)
                        else:
                            read.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await read
                        self.assertFalse(loop.remove_reader(read_fd))
                finally:
                    if write_fd is not None:
                        os.close(write_fd)

    async def test_one_shot_preserves_partial_response_and_reports_actual_status(self):
        for status in ("completed", "blocked", "budget_exhausted", "failed", "cancelled"):
            with self.subTest(status=status):
                output = io.StringIO()
                runtime = SimpleNamespace(
                    messages=[],
                    approvals=ToolApprovalStore(),
                    run_agent_turn=AsyncMock(return_value=SimpleNamespace(
                        response="Useful evidence and a next step", status=status,
                    )),
                )
                async def turn(_subscriber, new_input):
                    self.assertEqual(new_input, "test request")
                    runtime.messages.append(Message("assistant", "Useful evidence and a next step"))
                    return SimpleNamespace(response="Useful evidence and a next step", status=status)
                runtime.run_agent_turn = turn
                code = await _run_one_shot(runtime, ConsoleUI(output), "test request")
                self.assertEqual(code, 0 if status == "completed" else 1)
                self.assertEqual(runtime.messages[-1].text, "Useful evidence and a next step")
                if status != "completed":
                    self.assertNotIn("Completed", output.getvalue())
                self.assertIn(status.replace("_", " ").capitalize(), output.getvalue())

    async def test_noninteractive_pending_approval_does_not_consume_stdin(self):
        store = ToolApprovalStore()
        request = store.request("test", {}, "Needs approval")

        async def blocked_work():
            await asyncio.sleep(0)
            return SimpleNamespace(status="blocked")

        with patch("sys.stdin.isatty", return_value=False), patch(
            "builtins.input", side_effect=AssertionError("must not read piped input")
        ):
            result = await ConsoleUI(io.StringIO()).track(
                "Test", blocked_work(), EventBufferSubscriber(), approval_store=store,
            )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(store.get(request.approval_id).status, "pending")

    async def test_cancellation_waits_for_owned_operation_cleanup(self):
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def work():
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                cleaned.set()

        tracked = asyncio.create_task(ConsoleUI(io.StringIO()).track(
            "Test", work(), EventBufferSubscriber(),
        ))
        await entered.wait()
        tracked.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await tracked
        self.assertTrue(cleaned.is_set())

    async def test_approval_wait_keeps_event_loop_responsive(self):
        store = ToolApprovalStore(decision_timeout_seconds=1)
        request = store.request("test", {}, "Needs approval")
        progressed = asyncio.Event()

        async def work():
            progressed.set()
            await store.wait_for_decision(request.approval_id)
            return SimpleNamespace(status="completed")

        async def choose():
            await asyncio.wait_for(progressed.wait(), timeout=0.5)
            return "1"

        ui = ConsoleUI(io.StringIO())
        with patch("sys.stdin.isatty", return_value=True), patch.object(
            ui, "_read_input", side_effect=choose, create=True
        ):
            await ui.track("Test", work(), EventBufferSubscriber(), approval_store=store)
        self.assertEqual(store.get(request.approval_id).status, "approved")

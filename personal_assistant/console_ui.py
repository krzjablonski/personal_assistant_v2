from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import time
from collections.abc import Awaitable, Iterable
from contextlib import suppress
from typing import TypeVar

from agent.agent_event import AgentEvent, AgentEventType
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tool_framework.approval import ToolApprovalRequest, ToolApprovalStore
from personal_assistant.cli_input import ApprovalInput, supports_interactive_input


_T = TypeVar("_T")
# C0/C1 controls (keeping newline and tab), ESC included, plus invisible
# format characters: bidi embeddings/overrides/isolates, zero-width
# characters, line/paragraph separators and the byte-order mark.
_UNSAFE_RE = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]"
)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_TERMINAL_DISPLAY = {
    "completed": ("✓", "Completed", _GREEN),
    "blocked": ("?", "Blocked", _YELLOW),
    "budget_exhausted": ("!", "Budget exhausted", _YELLOW),
    "failed": ("!", "Failed", _YELLOW),
    "cancelled": ("×", "Cancelled", _DIM),
}
_EVENT_MARKERS = {
    AgentEventType.TOOL_CALL: "●",
    AgentEventType.TOOL_RESULT: "✓",
    AgentEventType.RETRY_SCHEDULED: "↻",
    AgentEventType.TOOL_APPROVAL_REQUIRED: "?",
    AgentEventType.COMMAND_FINISHED: "✓",
    AgentEventType.ERROR: "!",
}
_VISIBLE_EVENTS = {
    AgentEventType.TOOL_CALL,
    AgentEventType.TOOL_RESULT,
    AgentEventType.RETRY_SCHEDULED,
    AgentEventType.TOOL_APPROVAL_REQUIRED,
    AgentEventType.COMMAND_FINISHED,
    AgentEventType.ERROR,
}


def _escape_character(match: re.Match) -> str:
    """Render one unsafe character as a visible escape."""
    code = ord(match.group())
    if code == 0x0D:
        return "\\r"
    return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"


def _safe_text(value: object) -> str:
    """Render terminal controls and invisible characters as visible escapes.

    Deleting them would misrepresent content: a lone carriage return is a line
    break to Python and a shell, so dropping it can make live code look commented out.
    """
    return _UNSAFE_RE.sub(_escape_character, str(value))


def _has_hidden_characters(value: object) -> bool:
    """Report whether any string in a scope needs escapes for controls or invisible characters."""
    if isinstance(value, str):
        return bool(_UNSAFE_RE.search(value))
    if isinstance(value, dict):
        return any(_has_hidden_characters(key) or _has_hidden_characters(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_hidden_characters(item) for item in value)
    return False


def _discard_type_ahead() -> None:
    """Drop pending terminal input so earlier keystrokes cannot answer a new prompt."""
    try:
        import termios

        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        # Windows, pipes, mocked or closed stdin: nothing buffered to discard.
        pass


_SCOPE_LABELS = (("recipients", "To"), ("google_account", "Account"), ("calendar", "Calendar"))


def _scope_summary(arguments: dict) -> list[str]:
    """Return recipient/account/calendar lines present anywhere a scope names them."""
    environment = arguments.get("environment") if isinstance(arguments.get("environment"), dict) else {}
    values = {**({"calendar": environment["GOOGLE_CALENDAR_ID"]} if "GOOGLE_CALENDAR_ID" in environment else {}),
              **arguments}
    return [f"{label}: {_safe_text(values[key])}" for key, label in _SCOPE_LABELS if values.get(key) is not None]


def _single_line(value: object) -> str:
    """Convert display content to one line after removing recognized terminal controls."""
    return " ".join(_safe_text(value).splitlines())


def _tool_label(event: AgentEvent) -> str:
    """Produce a short readable tool label from an event for progress display."""
    name = (event.data or {}).get("tool_name", "tool")
    return _single_line(name).replace("__", " / ").replace("_", " ")[:48]


def _activity_text(event: AgentEvent | None) -> str:
    """Map the latest runtime event to a brief user-facing activity label."""
    if event is None:
        return "Thinking"
    if event.event_type == AgentEventType.USER_MESSAGE:
        return "Understanding request"
    if event.event_type == AgentEventType.TOOL_CALL:
        return f"Using {_tool_label(event)}"
    if event.event_type == AgentEventType.TOOL_RESULT:
        return f"Reviewing {_tool_label(event)} result"
    if event.event_type == AgentEventType.RESULT_CLASSIFIED:
        return "Result classified"
    if event.event_type == AgentEventType.RETRY_SCHEDULED:
        return "Retry scheduled"
    if (event.event_type == AgentEventType.REASONING
            and (event.data or {}).get("purpose") == "interrupted_run_report"):
        return "Summarizing interrupted run"
    if event.event_type == AgentEventType.TERMINAL_STATE:
        return "Finishing"
    if event.event_type == AgentEventType.TOOL_APPROVAL_REQUIRED:
        return f"Waiting for approval: {_tool_label(event)}"
    if event.event_type == AgentEventType.COMMAND_FINISHED:
        return "Workspace command finished"
    if event.event_type == AgentEventType.STATUS_CHANGE:
        return "Finishing"
    if event.event_type == AgentEventType.ERROR:
        return "Error"
    return "Thinking"


def _event_line(event: AgentEvent, agent_name: object) -> str | None:
    """Render a compact progress line for a visible event, or None for hidden event types."""
    if event.event_type not in _VISIBLE_EVENTS:
        return None
    marker = _EVENT_MARKERS.get(event.event_type, "·")
    return f"│  {marker} {_single_line(agent_name)} · {_activity_text(event)}"


class ConsoleUI:
    def __init__(
        self,
        stream=None,
        *,
        width: int | None = None,
        color: bool | None = None,
        plain: bool = False,
    ) -> None:
        """Configure console output width, interactivity, and color for the selected stream."""
        self.stream = sys.stdout if stream is None else stream
        self.plain = plain
        terminal_size = shutil.get_terminal_size(fallback=(88, 24))
        self.width = max(20, min(width or terminal_size.columns, 100))
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        detected_color = bool(
            self.interactive and "NO_COLOR" not in os.environ
        )
        self.color = detected_color if color is None else color
        self.console = Console(
            file=self.stream,
            width=self.width,
            height=terminal_size.lines,
            color_system="standard" if self.color else None,
            force_terminal=self.interactive or color is True,
            no_color=not self.color,
            highlight=False,
        )

    def _paint(self, text: str, *styles: str) -> str:
        """Apply supplied terminal styles when color is enabled, otherwise return plain text."""
        return f"{''.join(styles)}{text}{_RESET}" if self.color else text

    def _write(self, text: str) -> None:
        """Write and flush console text immediately for live progress updates."""
        self.stream.write(text)
        self.stream.flush()

    def header(self, title: str) -> None:
        """Display a width-limited heading for the assistant session."""
        label = f" {_safe_text(title).upper()} "
        line = f"{label}{'─' * max(0, self.width - len(label))}"
        self._write(self._paint(line[: self.width], _BOLD, _BLUE) + "\n")

    def error(self, message: str) -> None:
        """Display an error message with the console's red emphasis style."""
        text = f"ERROR: {message}"
        self.console.print(f"[red bold]{text}")

    def prompt(self) -> str:
        """Return the user-input prompt with optional terminal color."""
        return f"{_BOLD}{_CYAN}YOU › " if self.color else "YOU › "

    def end_prompt(self) -> None:
        """Reset terminal styling after input when colored prompts are enabled."""
        if self.color:
            self._write(_RESET)

    def message(self, role: str, content: object) -> None:
        """Display user text or assistant Markdown in a labeled conversation panel."""
        label, color = (
            ("YOU", "cyan") if role == "user" else ("ASSISTANT", "green")
        )
        text = _safe_text(content)
        body = (
            Text(text, style=color)
            if role == "user"
            else Markdown(text, style=color)
        )
        self.console.print(
            Panel(
                body,
                title=Text(label, style=f"bold {color}"),
                title_align="left",
                border_style=color,
                padding=(0, 1),
            )
        )

    def table(self, title: str, rows: Iterable[tuple[object, object]]) -> None:
        """Display nonempty key-value rows in a titled two-column table."""
        clean_rows = [(_safe_text(key), _safe_text(value)) for key, value in rows]
        if not clean_rows:
            return
        table = Table(box=box.SQUARE, expand=True, show_header=False)
        table.add_column()
        table.add_column()
        for key, value in clean_rows:
            table.add_row(Text(key, style="bold"), Text(value))
        self._table_title(title)
        self.console.print(table)

    def summary(self, title: str, rows: Iterable[tuple[object, object]]) -> None:
        """Display nonempty session details as labeled cells in a three-column grid."""
        clean_rows = [(_safe_text(key), _safe_text(value)) for key, value in rows]
        if not clean_rows:
            return
        self._table_title(title)
        table = Table(
            box=box.SQUARE,
            expand=True,
            show_header=False,
            show_lines=True,
            padding=(0, 1),
        )
        for ratio in (3, 2, 2):
            table.add_column(ratio=ratio)
        cells = []
        for label, value in clean_rows:
            cell = Text(label, style="bold blue")
            cell.append("\n" + value)
            cells.append(cell)
        for offset in range(0, len(cells), 3):
            table.add_row(*(cells[offset : offset + 3] + [Text()] * 3)[:3])
        self.console.print(table)

    def _table_title(self, title: object) -> None:
        """Display a sanitized uppercase title limited to the console width."""
        self.console.print(Text(_safe_text(title).upper()[: self.width], style="bold blue"))

    async def _read_input(self) -> str:
        """Read one terminal line without blocking the loop or retaining a reader."""
        self._write("Select: ")
        loop = asyncio.get_running_loop()
        result = loop.create_future()
        descriptor = sys.stdin.fileno()
        buffer = bytearray()

        def ready() -> None:
            if result.done():
                return
            try:
                # Read only the byte known to be ready. TextIO.readline can
                # block the loop when a pipe or raw terminal has a partial line.
                chunk = os.read(descriptor, 1)
                if not chunk:
                    raise EOFError("Approval input closed")
                buffer.extend(chunk)
            except Exception as error:
                result.set_exception(error)
            else:
                if chunk == b"\n":
                    result.set_result(buffer.decode("utf-8", errors="replace"))

        loop.add_reader(descriptor, ready)
        try:
            return await result
        finally:
            loop.remove_reader(descriptor)

    async def approval_prompt(
        self,
        request: ToolApprovalRequest,
        store: ToolApprovalStore,
    ) -> None:
        """Ask the user to allow once or deny the currently waiting action.

        Record the decision in the approval store and retry invalid or unavailable choices.
        """
        # Approval shows the exact model-authored scope: masking credential-like
        # patterns here could hide code or message content. Logs redact separately.
        arguments = request.arguments
        command = arguments.get("argv", arguments.get("command"))
        if isinstance(command, list):
            action = shlex.join(str(part) for part in command)
        else:
            action = request.tool_name
        scope = _safe_text(json.dumps(arguments, ensure_ascii=False, indent=2, default=str))
        warning = ("Warning: contains control or invisible characters, shown as \\r, \\xNN or \\uNNNN escapes."
                   if _has_hidden_characters(arguments) else "")
        unreviewed = arguments.get("unreviewed_files") or []
        if not self.plain and self.stream is sys.stdout and supports_interactive_input():
            if isinstance(command, list):
                summary = f"Action: {_safe_text(action)}\nReason: {_safe_text(request.reason)}"
                for line in _scope_summary(arguments):
                    summary += "\n" + line
                if arguments.get("cwd") is not None:
                    summary += f"\nDirectory: {_safe_text(arguments['cwd'])}"
                if arguments.get("stdin") is not None:
                    summary += f"\nStdin / code:\n{_safe_text(arguments['stdin'])}"
                if arguments.get("reviewed_files"):
                    summary += "\nReviewed files: inspect full action details (d)."
                if unreviewed:
                    summary += "\nUNREVIEWED files (over 1 MB, not shown): " + ", ".join(
                        _safe_text(item.get("path")) for item in unreviewed)
            else:
                summary = f"Action: {_single_line(action)}\nReason: {_safe_text(request.reason)}\n{scope}"
            if warning:
                summary += "\n" + warning
            _discard_type_ahead()
            self._write("\r\033[2K")
            allowed = await ApprovalInput(summary, scope, color=self.color).read()
            if allowed:
                recorded = store.approve(request.approval_id)
            else:
                recorded = store.deny(request.approval_id)
            outcome = ("Allowed once" if allowed else "Denied") if recorded else "Request no longer available"
            self.console.print(Text(f"{outcome}: {_single_line(action)}"))
            return
        self._write(
            "\r\033[2K\nApproval required\n"
            f"  Action: {_single_line(action)}\n"
            f"  Reason: {_single_line(request.reason)}\n"
            + "".join(f"  {line}\n" for line in _scope_summary(arguments))
            + (f"  {warning}\n" if warning else "")
            + f"  Resolved action:\n{scope}\n"
            "  1. Allow once\n"
            "  2. Deny\n"
        )
        _discard_type_ahead()
        while True:
            try:
                choice = (await self._read_input()).strip().casefold()
            except EOFError:
                store.deny(request.approval_id)
                return
            if choice in {"1", "once", "o"}:
                store.approve(request.approval_id)
                return
            if choice in {"2", "deny", "d"}:
                store.deny(request.approval_id)
                return
            self._write("Choose 1 or 2.\n")

    async def track(
        self,
        agent_name: str,
        operation: Awaitable[_T],
        subscriber,
        *,
        approval_store: ToolApprovalStore | None = None,
        refresh_interval: float = 0.1,
    ) -> _T:
        """Run an awaitable while displaying agent activity, new events, and pending approval prompts.

        Return its result on completion; cancel unfinished work and propagate
        errors or interruption while clearing interactive progress output.
        """
        task = asyncio.ensure_future(operation)
        started = time.monotonic()
        frame = 0
        last_activity = ""
        cursor = 0
        latest_event = None
        try:
            while not task.done():
                if approval_store is not None and sys.stdin.isatty():
                    for request in approval_store.pending():
                        prompt = asyncio.create_task(self.approval_prompt(request, approval_store))
                        try:
                            await asyncio.wait({task, prompt}, return_when=asyncio.FIRST_COMPLETED)
                            if prompt.done():
                                await prompt
                        finally:
                            if not prompt.done():
                                prompt.cancel()
                                with suppress(asyncio.CancelledError):
                                    await prompt
                        if task.done():
                            break
                cursor, events = subscriber.events_since(cursor)
                if events:
                    latest_event = events[-1]
                activity = _activity_text(latest_event)
                new_lines = [
                    line
                    for event in events
                    if (line := _event_line(event, agent_name)) is not None
                ]
                if new_lines:
                    if self.interactive:
                        self._write("\r\033[2K")
                    for line in new_lines:
                        self._write(self._paint(line[: self.width], _DIM) + "\n")
                elapsed = time.monotonic() - started
                status = self._status_line(
                    _SPINNER[frame % len(_SPINNER)], agent_name, activity, elapsed
                )
                if self.interactive:
                    self._write("\r\033[2K" + self._paint(status, _YELLOW, _DIM))
                elif activity != last_activity and not new_lines:
                    self._write(status + "\n")
                last_activity = activity
                frame += 1
                await asyncio.wait({task}, timeout=refresh_interval)
            result = await task
        except BaseException:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            if self.interactive:
                self._write("\r\033[2K")
        _, remaining_events = subscriber.events_since(cursor)
        for event in remaining_events:
            line = _event_line(event, agent_name)
            if line is not None:
                self._write(self._paint(line[: self.width], _DIM) + "\n")
        elapsed = time.monotonic() - started
        status = getattr(result, "status", "completed")
        marker, label, color = _TERMINAL_DISPLAY.get(
            status, ("!", _single_line(status), _YELLOW)
        )
        done = self._status_line(marker, agent_name, label, elapsed)
        self._write(self._paint(done, color, _DIM) + "\n")
        return result

    def _status_line(
        self, marker: str, agent_name: object, activity: object, elapsed: float
    ) -> str:
        """Format a width-limited activity line with agent name and elapsed time."""
        text = f"{marker} {_single_line(agent_name)} · {_single_line(activity)} · {elapsed:.1f}s"
        return text[: self.width]

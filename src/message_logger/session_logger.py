from message_logger.agent_event_subscriber import AgentEventSubscriber
from typing import Optional

import json
import hashlib
import os
import sys
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from message_logger.redaction import redact_text, redact_value
from config_service.paths import private_directory

if TYPE_CHECKING:
    from agent.agent_event import AgentEvent
    from llm.messages import Message


class SessionLogger(AgentEventSubscriber):
    def __init__(self, session_id: str, log_dir: Path, clean: bool = False) -> None:
        """Prepare text and JSONL logs for a session and append its start record.

        When clean is true, delete existing logs for this session first.
        """
        self._session_id = session_id
        self._history_fingerprints: list[bytes] = []
        self._history_messages: list["Message"] = []
        self._streams = {}
        self.closed = False
        private_directory(log_dir)
        self._log_path = log_dir / f"{session_id}.log"
        self._jsonl_path = log_dir / f"{session_id}.jsonl"
        if clean:
            for p in (self._log_path, self._jsonl_path):
                if p.exists():
                    p.unlink()
        self._is_new = not self._log_path.exists() or self._log_path.stat().st_size == 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_json({"type": "session_start", "ts": timestamp})

    def on_event(self, event: "AgentEvent") -> None:
        """Record a subscribed agent event in this session's logs."""
        self.log_event(event)

    def log_system_prompt(self, prompt: str, agent_name: Optional[str] = None) -> None:
        """Record the system prompt in both logs when the text log was initially new or empty."""
        if not self._is_new:
            return
        prompt = self._redact_secrets(prompt)
        if agent_name:
            self._append(f"[{agent_name.upper()}]")
        self._append("[SYSTEM PROMPT]")
        self._append(prompt)
        self._append("")
        self._append_json(
            {
                "type": "system_prompt",
                "ts": datetime.now().isoformat(),
                "data": {"text": prompt},
            }
        )

    def log_messages(self, messages: list["Message"], *, immutable: bool = False) -> None:
        """Append new history; mark replacement after compaction or reset.

        A caller that never mutates admitted messages may opt into fingerprint
        reuse for identical objects. General callers retain content comparison.
        """
        fingerprints = [
            self._history_fingerprints[index]
            if immutable and index < len(self._history_messages) and message is self._history_messages[index]
            else hashlib.sha256(json.dumps(message.to_dict(), sort_keys=True, default=str).encode()).digest()
            for index, message in enumerate(messages)
        ]
        prefix = 0
        for previous, current in zip(self._history_fingerprints, fingerprints):
            if previous != current:
                break
            prefix += 1
        if prefix < len(self._history_fingerprints):
            self._append("[CONTEXT REPLACED]")
            self._append_json({"type": "context_replaced", "ts": datetime.now().isoformat()})
        for message in messages[prefix:]:
            self.log_message(message)
        self._history_fingerprints = fingerprints
        self._history_messages = list(messages)

    def log_message(self, message: "Message") -> None:
        """Record a conversation message in text and JSONL form after heuristic credential redaction."""
        safe_message = self._redact_value(message.to_dict())
        content = safe_message["text"]
        if any(key not in ("role", "text") for key in safe_message):
            content = json.dumps(safe_message, ensure_ascii=False, indent=2)
        self._append(f"[{message.role.upper()}] {content}")
        self._append_json(
            {
                "type": "message",
                "ts": datetime.now().isoformat(),
                "data": safe_message,
            }
        )

    def log_exception(self, exc: BaseException, context: str = "Agent turn failed") -> None:
        """Record an exception and local traceback with recognized credential patterns masked."""
        timestamp = datetime.now()
        exception_type = type(exc).__name__
        message = self._redact_secrets(str(exc))
        traceback_text = self._redact_secrets(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        ).rstrip()
        self._append(
            f"  [{timestamp.strftime('%H:%M:%S')}] [RUNTIME_ERROR] "
            f"{exception_type}: {message}"
        )
        self._append(traceback_text)
        self._append_json(
            {
                "type": "runtime_error",
                "ts": timestamp.isoformat(),
                "session_id": self._session_id,
                "data": {
                    "context": context,
                    "exception_type": exception_type,
                    "message": message,
                    "traceback": traceback_text,
                },
            }
        )

    def log_event(self, event: "AgentEvent") -> None:
        """Record runtime events in readable text and structured JSONL for session diagnostics.

        Apply credential-redaction helpers to event data and message text where
        used; user-message events are left to log_message to avoid duplicates.
        """
        from agent.agent_event import AgentEventType

        ts = event.timestamp.strftime("%H:%M:%S")
        data = self._redact_value(event.data or {})
        safe_message = self._redact_secrets(event.message)
        a = f"[{event.agent_name}] " if event.agent_name else ""

        # --- Text log ---
        if event.event_type == AgentEventType.LLM_RESPONSE:
            usage = data.get("usage") or {}
            tokens_in = usage.get("input_tokens", "?")
            tokens_out = usage.get("output_tokens", "?")
            tools = data.get("tools_to_be_used", [])
            tools_str = f" | tools: {tools}" if tools else ""
            self._append(
                f"  [{ts}] {a}[LLM] stop={data.get('stop_reason')} "
                f"| tokens: {tokens_in}→{tokens_out}{tools_str}"
            )

        elif event.event_type == AgentEventType.TOOL_CALL:
            raw_args = data.get("args", {})
            unpacked_args = self._unpack_json_strings(raw_args)
            args = json.dumps(unpacked_args, ensure_ascii=False, indent=2, default=str)
            self._append(f"  [{ts}] {a}[TOOL_CALL] {data.get('tool_name')}(\n{args}\n)")

        elif event.event_type == AgentEventType.TOOL_RESULT:
            result = data.get("result", "")
            error = " [ERROR]" if data.get("is_error") else ""
            result = self._try_format_json(result)
            self._append(
                f"  [{ts}] {a}[TOOL_RESULT]{error} {data.get('tool_name')} →\n{result}"
            )

        elif event.event_type == AgentEventType.TOOL_APPROVAL_REQUIRED:
            self._append(
                f"  [{ts}] {a}[APPROVAL_REQUIRED] {data.get('tool_name')} "
                f"id={data.get('approval_id')}"
            )

        elif event.event_type == AgentEventType.COMMAND_FINISHED:
            self._append(
                f"  [{ts}] {a}[COMMAND] exit={data.get('exit_code')} "
                f"cmd={data.get('command')}"
            )

        elif event.event_type == AgentEventType.COMMAND_TIMED_OUT:
            self._append(f"  [{ts}] {a}[COMMAND_TIMED_OUT] {data.get('command')}")

        elif event.event_type == AgentEventType.STATUS_CHANGE:
            self._append(f"  [{ts}] {a}[STATUS] {data.get('status')}")

        elif event.event_type == AgentEventType.ASSISTANT_MESSAGE:
            self._append(f"  [{ts}] {a}[ASSISTANT] {data.get('text', safe_message)}")

        elif event.event_type == AgentEventType.ERROR:
            self._append(f"  [{ts}] {a}[ERROR] {safe_message}")

        elif event.event_type == AgentEventType.REASONING:
            self._append(f"  [{ts}] {a}[REASONING] {data.get('text', safe_message)}")

        elif event.event_type == AgentEventType.RETRY_SCHEDULED:
            self._append(
                f"  [{ts}] {a}[RETRY] action={data.get('logical_action_id')} "
                f"attempt={data.get('next_attempt')} delay={data.get('delay_seconds')}s"
            )

        elif event.event_type in {
            AgentEventType.RESULT_CLASSIFIED,
            AgentEventType.TERMINAL_STATE,
        }:
            self._append(
                f"  [{ts}] {a}[{event.event_type.value.upper()}] {safe_message}"
            )

        elif event.event_type == AgentEventType.USER_MESSAGE:
            pass  # already logged via log_message

        else:
            self._append(f"  [{ts}] {a}[{event.event_type.value}] {safe_message}")

        # --- JSONL log (structured) ---
        if event.event_type != AgentEventType.USER_MESSAGE:
            self._append_json(
                {
                    "type": event.event_type.value,
                    "ts": event.timestamp.isoformat(),
                    "iteration": event.iteration,
                    "agent_name": event.agent_name,
                    "message": safe_message,
                    "data": data,
                }
            )

    @staticmethod
    def _unpack_json_strings(obj: object) -> object:
        """Expand nested JSON object or array strings so tool arguments are readable in logs."""
        if isinstance(obj, dict):
            return {k: SessionLogger._unpack_json_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SessionLogger._unpack_json_strings(item) for item in obj]
        if isinstance(obj, str):
            stripped = obj.strip()
            if stripped and stripped[0] in ("{", "["):
                try:
                    parsed = json.loads(stripped)
                    return SessionLogger._unpack_json_strings(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass
        return obj

    @staticmethod
    def _try_format_json(text: str) -> str:
        """Format JSON object or array text for readable logs, preserving other text."""
        if not isinstance(text, str):
            return str(text)
        stripped = text.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                parsed = json.loads(stripped)
                unpacked = SessionLogger._unpack_json_strings(parsed)
                return json.dumps(unpacked, ensure_ascii=False, indent=2, default=str)
            except (json.JSONDecodeError, TypeError):
                pass
        return text

    _redact_secrets = staticmethod(redact_text)
    _redact_value = staticmethod(redact_value)

    @staticmethod
    def _open_private_append(path: Path):
        """Open a UTF-8 append stream with owner-only read and write permissions for a log file."""
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.chmod(path, 0o600)
            return os.fdopen(descriptor, "a", encoding="utf-8")
        except Exception:
            os.close(descriptor)
            raise

    def _append(self, line: str) -> None:
        """Append a text log line, reporting write failures as warnings without raising them."""
        self._write(self._log_path, line + "\n")

    def _append_json(self, entry: dict) -> None:
        """Append one structured JSONL entry, reporting write failures as warnings."""
        self._write(self._jsonl_path, json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _write(self, path: Path, text: str) -> None:
        try:
            if self.closed:
                raise ValueError("Session logger is closed")
            stream = self._streams.get(path)
            if stream is None:
                stream = self._streams[path] = self._open_private_append(path)
            stream.write(text)
            stream.flush()
        except Exception as error:
            with suppress(Exception):
                print(f"Warning: session log write failed ({type(error).__name__}).", file=sys.stderr)

    def close(self) -> None:
        """Flush and close both session-owned append streams exactly once."""
        if self.closed:
            return
        self.closed = True
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception as error:
                with suppress(Exception):
                    print(f"Warning: session log close failed ({type(error).__name__}).", file=sys.stderr)
        self._streams.clear()
        self._history_messages.clear()
        self._history_fingerprints.clear()

    @property
    def log_path(self) -> Path:
        """Return the text log path for this session."""
        return self._log_path

    @property
    def jsonl_path(self) -> Path:
        """Return the structured event log path for this session."""
        return self._jsonl_path

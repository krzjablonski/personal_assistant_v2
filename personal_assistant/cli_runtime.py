from __future__ import annotations

import sys
import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from agent import AgentEvent, AgentEventType
from llm.messages import Message
from agent.simple_agent.simple_agent import SimpleAgent
from config_service.paths import default_data_dir
from llm.langfuse_llm_client import LangfuseTrackedLLMClient, TraceContent, trace_agent_run
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from message_logger.session_logger import SessionLogger
from personal_assistant.cli_settings import (
    CLIENT_TYPES,
    RuntimeSettings,
    client_config,
    persist_settings,
)
from personal_assistant.services.agent_builder import AgentBuildRequest, build_agent
from personal_assistant.skill_references import SkillReferenceError, parse_skill_references
from tool_framework.approval import ToolApprovalStore


RUN_LOG_DIR = default_data_dir() / "agent_runs"


@dataclass
class AgentRunResult:
    response: str | dict
    status: str
    iterations: int
    events: list[AgentEvent]
    usage: Optional[dict]
    latest_context_usage: Optional[dict] = None
    usage_complete: bool = False
    model_call_count: int = 0
    log_path: Optional[str] = None
    jsonl_path: Optional[str] = None


def _create_session_logger(agent: SimpleAgent, log_dir: Path | None = None) -> SessionLogger:
    """Create session log files and record the agent's standing instructions."""
    logger = SessionLogger(session_id=agent.session_id, log_dir=log_dir or RUN_LOG_DIR)
    logger.log_system_prompt(agent.system_prompt, agent_name=agent.name)
    return logger


class CliRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        memory_provider,
        config,
        trace_content: TraceContent | None = None,
        workspace_root: Path | None = None,
        env_file: Path | None = None,
    ) -> None:
        """Build the CLI session's agent, memory, approvals, conversation history, and logging."""
        self.config = config
        if trace_content not in (None, "metadata", "redacted"):
            raise ValueError("Trace content must be metadata or redacted.")
        self.trace_content = trace_content
        self._closed = False
        self._lifecycle_lock = asyncio.Lock()
        self._active_turn: asyncio.Task | None = None
        self.memory_provider = memory_provider
        self.settings = settings
        self.working_directory = workspace_root.expanduser().resolve() if workspace_root is not None else None
        self.env_file = env_file
        config_path = getattr(config, "DB_PATH", None)
        self.data_directory = Path(config_path).parent if config_path is not None else RUN_LOG_DIR.parent
        self.log_directory = self.data_directory / "agent_runs" if config_path is not None else RUN_LOG_DIR
        self.agent, self.short_term_memory, self.approvals = self._build(
            settings
        )
        self.session_logger = _create_session_logger(self.agent, self.log_directory)
        self.agent.diagnostic_log_path = self.session_logger.log_path
        self.agent.subscribe(self.session_logger)

    def _build(
        self,
        settings: RuntimeSettings,
    ):
        """Construct an agent and fresh short-term memory and approval stores from runtime settings.

        Interactive sessions wait for an exact decision inside live execution.
        """
        approvals = ToolApprovalStore(
            decision_timeout_seconds=(float("inf") if sys.stdin.isatty() else 0),
        )
        request = AgentBuildRequest(
            client_type=CLIENT_TYPES[settings.provider],
            client_config=client_config(settings),
            max_iterations=settings.max_iterations,

        )
        agent = build_agent(
            request,
            memory_provider=self.memory_provider,
            approval_store=approvals,
            workspace_root=self.working_directory,
            env_file=self.env_file,
            config=self.config,
            output_root=self.data_directory / "outputs",
        )
        if self.trace_content is not None:
            agent.llm_client = LangfuseTrackedLLMClient(
                agent.llm_client, settings.model, content_mode=self.trace_content,
            )
            agent.short_term_memory._llm_client = agent.llm_client
        return agent, agent.short_term_memory, approvals

    async def rebuild(
        self,
        settings: RuntimeSettings,
        *,
        persist_fields: set[str] | None = None,
    ) -> None:
        """Replace the agent session after a settings change and optionally persist selected settings.

        Start fresh conversation and short-term memory state and clear old session approval requests.
        """
        async with self._lifecycle_lock:
            await self._cancel_turn()
            agent, short_term_memory, approvals = self._build(settings)
            session_logger = None
            try:
                session_logger = _create_session_logger(agent, self.log_directory)
                agent.diagnostic_log_path = session_logger.log_path
                agent.subscribe(session_logger)
                if persist_fields:
                    persist_settings(settings, self.config, persist_fields)
            except BaseException:
                if session_logger is not None:
                    session_logger.close()
                await agent.close_resources()
                await self._close_client(agent.llm_client)
                raise
            old_agent, old_logger = self.agent, self.session_logger
            await old_agent.clear_session()
            old_agent.unsubscribe(old_logger)
            self.approvals.clear()
            self.settings = settings
            self.agent = agent
            self.session_logger = session_logger
            self.short_term_memory = short_term_memory
            self.approvals = approvals
            old_logger.close()
            await self._close_client(old_agent.llm_client)

    @staticmethod
    async def _close_client(client) -> None:
        """Bound provider cleanup and report failures without losing session results."""
        try:
            async with asyncio.timeout(5):
                await client.aclose()
        except Exception as error:
            print(f"Warning: model client close failed ({type(error).__name__}).", file=sys.stderr)

    async def aclose(self) -> None:
        """Release the session's logger and provider; caller owns optional memory."""
        async with self._lifecycle_lock:
            await self._cancel_turn()
            if self._closed:
                return
            self._closed = True
            await self.agent.cancel_active_run()
            await self.agent.close_resources()
            try:
                self.agent.unsubscribe(self.session_logger)
            finally:
                self.session_logger.close()
                self.approvals.clear()
                await self._close_client(self.agent.llm_client)

    async def _cancel_turn(self) -> None:
        turn = self._active_turn
        if turn is not None and not turn.done():
            if not turn.cancelling():
                turn.cancel()
            try:
                await asyncio.shield(turn)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise

    async def run_agent_turn(
        self,
        subscriber: EventBufferSubscriber,
        new_input: str | list[Message],
        response_schema: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        """Admit one new turn after prior execution and logging have stopped."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("The session is closed.")
            await self._cancel_turn()
            turn = asyncio.create_task(self._run_agent_turn(subscriber, new_input, response_schema))
            self._active_turn = turn
        try:
            return await turn
        finally:
            if self._active_turn is turn:
                self._active_turn = None

    async def _run_agent_turn(
        self,
        subscriber: EventBufferSubscriber,
        new_input: str | list[Message],
        response_schema: type[BaseModel] | None = None,
    ) -> AgentRunResult:
        """Run the current conversation with temporary event collection and persistent session logging.

        The agent owns compaction and its consumed transcript. Sum every model
        call's known usage separately from the latest context sample. Preserve
        partial transcripts and remove the temporary subscriber on every exit.
        """
        cursor, _ = subscriber.events_since()
        self.agent.subscribe(subscriber)
        try:
            skills = getattr(self.agent, "skill_runtime", None)
            if isinstance(new_input, str):
                selected = parse_skill_references(new_input, skills.catalog.names() if skills else ())
                for name in selected:
                    result = skills.load_skill_instructions(name)
                    if result.is_error:
                        raise SkillReferenceError(f"Cannot load @{name}. Use /skills to inspect available skills.")
            incoming = [Message("user", new_input)] if isinstance(new_input, str) else new_input
            self.session_logger.log_messages([*self.agent.get_messages(), *incoming])
            with trace_agent_run(self.agent.llm_client, self.agent.session_id, self.agent.name):
                response = await self.agent.run(
                    user_query=new_input,
                    response_schema=response_schema,
                )
            _, events = subscriber.events_since(cursor)
            usage, latest_usage, complete, call_count = self._collect_usage(events)
            return AgentRunResult(
                response=response,
                status=self.agent.status.value,
                iterations=self.agent.iteration_count,
                events=events,
                usage=usage,
                latest_context_usage=latest_usage,
                usage_complete=complete,
                model_call_count=call_count,
                log_path=str(self.session_logger.log_path),
                jsonl_path=str(self.session_logger.jsonl_path),
            )
        except SkillReferenceError:
            raise
        except Exception as exc:
            self.session_logger.log_exception(exc)
            print(f"Full error log: {self.session_logger.log_path}", file=sys.stderr)
            raise
        finally:
            self.session_logger.log_messages(self.agent.get_messages())
            self.agent.unsubscribe(subscriber)

    @staticmethod
    def _collect_usage(events: list[AgentEvent]) -> tuple[dict | None, dict | None, bool, int]:
        """Sum observed counters; missing usage remains visibly incomplete."""
        total: dict = {}
        latest = None
        complete = True
        calls = 0
        for event in events:
            if event.event_type is not AgentEventType.LLM_RESPONSE:
                continue
            calls += 1
            usage = (event.data or {}).get("usage")
            if not isinstance(usage, dict):
                complete = False
                continue
            numeric = {key: value for key, value in usage.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)}
            complete = complete and all(key in numeric for key in ("input_tokens", "output_tokens"))
            for key, value in numeric.items():
                total[key] = total.get(key, 0) + value
            if numeric:
                latest = dict(usage)
        return total or None, latest, complete and calls > 0, calls

    def skill_rows(self) -> list[tuple[str, str]]:
        """Expose the current agent's catalog and activation state to CLI views."""
        skills = getattr(self.agent, "skill_runtime", None)
        if skills is None:
            return []
        return [("@" + name, f"{'loaded' if name in skills.loaded_names else 'available'} - {skills.catalog.get(name).description}")
                for name in skills.catalog.names()]

    async def clear_conversation(self) -> None:
        """Cancel work and reset the agent-owned session and approval requests."""
        async with self._lifecycle_lock:
            await self._cancel_turn()
            await self.agent.clear_session()
            self.approvals.clear()

    def update_loop_setting(self, field: str, value) -> None:
        """Persist and apply a supported loop setting without rebuilding the current conversation.

        Raise ValueError for unsupported fields.
        """
        if field != "max_iterations":
            raise ValueError(f"Unsupported live setting: {field}")
        settings = replace(self.settings, **{field: value})
        persist_settings(settings, self.config, {field})
        self.agent.config.max_iterations = settings.max_iterations
        self.settings = settings

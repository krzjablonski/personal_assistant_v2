from __future__ import annotations


import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from agent.agent_event import AgentEventType
from config_service import ConfigService
from config_service.paths import default_data_dir, migrate_legacy_data
from memory.long_term_memory import LongTermMemory
from message_logger.event_buffer_subscriber import EventBufferSubscriber
from personal_assistant.cli_commands import COMMANDS, _handle_command
from personal_assistant.cli_input import ChatInput
from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import (
    CLIENT_TYPES,
    RuntimeSettings,
    ensure_config_unlocked,
    ensure_provider_credentials,
    guided_setup,
    resolve_settings,
    status_rows,
)
from personal_assistant.console_ui import ConsoleUI
from personal_assistant.services.google_oauth import connect_google
from personal_assistant.services.skills_registry import SKILL_CATALOG
from personal_assistant.skill_references import SkillReferenceError


# Keys an implicit launch-directory .env may not set: they redirect host
# execution, private data, integration identities or model endpoints. An
# explicit --env-file is trusted and may set them.
_IMPLICIT_ENV_DENYLIST = frozenset({
    "PERSONAL_ASSISTANT_DATA_DIR", "XDG_DATA_HOME", "HOME", "TMPDIR", "PATH",
    "BROWSER_PYTHON_PATH", "BROWSER_EXECUTABLE_PATH", "ATTACHMENTS_DIR",
    "GOOGLE_OAUTH_CLIENT_JSON", "GOOGLE_OAUTH_TOKEN_JSON", "GOOGLE_ACCOUNT_EMAIL",
    "GOOGLE_CALENDAR_ID", "EMAIL_TO", "SSH_AUTH_SOCK", "BROWSER",
})
_IMPLICIT_ENV_DENIED_PATTERN = re.compile(
    r"(_BASE_URL|_API_BASE|_ENDPOINT|_PROXY)$|^(PYTHON|LD_|DYLD_|DOCKER_|SSL_|REQUESTS_CA|CURL_CA|LANGFUSE_)"
)


def _implicit_env_denied(key: str) -> bool:
    """Report whether an implicit launch-directory .env may not set this key."""
    upper = key.upper()
    return upper in _IMPLICIT_ENV_DENYLIST or bool(_IMPLICIT_ENV_DENIED_PATTERN.search(upper))


def _load_environment(env_file: Path, *, explicit: bool) -> list[str]:
    """Load configuration from an environment file without overriding the process environment.

    An explicit --env-file is loaded in full. The implicit launch-directory .env
    may come from an untrusted checkout, so security-sensitive keys are ignored
    with a warning; returns the ignored key names.
    """
    if explicit:
        load_dotenv(env_file, override=False)
        return []
    if not env_file.is_file():
        return []
    ignored = []
    for key, value in dotenv_values(env_file, interpolate=False).items():
        if value is None or key in os.environ:
            continue
        if _implicit_env_denied(key):
            if value:
                ignored.append(key)
            continue
        os.environ[key] = value
    if ignored:
        print(
            f"Warning: ignored security-sensitive settings in {env_file}: {', '.join(sorted(ignored))}. "
            "Use --env-file to trust this file.",
            file=sys.stderr,
        )
    return ignored


def build_parser() -> argparse.ArgumentParser:
    """Define command-line options for chat, one-shot requests, providers, skills, and Google connection."""
    parser = argparse.ArgumentParser(description="Run the personal assistant.", allow_abbrev=False)
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-shot prompt; omit for chat mode.",
    )
    parser.add_argument(
        "--client", "--provider", dest="client", choices=CLIENT_TYPES
    )
    parser.add_argument("--model")
    parser.add_argument("--trace", choices=("metadata", "redacted"), help="Opt in to Langfuse tracing: counts only, or redacted text (off by default)")
    parser.add_argument("--base-url")
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--workspace", type=Path, help="Explicit writable console folder (defaults to managed session storage)")
    parser.add_argument("--env-file", type=Path, help="Trusted configuration environment file (defaults to .env in the launch directory, which cannot set path, executable, account or endpoint keys; never grants console access)")
    parser.add_argument("--data-dir", type=Path, help="Private mutable data directory (defaults to the user data directory)")
    parser.add_argument("--migrate-data-from", type=Path, metavar="LEGACY_DATA_DIR", help="Copy legacy data to a new --data-dir and exit; preserve the source")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--list-skills", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Use line-based input without interactive completion")
    parser.add_argument(
        "--connect-google",
        nargs="?",
        const="",
        metavar="CLIENT_JSON",
        help=(
            "Connect Gmail and Google Calendar in a browser; provide a Google "
            "Desktop app client JSON file on first use"
        ),
    )
    return parser


async def _next_agent_prompt(
    runtime: CliRuntime,
    ui: ConsoleUI,
    reader: ChatInput | None = None,
) -> str | None:
    """Read interactive input until an agent request is ready or the user exits.

    Handle local commands first; return None on exit, EOF,
    or an interrupt while reading the prompt.
    """
    while True:
        try:
            line = (await reader.read(ui.prompt()) if reader is not None else input(ui.prompt())).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        finally:
            ui.end_prompt()

        if not line:
            continue

        prompt = line
        if prompt in {"/exit", "/quit"}:
            return None
        if await _handle_command(prompt, runtime, ui):
            continue
        return prompt


def _serialize_response(response: str | dict) -> str:
    """Convert an agent response to display text, formatting dictionaries as indented JSON."""
    if isinstance(response, str):
        return response
    return json.dumps(response, indent=2)


async def _execute_turn(runtime: CliRuntime, ui: ConsoleUI, prompt: str) -> bool:
    """Submit and display a turn whose transcript is retained by the runtime."""
    subscriber = EventBufferSubscriber()
    try:
        result = await ui.track(
            "Personal Assistant",
            runtime.run_agent_turn(subscriber, prompt),
            subscriber,
            approval_store=runtime.approvals,
        )
    except SkillReferenceError as error:
        ui.error(str(error))
        return False
    except asyncio.CancelledError:
        report = next((event for event in reversed(subscriber.snapshot())
                       if event.event_type is AgentEventType.ASSISTANT_MESSAGE
                       and (event.data or {}).get("interrupted")), None)
        if report is not None:
            ui.message("assistant", report.data["text"])
        raise
    except Exception:
        print(
            "Provider request failed. Check /status, credentials, "
            "model, and network."
        )
        return False

    answer = _serialize_response(result.response)
    ui.message("assistant", answer)
    return result.status == "completed"


async def _run_conversation(
    runtime: CliRuntime,
    ui: ConsoleUI,
    initial_prompt: str | None,
    *,
    plain: bool = False,
) -> int:
    """Run a supplied one-shot prompt or continue interactive turns until the user exits."""
    if initial_prompt is not None:
        return await _run_one_shot(runtime, ui, initial_prompt)

    reader = ChatInput(
        SKILL_CATALOG, COMMANDS, plain=plain, color=ui.color,
        toolbar=lambda: f" {runtime.settings.provider} / {runtime.settings.model}",
    )
    while True:
        prompt = await _next_agent_prompt(runtime, ui, reader)
        if prompt is None:
            return 0
        await _execute_turn(runtime, ui, prompt)


async def _run_one_shot(
    runtime: CliRuntime,
    ui: ConsoleUI,
    initial_prompt: str,
) -> int:
    """Process one prompt or local command and return a CLI exit code.

    Return one for execution failure and zero for a handled request
    or an empty or exit prompt.
    """
    if not initial_prompt:
        return 0

    prompt = initial_prompt

    ui.message("user", prompt)
    if prompt in {"/exit", "/quit"}:
        return 0
    if await _handle_command(prompt, runtime, ui):
        return 0
    return 0 if await _execute_turn(runtime, ui, prompt) else 1


def _connect_google_account(client_json: str, config) -> None:
    """Unlock configuration, connect Google credentials, and display the connected account."""
    ensure_config_unlocked(config)
    client_path = Path(client_json) if client_json else None
    email = connect_google(config, client_path)
    print(f"Google account connected: {email}")


def _prepare_runtime_configuration(
    args: argparse.Namespace,
    config,
) -> RuntimeSettings:
    """Resolve CLI settings, complete setup, and prepare credentials for the selected provider."""
    settings = guided_setup(
        args,
        resolve_settings(args, config),
        config,
    )
    ensure_provider_credentials(settings, config)
    return settings


def _initialize_session_ui(runtime: CliRuntime, ui: ConsoleUI) -> None:
    """Display the assistant heading, active session settings, and command-help hint."""
    ui.header("Personal Assistant")
    ui.summary(
        "Session",
        status_rows(
            runtime.settings,
            runtime.config,
        ),
    )
    print("Type /help for commands.")


@asynccontextmanager
async def _runtime_context(
    args: argparse.Namespace,
    settings: RuntimeSettings,
    *,
    config,
) -> AsyncIterator[CliRuntime]:
    """Own first-use memory storage and the runtime for the lifetime of a CLI session."""
    async with AsyncExitStack() as resources:
        workspace = args.workspace.expanduser().resolve() if args.workspace is not None else None
        env_file = (args.env_file or Path.cwd() / ".env").expanduser().absolute()
        if args.env_file is None and not env_file.exists():
            env_file = None
        memory = None
        active = True
        def close_memory():
            nonlocal active
            active = False
            if memory is not None:
                memory.close()
        resources.callback(close_memory)
        def memory_provider():
            nonlocal memory
            if not active:
                raise RuntimeError("The memory session is closed.")
            if memory is None:
                memory = LongTermMemory(Path(config.DB_PATH).with_name("agent_memory.db"))
            return memory
        runtime = CliRuntime(
            settings,
            memory_provider=memory_provider,
            config=config,
            trace_content=args.trace,
            workspace_root=workspace,
            env_file=env_file,
        )
        resources.push_async_callback(runtime.aclose)
        yield runtime


async def _run(args: argparse.Namespace) -> int:
    """Initialize CLI configuration and run Google connection setup or an assistant conversation."""
    workspace = (args.workspace or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("Workspace directory does not exist")
    env_file = (args.env_file or Path.cwd() / ".env").expanduser().absolute()
    if args.env_file is not None and not env_file.is_file():
        raise ValueError(f"Environment file does not exist: {env_file}")
    _load_environment(env_file, explicit=args.env_file is not None)
    data_dir = (args.data_dir or default_data_dir()).expanduser().resolve()
    if args.migrate_data_from is not None:
        try:
            migrate_legacy_data(args.migrate_data_from, data_dir, report_path=workspace / ".ai" / "inbox-triage.md")
        except (OSError, TimeoutError) as exc:
            raise ValueError(f"Data migration failed: {exc}") from exc
        print(f"Data copied to {data_dir}. Original data retained at {args.migrate_data_from}.")
        return 0
    legacy = workspace / "src" / "data"
    if legacy != data_dir and not (data_dir / "agent_config.db").exists() and any(
        (legacy / name).exists() for name in ("agent_config.db", "agent_memory.db")
    ):
        raise ValueError("Legacy data was found in WORKSPACE/src/data. Use --migrate-data-from with a new --data-dir, or --data-dir to keep using the legacy directory.")
    ui = ConsoleUI(color=False if args.plain else None, plain=args.plain)
    config = ConfigService(data_dir / "agent_config.db")
    try:
        if args.connect_google is not None:
            _connect_google_account(args.connect_google, config)
            return 0
        settings = _prepare_runtime_configuration(args, config)
        async with _runtime_context(args, settings, config=config) as runtime:
            _initialize_session_ui(runtime, ui)
            initial_prompt = " ".join(args.prompt) if args.prompt else None
            return await _run_conversation(runtime, ui, initial_prompt, plain=args.plain)
    finally:
        config.close()


def main() -> None:
    """Parse CLI arguments and launch the assistant with user-readable startup failures and exit codes.

    Handle skill listing directly and map keyboard interruption to exit code 130.
    """
    args = build_parser().parse_args()
    if args.list_skills:
        for name, skill in SKILL_CATALOG.skills.items():
            print(f"{name}: {skill.description}")
        return
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        raise SystemExit(
            "Unable to start the assistant. Check configuration and try again."
        ) from None

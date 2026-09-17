from __future__ import annotations

import inspect

from personal_assistant.cli_runtime import CliRuntime
from personal_assistant.cli_settings import (
    ensure_provider_credentials,
    prompt_provider_settings,
    status_rows,
)
from personal_assistant.console_ui import ConsoleUI


async def _change_model(runtime: CliRuntime) -> None:
    """Prompt for provider settings and rebuild a fresh conversation when they change.

    Prepare credentials and persist the relevant provider, model, and local-endpoint fields.
    """
    candidate = prompt_provider_settings(runtime.settings)
    if candidate == runtime.settings:
        print("Model settings unchanged.")
        return
    ensure_provider_credentials(candidate, runtime.config)
    fields = {"provider", "model"}
    if candidate.provider == "local":
        fields.update({"base_url", "context_window"})
    await runtime.rebuild(candidate, persist_fields=fields)
    print("Model changed. Started a new conversation.")


def _change_max_iterations(runtime: CliRuntime) -> None:
    """Prompt for an iteration limit from 1 to 100 and update the live runtime.

    Leave the limit unchanged on empty input and retry invalid entries.
    """
    while True:
        raw = input(
            f"Maximum iterations [1-100, current {runtime.settings.max_iterations}]: "
        ).strip()
        if not raw:
            return
        try:
            value = int(raw)
            if 1 <= value <= 100:
                runtime.update_loop_setting("max_iterations", value)
                print(f"Maximum iterations set to {value}.")
                return
        except ValueError:
            pass
        print("Maximum iterations must be between 1 and 100.")


async def _change_provider_details(runtime: CliRuntime) -> None:
    """Update hosted-provider credentials or local model settings and start a fresh conversation."""
    if runtime.settings.provider == "local":
        await _change_model(runtime)
        return
    ensure_provider_credentials(runtime.settings, runtime.config, force=True)
    await runtime.rebuild(runtime.settings)
    print("Credentials updated. Started a new conversation.")


async def _settings_menu(runtime: CliRuntime) -> None:
    """Offer repeated runtime-setting edits until the user leaves the menu.

    Report invalid selections or setting errors without ending the menu.
    """
    from personal_assistant.cli_web_settings import configure_web, configure_browser
    actions = {
        "1": lambda: _change_model(runtime),
        "2": lambda: _change_max_iterations(runtime),
        "3": lambda: _change_provider_details(runtime),
        "4": lambda: configure_web(runtime.config),
        "5": lambda: configure_browser(runtime),
    }
    while True:
        print(
            "\nSettings\n"
            "  1. Provider and model\n"
            "  2. Maximum iterations\n"
            "  3. Credentials or local endpoint\n"
            "  4. Web search and extraction providers\n"
            "  5. Browser settings\n"
            "  0. Done"
        )
        choice = input("Select: ").strip()
        if choice in {"", "0"}:
            return
        action = actions.get(choice)
        if action is None:
            print("Choose one of the listed numbers.")
            continue
        try:
            result = action()
            if inspect.isawaitable(result):
                await result
        except (KeyError, ValueError) as exc:
            print(f"Settings unchanged: {exc}")


COMMANDS = (
    ("/model", "Change provider or model"),
    ("/settings", "Change runtime settings"),
    ("/status", "Show active configuration"),
    ("/skills", "Show available and loaded skills"),
    ("/clear", "Clear conversation and loaded skills"),
    ("/help", "Show this help"),
    ("/commands", "List commands"),
    ("/exit", "End the session"),
    ("/quit", "End the session (alias)"),
)
HELP_TEXT = "Commands:\n" + "\n".join(
    f"  {command:<22} {description}" for command, description in COMMANDS
)


async def _handle_command(line: str, runtime: CliRuntime, ui: ConsoleUI) -> bool:
    """Handle local commands; ordinary text never grants tool approval."""
    if not line.startswith("/"):
        return False
    command, _, value = line.partition(" ")
    if command in {"/", "/commands"}:
        ui.table("Commands", COMMANDS)
        return True
    if command == "/clear":
        await runtime.clear_conversation()
        print("Conversation cleared.")
        return True
    if command == "/skills":
        ui.table("Skills", runtime.skill_rows())
        return True
    if command == "/status":
        ui.summary(
            "Session",
            status_rows(
                runtime.settings,
                runtime.config,
            ),
        )
        return True
    if command == "/help":
        print(HELP_TEXT)
        return True
    try:
        if command == "/model":
            await _change_model(runtime)
            return True
        if command == "/settings":
            await _settings_menu(runtime)
            return True
    except (KeyError, ValueError) as exc:
        print(f"Settings unchanged: {exc}")
        return True
    print(f"Unknown command: {command}. Type /help for available commands.")
    return True

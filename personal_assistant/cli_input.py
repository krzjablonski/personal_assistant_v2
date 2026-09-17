"""Session-owned terminal composer; releases stdin before command execution."""

import os
import sys
from collections.abc import Callable, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, CompleteEvent, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from agent_skills.catalog import SkillCatalog
from personal_assistant.skill_references import skill_mentions


def supports_interactive_input() -> bool:
    return (sys.stdin.isatty() and sys.stdout.isatty()
            and os.environ.get("TERM", "") not in {"dumb", "unknown"})


class ChatCompleter(Completer):
    def __init__(self, catalog: SkillCatalog, commands: Sequence[tuple[str, str]]):
        self.skills = tuple(("@" + skill.name, skill.description) for skill in catalog.skills.values())
        self.commands = tuple(commands)

    def get_completions(self, document, complete_event):
        text, cursor = document.text, document.cursor_position
        if text.startswith("/"):
            end = next((i for i, char in enumerate(text) if char.isspace()), len(text))
            if cursor > end:
                return
            start, candidates = 0, self.commands
        else:
            mention = next((match for match in skill_mentions(text)
                            if match.start() < cursor <= match.end()), None)
            if mention is None:
                return
            start, end, candidates = mention.start(), mention.end(), self.skills
        prefix, suffix = text[start:cursor], text[cursor:end]
        for name, description in candidates:
            # prompt_toolkit replaces only text before the cursor. Retain a
            # compatible suffix rather than duplicating or deleting user text.
            if name.startswith(prefix) and name.endswith(suffix) and len(prefix) + len(suffix) <= len(name):
                replacement = name[:-len(suffix)] if suffix else name
                yield Completion(replacement, start_position=start - cursor,
                                 display=name, display_meta=" ".join(description.split()))


def _key_bindings(completer: ChatCompleter) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event):
        buffer = event.current_buffer
        state = buffer.complete_state
        if state and state.current_completion:
            buffer.apply_completion(state.current_completion)
        else:
            buffer.complete_state = None
            buffer.validate_and_handle()

    @bindings.add("tab")
    def complete(event):
        buffer = event.current_buffer
        state = buffer.complete_state
        choices = state.completions if state else list(completer.get_completions(
            buffer.document, CompleteEvent(completion_requested=True)))
        if choices:
            buffer.apply_completion(state.current_completion if state and state.current_completion else choices[0])

    @bindings.add("escape", eager=True)
    def dismiss(event):
        event.current_buffer.cancel_completion()

    @bindings.add("c-j")
    def newline(event):
        event.current_buffer.insert_text("\n")

    return bindings


class ChatInput:
    def __init__(
        self, catalog: SkillCatalog, commands: Sequence[tuple[str, str]], *,
        plain: bool = False, color: bool = True, toolbar: Callable[[], str] | None = None,
        input=None, output=None,
    ):
        self.session = None
        if plain or (input is None and not supports_interactive_input()):
            return
        completer = ChatCompleter(catalog, commands)
        use_color = color and "NO_COLOR" not in os.environ
        styles = {
            "prompt": "bold ansicyan",
            "completion-menu.completion.current": "bg:ansicyan ansiblack",
            "completion-menu.meta.completion.current": "bg:ansicyan ansiblack",
            "bottom-toolbar": "noreverse ansibrightblack",
        } if use_color else {}
        self.session = PromptSession(
            message=[("class:prompt", "YOU > ")],
            completer=completer, complete_while_typing=True,
            history=InMemoryHistory(), multiline=True,
            key_bindings=_key_bindings(completer),
            bottom_toolbar=toolbar, style=Style.from_dict(styles),
            color_depth=None if use_color else ColorDepth.DEPTH_1_BIT,
            include_default_pygments_style=False,
            input=input, output=output,
        )

    async def read(self, prompt: str = "YOU > ") -> str:
        if self.session is None:
            return input(prompt)
        return await self.session.prompt_async()

"""Session-owned terminal composer; releases stdin before command execution."""

import asyncio
import os
import sys
from collections.abc import Callable, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, CompleteEvent, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.scroll import scroll_page_down, scroll_page_up
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

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


class ApprovalInput:
    """Inline decision picker with a scrollable, complete action review."""

    def __init__(self, summary: str, details: str, *, color: bool = True, input=None, output=None):
        self.selected = 1
        self.show_details = False
        self.body = TextArea(text=summary, read_only=True, scrollbar=True,
                             wrap_lines=True, height=Dimension(min=2, preferred=8, max=12))
        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("down")
        @bindings.add("tab")
        def select(event):
            self.selected = 1 - self.selected

        @bindings.add("1")
        @bindings.add("2")
        def numbered(event):
            self.selected = int(event.data) - 1

        @bindings.add("enter")
        @bindings.add("c-j")
        def confirm(event):
            event.app.exit(result=self.selected == 0)

        @bindings.add("escape", eager=True)
        @bindings.add("c-d")
        def deny(event):
            event.app.exit(result=False)

        @bindings.add("c-c")
        def cancel(event):
            event.app.exit(exception=asyncio.CancelledError())

        @bindings.add("d")
        def details_view(event):
            self.show_details = not self.show_details
            self.body.text = details if self.show_details else summary
            self.body.buffer.cursor_position = 0
            self.body.window.vertical_scroll = 0

        bindings.add("pagedown")(scroll_page_down)
        bindings.add("pageup")(scroll_page_up)

        def choices():
            return [("class:selected" if i == self.selected else "", f"{'>' if i == self.selected else ' '} {i + 1}. {label}\n")
                    for i, label in enumerate(("Allow once", "Deny"))]

        use_color = color and "NO_COLOR" not in os.environ
        self.app = Application(
            layout=Layout(HSplit([
                Window(FormattedTextControl([("class:title", "APPROVAL REQUIRED")]), height=1),
                Window(FormattedTextControl(lambda: "Full action details" if self.show_details else "Action summary"), height=1),
                self.body,
                Window(FormattedTextControl(choices), height=2),
                Window(FormattedTextControl("Enter confirm | d details | PgUp/PgDn scroll | Esc deny"), wrap_lines=True),
            ]), focused_element=self.body),
            key_bindings=bindings, full_screen=False, erase_when_done=True,
            style=Style.from_dict({"title": "bold ansicyan", "selected": "reverse bold"} if use_color else {"selected": "reverse"}),
            color_depth=None if use_color else ColorDepth.DEPTH_1_BIT,
            input=input, output=output,
        )

    async def read(self) -> bool:
        return await self.app.run_async()

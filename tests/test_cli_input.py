import asyncio
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent_skills.catalog import SkillCatalog, SkillDefinition
from personal_assistant.cli_input import ChatCompleter, ChatInput, supports_interactive_input


def catalog():
    return SkillCatalog({name: SkillDefinition(name, description, Path("SKILL.md"), "instructions")
                         for name, description in (("calendar", "Calendar events"), ("email", "Email messages"))})


COMMANDS = (("/status", "Show configuration"), ("/settings", "Edit settings"))


class CompletionTests(unittest.TestCase):
    def complete(self, text, cursor=None):
        return list(ChatCompleter(catalog(), COMMANDS).get_completions(
            Document(text, cursor_position=cursor), CompleteEvent(completion_requested=True)))

    def test_command_menu_and_descriptions(self):
        self.assertEqual([c.text for c in self.complete("/")], ["/status", "/settings"])
        completion, = self.complete("/sta")
        self.assertEqual((completion.text, completion.start_position, completion.display_meta_text),
                         ("/status", -4, "Show configuration"))

    def test_skills_complete_in_a_sentence_and_preserve_suffix(self):
        completion, = self.complete("Check @ca")
        self.assertEqual((completion.text, completion.start_position, completion.display_meta_text),
                         ("@calendar", -3, "Calendar events"))
        completion, = self.complete("Check @calendar tomorrow", cursor=9)
        self.assertEqual("Check @calendar tomorrow"[:6] + completion.text + "lendar tomorrow",
                         "Check @calendar tomorrow")

    def test_completion_ignores_literals_and_command_arguments(self):
        for text in ("user@ca", "`@ca", "```\n@ca", r"\@ca", "/status @ca", "Use /st", "@calendar.com", "`literal`@ca"):
            with self.subTest(text=text):
                self.assertEqual(self.complete(text), [])

    def test_terminal_detection_respects_streams_and_dumb_terminal(self):
        from unittest.mock import Mock
        terminal = Mock(isatty=lambda: True)
        with patch("sys.stdin", terminal), patch("sys.stdout", terminal), patch.dict("os.environ", {"TERM": "xterm"}):
            self.assertTrue(supports_interactive_input())
            with patch("sys.stdout", io.StringIO()):
                self.assertFalse(supports_interactive_input())
            with patch.dict("os.environ", {"TERM": "dumb"}):
                self.assertFalse(supports_interactive_input())


class KeyboardTests(unittest.IsolatedAsyncioTestCase):
    async def read_keys(self, keys):
        with create_pipe_input() as pipe:
            reader = ChatInput(catalog(), COMMANDS, input=pipe, output=DummyOutput())
            pipe.send_text(keys)
            async with asyncio.timeout(2):
                return await reader.read()

    async def test_tab_completion_then_submit(self):
        self.assertEqual(await self.read_keys("/sta\t\r"), "/status")
        self.assertEqual(await self.read_keys("Check @ca\t\r"), "Check @calendar")

    async def test_newline_and_bracketed_paste_do_not_submit_early(self):
        self.assertEqual(await self.read_keys("first\nsecond\r"), "first\nsecond")
        self.assertEqual(await self.read_keys("\x1b[200~first\nsecond\x1b[201~\r"), "first\nsecond")

    async def test_eof_and_interrupt(self):
        for keys, error in (("\x04", EOFError), ("\x03", KeyboardInterrupt)):
            with self.subTest(keys=keys), self.assertRaises(error):
                await self.read_keys(keys)

    async def test_session_retains_history_between_prompts(self):
        with create_pipe_input() as pipe:
            reader = ChatInput(catalog(), COMMANDS, input=pipe, output=DummyOutput())
            pipe.send_text("first\r")
            self.assertEqual(await asyncio.wait_for(reader.read(), 2), "first")
            pipe.send_text("\x1b[A\r")
            self.assertEqual(await asyncio.wait_for(reader.read(), 2), "first")

    async def test_plain_input_never_constructs_terminal_session(self):
        with patch("personal_assistant.cli_input.PromptSession") as session, patch("builtins.input", return_value="hello"):
            reader = ChatInput(catalog(), COMMANDS, plain=True)
            self.assertEqual(await reader.read(), "hello")
            session.assert_not_called()

    async def test_arrows_select_enter_accepts_and_escape_dismisses(self):
        for dismiss in (False, True):
            with self.subTest(dismiss=dismiss), create_pipe_input() as pipe:
                reader = ChatInput(catalog(), COMMANDS, input=pipe, output=DummyOutput())
                task = asyncio.create_task(reader.read())
                try:
                    pipe.send_text("/")
                    async with asyncio.timeout(2):
                        while reader.session.default_buffer.complete_state is None:
                            await asyncio.sleep(0)
                    pipe.send_text("\x1b[B")
                    async with asyncio.timeout(2):
                        while reader.session.default_buffer.complete_state.current_completion is None:
                            await asyncio.sleep(0)
                    pipe.send_text("\x1b" if dismiss else "\r")
                    async with asyncio.timeout(2):
                        while reader.session.default_buffer.complete_state is not None:
                            await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    pipe.send_text("\r")
                    self.assertEqual(await asyncio.wait_for(task, 2), "/" if dismiss else "/status")
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()

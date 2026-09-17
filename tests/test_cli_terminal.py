"""Exercise terminal rendering and stdin handoff without credentials or a model."""

import os
import re
import select
import signal
import struct
import subprocess
import sys
import time
import unittest


CHILD = r'''
import asyncio
import json
import termios
from personal_assistant.cli_input import ChatInput
from personal_assistant.cli_commands import COMMANDS
from personal_assistant.console_ui import ConsoleUI
from personal_assistant.services.skills_registry import SKILL_CATALOG
from tool_framework.approval import ToolApprovalStore

async def main():
    before = termios.tcgetattr(0)
    reader = ChatInput(SKILL_CATALOG, COMMANDS, toolbar=lambda: "local / fixture")
    try:
        value = await reader.read()
        print("RESULT=" + json.dumps(value, ensure_ascii=True), flush=True)
        store = ToolApprovalStore()
        request = store.request("fixture", {}, "Terminal test")
        await ConsoleUI().approval_prompt(request, store)
        print("APPROVAL=" + store.get(request.approval_id).status, flush=True)
    except (KeyboardInterrupt, EOFError):
        print("CLOSED", flush=True)
    after = termios.tcgetattr(0)
    # macOS may set its pending-input redisplay flag when leaving raw mode.
    # Compare terminal settings without this kernel-maintained transient bit.
    before[3] &= ~getattr(termios, "PENDIN", 0)
    after[3] &= ~getattr(termios, "PENDIN", 0)
    print("RESTORED=" + str(before == after), flush=True)

asyncio.run(main())
'''


@unittest.skipUnless(os.name == "posix", "Requires a POSIX pseudoterminal")
class TerminalTests(unittest.TestCase):
    def start_terminal(self, *, no_color=False):
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 100, 0, 0))
        env = {**os.environ, "TERM": "xterm-256color", "PROMPT_TOOLKIT_NO_CPR": "1"}
        if no_color:
            env["NO_COLOR"] = "1"
        else:
            env.pop("NO_COLOR", None)
        try:
            process = subprocess.Popen([sys.executable, "-c", CHILD], stdin=slave, stdout=slave,
                                       stderr=slave, env=env, start_new_session=True)
        finally:
            os.close(slave)

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

        self.addCleanup(cleanup)
        self.transcript = b""
        return master, process

    def read_until(self, master, marker):
        deadline = time.monotonic() + 5
        while marker not in self.transcript:
            self.assertLess(time.monotonic(), deadline, self.transcript.decode(errors="replace"))
            if select.select([master], [], [], 0.1)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    self.fail(self.transcript.decode(errors="replace"))
                self.assertTrue(data, self.transcript.decode(errors="replace"))
                self.transcript += data

    def test_visible_completion_resize_unicode_and_approval_handoff(self):
        import fcntl
        import termios

        for no_color in (False, True):
            with self.subTest(no_color=no_color):
                master, process = self.start_terminal(no_color=no_color)
                self.read_until(master, b"YOU >")
                os.write(master, b"/sta")
                self.read_until(master, b"Show active configuration")
                # Clear the command and exercise a skill menu in the same prompt.
                os.write(master, b"\x15@ca")
                self.read_until(master, b"@calendar")
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 20, 40, 0, 0))
                os.kill(process.pid, signal.SIGWINCH)
                os.write(master, "\t Sprawd\u017a\x1b[200~\ndruga linia\x1b[201~\r".encode())
                self.read_until(master, b'RESULT="@calendar Sprawd\\u017a\\ndruga linia"')
                self.read_until(master, b"Select:")
                os.write(master, b"2\n")
                self.read_until(master, b"APPROVAL=denied")
                self.read_until(master, b"RESTORED=True")
                self.assertEqual(process.wait(timeout=5), 0)
                if no_color:
                    codes = re.findall(rb"\x1b\[([0-9;]*)m", self.transcript)
                    self.assertFalse(any(30 <= int(n) <= 49 or 90 <= int(n) <= 107
                                         for code in codes for n in code.split(b";") if n))

    def test_ctrl_c_restores_terminal(self):
        master, process = self.start_terminal()
        self.read_until(master, b"YOU >")
        os.write(master, b"\x03")
        self.read_until(master, b"CLOSED")
        self.read_until(master, b"RESTORED=True")
        self.assertEqual(process.wait(timeout=5), 0)


if __name__ == "__main__":
    unittest.main()

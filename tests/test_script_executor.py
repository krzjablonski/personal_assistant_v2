import asyncio
import contextlib
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_skills.script_executor import _run_process, run_script, build_subprocess_env


class TestOutputCapture(unittest.IsolatedAsyncioTestCase):
    async def test_stream_overflow_reports_retained_and_omitted_bytes(self) -> None:
        """A successful process must identify discarded stdout and stderr explicitly."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "large.py"
            script.write_text(
                "import sys\n"
                "sys.stdout.write('x' * 1000017)\n"
                "sys.stderr.write('y' * 1000009)\n"
            )
            result = await run_script(skill_root=root, script_path=script.name)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.stdout), 1_000_000)
        self.assertEqual(len(result.stderr), 1_000_000)
        self.assertEqual(result.stdout_bytes_retained, 1_000_000)
        self.assertEqual(result.stderr_bytes_retained, 1_000_000)
        self.assertEqual(result.stdout_bytes_omitted, 17)
        self.assertEqual(result.stderr_bytes_omitted, 9)
        self.assertFalse(result.output_complete)

    async def test_unicode_capture_does_not_invent_a_replacement_character(self) -> None:
        """Discard an incomplete UTF-8 boundary and count its bytes as omitted."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "unicode.py"
            script.write_text("import sys\nsys.stdout.buffer.write('a🙂z'.encode('utf-8'))\n")
            result = await run_script(
                skill_root=root, script_path=script.name, max_capture_bytes=3,
            )

        self.assertEqual(result.stdout, "a")
        self.assertEqual(result.stdout_bytes_retained, 1)
        self.assertEqual(result.stdout_bytes_omitted, 5)
        self.assertFalse(result.output_complete)


class TestProcessLifetime(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_is_bounded_when_a_detached_child_keeps_pipes_open(self) -> None:
        """A child outside the managed group cannot make timeout or cancellation hang."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "detach_child.py"
            script.write_text(
                "import subprocess, sys\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen(\n"
                "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
                "    start_new_session=True,\n"
                ")\n"
                "Path('child.pid').write_text(str(child.pid))\n"
            )
            for interruption in ("timeout", "cancel"):
                with self.subTest(interruption=interruption):
                    started = asyncio.Event()
                    processes = []
                    spawn = asyncio.create_subprocess_exec

                    async def record_process(*args, **kwargs):
                        proc = await spawn(*args, **kwargs)
                        processes.append(proc)
                        started.set()
                        return proc

                    with (
                        patch("asyncio.create_subprocess_exec", side_effect=record_process),
                        patch("agent_skills.script_executor.CLEANUP_TIMEOUT_SECONDS", 0.05, create=True),
                    ):
                        task = asyncio.create_task(run_script(
                            skill_root=root, script_path=script.name,
                            timeout_seconds=0.2 if interruption == "timeout" else 60,
                        ))
                        try:
                            await asyncio.wait_for(started.wait(), timeout=3)
                            proc = processes[0]
                            async with asyncio.timeout(3):
                                while proc.returncode is None:
                                    await asyncio.sleep(0.01)
                            if interruption == "cancel":
                                task.cancel()
                                with self.assertRaises(asyncio.CancelledError):
                                    await asyncio.wait_for(asyncio.shield(task), timeout=1)
                            else:
                                result = await asyncio.wait_for(asyncio.shield(task), timeout=1)
                                self.assertTrue(result.timed_out)
                            self.assertIsNotNone(proc.returncode)
                            self.assertTrue(proc._transport.is_closing())
                        finally:
                            child_pid = int((root / "child.pid").read_text())
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(child_pid, signal.SIGKILL)
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                            for proc in processes:
                                with contextlib.suppress(ProcessLookupError):
                                    os.killpg(proc.pid, signal.SIGKILL)
                                await asyncio.wait_for(proc.communicate(), timeout=3)

    async def test_timeout_and_cancellation_stop_children_of_an_exited_leader(self) -> None:
        """Close inherited pipes by terminating descendants even after their parent exits."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "spawn_child.py"
            script.write_text(
                "import subprocess, sys\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            )
            for kind in ("script", "process"):
                for interruption in ("timeout", "cancel"):
                    with self.subTest(kind=kind, interruption=interruption):
                        started = asyncio.Event()
                        processes = []
                        spawn = asyncio.create_subprocess_exec

                        async def record_process(*args, **kwargs):
                            proc = await spawn(*args, **kwargs)
                            processes.append(proc)
                            started.set()
                            return proc

                        with patch("asyncio.create_subprocess_exec", side_effect=record_process):
                            timeout = 0.5 if interruption == "timeout" else 60
                            operation = (
                                run_script(
                                    skill_root=root, script_path=script.name,
                                    timeout_seconds=timeout,
                                )
                                if kind == "script" else
                                _run_process(
                                    argv=[sys.executable, str(script)], cwd=root, env=build_subprocess_env((), {}), stdin=None,
                                    timeout_seconds=timeout, max_capture_bytes=1_000_000, timeout_label="Fixture",
                                )
                            )
                            task = asyncio.create_task(operation)
                            try:
                                await asyncio.wait_for(started.wait(), timeout=3)
                                proc = processes[0]
                                async with asyncio.timeout(3):
                                    while proc.returncode is None:
                                        await asyncio.sleep(0.01)
                                self.assertEqual(proc.returncode, 0)
                                if interruption == "cancel":
                                    task.cancel()
                                    with self.assertRaises(asyncio.CancelledError):
                                        await task
                                else:
                                    result = await task
                                    self.assertTrue(result.timed_out)

                                # The fixture child keeps both inherited pipes open
                                # for its lifetime; EOF proves it no longer runs.
                                self.assertTrue(proc.stdout.at_eof())
                                self.assertTrue(proc.stderr.at_eof())
                            finally:
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                                for proc in processes:
                                    try:
                                        os.killpg(proc.pid, signal.SIGKILL)
                                    except ProcessLookupError:
                                        pass
                                    await asyncio.wait_for(proc.communicate(), timeout=3)

    async def test_cancellation_reaps_script_and_control_processes(self) -> None:
        """Verify cancelling script and host control-process execution terminates and reaps each child process."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "wait.py"
            script.write_text("import time\ntime.sleep(60)\n")
            for kind in ("script", "process"):
                with self.subTest(kind=kind):
                    started = asyncio.Event()
                    processes = []
                    spawn = asyncio.create_subprocess_exec

                    async def record_process(*args, **kwargs):
                        """Capture each spawned child and signal readiness so cancellation happens after process creation."""
                        proc = await spawn(*args, **kwargs)
                        processes.append(proc)
                        started.set()
                        return proc

                    with patch("asyncio.create_subprocess_exec", side_effect=record_process):
                        operation = (
                            run_script(skill_root=root, script_path="wait.py")
                            if kind == "script" else
                            _run_process(argv=[sys.executable, str(script)], cwd=root, env=build_subprocess_env((), {}), stdin=None,
                                         timeout_seconds=120, max_capture_bytes=1_000_000, timeout_label="Fixture")
                        )
                        task = asyncio.create_task(operation)
                        try:
                            await asyncio.wait_for(started.wait(), timeout=3)
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                            self.assertIsNotNone(processes[0].returncode)
                            with self.assertRaises(ProcessLookupError):
                                os.kill(processes[0].pid, 0)
                        finally:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                            for proc in processes:
                                if proc.returncode is None:
                                    proc.kill()
                                await proc.wait()


if __name__ == "__main__":
    unittest.main()

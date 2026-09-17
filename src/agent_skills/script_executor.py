from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


DEFAULT_TIMEOUT_SECONDS = 120
MAX_CAPTURE_BYTES = 1_000_000
CLEANUP_TIMEOUT_SECONDS = 1.0

# Baseline environment variables copied from the parent process when present.
# Everything else must be supplied through the per-script ``environment``
# allowlist so bundled CLI scripts never inherit the full parent environment.
_PASSTHROUGH_ENV = ("PATH", "HOME", "LANG", "LC_ALL")


@dataclass(frozen=True)
class ScriptResult:
    """Outcome of a bundled CLI script execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_bytes_retained: int = 0
    stderr_bytes_retained: int = 0
    stdout_bytes_omitted: int = 0
    stderr_bytes_omitted: int = 0
    capture_interrupted: bool = False

    @property
    def output_complete(self) -> bool:
        """Report whether the process finished and neither output stream lost bytes."""
        return not (self.capture_interrupted or self.timed_out or self.stdout_bytes_omitted or self.stderr_bytes_omitted)

    def capture_metadata(self) -> dict[str, int | bool]:
        """Expose small completeness and size fields without duplicating stream content."""
        return {
            "output_complete": self.output_complete,
            "stdout_bytes_retained": self.stdout_bytes_retained,
            "stderr_bytes_retained": self.stderr_bytes_retained,
            "stdout_bytes_omitted": self.stdout_bytes_omitted,
            "stderr_bytes_omitted": self.stderr_bytes_omitted,
        }

    def capture_notice(self) -> str:
        """Warn readers when a successful capture is only partial evidence."""
        if not (self.stdout_bytes_omitted or self.stderr_bytes_omitted):
            return ""
        return (
            "\n\n[Output capture incomplete: "
            f"stdout omitted {self.stdout_bytes_omitted} bytes; "
            f"stderr omitted {self.stderr_bytes_omitted} bytes. "
            "This is partial evidence, not a complete structured response.]"
        )


@dataclass(frozen=True)
class _StreamCapture:
    text: str
    retained_bytes: int
    omitted_bytes: int


def resolve_script_path(skill_root: Path, script_path: str) -> Path:
    """Resolve ``script_path`` strictly under ``skill_root``.

    The requested path must be relative, must not contain ``..`` and must resolve
    to an existing file inside the skill directory. A :class:`ValueError` is
    raised otherwise.
    """
    posix = PurePosixPath(script_path)
    if not posix.parts:
        raise ValueError("Script path must not be empty.")
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError("Script path must be relative and must not contain '..'.")

    root = skill_root.resolve()
    resolved = (root / Path(*posix.parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Script path escapes the skill directory.")
    if not resolved.is_file():
        raise ValueError(f"Script not found: {script_path}")
    return resolved


def build_subprocess_env(
    environment: Iterable[str],
    env_provider: Mapping[str, str],
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct a minimal subprocess environment.

    Only ``PATH``/``HOME``/``LANG``/``LC_ALL`` from the parent process plus the
    allowlisted variables (resolved from ``env_provider``, never blindly from
    ``os.environ``) are included.
    """
    env: dict[str, str] = {}
    baseline = os.environ if base_environment is None else base_environment
    for key in _PASSTHROUGH_ENV:
        value = baseline.get(key)
        if value is not None:
            env[key] = value
    for name in environment:
        if name in env_provider and env_provider[name] is not None:
            env[name] = str(env_provider[name])
    return env


async def run_script(
    *,
    skill_root: Path,
    script_path: str,
    args: Iterable[str] = (),
    stdin: str | None = None,
    environment: Iterable[str] = (),
    env_provider: Mapping[str, str] | None = None,
    base_environment: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_capture_bytes: int = MAX_CAPTURE_BYTES,
) -> ScriptResult:
    """Run a confined Python skill script with its permitted environment and bounded output capture.

    Return exit status and separate stdout/stderr text. On timeout, terminate
    the process group and return a timed-out result without partial output.
    """
    resolved = resolve_script_path(skill_root, script_path)
    env = build_subprocess_env(environment, env_provider or {}, base_environment=base_environment)

    return await _run_process(
        argv=[sys.executable, str(resolved), *map(str, args)],
        cwd=skill_root.resolve(),
        env=env,
        stdin=stdin,
        timeout_seconds=timeout_seconds,
        max_capture_bytes=max_capture_bytes,
        timeout_label="Script",
    )


async def _run_process(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    stdin: str | None,
    timeout_seconds: float,
    max_capture_bytes: int,
    timeout_label: str,
) -> ScriptResult:
    """Run a prepared subprocess with bounded capture and timeout or cancellation cleanup.

    Return decoded output on completion, a timeout result when time expires,
    and re-raise cancellation after terminating the process group.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdin_bytes = stdin.encode("utf-8") if stdin is not None else None
    try:
        stdout, stderr = await asyncio.wait_for(
            _communicate_capped(proc, stdin_bytes, max_capture_bytes),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        await _stop_process(proc)
        return ScriptResult(
            exit_code=-1,
            stdout="",
            stderr=f"{timeout_label} timed out after {timeout_seconds} seconds.",
            timed_out=True,
        )
    except asyncio.CancelledError:
        await _stop_process(proc)
        raise

    return ScriptResult(
        exit_code=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout.text,
        stderr=stderr.text,
        stdout_bytes_retained=stdout.retained_bytes,
        stderr_bytes_retained=stderr.retained_bytes,
        stdout_bytes_omitted=stdout.omitted_bytes,
        stderr_bytes_omitted=stderr.omitted_bytes,
    )


async def _stop_process(proc: asyncio.subprocess.Process) -> None:
    """Kill and reap the managed group without waiting forever for inherited pipes."""
    _kill_process_group(proc)
    try:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                _communicate_capped(proc, None, 0),
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
    finally:
        # asyncio.Process exposes no public close method. A detached descendant
        # may retain these pipes, so close the owned transport after the deadline.
        proc._transport.close()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=CLEANUP_TIMEOUT_SECONDS)


async def _communicate_capped(
    proc: asyncio.subprocess.Process,
    stdin_bytes: bytes | None,
    cap: int,
) -> tuple[_StreamCapture, _StreamCapture]:
    """Exchange standard input and bounded output with a child process and wait for its exit."""
    stdout_bytes, stderr_bytes, _ = await asyncio.gather(
        _read_capped(proc.stdout, cap),
        _read_capped(proc.stderr, cap),
        _feed_stdin(proc, stdin_bytes),
    )
    await proc.wait()
    return stdout_bytes, stderr_bytes


async def _read_capped(stream: asyncio.StreamReader | None, cap: int) -> _StreamCapture:
    """Drain an output stream while retaining only its first cap bytes.

    Continue reading discarded data so a verbose subprocess can finish.
    """
    if stream is None:
        return _StreamCapture("", 0, 0)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        if total < cap:
            chunks.append(chunk[: cap - total])
        total += len(chunk)
    data = b"".join(chunks)
    # When capture cuts a UTF-8 sequence, leave its incomplete bytes out of the
    # preview rather than introducing a replacement character at the boundary.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(data, final=total <= cap)
    retained = len(data) - len(decoder.getstate()[0])
    return _StreamCapture(text, retained, total - retained)


async def _feed_stdin(
    proc: asyncio.subprocess.Process, data: bytes | None
) -> None:
    """Send optional input to the child and close its input stream, tolerating early pipe closure."""
    if proc.stdin is None:
        return
    try:
        if data:
            proc.stdin.write(data)
            await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):
            proc.stdin.close()


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate the original group even if its leader has already exited."""
    try:
        # start_new_session=True makes the child's PID its process-group ID.
        # Looking up that ID after the leader exits can miss surviving children.
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

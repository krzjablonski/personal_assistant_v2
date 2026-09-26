"""One approval-gated argv tool for the fixed local Docker console."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import hashlib
import posixpath
import errno
import re
from pathlib import Path, PurePosixPath

from personal_assistant.services.docker_console import DockerConsole, ConsoleCancelled, ConsoleResult, MAX_TIMEOUT
from tool_framework.i_tool import ITool, PreparedAction, ToolPolicy, ToolResult
from tool_framework.output_files import save_output
from tool_framework.tool_executor import ToolExecutionCancelled
from config_service.paths import private_directory

INPUT_BYTES = 1_000_000
# Linux MAX_ARG_STRLEN (131072) includes the terminating NUL.
ARGUMENT_BYTES = 131_071


def container_path(cwd: str, operand: str) -> PurePosixPath:
    """Join an operand to cwd the way the kernel reads it, keeping '..' for symlink resolution.

    Repeated slashes (including a leading '//') and '.' segments are purely
    lexical, so collapsing them never changes which file is opened.
    """
    joined = posixpath.join(cwd, operand)
    return PurePosixPath("/" + "/".join(part for part in joined.split("/") if part not in ("", ".")))


def reviewed_files(argv: list[str], cwd: str, mounts: tuple[dict, ...],
                   unreviewed: list[dict] | None = None) -> list[dict]:
    """Inspect direct text-file operands without interpreting arbitrary programs.

    Execute original paths to preserve __file__, imports, and shell $0 semantics.
    Rechecking these snapshots detects edits, not filesystem races or changes in
    imported code or filenames embedded inside shell/code expressions. Oversized
    operands that may still execute are appended to ``unreviewed`` when given.
    """
    files = []
    seen = set()
    reviewed_bytes = 0
    interpreter = PurePosixPath(argv[0]).name
    file_interpreter = bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?|sh|bash|dash", interpreter))
    targets = [PurePosixPath(mount["target"]) for mount in mounts]
    for index, operand in enumerate(argv):
        # Keep '..' until the filesystem resolves symlinks, matching chdir/open.
        path = container_path(cwd, operand)
        if path in seen:
            continue
        seen.add(path)
        if ".." in path.parts and not any(path.is_relative_to(target) for target in targets):
            # e.g. /tmp/../workspace/x.py: container symlinks outside the mounts
            # decide where '..' lands, so the reviewed file could differ from the opened one.
            raise ValueError("Operands must not reach mounted directories through '..' from outside them; "
                             "use a normalized container path such as /workspace/script.py.")
        for mount, target in zip(mounts, targets):
            if not path.is_relative_to(target):
                continue
            root = Path(mount["source"])
            original = root / str(path.relative_to(target))
            try:
                resolved = original.resolve()
                exists = resolved.is_file()
            except OSError as error:
                # Inline code can be an argv value too, not a filesystem path.
                if error.errno == errno.ENAMETOOLONG:
                    continue
                raise
            if not resolved.is_relative_to(root):
                raise ValueError("A file operand resolves outside its host mount; use an explicit container path with no escaping host symlink.")
            if not exists:
                continue
            with resolved.open("rb") as stream:
                data = stream.read(INPUT_BYTES + 1)
            if len(data) > INPUT_BYTES:
                inline_before = any(value == "-" or value.startswith(("-c", "-m")) for value in argv[1:index])
                if (index == 0 or data.startswith(b"#!") or resolved.suffix in {".py", ".sh"}
                        or (file_interpreter and not inline_before)):
                    raise ValueError("Generated script review is limited to 1 MB; reduce or split the script before approval.")
                if unreviewed is not None:
                    # Wrappers (env, timeout, nice) and other interpreters may still run it.
                    unreviewed.append({"path": str(path), "host_path": str(resolved), "bytes": resolved.stat().st_size,
                                       "reason": "Larger than the 1 MB review limit; content not shown and not rechecked. "
                                                 "If a program executes or interprets it, it runs unreviewed."})
                continue
            reviewed_bytes += len(data)
            if reviewed_bytes > INPUT_BYTES or len(files) >= 128:
                raise ValueError("Complete file review exceeds 1 MB or 128 files; narrow the command before approval.")
            encoding = "utf-8"
            try:
                content = data.decode(encoding)
            except UnicodeDecodeError:
                encoding = "latin-1 (byte-preserving view)"
                content = data.decode("latin-1")
            stat = resolved.stat()
            files.append({"path": str(path), "host_path": str(resolved), "content": content, "encoding": encoding,
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "identity": [stat.st_dev, stat.st_ino, stat.st_mode]})
    return files


class RunCommandTool(ITool):
    def __init__(self, console: DockerConsole):
        self.console = console
        super().__init__(
            name="run_command",
            description=("Run an approved argv command in a fresh local Linux Docker container. "
                         "Use python - with stdin for code, or sh -c explicitly for shell syntax. "
                         "/workspace is writable; /outputs and configured /inputs are read-only. "
                         "No network or integration credentials. Every call needs approval; never automatically retry. "
                         "State persists only in workspace files. Missing utilities require explicit setup."),
            parameters=[],
            input_schema={"type": "object", "properties": {
                "argv": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
                "stdin": {"type": "string", "description": "Exact UTF-8 standard input, including generated code."},
                "cwd": {"type": "string", "default": "/workspace", "description": "Absolute container directory."},
                "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": MAX_TIMEOUT, "default": 30},
            }, "required": ["argv"], "additionalProperties": False},
            policy=ToolPolicy(mutates_local=True, requires_approval=True, can_parallel=False,
                              max_output_chars=12_000,
                              approval_reason="Arbitrary console code may change approved workspace files. Review the exact command and code."),
        )

    def validate_parameters(self, args: dict) -> None:
        super().validate_parameters(args)
        argv = args["argv"]
        if any("\x00" in value for value in argv):
            raise ValueError("argv must not contain NUL bytes.")
        if sum(len(value.encode("utf-8")) for value in argv) > INPUT_BYTES:
            raise ValueError("argv exceeds the 1 MB input limit.")
        if any(len(value.encode("utf-8")) > ARGUMENT_BYTES for value in argv):
            raise ValueError("An argv element exceeds the 128 KB per-argument Linux limit; pass large code or data through stdin instead.")
        if len(args.get("stdin", "").encode("utf-8")) > INPUT_BYTES:
            raise ValueError("stdin exceeds the 1 MB input limit.")
        cwd = args.get("cwd", "/workspace")
        if not PurePosixPath(cwd).is_absolute() or "\x00" in cwd:
            raise ValueError("cwd must be an absolute container path without NUL bytes.")
        if ".." in PurePosixPath(cwd).parts:
            raise ValueError("cwd must not contain '..' segments; use a normalized container path such as /workspace/project.")
        timeout = args.get("timeout", 30)
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
            raise ValueError(f"timeout must be a finite number greater than zero and at most {MAX_TIMEOUT} seconds.")

    def prepare_action(self, args: dict) -> PreparedAction:
        self.validate_input(args)
        execution = deepcopy({"stdin": None, "cwd": "/workspace", "timeout": 30, **args})
        # Only private session directories are created during preparation;
        # no container or user command starts before the gate grants approval.
        self.console.prepare_workspace()
        private_directory(self.console.outputs)
        environment = self.console.prepare_environment()
        unreviewed: list[dict] = []
        files = reviewed_files(execution["argv"], execution["cwd"], environment.mounts, unreviewed)
        scope = {**deepcopy(execution), **environment.approval_details(),
                 "executable": execution["argv"][0],
                 "reviewed_files": files,
                 **({"unreviewed_files": unreviewed} if unreviewed else {}),
                 "mutable_code_scope": "Direct file operands up to 1 MB are inspected and rechecked; non-UTF-8 bytes use a Latin-1 review view. Original paths execute to preserve imports and path semantics. Imports, large data, and paths embedded in code remain live inputs; this is not a filesystem-race-proof snapshot."}
        review = json.dumps(scope, ensure_ascii=False, indent=2)
        policy = self.policy
        if len(review) > 8000:
            saved = save_output(self.console.outputs, review, suffix=".json")
            # Presentation paths vary across preparations; they must not change
            # the exact execution identity used to remember a denial.
            policy = replace(policy, approval_reason=policy.approval_reason +
                             " Read before allowing. Full command/code review: " + saved["host_output_path"])

        async def execute() -> ToolResult:
            try:
                if reviewed_files(execution["argv"], execution["cwd"], environment.mounts) != files:
                    raise ValueError("Reviewed file contents or identity changed; prepare again and obtain fresh approval.")
                result = await self.console.execute(environment, execution["argv"],
                    stdin=execution["stdin"], cwd=execution["cwd"], timeout=execution["timeout"])
            except (ValueError, OSError) as error:
                # The recheck and Docker's scope validation both run before any
                # container exists; execute() reports post-start failures itself.
                return ToolResult(self.name, execution, f"Command not executed: {error}", True, {"not_executed": True})
            except ConsoleCancelled as error:
                raise ToolExecutionCancelled(self._result(execution, error.result)) from error
            return self._result(execution, result)

        return PreparedAction(self.name, execution, policy, scope, execute)

    async def run(self, args: dict) -> ToolResult:
        raise RuntimeError("run_command must be prepared and executed through the approval gate.")

    def _result(self, args: dict, result: ConsoleResult) -> ToolResult:
        process = result.process
        failed = not result.exit_verified or process.exit_code != 0 or process.timed_out or result.cancelled or not result.cleanup_verified
        metadata = {
            "argv": args["argv"], "cwd": args["cwd"], "exit_code": process.exit_code if result.exit_verified else None,
            "exit_verified": result.exit_verified,
            "stdout": process.stdout[:4000], "stderr": process.stderr[:4000],
            "stream_preview_truncated": len(process.stdout) > 4000 or len(process.stderr) > 4000,
            **process.capture_metadata(), "timeout": process.timed_out, "cancelled": result.cancelled,
            "container_name": result.container_name, "execution_started": result.started,
            "cleanup_verified": result.cleanup_verified,
        }
        status = f"Process exit code: {process.exit_code}." if result.exit_verified else "Process exit status is unknown."
        if not result.started:
            metadata["not_executed"] = True
            status = "The command did not start. " + status
        elif failed:
            metadata["uncertain_changes"] = ["Console execution may have changed workspace files; inspect them before deciding what to do next. Do not automatically repeat this command."]
        else:
            metadata["confirmed_changes"] = ["Console process completed with exit code zero; workspace effects and the requested outcome still require inspection when relevant."]
        if not result.cleanup_verified:
            status += f" Container cleanup could not be verified; check {result.container_name} with local Docker."
        notice = process.capture_notice().strip()
        # Lead with the notice so output limits cannot cut it off.
        body = (notice + "\n" if notice else "") + f"{status}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        return ToolResult(self.name, args, body, failed, metadata)

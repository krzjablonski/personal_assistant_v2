"""Local Docker execution with fixed isolation, narrow mounts, and exact cleanup.

Only metadata inspection runs on the host. User argv is passed to a fresh Linux
container after approval; no host execution or image-install fallback exists.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

import agent_skills
import config_service
from config_service.paths import default_data_dir, private_directory
from agent_skills.script_executor import ScriptResult, _run_process

IMAGE = "personal-assistant-console:1"
MAX_TIMEOUT = 120
CONTROL_TIMEOUT = 5
CAPTURE_BYTES = 1_000_000
ENTRYPOINT = "/usr/local/bin/python"
LAUNCHER = "/opt/personal-assistant-console/exec.py"
FIXED_POLICY = {
    "network": "none", "root_filesystem": "read_only", "capabilities": "none",
    "no_new_privileges": True, "cpus": 1, "memory": "512m", "pids": 64,
    "temporary_storage": "64m", "shared_memory": "16m", "path": "/usr/local/bin:/usr/bin:/bin",
    "shell": "/bin/sh", "timezone": "UTC", "automatic_retry": False,
}


def _host_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}


def _overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def protected_host_paths() -> tuple[Path, ...]:
    home = Path.home()
    application = Path(__file__).resolve().parents[1]
    return tuple(path.resolve() for path in (
        application, Path(agent_skills.__file__).parent.parent,
        Path(config_service.__file__).parent, Path(sys.prefix), default_data_dir(),
        application.parent / ".env", application.parent / "client_secret.json",
        home / ".ssh", home / ".aws", home / ".docker", home / ".config",
        home / ".codex", home / ".agents", home / ".gnupg", home / ".kube", home / ".mozilla",
        home / "Library/Keychains", home / "Library/Safari",
        home / "Library/Containers", home / "Library/Group Containers",
        home / "Library/Application Support/Google", home / "Library/Application Support/Firefox",
        home / "Library/Application Support/Microsoft Edge", home / "Library/Application Support/BraveSoftware",
    ))


@dataclass(frozen=True)
class ConsoleEnvironment:
    docker: str
    endpoint: str
    image_id: str
    architecture: str
    mounts: tuple[dict, ...]
    uid: int
    gid: int

    def approval_details(self) -> dict:
        return {"image": self.image_id, "platform": "linux/" + self.architecture,
                "environment": "Debian 12, Python 3.12, /bin/sh, find, rg, coreutils",
                "entrypoint": [ENTRYPOINT, "-I", LAUNCHER],
                "mounts": list(self.mounts), "user": f"{self.uid}:{self.gid}",
                "docker_endpoint": self.endpoint, "policy": dict(FIXED_POLICY)}


@dataclass(frozen=True)
class ConsoleResult:
    process: ScriptResult
    container_name: str
    started: bool
    cleanup_verified: bool
    cancelled: bool = False
    exit_verified: bool = False


class ConsoleCancelled(asyncio.CancelledError):
    def __init__(self, result: ConsoleResult):
        super().__init__("Console cancelled")
        self.result = result


class DockerConsole:
    def __init__(self, workspace: Path | None, outputs: Path, *, inputs: tuple[tuple[str, Path], ...] = (),
                 protected_paths: tuple[Path, ...] = (), data_directory: Path | None = None):
        self.data_directory = Path(data_directory or default_data_dir()).resolve()
        self.outputs = Path(outputs).absolute()
        self._managed_workspace = None
        if workspace is None:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", self.outputs.name):
                raise ValueError("Managed workspace requires a simple output session name.")
            self._managed_workspace = self.data_directory / "workspaces" / self.outputs.name
        self.workspace = Path(workspace).absolute() if workspace is not None else self._managed_workspace
        self.inputs = tuple((name, Path(path).absolute()) for name, path in inputs)
        # Retain both configured aliases and currently active targets. Later
        # symlink retargeting must not expose either an open database or its next target.
        self.protected_paths = tuple(dict.fromkeys(
            path for p in protected_paths for path in (Path(p).absolute(), Path(p).resolve())
        ))

    def prepare_workspace(self) -> None:
        """Lazily create only this session's managed scratch directory."""
        if self._managed_workspace is not None:
            self.validate_workspace(require_exists=False)
            private_directory(self.workspace)
        self.validate_workspace()

    def validate_workspace(self, *, require_exists: bool = True) -> None:
        """Check workspace configuration without contacting Docker or a provider."""
        self._validate_mount_path(self.workspace, "/workspace", require_exists=require_exists)

    def _validate_mount_path(self, original: Path, target: str, *, require_exists: bool = True) -> Path:
        try:
            path = original.resolve(strict=require_exists)
        except OSError as error:
            raise ValueError(f"Console mount {target} directory is unavailable: {original}.") from error
        if target == "/workspace" and self._managed_workspace is not None and path != self._managed_workspace:
            raise ValueError("The managed workspace must not be redirected through a symlink.")
        if (require_exists and not path.is_dir()) or Path.home().resolve().is_relative_to(path):
            raise ValueError("Console mounts must be narrow directories, never host root or home.")
        data_roots = {default_data_dir().resolve(), self.data_directory}
        protected = (*protected_host_paths(), self.data_directory, *(p.resolve() for p in self.protected_paths))
        for blocked in protected:
            if blocked in data_roots:
                # Only this console's managed workspace and output subtree may
                # cross the private-data boundary; other protected paths still apply.
                if target == "/workspace" and path == self._managed_workspace:
                    continue
                if target == "/outputs" and path.is_relative_to(blocked / "outputs") and path != blocked / "outputs":
                    continue
            if _overlaps(path, blocked):
                raise ValueError(f"Console mount {target} ({path}) overlaps protected host path {blocked}. Select a separate workspace or omit --workspace to use managed session storage.")
        if target != "/outputs" and _overlaps(path, self.outputs.resolve()):
            raise ValueError("Workspace/input mounts must not overlap session outputs.")
        if any(char in str(path) for char in (",", "\n", "\r", "\x00")):
            raise ValueError("Mount paths cannot contain Docker mount separators or control characters.")
        return path

    def context(self) -> str:
        """Describe configuration without starting Docker or resolving credentials."""
        return (f"Console environment: local Docker, Linux Debian 12 image {IMAGE}; "
                "Python 3.12, /bin/sh, find, rg, coreutils. Image identity and Linux architecture are resolved before approval. "
                f"{'Managed session workspace' if self._managed_workspace is not None else 'Host workspace'} {self.workspace} maps to writable /workspace. "
                "Session output files remain readable at /outputs; inspect them with approved commands instead of repeating source operations. "
                "Only explicitly configured inputs appear read-only under /inputs. Fresh container per invocation; "
                "no network, no integration credentials, read-only root, bounded /tmp. Container timezone is UTC. "
                "These are Linux commands, not native host commands.")

    def mounts(self) -> list[dict]:
        grants = [(self.workspace, "/workspace", False)]
        names = set()
        for name, source in self.inputs:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
                raise ValueError("Input mounts require unique simple names under /inputs.")
            names.add(name)
            grants.append((source, "/inputs/" + name, True))
        grants.append((self.outputs, "/outputs", True))
        result = []
        for original, target, readonly in grants:
            path = self._validate_mount_path(original, target)
            if any(_overlaps(path, Path(mount["source"])) for mount in result):
                raise ValueError("Console mount grants must not overlap; read-only inputs cannot also have writable aliases.")
            stat = path.stat()
            result.append({"source": str(path), "target": target, "read_only": readonly,
                           "device": stat.st_dev, "inode": stat.st_ino})
        return result

    @staticmethod
    def _exclude_sockets(mounts: tuple[dict, ...], endpoint: str) -> None:
        sockets = [Path(endpoint[len("unix://"):]).resolve(), Path("/var/run/docker.sock").resolve()]
        if os.environ.get("SSH_AUTH_SOCK"):
            sockets.append(Path(os.environ["SSH_AUTH_SOCK"]).resolve())
        for mount in mounts:
            if any(socket.is_relative_to(Path(mount["source"])) for socket in sockets):
                raise ValueError("Console mounts must not expose Docker or SSH-agent sockets, including custom local endpoints.")

    @staticmethod
    def _inspect(argv: list[str]) -> str:
        try:
            result = subprocess.run(argv, env=_host_environment(), stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=CONTROL_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError("Local Docker is unavailable. Start Docker Desktop and run python -m personal_assistant.console_setup.") from error
        if result.returncode:
            raise ValueError("Local Docker or the console image is unavailable. Start Docker Desktop and run python -m personal_assistant.console_setup.")
        return result.stdout

    def connection(self) -> tuple[str, str]:
        explicit = os.environ.get("DOCKER_HOST")
        if explicit and not explicit.startswith("unix://"):
            raise ValueError("The console requires a local Unix Docker endpoint; remote endpoints are not supported.")
        executable = shutil.which("docker")
        if not executable:
            raise ValueError("Docker CLI is unavailable. Install and start Docker Desktop, then run python -m personal_assistant.console_setup.")
        executable = str(Path(executable).resolve())
        if explicit:
            endpoint = explicit
        else:
            args = [executable, "context", "inspect"]
            if os.environ.get("DOCKER_CONTEXT"):
                args.append(os.environ["DOCKER_CONTEXT"])
            try:
                endpoint = json.loads(self._inspect(args))[0]["Endpoints"]["docker"]["Host"]
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise ValueError("Cannot resolve the local Docker context. Check Docker Desktop setup.") from error
        if not isinstance(endpoint, str) or not endpoint.startswith("unix:///"):
            raise ValueError("The console requires a local Unix Docker endpoint; remote endpoints are not supported.")
        return executable, "unix://" + str(Path(endpoint[len("unix://"):]).resolve())

    def prepare_environment(self) -> ConsoleEnvironment:
        mounts = tuple(self.mounts())
        docker, endpoint = self.connection()
        self._exclude_sockets(mounts, endpoint)
        with tempfile.TemporaryDirectory(prefix="pa-docker-config-") as config:
            image = json.loads(self._inspect([docker, "--config", config, "--host", endpoint,
                                              "image", "inspect", IMAGE]))[0]
        if image.get("Os") != "linux" or not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("Id", "")):
            raise ValueError("The console requires the explicitly built Linux console image.")
        if image.get("Config", {}).get("Volumes"):
            raise ValueError("The console image must not declare implicit volume mounts.")
        uid, gid = os.getuid(), os.getgid()
        if uid == 0:
            raise ValueError("Run the host application as a non-root user so private mounts remain accessible without container root.")
        return ConsoleEnvironment(docker, endpoint, image["Id"], image["Architecture"], mounts, uid, gid)

    async def _invoke(self, environment: ConsoleEnvironment, argv: list[str], *, stdin=None, timeout=CONTROL_TIMEOUT) -> ScriptResult:
        with tempfile.TemporaryDirectory(prefix="pa-docker-config-") as config:
            return await _run_process(
                argv=[environment.docker, "--config", config, "--host", environment.endpoint, *argv],
                cwd=Path(config), env=_host_environment(), stdin=stdin, timeout_seconds=timeout,
                max_capture_bytes=CAPTURE_BYTES, timeout_label="Docker console",
            )

    async def _cleanup(self, environment: ConsoleEnvironment, name: str) -> bool:
        try:
            await self._invoke(environment, ["rm", "--force", name])
            state = await self._invoke(environment, ["container", "ls", "--all", "--filter", "name=^/" + name + "$", "--format", "{{.ID}}"])
            return state.exit_code == 0 and not state.stdout.strip()
        except Exception:
            return False

    async def execute(self, environment: ConsoleEnvironment, argv: list[str], *, stdin: str | None,
                      cwd: str, timeout: float) -> ConsoleResult:
        if tuple(self.mounts()) != environment.mounts:
            raise ValueError("The approved mount scope changed; prepare again and obtain fresh approval.")
        if self.connection() != (environment.docker, environment.endpoint):
            raise ValueError("The approved Docker endpoint changed; prepare again and obtain fresh approval.")
        self._exclude_sockets(environment.mounts, environment.endpoint)
        name = "pa-console-" + uuid.uuid4().hex
        args = ["create", "--name=" + name, "--pull=never", "--interactive", "--read-only", "--network=none",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--cpus=1", "--memory=512m", "--memory-swap=512m",
                "--pids-limit=64", "--shm-size=16m", "--tmpfs=/tmp:rw,nosuid,nodev,size=64m,mode=1777",
                "--ulimit=nofile=1024:1024", "--ulimit=core=0:0", "--log-driver=none", "--init",
                f"--user={environment.uid}:{environment.gid}", "--workdir=/workspace",
                "--env=HOME=/tmp", "--env=TZ=UTC", "--env=PYTHONDONTWRITEBYTECODE=1",
                "--entrypoint=" + ENTRYPOINT]
        for mount in environment.mounts:
            option = f"type=bind,source={mount['source']},target={mount['target']},bind-propagation=rprivate"
            # Disable nested host mount propagation/inclusion on supported Docker versions.
            option += ",bind-recursive=disabled"
            if mount["read_only"]: option += ",readonly"
            args.extend(["--mount", option])
        args.extend([environment.image_id, "-I", LAUNCHER, cwd, *argv])
        started = cancelled = creation_acknowledged = exit_verified = False
        process = ScriptResult(-1, "", "Container did not start.")
        try:
            process = await self._invoke(environment, args)
            if process.exit_code == 0:
                creation_acknowledged = True
                started = True  # A failed start response may still hide actual execution.
                process = await self._invoke(environment, ["start", "--attach", "--interactive", name], stdin=stdin, timeout=timeout)
                if not process.timed_out:
                    inspected = await self._invoke(environment, ["inspect", "--format", "{{json .State}}", name])
                    try:
                        state = json.loads(inspected.stdout) if inspected.exit_code == 0 else {}
                        exit_verified = (state.get("Status") == "exited" and state.get("Running") is False
                                         and type(state.get("ExitCode")) is int)
                    except (ValueError, AttributeError):
                        exit_verified = False
                    complete = exit_verified and process.exit_code == 0 and state["ExitCode"] == 0
                    process = replace(process, exit_code=state["ExitCode"] if exit_verified else process.exit_code,
                                      capture_interrupted=process.capture_interrupted or not complete)
        except asyncio.CancelledError:
            cancelled = True
            process = ScriptResult(-1, "", "Console invocation cancelled.", capture_interrupted=True)
        except Exception as error:
            process = ScriptResult(-1, "", f"Docker transport failed ({type(error).__name__}); check the local daemon.", capture_interrupted=True)
        finally:
            cleanup = asyncio.create_task(self._cleanup(environment, name))
            while True:
                try:
                    cleaned = await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if cleanup.done():
                        cleaned = False
                        break
        # An unacknowledged create request can complete after cleanup, including
        # ordinary transport EOF/reset errors, timeouts, and cancellation.
        # Absence observed now does not prove the daemon abandoned that request.
        if not creation_acknowledged:
            cleaned = False
        result = ConsoleResult(process, name, started, cleaned, cancelled, exit_verified)
        if cancelled:
            raise ConsoleCancelled(result)
        return result

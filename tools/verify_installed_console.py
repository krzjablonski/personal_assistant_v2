"""Offline wheel + CLI smoke against real local Docker, using only synthetic data.

Run explicitly after console_setup; never pulls an image or contacts a provider.
Only the model adapter and in-process approval decision are replaced. The CLI,
installed package, gate, command executor, Docker daemon and filesystem are real.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


async def installed_smoke(site: Path) -> None:
    from unittest.mock import patch
    import personal_assistant.cli as cli
    from llm.i_llm_client import LLMResponse
    from llm.messages import Message, ToolCall
    from personal_assistant.services import agent_builder, console_tools, docker_console

    for module in (cli, agent_builder, console_tools, docker_console):
        assert Path(module.__file__).resolve().is_relative_to(site), module.__file__

    workspace = Path.cwd() / "workspace"
    workspace.mkdir()
    data = Path.cwd() / "private-data"
    code = "from pathlib import Path\nPath('wheel-proof.txt').write_text('installed-ready')\nprint('installed-ready')\n"

    class SyntheticClient:
        context_window = 100000
        model = "offline-smoke"

        def __init__(self):
            self.calls = []
            self.closed = False

        async def chat(self, **request):
            self.calls.append({"messages": deepcopy(request["messages"]), "system": request["system"]})
            if len(self.calls) == 1:
                names = {tool.name for tool in request["tools"]}
                assert names == {"run_command", "load_skill_instructions", "read_skill_resource",
                                 "run_skill_command", "save_memory", "recall_memory"}, names
                for skill in ("email", "calendar", "web-research", "wiki", "memory"):
                    assert skill in request["system"]
                message = Message("assistant", tool_calls=[ToolCall("wheel-command", "run_command",
                                  {"argv": ["python", "-"], "stdin": code})])
                return LLMResponse(message, "tool_calls", {"input_tokens": 10, "output_tokens": 10})
            assert len(self.calls) == 2
            results = [m for m in request["messages"] if m.role == "tool"]
            assert len(results) == 1 and not results[0].is_error, results
            assert "installed-ready" in results[0].text
            return LLMResponse(Message("assistant", "Installed console smoke complete."), "end_turn",
                               {"input_tokens": 20, "output_tokens": 5})

        async def aclose(self):
            self.closed = True

    approvals = []
    runtimes = []

    async def approve(request):
        approvals.append(request)
        assert request.tool_name == "run_command"
        assert request.arguments["stdin"] == code
        return True

    class ApprovedRuntime(cli.CliRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.approvals.approval_handler = approve
            runtimes.append(self)

    arguments = ["--client", "local", "--model", "offline-smoke", "--workspace", str(workspace),
                 "--data-dir", str(data), "Create the synthetic wheel proof"]
    client = SyntheticClient()
    with patch.object(agent_builder, "create_client", return_value=client), patch.object(cli, "CliRuntime", ApprovedRuntime):
        assert await cli._run(cli.build_parser().parse_args(arguments)) == 0
    assert (workspace / "wheel-proof.txt").read_text() == "installed-ready"
    assert len(approvals) == 1 and len(client.calls) == 2 and client.closed
    assert not runtimes[0].approvals.pending()
    assert not (data / "agent_memory.db").exists()

    # A fresh noninteractive invocation cannot reuse the earlier authorization.
    (workspace / "wheel-proof.txt").unlink()
    blocked = SyntheticClient()
    with patch.object(agent_builder, "create_client", return_value=blocked):
        assert await cli._run(cli.build_parser().parse_args(arguments)) == 1
    assert len(blocked.calls) == 1 and blocked.closed
    assert not (workspace / "wheel-proof.txt").exists()
    assert not (data / "agent_memory.db").exists()
    print(json.dumps({"installed_imports": True, "approved_command": "completed",
                      "fresh_unavailable_approval": "not_executed", "memory_created": False}))


def verify() -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "src")]
    from personal_assistant.services.docker_console import DockerConsole
    _, endpoint = DockerConsole(repo, repo).connection()
    with tempfile.TemporaryDirectory(prefix="pa-wheel-console-") as temporary:
        root = Path(temporary)
        build = root / "build"
        build.mkdir()
        for name in ("src", "personal_assistant"):
            shutil.copytree(repo / name, build / name,
                            ignore=shutil.ignore_patterns("data", "__pycache__", "*.egg-info"))
        shutil.copy2(repo / "pyproject.toml", build / "pyproject.toml")
        (root / "home").mkdir()
        env = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR") if key in os.environ}
        env.update(HOME=str(root / "home"), DOCKER_HOST=endpoint, PYTHONDONTWRITEBYTECODE="1",
                   PERSONAL_ASSISTANT_DATA_DIR=str(root / "private-data"))

        def run(*args):
            result = subprocess.run([sys.executable, *args], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout

        run("-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index", "--no-cache-dir",
            "--wheel-dir", str(root / "wheels"), str(build))
        site = root / "installed"
        run("-m", "pip", "install", "--no-deps", "--no-index", "--no-cache-dir", "--target", str(site),
            str(next((root / "wheels").glob("*.whl"))))
        env["PYTHONPATH"] = str(site)
        assert "--workspace" in run("-m", "personal_assistant", "--help")
        assert len(run("-m", "personal_assistant", "--list-skills").splitlines()) == 5
        smoke = root / "smoke.py"
        shutil.copy2(__file__, smoke)
        print(run(str(smoke), "--installed", str(site)))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--installed":
        asyncio.run(installed_smoke(Path(sys.argv[2]).resolve()))
    else:
        verify()

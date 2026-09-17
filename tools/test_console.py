"""Explicit real-Docker boundary tests; never pulls images or starts a daemon."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo), str(repo / "src")]
from personal_assistant.services.docker_console import DockerConsole
_, endpoint = DockerConsole(repo, repo).connection()
with tempfile.TemporaryDirectory(prefix="pa-docker-tests-") as temporary:
    root = Path(temporary)
    (root / "home").mkdir()
    env = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR") if key in os.environ}
    env.update(HOME=str(root / "home"), DOCKER_HOST=endpoint, PERSONAL_ASSISTANT_DATA_DIR=str(root / "private-data"),
               PYTHONPATH=f"{repo}:{repo / 'src'}", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "unittest", "-v", "tests.docker_console_integration", *sys.argv[1:]],
                            cwd=root, env=env)
    raise SystemExit(result.returncode)

"""Run the current working sources without real configuration, data or credentials."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
python = Path(sys.executable)
with tempfile.TemporaryDirectory(prefix="pa-current-tests-") as directory:
    root = Path(directory)
    for name in ("src", "personal_assistant", "tests", "evaluations", "tools", ".docs", ".ai"):
        source = repo / name
        if source.exists():
            shutil.copytree(source, root / name, ignore=shutil.ignore_patterns("data", "__pycache__", ".pytest_cache", ".fetch_cache", ".analyze_image_cache"))
    for name in ("pyproject.toml", "requirements.txt", "requirements-lock.txt", "README.md", ".gitignore", ".env.example"):
        shutil.copy2(repo / name, root / name)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR", "LC_ALL") if key in os.environ}
    env.update(HOME=str(root / "home"), PERSONAL_ASSISTANT_DATA_DIR=str(root / "private-data"), XDG_DATA_HOME=str(root / "home" / "data"), PYTHONPATH=f"{root}:{root / 'src'}", PYTHONDONTWRITEBYTECODE="1", LANGFUSE_TRACING_ENABLED="false")
    arguments = sys.argv[1:] or ["discover", "-s", "tests", "-t", "."]
    command = [str(python), "-m", "unittest", *arguments]
    print("Isolated current-source tests:", " ".join(arguments), flush=True)
    result = subprocess.run(command, cwd=root, env=env)
    raise SystemExit(result.returncode)

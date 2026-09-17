from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class TestInstalledDistribution(unittest.TestCase):
    def test_clean_wheel_contains_skills_and_runs_outside_checkout_without_writes(self):
        source = Path(__file__).resolve().parents[1]
        expected = {"calendar", "email", "memory", "web-research", "wiki", "browser"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            for name in ("src", "personal_assistant", "skills"):
                if (source / name).exists():
                    shutil.copytree(source / name, build / name,
                                    ignore=shutil.ignore_patterns("data", "__pycache__", "*.egg-info"))
            shutil.copy2(source / "pyproject.toml", build / "pyproject.toml")
            env = {key: os.environ[key] for key in ("PATH", "LANG", "TMPDIR") if key in os.environ}
            env.update(HOME=str(root / "home"), PYTHONDONTWRITEBYTECODE="1",
                       PERSONAL_ASSISTANT_DATA_DIR=str(root / "private-data"))

            def run(*args, cwd=root):
                result = subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                                        text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                return result.stdout

            run("-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
                "--no-cache-dir", "--wheel-dir", str(root / "wheels"), str(build))
            wheel = next((root / "wheels").glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
            self.assertFalse(any(name.startswith("mcp_client/") or "/sandbox" in name or "/inbox_triage/" in name or name.endswith("agent_profiles.py") for name in names))
            for name in expected:
                self.assertIn(f"agent_skills/bundled/{name}/SKILL.md", names)
            self.assertIn("personal_assistant/console_image/Dockerfile", names)
            self.assertIn("personal_assistant/console_image/exec.py", names)
            self.assertIn("personal_assistant/browser-requirements.txt", names)
            self.assertIn("personal_assistant/services/browser_worker.py", names)
            self.assertIn("personal_assistant/cli_input.py", names)
            self.assertIn("personal_assistant/skill_references.py", names)
            self.assertIn("agent_skills/web_research/providers.py", names)
            self.assertIn("agent_skills/bundled/email/scripts/google_credentials.py", names)
            self.assertIn("agent_skills/bundled/email/references/inbox-triage.md", names)
            self.assertFalse(any(f"bundled/{retired}/" in name for name in names for retired in ("filesystem", "coding-workflow", "date-time")))
            self.assertFalse(any("agent_config.db" in name or "agent_memory.db" in name for name in names))

            site = root / "installed"
            run("-m", "pip", "install", "--no-deps", "--no-index", "--no-cache-dir",
                "--target", str(site), str(wheel))
            env["PYTHONPATH"] = str(site)
            output = run("-m", "personal_assistant", "--list-skills")
            self.assertEqual({line.split(":", 1)[0] for line in output.splitlines()}, expected)
            self.assertIn("--help", run("-m", "personal_assistant", "--help"))
            self.assertIn("--plain", run("-m", "personal_assistant", "--help"))
            self.assertIn("--help", run("-m", "personal_assistant.console_setup", "--help"))
            self.assertIn("--install", run("-m", "personal_assistant.browser_setup", "--help"))
            run("-c", "import importlib.util; from agent_skills.web_research.providers import WebProvider; from personal_assistant.services.browser_session import BrowserSession; assert 'browser_use' not in __import__('sys').modules")
            for facade in ('search', 'extract'):
                self.assertIn('--help', run(str(site / 'agent_skills' / 'bundled' / 'web-research' / 'scripts' / (facade + '.py')), '--help'))
            self.assertFalse((root / "private-data").exists())
            self.assertFalse(any(site.rglob("*.db")))

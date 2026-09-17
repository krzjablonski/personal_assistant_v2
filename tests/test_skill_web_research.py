from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from agent_skills.catalog import SkillCatalog

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "src" / "agent_skills" / "bundled"
WEB_DIR = SKILLS_ROOT / "web-research" / "scripts"
SEARCH = WEB_DIR / "search.py"
EXTRACT = WEB_DIR / "extract.py"


def _env_without_tavily() -> dict[str, str]:
    """Copy the process environment without the Tavily key for missing-credential tests."""
    env = dict(os.environ)
    env.pop("TAVILY_API_KEY", None)
    return env


def _run(script: Path, args: list[str], env: dict[str, str] | None = None):
    """Run a research script with captured output and an optional test environment."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=env,
    )


class TestWebResearchCliValidation(unittest.TestCase):
    """Hermetic tests: argument validation and the missing-API-key path exit
    before any network call, so no live requests are made."""

    def test_search_missing_query_exits_nonzero(self) -> None:
        """Ensure search rejects an omitted query with a CLI usage error."""
        result = _run(SEARCH, [])  # missing --query
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

    def test_search_missing_api_key(self) -> None:
        """Ensure search reports its missing Tavily credential before requesting results."""
        result = _run(SEARCH, ["--query", "cats"], env=_env_without_tavily())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TAVILY_API_KEY", result.stderr)

    def test_extract_missing_url_exits_nonzero(self) -> None:
        """Ensure extraction rejects an omitted URL with a CLI usage error."""
        result = _run(EXTRACT, [])  # missing --url
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

    def test_extract_missing_api_key(self) -> None:
        """Ensure extraction reports its missing Tavily credential before requesting content."""
        result = _run(
            EXTRACT, ["--url", "https://example.com"], env=_env_without_tavily()
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TAVILY_API_KEY", result.stderr)


class TestWebResearchCatalog(unittest.TestCase):
    def test_metadata_scripts_parsed_with_environment(self) -> None:
        """Verify both research commands declare read-only access and their Tavily key requirement."""
        catalog = SkillCatalog.discover([SKILLS_ROOT])
        skill = catalog.get("web-research")
        for rel in ("scripts/search.py", "scripts/extract.py"):
            spec = skill.scripts.get(rel)
            self.assertIsNotNone(spec, f"{rel} must parse into a ScriptSpec")
            self.assertEqual(spec.safety, "read-only")
            capability = "SEARCH" if "search.py" in rel else "EXTRACT"
            self.assertEqual(spec.environment, (f"WEB_{capability}_PROVIDER", f"WEB_{capability}_API_KEY", "SESSION_OUTPUT_DIR"))


class TestEnvProviderWiring(unittest.TestCase):
    def test_provider_supplies_tavily_key_from_config(self) -> None:
        """Verify the skill environment provider uses its explicitly supplied configuration."""
        from personal_assistant.services.agent_builder import build_skill_env_provider
        from types import SimpleNamespace

        values = {"web_search.tavily_api_key": "test-key-123"}
        provider = build_skill_env_provider(SimpleNamespace(get=values.get))(("TAVILY_API_KEY",))
        self.assertEqual(provider.get("TAVILY_API_KEY"), "test-key-123")


if __name__ == "__main__":
    unittest.main()

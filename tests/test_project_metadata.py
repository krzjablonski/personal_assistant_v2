import tomllib
import unittest
from pathlib import Path


class TestProjectMetadata(unittest.TestCase):
    def test_installation_has_no_retired_integration_dependencies(self) -> None:
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
        dependencies = project["dependencies"]
        self.assertIn("jsonschema", dependencies)
        self.assertFalse(any(item.split(">=")[0] in {"mcp", "e2b"} for item in dependencies))
        self.assertEqual(set(project["optional-dependencies"]), {"observability", "dev"})


if __name__ == "__main__":
    unittest.main()

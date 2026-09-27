import re
import tomllib
import unittest
from pathlib import Path


class TestProjectMetadata(unittest.TestCase):
    def test_installation_has_no_retired_integration_dependencies(self) -> None:
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
        dependencies = project["dependencies"]
        names = {re.split(r"[<>=!~\[ ]", item, maxsplit=1)[0] for item in dependencies}
        self.assertIn("jsonschema", names)
        self.assertFalse(any(item.split(">=")[0] in {"mcp", "e2b"} for item in dependencies))
        self.assertEqual(set(project["optional-dependencies"]), {"observability", "dev"})

    def test_runtime_dependencies_declare_lower_bounds(self) -> None:
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
        unbounded = [item for item in project["dependencies"] if ">=" not in item and "==" not in item]
        self.assertEqual(unbounded, [])


if __name__ == "__main__":
    unittest.main()

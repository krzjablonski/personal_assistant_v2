from __future__ import annotations

import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_MODULES = (
    "agent_builder",
    "google_oauth",
    "skills_registry",
    "console_tools",
    "docker_console",
)


class TestServicesLayout(unittest.TestCase):
    def test_personal_ops_modules_are_importable_from_services_only(self) -> None:
        """Verify the retired personal_ops directory is absent and service modules import from the new package."""
        self.assertFalse((ROOT / "personal_ops").exists())

        for module_name in SERVICE_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(
                    f"personal_assistant.services.{module_name}"
                )


if __name__ == "__main__":
    unittest.main()

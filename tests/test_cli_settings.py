import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from personal_assistant.cli_settings import RuntimeSettings, persist_settings


class TestAtomicSettingsPersistence(unittest.TestCase):
    def test_selected_settings_are_normalized_and_saved_in_one_batch(self):
        config = SimpleNamespace(set_many=Mock())
        settings = RuntimeSettings(provider="local", model="test-model", max_iterations=20)

        persist_settings(settings, config, {"provider", "model", "max_iterations", "context_window"})

        config.set_many.assert_called_once_with({
            "cli.provider": "local",
            "cli.model": "test-model",
            "cli.max_iterations": "20",
            "cli.local_context_window": "",
        })

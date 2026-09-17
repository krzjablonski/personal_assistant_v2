from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from personal_assistant.services.google_oauth import GOOGLE_SCOPES, connect_google


class FakeConfig:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        """Create an in-memory configuration with a record of writes for OAuth assertions."""
        self.values = dict(values or {})
        self.set_calls: list[tuple[str, str]] = []

    def get(self, key: str) -> str | None:
        """Return a saved fake configuration value, or None when absent."""
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a fake configuration value and record the write for later assertions."""
        self.values[key] = value
        self.set_calls.append((key, value))

    def set_many(self, values: dict[str, str]) -> None:
        for key, value in values.items():
            self.set(key, value)


class FakeCredentials:
    def to_json(self) -> str:
        """Supply predictable serialized OAuth credentials without contacting Google."""
        return json.dumps({"refresh_token": "refresh", "scopes": GOOGLE_SCOPES})


class FakeFlow:
    def __init__(self) -> None:
        """Initialize the fake OAuth flow before any loopback-server call."""
        self.run_kwargs = None

    def run_local_server(self, **kwargs):
        """Record loopback options and return fake credentials without opening a server."""
        self.run_kwargs = kwargs
        return FakeCredentials()


class TestGoogleOAuth(unittest.TestCase):
    def test_connect_uses_desktop_loopback_flow_and_stores_credentials(self) -> None:
        """Verify desktop OAuth uses a loopback redirect and saves client and token configuration."""
        client = {
            "installed": {
                "client_id": "client.apps.googleusercontent.com",
                "client_secret": "client-secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        config = FakeConfig()
        flow = FakeFlow()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "client.json"
            path.write_text(json.dumps(client), encoding="utf-8")
            email = connect_google(
                config,
                path,
                flow_factory=lambda loaded, scopes: (
                    self.assertEqual(loaded, client)
                    or self.assertEqual(tuple(scopes), GOOGLE_SCOPES)
                    or flow
                ),
                profile_getter=lambda _credentials: "me@example.com",
            )

        self.assertEqual(email, "me@example.com")
        self.assertEqual(
            flow.run_kwargs,
            {"host": "127.0.0.1", "port": 0, "open_browser": True},
        )
        self.assertEqual(
            [key for key, _value in config.set_calls],
            ["google.oauth_client_json", "google.oauth_token_json", "google.account_email", "google.account_credential_binding"],
        )
        self.assertEqual(config.values["google.account_email"], "me@example.com")
        self.assertEqual(json.loads(config.values["google.oauth_client_json"]), client)

    def test_connect_reuses_saved_client_config(self) -> None:
        """Ensure reconnecting can use saved client configuration without a new file."""
        client = {"installed": {"client_id": "saved-client"}}
        config = FakeConfig({"google.oauth_client_json": json.dumps(client)})

        email = connect_google(
            config,
            None,
            flow_factory=lambda loaded, _scopes: (
                self.assertEqual(loaded, client) or FakeFlow()
            ),
            profile_getter=lambda _credentials: "me@example.com",
        )

        self.assertEqual(email, "me@example.com")

    def test_connect_rejects_non_desktop_client_before_opening_browser(self) -> None:
        """Ensure web-client credentials are rejected by the desktop connection flow."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "client.json"
            path.write_text(json.dumps({"web": {"client_id": "wrong"}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Desktop app"):
                connect_google(FakeConfig(), path)


if __name__ == "__main__":
    unittest.main()

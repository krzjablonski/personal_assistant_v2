from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable


GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events",
)
MAX_CLIENT_CONFIG_BYTES = 64 * 1024


def connect_google(
    config,
    client_json_path: Path | None,
    *,
    flow_factory: Callable | None = None,
    profile_getter: Callable | None = None,
) -> str:
    """Authorize a Google account, persist its client configuration and token, and return its email address.

    Use the desktop browser flow and reject a missing account email before saving.
    """
    client_config = _load_client_config(config, client_json_path)
    if flow_factory is None:
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow_factory = InstalledAppFlow.from_client_config
    if profile_getter is None:
        profile_getter = _profile_email

    flow = flow_factory(client_config, GOOGLE_SCOPES)
    # Google recommends a loopback callback for desktop applications.
    # https://developers.google.com/identity/protocols/oauth2/native-app#redirect-uri_loopback
    credentials = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        open_browser=True,
    )
    email = profile_getter(credentials)
    if not isinstance(email, str) or not email.strip():
        raise ValueError("Google did not return an account email address")

    token_json = credentials.to_json()
    config.set_many({
        "google.oauth_client_json": json.dumps(client_config),
        "google.oauth_token_json": token_json,
        "google.account_email": email.strip(),
        "google.account_credential_binding": hashlib.sha256(token_json.encode("utf-8")).hexdigest(),
    })
    return email


def _load_client_config(config, path: Path | None) -> dict:
    """Load saved or file-based Google desktop OAuth configuration and validate its basic shape.

    Reject unreadable, oversized file input, invalid JSON, or missing installed
    client identity with ValueError.
    """
    if path is None:
        raw = config.get("google.oauth_client_json")
        if not raw:
            raise ValueError("Provide the downloaded Google Desktop app client JSON file")
    else:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("Could not read the Google OAuth client JSON file") from exc
        if len(raw.encode("utf-8")) > MAX_CLIENT_CONFIG_BYTES:
            raise ValueError("Google OAuth client JSON file is too large")

    try:
        client_config = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Google OAuth client file is not valid JSON") from exc
    installed = client_config.get("installed") if isinstance(client_config, dict) else None
    if not isinstance(installed, dict) or not installed.get("client_id"):
        raise ValueError("Google OAuth client must use the Desktop app application type")
    return client_config


def _profile_email(credentials) -> str:
    """Fetch the authenticated Gmail profile and return its email address, or an empty fallback."""
    from googleapiclient.discovery import build

    profile = (
        build("gmail", "v1", credentials=credentials, cache_discovery=False)
        .users()
        .getProfile(userId="me")
        .execute()
    )
    return profile.get("emailAddress", "") if isinstance(profile, dict) else ""

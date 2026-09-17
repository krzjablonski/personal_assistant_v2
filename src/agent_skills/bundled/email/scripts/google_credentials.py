"""Load the connected user's Google credentials from the skill environment."""
from __future__ import annotations

import json
import os


GMAIL_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
)
CONNECT_MESSAGE = (
    "Google account is not connected. Run "
    "personal-assistant --connect-google CLIENT_JSON."
)


class GoogleCredentialsError(ValueError):
    pass


def load_google_credentials():
    """Load user OAuth credentials from the skill environment and require Gmail scopes.

    Raise GoogleCredentialsError for unavailable or invalid configuration.
    Token refresh and authenticated requests happen when the API client uses them.
    """
    raw = os.environ.get("GOOGLE_OAUTH_TOKEN_JSON")
    if not raw:
        raise GoogleCredentialsError(CONNECT_MESSAGE)
    try:
        info = json.loads(raw)
        from google.oauth2.credentials import Credentials

        credentials = Credentials.from_authorized_user_info(info)
    except (ImportError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GoogleCredentialsError(CONNECT_MESSAGE) from exc
    if not credentials.has_scopes(GMAIL_SCOPES):
        raise GoogleCredentialsError(CONNECT_MESSAGE)
    return credentials

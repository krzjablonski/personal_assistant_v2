from __future__ import annotations

import argparse
import ipaddress
import sys
from dataclasses import dataclass, replace
from getpass import getpass
from urllib.parse import urlparse



CLIENT_TYPES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "local": "Local",
}
DEFAULT_MODELS = {
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3-flash-preview",
    "openrouter": "openai/gpt-5.4",
    "local": "local-model",
}
DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
PROVIDER_CREDENTIALS = {
    "openai": ("llm.openai_api_key", "OPEN_AI_API_KEY"),
    "anthropic": ("llm.anthropic_api_key", "ANTHROPIC_API_KEY"),
    "gemini": ("llm.google_gemini_api_key", "GEMINI_API_KEY"),
    "openrouter": ("llm.openrouter_api_key", "OPENROUTER_API_KEY"),
}
SETTING_KEYS = {
    "provider": "cli.provider",
    "model": "cli.model",
    "max_iterations": "cli.max_iterations",

    "base_url": "cli.local_base_url",
    "context_window": "cli.local_context_window",
}
_LOCAL_NAME_SUFFIXES = (".localhost", ".local", ".lan", ".internal", ".home.arpa")


def _is_local_network_host(host: str | None) -> bool:
    """Report whether plain HTTP to this host stays on the machine or a private network.

    Accept loopback and private/link-local IP literals, ``localhost``, single-label
    names and common local-only suffixes; anything else must use HTTPS.
    """
    if not host:
        return False
    host = host.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(_LOCAL_NAME_SUFFIXES)
    return (address.is_loopback or address.is_private or address.is_link_local) and not address.is_unspecified


@dataclass(frozen=True)
class RuntimeSettings:
    provider: str = "openai"
    model: str = "gpt-5.4"
    max_iterations: int = 10

    base_url: str = DEFAULT_LOCAL_BASE_URL
    context_window: int | None = None

    def __post_init__(self) -> None:
        """Reject unsupported runtime choices and invalid model, budget, or local-endpoint settings."""
        if self.provider not in CLIENT_TYPES:
            raise ValueError(f"Unknown provider: {self.provider}")
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("Model must contain 1-200 characters")
        if not 1 <= self.max_iterations <= 100:
            raise ValueError("max_iterations must be between 1 and 100")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("Context window must be positive")
        if self.provider == "local":
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Local base URL must be an HTTP(S) URL")
            if parsed.scheme == "http" and not _is_local_network_host(parsed.hostname):
                raise ValueError(
                    "Local base URL must use https:// unless it targets localhost or a private network host"
                )


def _saved_int(
    config,
    key: str,
    default: int | None,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int | None:
    """Read an integer setting within the requested bounds, falling back on invalid or absent values."""
    try:
        value = int(config.get(key))
    except (TypeError, ValueError):
        return default
    if value < minimum or (maximum is not None and value > maximum):
        return default
    return value


def resolve_settings(args: argparse.Namespace, config) -> RuntimeSettings:
    """Combine CLI overrides, saved preferences, and defaults into validated runtime settings.

    Reuse a saved model only for its saved provider.
    """
    saved_provider = config.get("cli.provider")
    provider = args.client or (
        saved_provider if saved_provider in CLIENT_TYPES else "openai"
    )
    saved_model = config.get("cli.model")
    model = args.model or (
        saved_model
        if saved_provider == provider and saved_model and len(saved_model) <= 200
        else DEFAULT_MODELS[provider]
    )
    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = _saved_int(
            config, "cli.max_iterations", 10, maximum=100
        )
    base_url = (
        args.base_url
        or config.get("cli.local_base_url")
        or DEFAULT_LOCAL_BASE_URL
    )
    context_window = args.context_window
    if context_window is None and provider == "local":
        context_window = _saved_int(
            config, "cli.local_context_window", None
        )
    return RuntimeSettings(
        provider=provider,
        model=model,
        max_iterations=max_iterations,
        base_url=base_url,
        context_window=context_window,
    )


def ensure_provider_credentials(
    settings: RuntimeSettings,
    config,
    *,
    force: bool = False,
) -> None:
    """Ensure a hosted provider has a configured API key, prompting and storing one when needed.

    Skip local providers; forced setup prompts for replacement credentials.
    Raise ValueError when interactive setup is unavailable or the new key is empty.
    """
    if settings.provider == "local":
        return
    config_key, env_name = PROVIDER_CREDENTIALS[settings.provider]
    if not force and config.get(config_key):
        return
    if not sys.stdin.isatty():
        raise ValueError(
            f"Missing provider credentials. Set {env_name} or run "
            "personal-assistant interactively to configure it."
        )

    ensure_config_unlocked(config)
    if not force and config.get(config_key):
        return

    api_key = getpass(f"{CLIENT_TYPES[settings.provider]} API key: ").strip()
    if not api_key:
        raise ValueError("API key cannot be empty")
    config.set(config_key, api_key)


def ensure_config_unlocked(config) -> None:
    """Interactively unlock encrypted settings or initialize a new master password.

    Return immediately when unlocked; reject noninteractive setup or user cancellation.
    """
    if not config.is_locked():
        return
    if not sys.stdin.isatty():
        raise ValueError(
            "Encrypted settings are locked. Run personal-assistant interactively."
        )
    if config.has_master_password():
        while True:
            password = getpass("Master password (blank to cancel): ")
            if not password:
                raise ValueError("Credential setup cancelled")
            if config.unlock(password):
                return
            print("Incorrect master password.")

    while True:
        password = getpass("Create master password (minimum 8 characters): ")
        if not password:
            raise ValueError("Credential setup cancelled")
        if len(password) < 8:
            print("Master password must contain at least 8 characters.")
            continue
        if password != getpass("Confirm master password: "):
            print("Master passwords do not match.")
            continue
        config.set_master_password(password)
        return


def status_rows(
    settings: RuntimeSettings,
    config,
) -> tuple[tuple[str, str], ...]:
    """Describe active settings and credential presence for the CLI session summary.

    Credential labels reflect local configuration presence, not a live service check.
    """
    if settings.provider == "local":
        credential_status = "not required"
    else:
        credential_status = (
            "configured"
            if config.get(PROVIDER_CREDENTIALS[settings.provider][0])
            else "missing"
        )
    from personal_assistant.cli_web_settings import web_status
    return (
        ("Provider", f"{CLIENT_TYPES[settings.provider]} ({settings.model})"),
        ("Capabilities", "Console + browser + 6 trusted skills"),
        ("Max iterations", str(settings.max_iterations)),
        ("Credentials", credential_status),
        (
            "Google",
            "available on first use; connect with --connect-google",
        ),
    ) + web_status(config)


def select_option(
    title: str,
    options: tuple[tuple[str, str], ...],
    current: str,
) -> str:
    """Prompt for an option and return its value, keeping the current value on empty input.

    Repeat input that cannot be converted to a valid sequence index.
    """
    print(f"\n{title}")
    for index, (value, label) in enumerate(options, start=1):
        marker = " *" if value == current else ""
        print(f"  {index}. {label}{marker}")
    while True:
        choice = input("Select (Enter keeps current): ").strip()
        if not choice:
            return current
        try:
            return options[int(choice) - 1][0]
        except (ValueError, IndexError):
            print("Choose one of the listed numbers.")


def prompt_provider_settings(settings: RuntimeSettings) -> RuntimeSettings:
    """Prompt for provider, model, and relevant local-endpoint settings and return a revised settings object."""
    provider = select_option(
        "Provider",
        tuple((value, label) for value, label in CLIENT_TYPES.items()),
        settings.provider,
    )
    same_provider = provider == settings.provider
    default_model = settings.model if same_provider else DEFAULT_MODELS[provider]
    while True:
        model = input(f"Model [{default_model}]: ").strip() or default_model
        if 1 <= len(model) <= 200:
            break
        print("Model must contain 1-200 characters.")

    base_url = settings.base_url
    context_window = settings.context_window
    if provider == "local":
        base_url = (
            input(f"Base URL [{settings.base_url}]: ").strip()
            or settings.base_url
        )
        default_context = (
            str(settings.context_window)
            if same_provider and settings.context_window is not None
            else ""
        )
        while True:
            raw_context = input(
                f"Context window [{default_context or 'automatic'}]: "
            ).strip()
            if not raw_context:
                context_window = int(default_context) if default_context else None
                break
            try:
                context_window = int(raw_context)
                if context_window > 0:
                    break
            except ValueError:
                pass
            print("Context window must be a positive integer.")

    return replace(
        settings,
        provider=provider,
        model=model,
        base_url=base_url,
        context_window=context_window,
    )


def guided_setup(
    args: argparse.Namespace,
    settings: RuntimeSettings,
    config,
) -> RuntimeSettings:
    """Collect and save missing provider choices for an interactive first run.

    Preserve supplied settings unchanged in noninteractive sessions.
    """
    if not sys.stdin.isatty():
        return settings
    fields: set[str] = set()
    if args.client is None and not config.get("cli.provider"):
        settings = prompt_provider_settings(settings)
        fields.update({"provider", "model"})
        if settings.provider == "local":
            fields.update({"base_url", "context_window"})
    if fields:
        persist_settings(settings, config, fields)
    return settings


def persist_settings(
    settings: RuntimeSettings,
    config,
    fields: set[str],
) -> None:
    """Store selected runtime fields together using normalized string values."""
    values = {}
    for field in fields:
        value = getattr(settings, field)
        if isinstance(value, bool):
            stored = str(value).lower()
        elif value is None:
            stored = ""
        else:
            stored = str(value)
        values[SETTING_KEYS[field]] = stored
    config.set_many(values)


def client_config(settings: RuntimeSettings) -> dict:
    """Build model settings with a context override for every provider."""
    config = {"model": settings.model}
    if settings.context_window is not None:
        config["context_window"] = settings.context_window
    if settings.provider == "local":
        config.update(
            base_url=settings.base_url,
            context_window=settings.context_window,
        )
    return config

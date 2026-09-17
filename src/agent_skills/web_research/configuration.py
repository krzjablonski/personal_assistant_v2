"""Explicit capability selection and lazy, narrowly scoped credentials."""
from collections.abc import Callable, Mapping

PROVIDERS = {
    "tavily": ("web_search.tavily_api_key", "TAVILY_API_KEY"),
    "parallel": ("web.parallel_api_key", "PARALLEL_API_KEY"),
    "firecrawl": ("web.firecrawl_api_key", "FIRECRAWL_API_KEY"),
    "brave": ("web.brave_api_key", "BRAVE_SEARCH_API_KEY"),
}


def validate_provider(provider: str, capability: str) -> str:
    if capability not in {"search", "extract"}:
        raise ValueError("Unknown web capability")
    if provider not in PROVIDERS or (capability == "extract" and provider == "brave"):
        raise ValueError(f"Provider '{provider}' does not support {capability}. Configure web.{capability}_provider.")
    return provider


def resolve_environment(capability: str, get: Callable) -> dict[str, str]:
    provider = validate_provider(get(f"web.{capability}_provider") or "tavily", capability)
    key = get(PROVIDERS[provider][0])
    prefix = f"WEB_{capability.upper()}"
    return {f"{prefix}_PROVIDER": provider, **({f"{prefix}_API_KEY": key} if key else {})}


def from_environment(capability: str, env: Mapping[str, str]) -> tuple[str, str]:
    prefix = f"WEB_{capability.upper()}"
    provider = validate_provider(env.get(f"{prefix}_PROVIDER", "tavily"), capability)
    # A trusted capability snapshot must not fall back to another ambient key.
    key = env.get(f"{prefix}_API_KEY", "") if f"{prefix}_PROVIDER" in env else env.get(PROVIDERS[provider][1], "")
    if not key:
        raise ValueError(f"Missing {PROVIDERS[provider][1]}. Configure {PROVIDERS[provider][0]}.")
    return provider, key

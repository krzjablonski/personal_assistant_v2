from llm.i_llm_client import ILLMClient


def create_client(
    client_type: str, config: dict, *, config_source
) -> ILLMClient:
    """Create a supported provider client from model settings and the selected configuration source.

    Use provider credentials for hosted models and the supplied endpoint for
    Local. Raise ValueError for an unknown client type.
    """
    context_options = (
        {"context_window": config["context_window"]}
        if config.get("context_window") is not None else {}
    )
    if client_type == "OpenAI":
        from llm.openai_compatible_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            model=config["model"],
            api_key=config_source.get("llm.openai_api_key") or "",
            **context_options,
        )
    elif client_type == "Anthropic":
        from llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            model=config["model"],
            api_key=config_source.get("llm.anthropic_api_key") or "",
            **context_options,
        )
    elif client_type == "Google Gemini":
        from llm.gemini_client import GeminiClient

        return GeminiClient(
            model=config["model"],
            api_key=config_source.get("llm.google_gemini_api_key") or "",
            **context_options,
        )
    elif client_type == "OpenRouter":
        from llm.openai_compatible_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url="https://openrouter.ai/api/v1",
            model=config["model"],
            api_key=config_source.get("llm.openrouter_api_key") or "",
            **context_options,
        )
    elif client_type == "Local":
        from llm.openai_compatible_client import OpenAICompatibleClient

        return OpenAICompatibleClient(
            base_url=config["base_url"],
            model=config["model"],
            api_key="ollama",
            context_window=config.get("context_window"),
        )
    raise ValueError(f"Unknown LLM client type: {client_type}")

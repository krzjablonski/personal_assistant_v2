"""Provider construction is explicit and does not own global configuration."""
import unittest
from unittest.mock import patch

from llm.client_factory import create_client


class TestActiveClientFactory(unittest.TestCase):
    def test_configuration_is_required_even_for_local_provider(self):
        with self.assertRaises(TypeError):
            create_client("Local", {"model": "local-test", "base_url": "http://localhost/v1"})

    def test_injected_credentials_reach_all_provider_adapters(self):
        cases = (
            ("Anthropic", "llm.anthropic_api_key", "llm.anthropic_client.anthropic.AsyncAnthropic"),
            ("Google Gemini", "llm.google_gemini_api_key", "llm.gemini_client.genai.Client"),
            ("OpenAI", "llm.openai_api_key", "llm.openai_compatible_client.AsyncOpenAI"),
            ("OpenRouter", "llm.openrouter_api_key", "llm.openai_compatible_client.AsyncOpenAI"),
        )
        for provider, key, sdk in cases:
            with self.subTest(provider=provider), patch(sdk) as constructor:
                client = create_client(provider, {"model": "test"}, config_source={key: "injected-key"})
                constructor.assert_not_called()
                self.assertIs(client.client, constructor.return_value)
                self.assertEqual(constructor.call_args.kwargs["api_key"], "injected-key")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown LLM client type"):
            create_client("Typo", {"model": "test"}, config_source={})

    def test_empty_gemini_configuration_fails(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            create_client("Google Gemini", {"model": "test"}, config_source={})

    def test_compatible_provider_endpoints_and_keys(self):
        for provider, host, key in (("OpenAI", "api.openai.com", "llm.openai_api_key"),
                                    ("OpenRouter", "openrouter.ai/api", "llm.openrouter_api_key")):
            with self.subTest(provider=provider), patch("llm.openai_compatible_client.OpenAICompatibleClient") as ctor:
                create_client(provider, {"model": "test"}, config_source={key: "key"})
                ctor.assert_called_once_with(base_url="https://"+host+"/v1", model="test", api_key="key")

    def test_local_endpoint_context_and_model_are_explicit(self):
        with patch("llm.openai_compatible_client.OpenAICompatibleClient") as ctor:
            create_client("Local", {"base_url": "http://localhost:4321/v1", "model": "local-test", "context_window": 32768}, config_source={})
            ctor.assert_called_once_with(base_url="http://localhost:4321/v1", model="local-test", api_key="ollama", context_window=32768)

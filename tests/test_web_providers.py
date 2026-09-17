import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from pathlib import Path

import httpx

from agent_skills.web_research.providers import WebProvider, WebError
from agent_skills.web_research.content import document_content
from agent_skills.web_research.configuration import resolve_environment


class WebProviderTests(unittest.TestCase):
    def test_parallel_v1_request_contract_and_trimmed_query(self):
        p = self.provider("parallel", {"results": []})
        result = p.search(" " * 100000 + "q", limit=3)
        self.assertEqual(result["query"], "q")
        self.assertEqual(json.loads(self.requests[-1].content), {
            "objective": "q", "search_queries": ["q"], "mode": "fast",
            "advanced_settings": {"max_results": 3, "excerpt_settings": {"max_chars_per_result": 1500}}})
        p.extract(["https://example.com"])
        self.assertEqual(json.loads(self.requests[-1].content), {
            "urls": ["https://example.com"], "advanced_settings": {"full_content": True}})

    def provider(self, name, payload, status=200):
        self.requests = []
        def respond(request):
            self.requests.append(request)
            return httpx.Response(status, json=payload)
        self.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.client.close)
        return WebProvider(name, "secret-fixture", client=self.client)

    def test_each_search_maps_source_content_and_authentication(self):
        rows = {
            "tavily": {"results": [{"title": "T", "url": "https://example.com", "content": "Fact"}]},
            "parallel": {"results": [{"title": "T", "url": "https://example.com", "excerpts": ["Fact"]}]},
            "firecrawl": {"success": True, "data": {"web": [{"title": "T", "url": "https://example.com", "description": "Fact"}]}},
            "brave": {"web": {"results": [{"title": "T", "url": "https://example.com", "description": "Fact"}]}},
        }
        for name, payload in rows.items():
            with self.subTest(provider=name):
                result = self.provider(name, payload).search("question", limit=3)
                self.assertEqual(result["results"][0]["content"], "Fact")
                self.assertEqual(result["provider"], name)
                self.assertIsNone(result["results"][0]["published_at"])
                self.assertEqual(len(self.requests), 1)
                self.assertNotIn("secret-fixture", str(self.requests[0].url))

    def test_explicit_tavily_depth_is_not_silently_mapped(self):
        provider = self.provider("parallel", {})
        with self.assertRaisesRegex(WebError, "search-depth"):
            provider.search("q", depth="advanced")
        self.assertEqual(self.requests, [])

    def test_missing_or_unknown_provider_and_brave_extract_fail_before_io(self):
        with self.assertRaises(WebError):
            WebProvider("unknown", "k")
        with self.assertRaises(WebError):
            WebProvider("tavily", "")
        p = self.provider("brave", {})
        with self.assertRaisesRegex(WebError, "extract"):
            p.extract(["https://example.com"])
        self.assertEqual(self.requests, [])

    def test_rate_limit_is_bounded_and_does_not_disclose_response(self):
        p = self.provider("tavily", {"error": "secret-fixture"}, 429)
        with self.assertRaises(WebError) as caught:
            p.search("q")
        self.assertEqual(caught.exception.category, "rate_limit")
        self.assertNotIn("secret-fixture", str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    def test_extract_retains_partial_results_and_reports_missing_urls(self):
        p = self.provider("tavily", {"results": [{"url": "https://example.com/a", "raw_content": "Full"}], "failed_results": []})
        result = p.extract(["https://example.com/a", "https://example.com/b"])
        self.assertEqual(result["results"][0]["content"], "Full")
        self.assertEqual(result["failed"][0]["url"], "https://example.com/b")
        self.assertTrue(result["partial"])

    def test_malformed_search_is_not_empty_success(self):
        with self.assertRaises(WebError):
            self.provider("tavily", {"unexpected": []}).search("q")
        with self.assertRaises(WebError):
            self.provider('brave', {'unexpected': []}).search('q')
        with self.assertRaises(WebError):
            self.provider('parallel', {'results': [{'url': 'https://example.com', 'excerpts': None}]}).search('q')

    def test_parallel_extract_and_firecrawl_extract(self):
        payloads = {
            "parallel": {"results": [{"url": "https://example.com", "full_content": "Text"}], "errors": []},
            "firecrawl": {"success": True, "data": {"markdown": "Text", "metadata": {"sourceURL": "https://example.com"}}},
        }
        for name, payload in payloads.items():
            with self.subTest(provider=name):
                r = self.provider(name, payload).extract(["https://example.com"])
                self.assertEqual(r["results"][0]["content"], "Text")
                self.assertFalse(r["failed"])

    def test_untrusted_urls_and_oversized_batches_fail(self):
        p = self.provider("tavily", {})
        for urls in (["file:///etc/passwd"], ["http://127.0.0.1"], ["https://user:pass@example.com"], ["https://example.com"] * 6):
            with self.subTest(urls=urls), self.assertRaises(WebError):
                p.extract(urls)
        self.assertEqual(self.requests, [])


class ContentTests(unittest.TestCase):
    def test_all_failed_cli_retains_json_and_exits_nonzero(self):
        from agent_skills.web_research.cli import main
        output, error = io.StringIO(), io.StringIO()
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'fixture'}, clear=True), \
                patch('agent_skills.web_research.cli.WebProvider') as provider, redirect_stdout(output), redirect_stderr(error):
            provider.return_value.extract.return_value = {'results': [], 'failed': [{'url': 'https://example.com'}]}
            self.assertEqual(main('extract', ['--url', 'https://example.com']), 1)
        self.assertEqual(json.loads(output.getvalue())['results'], [])
        self.assertEqual(json.loads(error.getvalue())['category'], 'page_error')

    def test_storage_failure_keeps_preview_and_never_claims_full_copy(self):
        with patch('agent_skills.web_research.content.save_output', side_effect=OSError):
            doc = document_content('x' * 5000, Path('/tmp/unused'))
        self.assertFalse(doc['stored_content_complete'])
        self.assertNotIn('output_path', doc)
        self.assertIn('output_save_error', doc)

    def test_long_unicode_page_is_saved_before_stdout_capture(self):
        with tempfile.TemporaryDirectory() as d:
            text = "początek\n" + "środek\n" * 170000 + "koniec"
            doc = document_content(text, Path(d))
            self.assertLess(len(json.dumps(doc)), 15000)
            self.assertTrue(doc["truncated"])
            self.assertTrue(doc["stored_content_complete"])
            self.assertEqual(Path(doc["host_output_path"]).read_text(), text)
            self.assertTrue(doc["output_path"].startswith("/outputs/"))

    def test_storage_cap_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            doc = document_content("x" * 2100000, Path(d))
            self.assertFalse(doc["stored_content_complete"])


class ConfigurationTests(unittest.TestCase):
    def test_two_mixed_configurations_pass_only_the_selected_secret_to_runtime(self):
        from personal_assistant.services.agent_builder import build_skill_env_provider
        from types import SimpleNamespace
        for search, extract in [('parallel', 'tavily'), ('brave', 'firecrawl')]:
            values = {'web.search_provider': search, 'web.extract_provider': extract,
                      'web.parallel_api_key': 'p', 'web.brave_api_key': 'b',
                      'web.firecrawl_api_key': 'f', 'web_search.tavily_api_key': 't'}
            resolver = build_skill_env_provider(SimpleNamespace(get=values.get))
            self.assertEqual(set(resolver(['WEB_SEARCH_PROVIDER', 'WEB_SEARCH_API_KEY'])),
                             {'WEB_SEARCH_PROVIDER', 'WEB_SEARCH_API_KEY'})
            self.assertEqual(resolver(['WEB_EXTRACT_PROVIDER', 'WEB_EXTRACT_API_KEY'])['WEB_EXTRACT_PROVIDER'], extract)

    def test_only_selected_capability_secret_is_resolved(self):
        calls = []
        values = {"web.search_provider": "brave", "web.brave_api_key": "brave-key"}
        def get(key):
            calls.append(key)
            return values.get(key)
        env = resolve_environment("search", get)
        self.assertEqual(env, {"WEB_SEARCH_PROVIDER": "brave", "WEB_SEARCH_API_KEY": "brave-key"})
        self.assertEqual(calls, ["web.search_provider", "web.brave_api_key"])

    def test_legacy_tavily_key_remains_default(self):
        env = resolve_environment("extract", {"web_search.tavily_api_key": "old-key"}.get)
        self.assertEqual(env["WEB_EXTRACT_API_KEY"], "old-key")
        self.assertEqual(env["WEB_EXTRACT_PROVIDER"], "tavily")


class WebRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_providers_through_loaded_skill_and_command_facades(self):
        from agent_skills.catalog import SkillCatalog
        from agent_skills.runtime import SkillRuntime
        from agent_skills.script_executor import ScriptResult
        from agent_skills.web_research.cli import main
        from personal_assistant.services.agent_builder import build_skill_env_provider
        from types import SimpleNamespace
        requests = []
        def respond(request):
            requests.append(request)
            if request.url.host == 'api.parallel.ai':
                payload = {'results': [{'title': 'Source', 'url': 'https://example.com', 'excerpts': ['Fact']}]}
            elif request.url.host == 'api.search.brave.com':
                payload = {'web': {'results': [{'url': 'https://example.com', 'description': 'Fact'}]}}
            elif request.url.host == 'api.firecrawl.dev':
                payload = {'success': True, 'data': {'markdown': 'Document'}}
            else:
                payload = {'results': [{'url': 'https://example.com', 'raw_content': 'Document'}]}
            return httpx.Response(200, json=payload)
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, tempfile.TemporaryDirectory() as directory:
            async def boundary(*, script_path, args, env_provider, **kwargs):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.dict('os.environ', env_provider, clear=True), redirect_stdout(stdout), redirect_stderr(stderr), \
                        patch('agent_skills.web_research.cli.WebProvider', side_effect=lambda name, key: WebProvider(name, key, client=client)):
                    code = main('search' if script_path.endswith('search.py') else 'extract', args)
                return ScriptResult(code, stdout.getvalue(), stderr.getvalue())
            for search, extract in [('parallel', 'tavily'), ('brave', 'firecrawl')]:
                values = {'web.search_provider': search, 'web.extract_provider': extract,
                          'web.parallel_api_key': 'p', 'web.brave_api_key': 'b',
                          'web.firecrawl_api_key': 'f', 'web_search.tavily_api_key': 't'}
                runtime = SkillRuntime(SkillCatalog.discover(), output_directory=Path(directory),
                                       env_provider=build_skill_env_provider(SimpleNamespace(get=values.get)))
                runtime.load_skill_instructions('web-research')
                with patch('agent_skills.runtime.run_script', side_effect=boundary):
                    found = await runtime.run_command('web-research', ['scripts/search.py', '--query', 'question'])
                    read = await runtime.run_command('web-research', ['scripts/extract.py', '--url', 'https://example.com'])
                self.assertFalse(found.is_error, found.result)
                self.assertFalse(read.is_error, read.result)
                self.assertIn(f'"provider": "{search}"', found.result)
                self.assertIn(f'"provider": "{extract}"', read.result)
            self.assertEqual(len(requests), 4)


if __name__ == "__main__":
    unittest.main()

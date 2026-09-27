import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

from personal_assistant.services.browser_network import PublicProxy
from personal_assistant.services.browser_session import BrowserSession, BrowserSettings, BrowserError
from personal_assistant.services.browser_tools import BrowserTool


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_keeps_cleanup_serialized(self):
        manager = BrowserSession(BrowserSettings(), Path('/tmp/unused'))
        entered, release = asyncio.Event(), asyncio.Event()
        async def wait():
            entered.set()
            await release.wait()
        profile = Mock()
        manager._profile = profile
        manager._process = SimpleNamespace(pid=123456, returncode=None, terminate=Mock(), wait=wait)
        with patch('personal_assistant.services.browser_session.os.killpg') as kill:
            closing = asyncio.create_task(manager.aclose())
            await entered.wait()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            self.assertTrue(manager._lock.locked())
            self.assertFalse(closing.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            profile.cleanup.assert_called_once()
            kill.assert_called_once()
            self.assertIsNone(manager._profile)

    async def test_denial_dispatches_nothing_and_lost_response_is_uncertain(self):
        from tool_framework.approval import ToolApprovalStore
        from tool_framework.tool_executor import ToolExecutor
        from tool_framework.tool_collection import ToolCollection
        manager = BrowserSession(BrowserSettings(), Path('/tmp/unused'))
        manager.observation = {'identity': 'one', 'generation': 1, 'url': 'https://example.com',
                               'fingerprint': 'abc', 'elements': [{'ref': 'e1', 'label': 'Send'}]}
        manager.execute = AsyncMock(side_effect=BrowserError('Worker stopped'))
        tool = BrowserTool(manager)
        executor = ToolExecutor(ToolCollection([tool]), ToolApprovalStore(approval_handler=AsyncMock(return_value=False)))
        result = await executor.execute('browser', {'action': 'click', 'ref': 'e1'})
        self.assertTrue(result.metadata['not_executed'])
        manager.execute.assert_not_awaited()
        executor = ToolExecutor(ToolCollection([tool]), ToolApprovalStore(approval_handler=AsyncMock(return_value=True)))
        result = await executor.execute('browser', {'action': 'click', 'ref': 'e1'})
        self.assertFalse(result.metadata['not_executed'])
        self.assertTrue(result.metadata['uncertain_changes'])
        manager.execute.assert_awaited_once()

    async def test_agent_keeps_resource_between_turns_but_closes_on_clear(self):
        from agent.simple_agent.simple_agent import SimpleAgent
        from tests.agent_fixtures import ScriptedClient, response
        resource = SimpleNamespace(aclose=AsyncMock())
        agent = SimpleAgent(llm_client=ScriptedClient([response('One'), response('Two')]), session_resources=(resource,))
        await agent.run('one')
        await agent.run('two')
        resource.aclose.assert_not_awaited()
        await agent.clear_session()
        resource.aclose.assert_awaited_once()

    async def test_private_and_mixed_dns_are_rejected_before_connect(self):
        proxy = PublicProxy()
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '::ffff:127.0.0.1'):
            with self.subTest(address=address), self.assertRaises(ValueError):
                await proxy.resolve(address, 443)
        loop = asyncio.get_running_loop()
        with patch.object(loop, 'getaddrinfo', AsyncMock(return_value=[(2,1,6,'',('93.184.216.34',443)),(2,1,6,'',('127.0.0.1',443))])):
            with self.assertRaises(ValueError):
                await proxy.resolve('example.com', 443)

    async def test_ref_and_approval_are_bound_to_observed_session(self):
        with TemporaryDirectory() as d:
            manager = BrowserSession(BrowserSettings(), Path(d))
            tool = BrowserTool(manager)
            manager.observation = {'identity': 'one', 'generation': 1, 'url': 'https://example.com',
                                   'fingerprint': 'abc', 'elements': [{'ref': 'e1', 'label': 'Send'}]}
            prepared = tool.prepare_action({'action': 'click', 'ref': 'e1'})
            self.assertTrue(prepared.policy.requires_approval)
            self.assertFalse(prepared.policy.retry_safe)
            self.assertEqual(prepared.approval_arguments['target']['label'], 'Send')
            opened = tool.prepare_action({'action': 'open', 'url': 'https://example.com/next?q=1'})
            self.assertTrue(opened.policy.requires_approval)
            self.assertEqual(opened.approval_arguments['url'], 'https://example.com/next?q=1')
            self.assertFalse(tool.prepare_action({'action': 'extract'}).policy.requires_approval)
            fill = tool.prepare_action({'action': 'fill', 'ref': 'e1', 'text': 'review this value'})
            self.assertEqual(fill.approval_arguments['text'], 'review this value')
            manager.observation = dict(manager.observation, identity='two')
            with self.assertRaisesRegex(ValueError, 'changed'):
                await prepared.execute()

    async def test_schema_rejects_arbitrary_code_and_targetless_keypress(self):
        tool = BrowserTool(BrowserSession(BrowserSettings(), Path('/tmp/unused')))
        for args in ({'action': 'evaluate', 'code': 'x'}, {'action': 'press', 'key': 'Enter'},
                     {'action': 'scroll', 'pixels': 50000}, {'action': 'open', 'url': 'file:///etc/passwd'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tool.prepare_action(args)

    async def test_missing_dependency_is_lazy_and_close_is_idempotent(self):
        manager = BrowserSession(BrowserSettings(python_path='/does-not-exist'), Path('/tmp/unused'))
        await manager.aclose()
        with self.assertRaisesRegex(ValueError, 'browser_setup'):
            await manager.execute({'action': 'snapshot'})
        await manager.aclose()


if __name__ == '__main__':
    unittest.main()

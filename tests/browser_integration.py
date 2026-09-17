"""Opt-in real Chromium fixture test, without API keys or external form writes.

BROWSER_TEST_PYTHON=/path/to/browser-env/bin/python python -m unittest discover -s tests -p browser_integration.py
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import tempfile
import threading
import unittest

from personal_assistant.services.browser_session import BrowserSession, BrowserSettings


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', 'http://127.0.0.1:8080/blocked')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        if self.path == '/guards':
            self.wfile.write(b'''<html><title>Guards</title><body>
            <form onsubmit="event.preventDefault();document.querySelector('#count').textContent=Number(document.querySelector('#count').textContent)+1">
            <input aria-label="Form input" onfocus="queueMicrotask(()=>document.querySelector('#other').focus())">
            <button>Submit</button></form><input id="other" aria-label="Other">
            <span id="count">0</span>
            <button onclick="if(confirm('Proceed?'))document.querySelector('#count').textContent='BAD'">Confirm</button>
            <div id="shadow"></div><script>
            document.querySelector('#shadow').attachShadow({mode:'open'}).innerHTML='<button>Shadow target</button>';
            Object.defineProperty(document.body,'innerText',{get:()=> 'FORGED PAGE'});
            </script></body></html>''')
            return
        self.wfile.write(b'''<html><head><title>Browser fixture</title></head><body>
        <label>Name <input aria-label="Name" id="name"></label>
        <button onclick="document.querySelector('#result').innerText=document.querySelector('#name').value">Send</button>
        <p id="result">Ready</p><img src="http://127.0.0.1:8080/blocked">
        <button onclick="setTimeout(()=>document.querySelector('#result').innerText='Changed',500)">Change later</button>
        <div style="height:3000px">Scroll fixture</div></body></html>''')


@unittest.skipUnless(os.environ.get('BROWSER_TEST_PYTHON'), 'opt-in isolated Browser Use environment required')
class BrowserIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.temp = tempfile.TemporaryDirectory()
        origin = ('127.0.0.1', self.server.server_port)
        self.url = f'http://{origin[0]}:{origin[1]}'
        self.session = BrowserSession(BrowserSettings(python_path=os.environ['BROWSER_TEST_PYTHON'],
            executable_path=os.environ.get('BROWSER_TEST_EXECUTABLE', ''), headless=True), Path(self.temp.name), test_origin=origin)

    async def asyncTearDown(self):
        await self.session.aclose()
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        self.temp.cleanup()

    def ref(self, label):
        return next(el['ref'] for el in self.session.observation['elements'] if el['label'] == label and el['tag'] in ('input', 'button'))

    async def test_observe_fill_click_extract_and_cleanup(self):
        first = await self.session.execute({'action': 'open', 'url': self.url})
        self.assertIn('Ready', first['observation']['text'])
        self.assertGreater(first['network_blocks'], 0)
        field = self.ref('Name')
        await self.session.execute({'action': 'fill', 'ref': field, 'text': 'Hello fixture'}, self.session.binding(field))
        button = self.ref('Send')
        await self.session.execute({'action': 'click', 'ref': button}, self.session.binding(button))
        result = await self.session.execute({'action': 'extract'})
        self.assertIn('Hello fixture', result['content'])
        process = self.session._process
        profile = self.session._profile.name
        await self.session.aclose()
        self.assertIsNotNone(process.returncode)
        self.assertFalse(Path(profile).exists())
        await self.session.aclose()

    async def test_redirect_is_blocked_and_cancellation_closes_worker(self):
        await self.session.execute({'action': 'open', 'url': self.url})
        result = await self.session.execute({'action': 'open', 'url': self.url + '/redirect'})
        self.assertGreater(result['network_blocks'], 0)
        # Cancel an in-progress RPC, including a possible navigation side effect.
        process = self.session._process
        task = asyncio.create_task(self.session.execute({'action': 'snapshot'}))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(self.session._process)
        self.assertIsNotNone(process.returncode)

    async def test_live_dom_change_invalidates_approved_reference(self):
        await self.session.execute({'action': 'open', 'url': self.url})
        change = self.ref('Change later')
        await self.session.execute({'action': 'click', 'ref': change}, self.session.binding(change))
        send = self.ref('Send')
        binding = self.session.binding(send)
        await asyncio.sleep(0.7)
        with self.assertRaisesRegex(ValueError, 'changed'):
            await self.session.execute({'action': 'click', 'ref': send}, binding)

    async def test_isolated_observation_targeted_enter_and_dialog_dismissal(self):
        await self.session.execute({'action': 'open', 'url': self.url + '/guards'})
        self.assertNotIn('FORGED PAGE', self.session.observation['text'])
        self.assertNotIn('Shadow target', [el['label'] for el in self.session.observation['elements']])
        field = self.ref('Form input')
        await self.session.execute({'action': 'press', 'ref': field, 'key': 'Enter'}, self.session.binding(field))
        self.assertIn('1', self.session.observation['text'])
        button = self.ref('Confirm')
        result = await self.session.execute({'action': 'click', 'ref': button}, self.session.binding(button))
        self.assertNotIn('BAD', result['observation']['text'])
        self.assertGreater(result['dialogs_dismissed'], 0)

    async def test_outer_agent_owns_approval_accounting_and_clear(self):
        from agent.simple_agent.simple_agent import SimpleAgent
        from personal_assistant.services.browser_tools import BrowserTool
        from tests.agent_fixtures import ScriptedClient, response
        from llm.messages import ToolCall
        from tool_framework.tool_collection import ToolCollection
        from tool_framework.approval import ToolApprovalStore
        from unittest.mock import AsyncMock
        await self.session.execute({'action': 'open', 'url': self.url + '/guards'})
        fixture = self
        class FixtureClient(ScriptedClient):
            async def chat(self, **request):
                self.calls.append(request)
                index = len(self.calls)
                if index == 1:
                    return response(calls=[ToolCall('observe', 'browser', {'action': 'snapshot'})])
                if index == 2:
                    return response(calls=[ToolCall('submit', 'browser', {'action': 'press', 'ref': fixture.ref('Form input'), 'key': 'Enter'})])
                if index == 3:
                    return response(calls=[ToolCall('read', 'browser', {'action': 'extract'})])
                return response('Fixture submitted once')
        client = FixtureClient([])
        approve = AsyncMock(return_value=True)
        agent = SimpleAgent(llm_client=client, tool_collection=ToolCollection([BrowserTool(self.session)]),
                            approval_store=ToolApprovalStore(approval_handler=approve),
                            session_resources=(self.session,))
        self.assertEqual(await agent.run('Submit the fixture once'), 'Fixture submitted once')
        self.assertEqual(len(client.calls), 4)
        approve.assert_awaited_once()
        self.assertIn('1', self.session.observation['text'])
        self.assertTrue(agent.effects['confirmed_changes'])
        process = self.session._process
        await agent.clear_session()
        self.assertIsNotNone(process.returncode)

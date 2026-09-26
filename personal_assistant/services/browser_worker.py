"""Private JSON-lines worker for the pinned Browser Use environment.

Only this module imports Browser Use. Fixed implementation scripts/CDP calls are
never supplied by the model. No Agent, remote browser, or secondary LLM is used.
"""
import asyncio
from contextlib import redirect_stdout, suppress
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import uuid

spec = importlib.util.spec_from_file_location('browser_network', Path(__file__).with_name('browser_network.py'))
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)

PAGE_STATE = """() => ({url: location.href, title: document.title,
 text: (document.body?.innerText || '').slice(0,2000001),
 html: document.documentElement.outerHTML.slice(0,2000001),
 input_count: document.querySelectorAll('input,textarea,select').length,
 inputs: Array.from(document.querySelectorAll('input,textarea,select')).slice(0,500).map(e =>
 [e.type,e.value,e.checked,e.disabled]), scroll: [scrollX,scrollY]})"""

TARGET_STATE = """function() {
 const norm=(s)=>(s||'').replace(/\\s+/g,' ').trim();
 const tag=this.tagName.toLowerCase(), type=this.getAttribute('type')||'';
 const buttonInput=tag==='input' && ['button','submit','reset'].includes(type.toLowerCase());
 const visible=norm(buttonInput ? this.value : this.innerText).slice(0,160);
 const aria=norm(this.getAttribute('aria-label')).slice(0,160);
 const placeholder=norm(this.getAttribute('placeholder')).slice(0,160);
 const name=norm(this.getAttribute('name')||this.textContent).slice(0,160);
 const a=visible.toLowerCase(), b=aria.toLowerCase();
 const form=tag==='form' ? this : (this.form || null);
 // Prototype getter: named form controls (e.g. name="action") cannot shadow it.
 const formAction=form ? Object.getOwnPropertyDescriptor(HTMLFormElement.prototype,'action').get.call(form) : '';
 return {html:this.outerHTML, tag, root:this.getRootNode()===document, value:this.value, checked:this.checked,
 disabled:!!this.disabled, type, label:visible||aria||placeholder||name, text:visible, aria_label:aria,
 placeholder, label_mismatch:!!(visible && aria && !a.includes(b) && !b.includes(a)),
 href:(typeof this.href==='string' ? this.href : '').slice(0,2000),
 form_action:String(this.hasAttribute('formaction') && this.formAction ? this.formAction : formAction||'').slice(0,2000)}; }"""

# Snapshot element fields shown to the model and in approval prompts.
TARGET_DISPLAY_FIELDS = (('text', 160), ('aria_label', 160), ('placeholder', 160),
                         ('label_mismatch', None), ('href', 2000), ('form_action', 2000))

TARGET_ACTION = """function(expectedPage, expectedTarget, action, value) {
 const currentPage = (""" + PAGE_STATE + """)();
 const target = (""" + TARGET_STATE + """).call(this);
 if (!this.isConnected || !target.root || JSON.stringify(currentPage)!==JSON.stringify(expectedPage)
     || JSON.stringify(target)!==JSON.stringify(expectedTarget)) return {error:'Browser state changed; take a new snapshot'};
 const rect=this.getBoundingClientRect();
 const hit=document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2);
 if (!rect.width || !rect.height || !hit || !(hit===this || this.contains(hit)) || this.disabled)
     return {error:'Target is not visible and unobscured; scroll or take a fresh snapshot'};
 if (action==='click') this.click();
 else if (action==='fill') {
   if (!['input','textarea'].includes(target.tag) || ['password','file','hidden','checkbox','radio','submit','button'].includes(target.type))
     return {error:'Only ordinary text fields can be filled'};
   const prototype=target.tag==='textarea'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;
   Object.getOwnPropertyDescriptor(prototype,'value').set.call(this,value);
   this.dispatchEvent(new Event('input',{bubbles:true}));
   this.dispatchEvent(new Event('change',{bubbles:true}));
 } else if (action==='press') {
   // Targeted DOM key events cannot be redirected by focus handlers. Native
   // keyboard emulation is intentionally unavailable. Implement only explicit
   // Enter/Space activation; Escape is a targeted event for page handlers.
   const key=value==='Space'?' ':value;
   const proceed=this.dispatchEvent(new KeyboardEvent('keydown',{key,bubbles:true,cancelable:true}));
   this.dispatchEvent(new KeyboardEvent('keyup',{key,bubbles:true}));
   if (proceed && (value==='Enter' || value==='Space')) {
     const now=(""" + TARGET_STATE + """).call(this);
     if (JSON.stringify(now)!==JSON.stringify(expectedTarget))
       return {error:'Target changed during key handling; no default activation performed',uncertain:true};
     if (target.tag==='button' || (target.tag==='a' && value==='Enter')) this.click();
     else if (value==='Enter' && target.tag==='input' && this.form) this.form.requestSubmit();
   }
 }
 return {dispatched:true};
}"""


class Worker:
    def __init__(self):
        self.browser = None
        self.proxy = None
        self.page = None
        self.identity = uuid.uuid4().hex
        self.generation = 0
        self.targets = {}
        self.fingerprint = None
        self.started_action = False
        self.dialogs_dismissed = 0
        self.background = set()

    async def start(self, args):
        from importlib.metadata import version
        if version('browser-use') != '0.13.10':
            raise ValueError('Expected browser-use==0.13.10; run browser_setup --install')
        from browser_use import Browser
        # The pinned library auto-accepts confirm dialogs. Replace that watchdog
        # before startup, only inside this isolated process, with dismissal.
        from browser_use.browser.watchdogs import popups_watchdog
        owner = self
        class DismissDialogs(popups_watchdog.PopupsWatchdog):
            async def on_TabCreatedEvent(self, event):
                session = await self.browser_session.get_or_create_cdp_session(event.target_id, focus=False)
                await session.cdp_client.send.Page.enable(session_id=session.session_id)
                def dismiss(event_data, session_id=None):
                    async def run():
                        with suppress(Exception):
                            await session.cdp_client.send.Page.handleJavaScriptDialog(
                                params={'accept': False}, session_id=session_id or session.session_id)
                            owner.dialogs_dismissed += 1
                    task = asyncio.create_task(run())
                    owner.background.add(task)
                    task.add_done_callback(owner.background.discard)
                session.cdp_client.register.Page.javascriptDialogOpening(dismiss)
        popups_watchdog.PopupsWatchdog = DismissDialogs
        self.proxy = network.PublicProxy(test_origin=tuple(args['test_origin']) if args.get('test_origin') else None)
        proxy = await self.proxy.start()
        self.browser = Browser(
            executable_path=args.get('executable_path') or None,
            headless=args['headless'], user_data_dir=str(Path(args['profile']) / 'profile'),
            enable_default_extensions=False, permissions=[], accept_downloads=False,
            auto_download_pdfs=False, chromium_sandbox=True, proxy={'server': proxy},
            args=['--proxy-bypass-list=<-loopback>', '--disable-quic',
                  '--force-webrtc-ip-handling-policy=disable_non_proxied_udp', '--disable-extensions',
                  '--disable-background-networking', '--disable-sync', '--no-first-run',
                  '--use-mock-keychain', '--password-store=basic'],
        )
        await self.browser.start()
        self.page = await self.browser.get_current_page()
        session = await self.page.session_id
        await self.browser.cdp_client.send.Browser.setDownloadBehavior(params={'behavior': 'deny'})
        await self.browser.cdp_client.send.Network.setBlockedURLs(
            params={'urls': ['file://*', 'ftp://*', 'filesystem:*']}, session_id=session)
        # Deny permission prompts, including camera/microphone/geolocation.
        await self.browser.cdp_client.send.Browser.grantPermissions(params={'permissions': []})
        return {'ready': True, 'version': '0.13.10'}

    async def isolated_context(self):
        session = await self.page.session_id
        frames = await self.browser.cdp_client.send.Page.getFrameTree(session_id=session)
        frame = frames['frameTree']['frame']['id']
        context = await self.browser.cdp_client.send.Page.createIsolatedWorld(
            params={'frameId': frame, 'worldName': 'personal-assistant-observation'}, session_id=session)
        return session, frame, context['executionContextId']

    async def state(self):
        session, _, context = await self.isolated_context()
        result = await self.browser.cdp_client.send.Runtime.evaluate(
            params={'expression': '(' + PAGE_STATE + ')()', 'contextId': context, 'returnByValue': True}, session_id=session)
        value = result['result']['value']
        encoded = json.dumps(value, sort_keys=True).encode()
        return value, hashlib.sha256(encoded).hexdigest()

    async def target_call(self, backend_id, function, *args):
        session, _, context = await self.isolated_context()
        resolved = await self.browser.cdp_client.send.DOM.resolveNode(
            params={'backendNodeId': backend_id, 'executionContextId': context}, session_id=session)
        object_id = resolved['object']['objectId']
        try:
            result = await self.browser.cdp_client.send.Runtime.callFunctionOn(
                params={'objectId': object_id, 'functionDeclaration': function, 'returnByValue': True,
                        'arguments': [{'value': arg} for arg in args]}, session_id=session)
            if result.get('exceptionDetails'):
                raise ValueError('Target operation failed; take a fresh snapshot')
            return result['result'].get('value')
        finally:
            with suppress(Exception):
                await self.browser.cdp_client.send.Runtime.releaseObject(params={'objectId': object_id}, session_id=session)

    async def snapshot(self):
        # DOM retrieval is bounded by the enclosing action timeout.
        pages = await self.browser.get_pages()
        page_id = (await self.page.get_target_info())['targetId']
        popups = max(0, len(pages) - 1)
        for page in pages:
            if (await page.get_target_info())['targetId'] != page_id:
                await self.browser.close_page(page)
        await self.browser.get_or_create_cdp_session(page_id, focus=True)
        state = await self.browser.get_browser_state_summary(include_screenshot=False, cached=False)
        value, fingerprint = await self.state()
        self.generation += 1
        self.targets = {}
        elements = []
        for index, node in list(state.dom_state.selector_map.items())[:30]:
            ref = f'{self.identity[:8]}-{self.generation}-{index}'
            try:
                target = await self.target_call(node.backend_node_id, TARGET_STATE)
            except Exception:
                continue  # cross-target frames cannot resolve in the main isolated world
            if not target['root'] or len(target['html']) > 20000:
                continue
            attrs = node.attributes or {}
            # Password/file inputs are deliberately not action targets.
            if attrs.get('type', '').lower() in ('password', 'file'):
                continue
            self.targets[ref] = (node.backend_node_id, target)
            element = {'ref': ref, 'tag': target['tag'], 'label': target['label'][:160],
                       'type': target['type'][:40]}
            for field, limit in TARGET_DISPLAY_FIELDS:
                if target.get(field):
                    element[field] = target[field] if limit is None else target[field][:limit]
            elements.append(element)
        self.fingerprint = fingerprint
        return {'observation': {'identity': self.identity, 'generation': self.generation,
                                'url': value['url'][:4000], 'title': value['title'][:300],
                                'fingerprint': fingerprint, 'text': value['text'][:3500],
                                'text_truncated': len(value['text']) > 3500, 'elements': elements},
                'popups_closed': popups, 'dialogs_dismissed': self.dialogs_dismissed, 'network_blocks': self.proxy.blocked}

    async def action(self, args):
        self.started_action = False
        action = args['action']
        if action == 'start':
            return await self.start(args)
        if action == 'open':
            from urllib.parse import urlsplit
            parsed = urlsplit(args['url'])
            if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
                raise ValueError('Only public HTTP(S) URLs are supported')
            await self.proxy.resolve(parsed.hostname or '', parsed.port or (443 if parsed.scheme == 'https' else 80))
            await self.page.goto(args['url'])
        elif action in ('click', 'fill', 'press') or (action == 'extract' and args.get('ref')):
            binding = args.get('binding') or {}
            value, fingerprint = await self.state()
            if (binding.get('identity') != self.identity or binding.get('generation') != self.generation
                    or binding.get('fingerprint') != fingerprint or len(value['html']) > 2000000
                    or value['input_count'] > 500
                    or args['ref'] not in self.targets):
                raise ValueError('Browser state changed; take a new snapshot and obtain fresh approval')
            backend_id, target = self.targets[args['ref']]
            live_target = await self.target_call(backend_id, TARGET_STATE)
            if live_target != target:
                raise ValueError('Target is detached or unsupported; take a new snapshot')
            if action == 'extract':
                return {'url': value['url'], 'content': await self.target_call(backend_id,
                    'function() { return (this.innerText || this.textContent || "").slice(0,2000001); }')}
            self.started_action = True
            result = await self.target_call(backend_id, TARGET_ACTION, value, target, action,
                                            args.get('key') if action == 'press' else args.get('text'))
            if result.get('error'):
                self.started_action = result.get('uncertain', False)
                raise ValueError(result['error'])
        elif action == 'extract':
            value, _ = await self.state()
            return {'url': value['url'], 'content': value['text']}
        elif action == 'scroll':
            await self.page.evaluate('(pixels) => { window.scrollBy(0,pixels); }', args.get('pixels', 600))
        elif action != 'snapshot':
            raise ValueError('Unsupported browser action')
        return await self.snapshot()

    async def close(self):
        if self.browser:
            with suppress(Exception):
                async with asyncio.timeout(5):
                    await self.browser.kill()
        if self.proxy:
            await self.proxy.aclose()
        for task in self.background:
            task.cancel()
        await asyncio.gather(*self.background, return_exceptions=True)


REQUEST_LIMIT = 32768
OVERSIZED = object()


async def read_request(reader):
    """Return the next request line, b'' at EOF, or OVERSIZED after discarding a too-long line."""
    try:
        return await reader.readuntil(b'\n')
    except asyncio.IncompleteReadError as error:
        return error.partial
    except asyncio.LimitOverrunError as error:
        consumed = error.consumed
    # Drop the oversized line, including bytes that have not arrived yet, so
    # the next request starts on a line boundary.
    while True:
        try:
            await reader.readexactly(consumed)
            await reader.readuntil(b'\n')
            return OVERSIZED
        except asyncio.IncompleteReadError:
            return OVERSIZED
        except asyncio.LimitOverrunError as error:
            consumed = error.consumed


async def main():
    protocol = sys.stdout
    worker = Worker()
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    reader = asyncio.StreamReader(limit=REQUEST_LIMIT)
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    try:
        with redirect_stdout(sys.stderr):
            while line := await read_request(reader):
                try:
                    if line is OVERSIZED:
                        worker.started_action = False
                        raise ValueError('Browser request exceeded the size limit and was ignored')
                    response = await worker.action(json.loads(line))
                except ValueError as error:
                    response = {'error': str(error)[:300], 'not_executed': not worker.started_action}
                except Exception:
                    response = {'error': 'Browser operation failed. Inspect a fresh snapshot; run browser_setup --check if startup failed.',
                                'not_executed': not worker.started_action}
                protocol.write(json.dumps(response, ensure_ascii=False) + '\n')
                protocol.flush()
    finally:
        with redirect_stdout(sys.stderr):
            await worker.close()
        transport.close()


if __name__ == '__main__':
    with suppress(asyncio.CancelledError):
        asyncio.run(main())

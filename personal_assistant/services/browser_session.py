"""Session-owned, serialized Python worker. Browser Use never enters app imports."""
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import tempfile

from config_service.paths import default_data_dir
from agent_skills.web_research.content import document_content


class BrowserError(ValueError):
    def __init__(self, message, *, not_executed=False):
        super().__init__(message)
        self.not_executed = not_executed


@dataclass(frozen=True)
class BrowserSettings:
    python_path: str = ''
    executable_path: str = ''
    headless: bool = False
    action_timeout_seconds: int = 30

    @classmethod
    def from_config(cls, config):
        headless = config.get('browser.headless') or 'false'
        if headless not in ('true', 'false'):
            raise ValueError('browser.headless must be true or false')
        timeout = int(config.get('browser.action_timeout_seconds') or 30)
        if not 5 <= timeout <= 120:
            raise ValueError('browser.action_timeout_seconds must be between 5 and 120')
        return cls(config.get('browser.python_path') or '', config.get('browser.executable_path') or '',
                   headless == 'true', timeout)

    @property
    def python(self):
        return Path(self.python_path).expanduser() if self.python_path else default_data_dir() / 'browser-env' / 'bin' / 'python'


class BrowserSession:
    def __init__(self, settings: BrowserSettings, output_directory: Path, *, test_origin=None):
        self._settings, self.output_directory = settings, output_directory
        self._test_origin = test_origin  # only injected by fixture tests, never a tool/config option
        self._process = None
        self._stderr_task = None
        self._profile = None
        self._lock = asyncio.Lock()
        self.observation = None

    @property
    def settings(self):
        if callable(self._settings):
            self._settings = self._settings()
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    def binding(self, ref=None):
        if not self.observation:
            raise ValueError('Take a browser snapshot before using an element reference')
        state = self.observation
        target = next((el for el in state['elements'] if el['ref'] == ref), None) if ref else None
        if ref and target is None:
            raise ValueError('Stale or unknown browser reference; take a new snapshot')
        return {key: state[key] for key in ('identity', 'generation', 'url', 'fingerprint')} | {'target': target}

    async def _rpc(self, request):
        process = self._process
        process.stdin.write(json.dumps(request, ensure_ascii=False).encode() + b'\n')
        await process.stdin.drain()
        line = await process.stdout.readline()
        if not line:
            raise BrowserError('Browser worker stopped. Run python -m personal_assistant.browser_setup --check')
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError):
            raise BrowserError('Browser worker returned an invalid response') from None
        if response.get('error'):
            raise BrowserError(response['error'], not_executed=response.get('not_executed', False))
        return response

    async def _start(self):
        if self._process:
            return
        if not self.settings.python.is_file():
            raise ValueError('Browser dependency is missing. Run python -m personal_assistant.browser_setup --install')
        self._profile = tempfile.TemporaryDirectory(prefix='personal-assistant-browser-')
        # Do not inherit application secrets, proxy settings or Python import hooks.
        env = {key: os.environ[key] for key in ('PATH', 'LANG', 'DISPLAY', 'XAUTHORITY', 'SYSTEMROOT', 'TMPDIR') if key in os.environ}
        env.update(HOME=self._profile.name, ANONYMIZED_TELEMETRY='false', BROWSER_USE_LOGGING_LEVEL='error',
                   BROWSER_USE_CLOUD_SYNC='false', PYTHONUNBUFFERED='1')
        self._process = await asyncio.create_subprocess_exec(
            str(self.settings.python), '-I', str(Path(__file__).with_name('browser_worker.py')),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=self._profile.name, env=env, start_new_session=True, limit=10_000_000)
        self._stderr_task = asyncio.create_task(self._drain_diagnostics(self._process.stderr))
        async with asyncio.timeout(60):
            await self._rpc({'action': 'start', 'profile': self._profile.name,
                             'headless': self.settings.headless, 'executable_path': self.settings.executable_path,
                             'test_origin': self._test_origin})

    async def _drain_diagnostics(self, reader):
        while await reader.read(8192):
            pass  # library logs may contain URLs or page contents; never retain by default

    async def execute(self, args, binding=None):
        async with self._lock:
            if args['action'] == 'close':
                await self._finish_cleanup()
                return {'closed': True}
            try:
                await self._start()
                if binding is not None and self.binding(args.get('ref')) != binding:
                    raise ValueError('Browser state changed; take a new snapshot and obtain fresh approval')
                async with asyncio.timeout(self.settings.action_timeout_seconds):
                    response = await self._rpc(dict(args, binding=binding))
                if 'observation' in response:
                    self.observation = response['observation']
                if 'content' in response:
                    response.update(document_content(response.pop('content'), self.output_directory))
                return response
            except BaseException:
                # Timeout/cancellation/unknown result cannot safely preserve stale references.
                await self._finish_cleanup()
                raise

    async def _finish_cleanup(self):
        task = asyncio.create_task(self._close())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        self.observation = None
        process, self._process = self._process, None
        stderr_task, self._stderr_task = self._stderr_task, None
        profile, self._profile = self._profile, None
        if process is not None:
            try:
                if process.returncode is None:
                    process.terminate()  # worker handles SIGTERM and closes Chromium
                await asyncio.wait_for(process.wait(), 8)
            except (ProcessLookupError, TimeoutError):
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            finally:
                # Worker exit does not prove its CDP cleanup killed Chromium.
                # Chrome inherits this dedicated process group; reap it even if
                # the worker suppressed a failure while closing the browser.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                if stderr_task:
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)
        if profile:
            profile.cleanup()

    async def aclose(self):
        async with self._lock:
            await self._finish_cleanup()

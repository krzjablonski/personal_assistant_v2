"""User-owned provider selection and browser preferences; no provider probing."""
from getpass import getpass

from agent_skills.web_research.configuration import PROVIDERS, validate_provider
from personal_assistant.cli_settings import ensure_config_unlocked, select_option


def web_status(config):
    rows = []
    for capability in ('search', 'extract'):
        selected = config.get(f'web.{capability}_provider') or 'tavily'
        try:
            validate_provider(selected, capability)
            present = config.contains(PROVIDERS[selected][0])
            status = 'key configured' if present else 'key missing'
        except ValueError:
            status = 'invalid selection'
        rows.append((f'Web {capability}', f'{selected} ({status}; no automatic fallback)'))
    rows.append(('Browser', 'headless' if config.get('browser.headless') == 'true' else 'visible; starts on first use'))
    return tuple(rows)


def configure_web(config):
    values = {}
    for capability in ('search', 'extract'):
        options = tuple((name, name.title()) for name in PROVIDERS if capability == 'search' or name != 'brave')
        current = config.get(f'web.{capability}_provider') or 'tavily'
        chosen = select_option(f'Web {capability} provider', options, current)
        values[f'web.{capability}_provider'] = validate_provider(chosen, capability)
    for selected in dict.fromkeys(values.values()):
        key, env = PROVIDERS[selected]
        value = getpass(f'{selected.title()} API key (Enter keeps current / {env}): ').strip()
        if value:
            ensure_config_unlocked(config)
            values[key] = value
    config.set_many(values)
    print('Web providers saved. New searches/extractions use these settings; no conversation reset needed.')


async def configure_browser(runtime):
    config = runtime.config
    headless = select_option('Browser mode', (('false', 'Visible (manual login available)'), ('true', 'Headless')),
                             config.get('browser.headless') or 'false')
    values = {'browser.headless': headless}
    for key, label in (('browser.python_path', 'Isolated Python path'), ('browser.executable_path', 'Chrome/Chromium path')):
        current = config.get(key) or ''
        values[key] = input(f'{label} [{current or "automatic"}]: ').strip() or current
    from personal_assistant.services.browser_session import BrowserSettings
    class Candidate:
        def get(self, key):
            return values.get(key, config.get(key))
    settings = BrowserSettings.from_config(Candidate())
    await runtime.agent.close_resources()
    config.set_many(values)
    # Browser manager is created once per agent and has not started again yet.
    from personal_assistant.services.browser_tools import BrowserTool
    for tool in runtime.agent.tool_collection.get_tools():
        if isinstance(tool, BrowserTool):
            tool.session.settings = settings
    print('Browser settings saved; the previous owned browser was closed.')

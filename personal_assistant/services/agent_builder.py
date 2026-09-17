from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
import os
from pathlib import Path
import uuid

from agent.prompts.react_prompts import REACT_SYSTEM_PROMPT
from agent.simple_agent.simple_agent import AgentConfig, SimpleAgent
from agent_skills.runtime import SkillRuntime
from agent_skills.tools import LoadSkillInstructionsTool, ReadSkillResourceTool, RunSkillCommandTool
from llm.client_factory import create_client
from memory.long_term_memory import LongTermMemory
from memory.tools import RecallMemoryTool, SaveMemoryTool
from personal_assistant.services.skills_registry import SKILL_CATALOG
from personal_assistant.services.console_tools import RunCommandTool
from personal_assistant.services.docker_console import DockerConsole
from personal_assistant.services.browser_session import BrowserSession, BrowserSettings
from personal_assistant.services.browser_tools import BrowserTool
from tool_framework.output_files import session_output_directory
from tool_framework.approval import ToolApprovalStore
from tool_framework.i_tool import ITool
from tool_framework.tool_collection import ToolCollection


@dataclass(frozen=True, kw_only=True)
class AgentBuildRequest:
    client_type: str
    client_config: dict
    max_iterations: int

    def __post_init__(self) -> None:
        """Keep provider constructors from mutating the caller's settings."""
        object.__setattr__(self, "client_config", dict(self.client_config))


def build_skill_env_provider(config) -> Callable[[Iterable[str]], dict[str, str]]:
    """Resolve only a trusted script's declared integration values at preparation."""
    settings = {
        "TAVILY_API_KEY": "web_search.tavily_api_key",
        "GOOGLE_OAUTH_TOKEN_JSON": "google.oauth_token_json",
        "GOOGLE_CALENDAR_ID": "calendar.calendar_id",
        "EMAIL_TO": "email.to",
        "ATTACHMENTS_DIR": "email.attachments_dir",
    }

    def resolve(names: Iterable[str]) -> dict[str, str]:
        from agent_skills.web_research.configuration import resolve_environment
        names = tuple(names)
        provider = {}
        def get_value(key):
            value = config.get(key)
            if (value is None and getattr(config, "is_locked", lambda: False)() and config.contains(key)):
                from personal_assistant.cli_settings import ensure_config_unlocked
                ensure_config_unlocked(config)
                value = config.get(key)
            return value
        for capability in ("search", "extract"):
            if f"WEB_{capability.upper()}_PROVIDER" in names:
                provider.update(resolve_environment(capability, get_value))
        for name in names:
            if name == "ALLOWED_EMAIL_RECIPIENTS":
                value = os.getenv(name)
            elif name in settings:
                value = get_value(settings[name])
            else:
                continue
            if value is not None:
                provider[name] = value
        token = provider.get("GOOGLE_OAUTH_TOKEN_JSON")
        if token:
            from hashlib import sha256
            account = config.get("google.account_email")
            if account and config.get("google.account_credential_binding") == sha256(token.encode("utf-8")).hexdigest():
                provider["GOOGLE_ACCOUNT_EMAIL"] = account
        return provider

    return resolve


def build_agent(
    request: AgentBuildRequest,
    *,
    config,
    memory_provider: Callable[[], LongTermMemory],
    approval_store: ToolApprovalStore | None = None,
    workspace_root: str | Path | None = None,
    output_root: Path | None = None,
    env_file: Path | None = None,
) -> SimpleAgent:
    """Compose the core console and complete trusted catalog without opening integrations."""
    approval_store = approval_store or ToolApprovalStore()
    session_id = str(uuid.uuid4())
    output_directory = output_root / session_id if output_root is not None else session_output_directory(session_id)
    workspace = Path(workspace_root).expanduser().absolute() if workspace_root is not None else None
    data_path = getattr(config, "DB_PATH", None)
    protected = (workspace / ".env",) if workspace is not None and (workspace / ".env").exists() else ()
    if env_file is not None:
        protected += (env_file,)
    if data_path is not None:
        protected += (Path(data_path), Path(data_path).with_name("agent_memory.db"))
    console = DockerConsole(workspace, output_directory, protected_paths=protected,
                            data_directory=Path(data_path).parent if data_path is not None else None)
    console.validate_workspace(require_exists=workspace is not None)
    skill_runtime = SkillRuntime(SKILL_CATALOG, approval_store=approval_store,
                                 output_directory=output_directory, env_provider=build_skill_env_provider(config))
    browser = BrowserSession(lambda: BrowserSettings.from_config(config), output_directory)
    tools: list[ITool] = [RunCommandTool(console), LoadSkillInstructionsTool(skill_runtime),
                         ReadSkillResourceTool(skill_runtime), RunSkillCommandTool(skill_runtime),
                         SaveMemoryTool(memory_provider), RecallMemoryTool(memory_provider), BrowserTool(browser)]
    system_prompt = f"{REACT_SYSTEM_PROMPT}\n\n{SKILL_CATALOG.build_catalog_prompt()}"
    agent_config = AgentConfig(
        max_iterations=request.max_iterations,
        agent_name="Personal Assistant",
        session_id=session_id,
    )
    return SimpleAgent(
        system_prompt=system_prompt,
        tool_collection=ToolCollection(tools),
        llm_client=create_client(request.client_type, request.client_config, config_source=config),
        config=agent_config,
        skill_runtime=skill_runtime,
        approval_store=approval_store,
        output_directory=output_directory,
        environment_context=console.context,
        session_resources=(browser,),
    )

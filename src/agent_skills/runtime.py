from __future__ import annotations

from pathlib import Path, PurePosixPath
import hashlib
import sys
from typing import Any, Mapping, Callable, Iterable

from agent_skills.catalog import ScriptSpec, SkillCatalog, SkillDefinition
from agent_skills.script_executor import (
    DEFAULT_TIMEOUT_SECONDS,
    ScriptResult,
    build_subprocess_env,
    resolve_script_path,
    run_script,
)
from tool_framework.approval import ToolApprovalStore, private_scope_binding, sanitized_approval_arguments
from tool_framework.output_files import session_output_directory
from tool_framework.i_tool import PreparedAction, ToolOutcome, ToolPolicy, ToolResult
from tool_framework.tool_executor import execute_prepared_action, preparation_failure


MAX_RESOURCE_CHARS = 20_000
MAX_COMMAND_OUTPUT_CHARS = 20_000
COMMAND_STDERR_TAIL_CHARS = 4_000


class SkillRuntime:
    """Session-scoped runtime for Agent Skill instruction loading and execution."""

    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        approval_store: ToolApprovalStore | None = None,
        output_directory: Path | None = None,
        env_provider: Mapping[str, str] | Callable[[Iterable[str]], Mapping[str, str]] | None = None,
    ):
        """Create session skill state with approvals, output files and tool environment values."""
        self.catalog = catalog
        self.approval_store = approval_store or ToolApprovalStore()
        self.output_directory = Path(output_directory) if output_directory is not None else session_output_directory()
        self.env_provider = env_provider if callable(env_provider) else dict(env_provider or {})
        self._loaded_skill_blocks: dict[str, str] = {}

    @property
    def loaded_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded_skill_blocks))

    def clear(self) -> None:
        """Forget instructions when the owning conversation is cleared."""
        self._loaded_skill_blocks.clear()

    def command_policy(self, skill_name: str, argv: object) -> ToolPolicy:
        """Classify a command's declared side effects for runtime retry and batching decisions.

        All commands are nonparallel. The prepared-action gate enforces approval.
        """
        if not isinstance(argv, list) or not argv:
            return ToolPolicy(can_parallel=False)
        try:
            spec = self.catalog.get(skill_name).scripts.get(str(argv[0]))
        except ValueError:
            spec = None
        safety = spec.safety if spec else "undeclared"
        return ToolPolicy(
            read_only=safety == "read-only",
            mutates_local=safety == "local-mutation",
            mutates_external=safety not in {"read-only", "local-mutation"},
            requires_approval=_requires_approval(spec),
            can_parallel=False,
            max_output_chars=MAX_COMMAND_OUTPUT_CHARS,
            approval_reason=f"Running '{safety}' script '{skill_name}/{argv[0]}' requires explicit human approval.",
        )

    def build_loaded_skill_prompt(self) -> str:
        """Render cached skill instructions so they remain available on later model requests.

        Return an empty string when no skill has been loaded.
        """
        if not self._loaded_skill_blocks:
            return ""
        lines = [
            "## Loaded Agent Skills",
            "The following skill instructions were loaded earlier in this session and remain in force. "
            "Keep following them for any further use of these skills — read their referenced files with "
            "`read_skill_resource` and, where a skill bundles scripts, run them with `run_skill_command`, "
            "using the exact paths and arguments the instructions specify. Do not reload a skill that "
            "already appears here.",
        ]
        for name in sorted(self._loaded_skill_blocks):
            lines.append(self._loaded_skill_blocks[name])
        return "\n\n".join(lines)

    def load_skill_instructions(self, name: str) -> ToolResult:
        """Activate a catalog skill for this session and return its instructions and resources.

        Repeated loads return a confirmation without duplicating the cached block;
        unknown skills return an actionable error result.
        """
        try:
            skill = self.catalog.get(name)
        except ValueError as exc:
            return ToolResult(
                tool_name="load_skill_instructions",
                parameters={"name": name},
                result=f"Error: {exc}",
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
            )

        if name in self._loaded_skill_blocks:
            return ToolResult(
                tool_name="load_skill_instructions",
                parameters={"name": name},
                result=f"Skill '{name}' instructions are already loaded.",
                outcome=ToolOutcome.USABLE,
            )

        block = self._format_skill_content(skill)
        self._loaded_skill_blocks[name] = block
        return ToolResult(
            tool_name="load_skill_instructions",
            parameters={"name": name},
            result=block,
            outcome=ToolOutcome.USABLE,
        )

    def read_resource(self, skill_name: str, resource_path: str) -> ToolResult:
        """Read a bounded text preview of a loaded skill's reference or asset file.

        Return actionable errors for unloaded skills, invalid paths, or read failures.
        """
        if skill_name not in self._loaded_skill_blocks:
            return ToolResult(
                tool_name="read_skill_resource",
                parameters={"skill": skill_name, "path": resource_path},
                result=(
                    f"Error: Skill '{skill_name}' instructions must be loaded before "
                    "reading its resources. Next step: call `load_skill_instructions` "
                    f'with name "{skill_name}", then retry this exact read_skill_resource '
                    "call."
                ),
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
            )
        try:
            skill = self.catalog.get(skill_name)
            path = self._resolve_resource(skill, resource_path)
            text = path.read_text(encoding="utf-8", errors="replace")
        except ValueError as exc:
            return ToolResult(
                tool_name="read_skill_resource",
                parameters={"skill": skill_name, "path": resource_path},
                result=f"Error: {exc}",
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
            )
        except OSError as exc:
            return ToolResult(
                tool_name="read_skill_resource",
                parameters={"skill": skill_name, "path": resource_path},
                result=f"Error: Failed to read resource - {exc}",
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
            )

        truncated = ""
        if len(text) > MAX_RESOURCE_CHARS:
            text = text[:MAX_RESOURCE_CHARS]
            truncated = f"\n\n[truncated to {MAX_RESOURCE_CHARS} characters]"

        return ToolResult(
            tool_name="read_skill_resource",
            parameters={"skill": skill_name, "path": resource_path},
            result=f"[Skill resource: {skill_name}/{resource_path}]\n{text}{truncated}",
            outcome=ToolOutcome.USABLE,
        )

    async def run_command(
        self, skill_name: str, argv: list[str], stdin: str | None = None
    ) -> ToolResult:
        """Prepare direct skill calls and use the same gate as the tool facade."""
        params = self._command_params(skill_name, argv, stdin)
        try:
            action = self.prepare_command(skill_name, argv, stdin)
        except Exception as error:
            return preparation_failure("run_skill_command", params, error)
        return await execute_prepared_action(action, self.approval_store, self.output_directory)

    def prepare_command(
        self, skill_name: str, argv: list[str], stdin: str | None = None
    ) -> PreparedAction:
        """Validate trusted script scope and capture configuration before approval.

        The executable closure retains only the declared environment and a stable
        baseline. No credential value enters the displayed or serialized scope.
        """
        if skill_name not in self._loaded_skill_blocks:
            raise ValueError(
                f"Skill '{skill_name}' instructions must be loaded before running its commands. "
                f'Next step: call `load_skill_instructions` with name "{skill_name}", '
                "then retry this exact run_skill_command call."
            )
        skill = self.catalog.get(skill_name)
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("command must include the script path as its first element and contain only strings.")
        if stdin is not None and not isinstance(stdin, str):
            raise ValueError("stdin must be a string when supplied.")
        argv = list(argv)
        script_path = argv[0]
        resolved = resolve_script_path(skill.root, script_path)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        spec = skill.scripts.get(script_path)
        environment = spec.environment if spec else ()
        provided = self.env_provider(environment) if callable(self.env_provider) else self.env_provider
        env = build_subprocess_env(environment, provided)
        if "SESSION_OUTPUT_DIR" in environment:
            env["SESSION_OUTPUT_DIR"] = str(self.output_directory.resolve())
        params = self._command_params(skill_name, argv, stdin)
        public_env_names = {"EMAIL_TO", "ALLOWED_EMAIL_RECIPIENTS", "GOOGLE_CALENDAR_ID", "ATTACHMENTS_DIR", "SESSION_OUTPUT_DIR"}
        scope = {
            **params,
            "working_directory": str(skill.root.resolve()),
            "executable": sys.executable,
            "script_path": str(resolved),
            "script_sha256": digest,
            "environment": {
                name: env.get(name, "unavailable")
                for name in environment if name in public_env_names
                and (name != "ALLOWED_EMAIL_RECIPIENTS" or name in env)
            },
            "private_environment_names": [name for name in environment if name not in public_env_names],
            "environment_binding": private_scope_binding(env),
        }
        if "GOOGLE_OAUTH_TOKEN_JSON" in environment:
            scope["google_account"] = provided.get("GOOGLE_ACCOUNT_EMAIL") or "unavailable (reconnect to identify account)"
        if "GOOGLE_CALENDAR_ID" in environment:
            scope["environment"]["GOOGLE_CALENDAR_ID"] = env.get("GOOGLE_CALENDAR_ID") or "primary"
        if "EMAIL_TO" in environment:
            # Show the effective recipient even when the script falls back to EMAIL_TO.
            explicit = _last_option(argv[1:], "--to")
            scope["recipients"] = (explicit if explicit is not None
                                   else env.get("EMAIL_TO") or "none (no --to and EMAIL_TO unset)")

        async def execute() -> ToolResult:
            # Recheck mutable filesystem state at the execution boundary.
            try:
                if skill_name not in self._loaded_skill_blocks:
                    raise ValueError("Skill instructions are no longer loaded; prepare a new action.")
                current = resolve_script_path(skill.root, script_path)
                if current != resolved or hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                    raise ValueError("The approved script changed; prepare a new action and obtain approval.")
            except (ValueError, OSError) as error:
                return ToolResult("run_skill_command", params, f"Command not executed: {error}", True,
                                  {"not_executed": True}, ToolOutcome.ACTIONABLE_FAILURE)
            result = await run_script(
                skill_root=skill.root, script_path=script_path, args=argv[1:], stdin=stdin,
                environment=environment, env_provider=env, base_environment=env,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            return self._shape_command_result(skill_name, argv, stdin, script_path, spec, result)

        return PreparedAction(
            "run_skill_command", params, self.command_policy(skill_name, argv),
            sanitized_approval_arguments(scope), execute,
            metadata={"skill": skill_name, "script": script_path},
        )

    def _format_skill_content(self, skill: SkillDefinition) -> str:
        """Render skill instructions with resource paths and declared command restrictions for agent context."""
        lines = [
            f'<skill_content name="{skill.name}">',
            skill.body,
            "",
            f"Skill directory: {skill.root}",
            "These instructions are now active. Follow them exactly, using the paths and "
            "commands they specify — do not guess. Every path below is relative to the "
            "skill directory.",
        ]

        resources = skill.list_resources()
        if resources:
            lines.append("<skill_resources>")
            lines.append(
                "Read a referenced file with `read_skill_resource`, passing its path "
                "exactly as listed here (only references/ and assets/ paths are readable):"
            )
            for resource in resources:
                lines.append(f"  <file>{resource}</file>")
            lines.append("</skill_resources>")

        if skill.scripts:
            lines.append("<skill_scripts>")
            lines.append(
                "Run these bundled scripts with `run_skill_command` (command = the "
                "script path exactly as listed below, relative to the skill directory, "
                "followed by its arguments):"
            )
            for path in sorted(skill.scripts):
                spec = skill.scripts[path]
                approval_note = " (requires human approval)" if _requires_approval(spec) else ""
                description = spec.description or f"{spec.safety} script."
                lines.append(f"  <script path=\"{path}\"{approval_note}>{description}</script>")
            lines.append("</skill_scripts>")

        lines.append("</skill_content>")
        return "\n".join(lines)

    def _resolve_resource(self, skill: SkillDefinition, resource_path: str) -> Path:
        """Resolve an existing reference or asset file confined to its skill directory.

        Raise ValueError for absolute paths, traversal, escaped targets, missing
        files, or unsupported resource directories.
        """
        posix = PurePosixPath(resource_path)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError("Resource path must be relative and must not contain '..'.")
        if not posix.parts or posix.parts[0] not in {"references", "assets"}:
            raise ValueError("Only references/ and assets/ resources may be read.")

        resolved = (skill.root / Path(*posix.parts)).resolve()
        root = skill.root.resolve()
        if root not in resolved.parents:
            raise ValueError("Resource path escapes the skill directory.")
        if not resolved.exists():
            raise ValueError(f"Resource not found: {resource_path}")
        if not resolved.is_file():
            raise ValueError(f"Resource is not a file: {resource_path}")
        return resolved

    @staticmethod
    def _command_params(
        skill_name: str, argv: list[str], stdin: str | None
    ) -> dict[str, Any]:
        """Build the exact skill-command payload used in results and approval requests."""
        return {"skill": skill_name, "command": list(argv), **({"stdin": stdin} if stdin is not None else {})}

    def _shape_command_result(
        self,
        skill_name: str,
        argv: list[str],
        stdin: str | None,
        script_path: str,
        spec: ScriptSpec | None,
        result: ScriptResult,
    ) -> ToolResult:
        """Translate script output and exit status into an agent tool result with actionable metadata.

        Distinguish read-only timeouts, possible mutation, and recognized authentication
        errors; apply output limits to completed commands.
        """
        params = self._command_params(skill_name, argv, stdin)
        base_metadata = {
            "skill": skill_name,
            "script": script_path,
            "exit_code": result.exit_code,
            "stdout": result.stdout[:COMMAND_STDERR_TAIL_CHARS],
            "stderr": result.stderr[-COMMAND_STDERR_TAIL_CHARS:],
            **result.capture_metadata(),
        }
        mutating = spec is None or spec.safety != "read-only"
        if mutating and not result.timed_out:
            if result.exit_code == 0:
                # A fixed summary: script output is untrusted and stays in the tool result.
                base_metadata["confirmed_changes"] = [
                    f"Skill command '{skill_name}/{script_path}' completed with exit 0 (output in tool result)."
                ]
            else:
                # A normal failure exit can follow a submitted external request.
                # Script stderr, including auth markers, cannot prove no effect.
                base_metadata["uncertain_changes"] = [
                    f"Skill command '{skill_name}/{script_path}' may have changed state before exiting "
                    f"with code {result.exit_code}; inspect state before repeating it."
                ]

        if result.timed_out:
            retry_safe = spec is not None and spec.safety == "read-only"
            metadata = {**base_metadata, "timeout": True}
            if not retry_safe:
                metadata["uncertain_changes"] = [
                    f"Skill command '{skill_name}/{script_path}' may have changed state before timing out."
                ]
            return ToolResult(
                tool_name="run_skill_command",
                parameters=params,
                result=f"Error: Command '{skill_name}/{script_path}' - {result.stderr}",
                is_error=True,
                metadata=metadata,
                outcome=(
                    ToolOutcome.TRANSIENT_FAILURE
                    if retry_safe
                    else ToolOutcome.ACTIONABLE_FAILURE
                ),
            )

        if result.exit_code != 0:
            stderr_tail = result.stderr[-COMMAND_STDERR_TAIL_CHARS:]
            authentication_required = any(
                marker in stderr_tail.casefold()
                for marker in (
                    "--connect-google",
                    "authorization expired",
                    "api_key environment variable is not set",
                )
            )
            if authentication_required:
                base_metadata["authentication_required"] = True
            text = (
                f"Error: Command '{skill_name}/{script_path}' exited with code "
                f"{result.exit_code}."
            )
            if stderr_tail.strip():
                text = f"{text}\n\n[stderr]\n{stderr_tail}"
            text += result.capture_notice()
            wrapped = ToolResult(
                tool_name="run_skill_command",
                parameters=params,
                result=text,
                is_error=True,
                metadata=dict(base_metadata),
                outcome=(
                    ToolOutcome.MISSING_INPUT
                    if authentication_required
                    else ToolOutcome.ACTIONABLE_FAILURE
                ),
            )
            return wrapped

        # Lead with the notice: output limits can cut a trailing notice off.
        notice = result.capture_notice().strip()
        wrapped = ToolResult(
            tool_name="run_skill_command",
            parameters=params,
            result=f"{notice}\n\n{result.stdout}" if notice else result.stdout,
            is_error=False,
            metadata=dict(base_metadata),
            outcome=ToolOutcome.USABLE,
        )
        return wrapped


def _last_option(args: list[str], name: str) -> str | None:
    """Return the final value of an option, following argparse's last-value-wins semantics."""
    value = None
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--":
            break
        if item == name and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        if item.startswith(name + "="):
            value = item[len(name) + 1:]
        index += 1
    return value


def _requires_approval(spec: ScriptSpec | None) -> bool:
    """Require approval for mutations and undeclared scripts unless trusted metadata overrides it."""
    if spec is None:
        return True
    if spec.requires_approval is not None:
        return spec.requires_approval
    return spec.safety != "read-only"

from __future__ import annotations

from tool_framework.i_tool import ITool, PreparedAction, ToolPolicy, ToolResult
from agent_skills.runtime import SkillRuntime


class LoadSkillInstructionsTool(ITool):
    def __init__(self, runtime: SkillRuntime):
        """Expose instruction loading as an agent tool constrained to catalog skill names."""
        self._runtime = runtime
        names = runtime.catalog.names()
        super().__init__(
            name="load_skill_instructions",
            description=(
                "Load the full SKILL.md instructions for an Agent Skill into context. "
                "This is the REQUIRED first step of every skill use: call it BEFORE any "
                "other use of the skill (before read_skill_resource or run_skill_command). "
                "It is idempotent — loading an already-loaded skill is safe and cheap. "
                f"Available skills: {', '.join(names)}."
            ),
            parameters=[],
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": list(names),
                        "description": "Name of the skill to load.",
                    }
                },
                "required": ["name"],
            },
        )

    async def run(self, args: dict) -> ToolResult:
        """Validate the requested skill name and delegate instruction activation to the session runtime."""
        self.validate_parameters(args)
        return self._runtime.load_skill_instructions(args["name"])


class ReadSkillResourceTool(ITool):
    def __init__(self, runtime: SkillRuntime):
        """Expose reference and asset reads for loaded skills through a named tool."""
        self._runtime = runtime
        names = runtime.catalog.names()
        super().__init__(
            name="read_skill_resource",
            description=(
                "Requires the skill's instructions to be already loaded via "
                "load_skill_instructions — call that first if you haven't. Reads a "
                "referenced Agent Skill resource (a file the loaded instructions point "
                "to). Only resources under references/ and assets/ are allowed."
            ),
            parameters=[],
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": list(names),
                        "description": "Loaded skill name.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative resource path such as references/REFERENCE.md.",
                    },
                },
                "required": ["skill", "path"],
            },
        )

    async def run(self, args: dict) -> ToolResult:
        """Validate resource-read parameters and return the runtime's confined file preview."""
        self.validate_parameters(args)
        return self._runtime.read_resource(args["skill"], args["path"])


class RunSkillCommandTool(ITool):
    def __init__(self, runtime: SkillRuntime):
        """Expose bundled script execution with command arguments and optional standard input."""
        self._runtime = runtime
        names = runtime.catalog.names()
        super().__init__(
            name="run_skill_command",
            description=(
                "Requires the skill's instructions to be already loaded via "
                "load_skill_instructions — call that first if you haven't. Runs a "
                "bundled CLI script from the skill. The `command` is the script path "
                "relative to the skill root (e.g. scripts/search.py) followed by its "
                "command-line arguments. Never guess the path or arguments — use exactly "
                "what the loaded instructions specify."
            ),
            parameters=[],
            input_schema={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "enum": list(names),
                        "description": "Loaded skill name.",
                    },
                    "command": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Script path relative to the skill root followed by its "
                            "arguments."
                        ),
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional standard input passed to the script.",
                    },
                },
                "required": ["skill", "command"],
            },
            policy=ToolPolicy(can_parallel=False),
        )

    async def run(self, args: dict) -> ToolResult:
        """Validate command parameters and execute the script through the skill runtime's policy checks."""
        self.validate_parameters(args)
        return await self._runtime.run_command(
            args["skill"],
            args["command"],
            args.get("stdin"),
        )

    def prepare_action(self, args: dict) -> PreparedAction:
        """Return the runtime action itself so execution uses one approval gate."""
        self.validate_input(args)
        return self._runtime.prepare_command(args["skill"], args["command"], args.get("stdin"))

    def policy_for(self, args: dict) -> ToolPolicy:
        """Return the runtime's side-effect classification for this concrete skill command."""
        return self._runtime.command_policy(
            str(args.get("skill", "")),
            args.get("command", []),
        )

"""Application tools for long-term memory.

These are client-specific tools: they depend on this application's
:class:`~memory.long_term_memory.LongTermMemory` and are registered directly on
the agent's tool collection (they are not portable Agent Skill CLI scripts).
The ``memory`` skill under ``skills/`` is instruction-only and guides the model
to call the ``save_memory`` / ``recall_memory`` tools defined here.
"""

from __future__ import annotations

from collections.abc import Callable

from memory.long_term_memory import LongTermMemory, MEMORY_CATEGORIES
from tool_framework.i_tool import ITool, ToolOutcome, ToolParameter, ToolPolicy, ToolResult


class SaveMemoryTool(ITool):
    """Tool for saving information to long-term memory."""

    def __init__(self, memory_provider: Callable[[], LongTermMemory]):
        """Expose persistent memory creation as a tool with content and category inputs."""
        self._memory = memory_provider
        super().__init__(
            name="save_memory",
            description=(
                "Save an important fact, user preference, or piece of information "
                "to long-term memory so it can be recalled in future conversations. "
                "Use this when the user shares personal preferences, important facts, "
                "or explicitly asks you to remember something."
            ),
            parameters=[
                ToolParameter(
                    name="content",
                    type="string",
                    required=True,
                    default=None,
                    description=(
                        "The information to remember. Be specific and self-contained, "
                        "e.g. 'User prefers morning meetings before 10 AM'."
                    ),
                ),
                ToolParameter(
                    name="category",
                    type="string",
                    required=False,
                    default="general",
                    description=(
                        f"Category for the memory. "
                        f"Allowed values: {', '.join(MEMORY_CATEGORIES)}."
                    ),
                ),
            ],
            policy=ToolPolicy(
                mutates_local=True,
                requires_approval=True,
                approval_reason=(
                    "Saved memories are recalled in future conversations; "
                    "confirm this content reflects what you want remembered."
                ),
                can_parallel=False,
                max_output_chars=20_000,
            ),
        )

    def approval_arguments(self, args: dict) -> dict:
        """Show exactly the content and effective category that will be stored."""
        return {"content": args.get("content"), "category": args.get("category", "general")}

    async def run(self, args: dict) -> ToolResult:
        """Validate and save information for future conversations, returning a confirmation or error text."""
        self.validate_parameters(args)
        content = args["content"]
        category = args.get("category", "general")

        try:
            memory_id = self._memory().save(content=content, category=category)
            return ToolResult(
                tool_name=self.name,
                parameters=args,
                result=f"Memory saved successfully (id={memory_id}, category='{category}').",
                metadata={"confirmed_changes": [f"Memory {memory_id} was saved."]},
            )
        except Exception as e:
            metadata = {"exception_type": type(e).__name__}
            if not isinstance(e, ValueError):
                metadata["uncertain_changes"] = [
                    "The memory save could not be confirmed; recall it before trying again."
                ]
            return ToolResult(
                tool_name=self.name,
                parameters=args,
                result="Memory could not be saved reliably. Check storage and recall it before retrying.",
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
                metadata=metadata,
            )


class RecallMemoryTool(ITool):
    """Tool for searching and recalling information from long-term memory."""

    def __init__(self, memory_provider: Callable[[], LongTermMemory]):
        """Expose memory search with optional category and result-count controls."""
        self._memory = memory_provider
        super().__init__(
            name="recall_memory",
            description=(
                "Search long-term memory for previously saved information. "
                "Use this when you need to recall facts, user preferences, "
                "or other information that was saved in a previous conversation."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    required=True,
                    default=None,
                    description="Search query to find relevant memories. Use keywords.",
                ),
                ToolParameter(
                    name="category",
                    type="string",
                    required=False,
                    default=None,
                    description=(
                        "Optional category filter. "
                        f"Allowed values: {', '.join(MEMORY_CATEGORIES)}."
                    ),
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    required=False,
                    default=5,
                    description="Maximum number of memories to return (default: 5).",
                ),
            ],
            policy=ToolPolicy(
                read_only=True,
                can_parallel=False,
                max_output_chars=20_000,
            ),
        )

    async def run(self, args: dict) -> ToolResult:
        """Search saved information and format matches, absence of matches, or storage errors for the agent."""
        self.validate_parameters(args)
        query = args["query"]
        category = args.get("category")
        limit = args.get("limit", 5)

        try:
            entries = self._memory().recall(
                query=query, category=category, limit=limit
            )

            if not entries:
                return ToolResult(
                    tool_name=self.name,
                    parameters=args,
                    result=f"No memories found matching '{query}'.",
                )

            lines = [f"Found {len(entries)} memory/memories:\n"]
            for entry in entries:
                lines.append(
                    f"- [{entry.category}] (id={entry.id}, saved={entry.created_at}): "
                    f"{entry.content}"
                )

            return ToolResult(
                tool_name=self.name,
                parameters=args,
                result="\n".join(lines),
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                parameters=args,
                result="Memories could not be read. Check that memory storage is available.",
                is_error=True,
                outcome=ToolOutcome.ACTIONABLE_FAILURE,
                metadata={"exception_type": type(e).__name__},
            )

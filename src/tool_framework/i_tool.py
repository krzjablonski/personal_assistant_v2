from typing import Any, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from jsonschema import Draft202012Validator

from llm.tool_schema_builder import build_parameters_schema
from tool_framework.approval import private_scope_binding, sanitized_approval_arguments


class ToolOutcome(Enum):
    USABLE = "usable"
    TRANSIENT_FAILURE = "transient_failure"
    MISSING_INPUT = "missing_input"
    ACTIONABLE_FAILURE = "actionable_failure"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass(frozen=True)
class ToolPolicy:
    """Execution policy metadata used by the agent runtime."""

    read_only: bool = False
    mutates_local: bool = False
    mutates_external: bool = False
    requires_approval: bool = False
    can_parallel: bool = True
    default_timeout_seconds: int | None = None
    max_output_chars: int | None = None
    approval_reason: str = ""
    idempotent: bool = False

    @property
    def retry_safe(self) -> bool:
        """Report whether read-only or idempotent policy permits automatic retries."""
        return self.read_only or self.idempotent


@dataclass
class ToolResult:
    tool_name: str
    parameters: dict
    result: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: ToolOutcome | None = None

    @property
    def effective_outcome(self) -> ToolOutcome:
        """Return the explicit outcome or infer a runtime outcome from error and approval metadata."""
        if self.outcome is not None:
            return self.outcome
        if not self.is_error:
            return ToolOutcome.USABLE
        if self.metadata.get("approval_required") is True:
            return ToolOutcome.MISSING_INPUT
        return ToolOutcome.ACTIONABLE_FAILURE


@dataclass
class ToolParameter:
    name: str
    type: str
    required: bool
    default: any
    description: str


@dataclass(frozen=True)
class PreparedAction:
    """One validated execution snapshot; private values stay inside the callable.

    Arguments are the resolved model call used for result reporting.
    Approval arguments are the exact reviewed scope; private configuration
    appears only as opaque bindings. Logs and traces redact separately.
    Execution and approval receive separate copies so neither can alter the other.
    """

    tool_name: str
    arguments: dict[str, Any]
    policy: ToolPolicy
    approval_arguments: dict[str, Any]
    execute: Callable[[], Awaitable[ToolResult]] = field(repr=False, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    _scope_binding: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", deepcopy(self.arguments))
        object.__setattr__(self, "approval_arguments", deepcopy(self.approval_arguments))
        object.__setattr__(self, "_scope_binding", self._current_binding())

    def _current_binding(self) -> str:
        return private_scope_binding((self.tool_name, self.arguments, self.approval_arguments, self.policy.__dict__))

    def validate_scope(self) -> None:
        if self._current_binding() != self._scope_binding:
            raise ValueError("The prepared action was changed; prepare it again and obtain fresh approval.")


class ITool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        policy: ToolPolicy | None = None,
        input_schema: dict | None = None,
    ):
        """Define a tool's model-facing identity, parameters, and default execution policy."""
        self.name = name
        self.description = description
        self.parameters = parameters
        self.input_schema: dict | None = deepcopy(input_schema)
        self.input_schema = build_parameters_schema(self)
        self.policy = policy or ToolPolicy()

    def __str__(self) -> str:
        """Return a compact tool name and description for display."""
        return f"`{self.name}`: {self.description}"

    @abstractmethod
    async def run(self, args: dict[str, any]) -> ToolResult:
        """Execute a concrete tool action and return its result for the agent runtime."""
        pass

    def approval_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the exact payload used to identify an approved call."""
        return args

    def policy_for(self, args: dict[str, Any]) -> ToolPolicy:
        """Return the execution policy for one concrete call."""
        return self.policy

    def prepare_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve ambient targets before approval, without performing side effects."""
        return deepcopy(args)

    def prepare_action(self, args: dict[str, Any]) -> PreparedAction:
        """Validate and freeze one call for the shared execution gate."""
        self.validate_input(args)
        resolved = self.prepare_arguments(deepcopy(args))
        self.validate_input(resolved)
        execution_args = deepcopy(resolved)
        policy = self.policy_for(resolved)
        scope = sanitized_approval_arguments(self.approval_arguments(deepcopy(resolved)))

        async def execute() -> ToolResult:
            self.validate_input(execution_args)
            current_scope = sanitized_approval_arguments(self.approval_arguments(deepcopy(execution_args)))
            if current_scope != scope or self.policy_for(execution_args) != policy:
                raise ValueError("The approved action scope changed; prepare a new action and obtain approval.")
            return await self.run(deepcopy(execution_args))

        return PreparedAction(
            self.name,
            deepcopy(resolved),
            policy,
            deepcopy(scope),
            execute,
        )

    def validate_input(self, args: dict[str, Any]) -> None:
        """Validate the canonical schema and retain subclass confinement checks."""
        self.validate_parameters(args)

    def validate_parameters(self, args: dict[str, Any]) -> None:
        """Validate the same JSON Schema published to every model provider."""
        schema = build_parameters_schema(self)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(args),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if not errors:
            return
        messages = []
        for error in errors:
            path = "$"
            for part in error.absolute_path:
                path += f"[{part}]" if isinstance(part, int) else f".{part}"
            messages.append(f"{path}: {error.message}")
        raise ValueError(
            f"Tool '{self.name}' was called with invalid parameters. "
            f"Errors: {'; '.join(messages)}"
        )

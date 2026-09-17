from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from llm.messages import Message

if TYPE_CHECKING:
    from typing import Type
    from pydantic import BaseModel
    from tool_framework.i_tool import ITool


class LLMResponseError(ValueError):
    """A provider response cannot be used safely; never include its private payload."""

    def __init__(self, provider: str, code: str):
        self.provider = provider
        self.code = code
        super().__init__(f"{provider} returned an unusable response ({code}).")


@dataclass
class LLMResponse:
    """Response from LLM."""

    message: Message
    stop_reason: str  # "end_turn", "tool_calls", "max_tokens"
    usage: dict  # {"input_tokens": int, "output_tokens": int}
    structured_data: Optional[dict] = field(default=None)

    @property
    def has_tool_calls(self) -> bool:
        """Report whether the provider-normalized stop reason requests tool execution."""
        return self.stop_reason == "tool_calls"


class ILLMClient(ABC):
    async def aclose(self) -> None:
        """Release resources owned by this client; stateless implementations need no cleanup."""

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Return the model's context window size in tokens."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List["Message"],
        system: str,
        tools: Optional[List["ITool"]] = None,
        max_tokens: int = 4096,
        response_schema: Optional["Type[BaseModel]"] = None,
    ) -> LLMResponse:
        """Send conversation messages and standing instructions to a model and return a normalized response.

        Tools describe available actions, and max_tokens requests a response limit.
        When response_schema is supplied, request structured output for that
        Pydantic model and expose parsed data through LLMResponse.structured_data.
        Raise LLMResponseError for refusals, empty completions, or malformed tool
        inputs, before returning any calls that could execute.
        Concrete clients define provider-specific handling and error behavior.
        """
        pass

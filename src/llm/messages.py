"""Application conversation data, independent of any model SDK or wire format."""

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class Image:
    data: str  # Base64-encoded image bytes.
    media_type: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """Plain text with optional images, calls, or a tool reply.

    provider_data is opaque, JSON-serializable continuation state. Only the
    originating LLM client interprets its own namespace; application code
    carries it through unchanged.
    """

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    images: list[Image] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    reasoning: str = ""
    provider_data: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant", "tool"):
            raise ValueError("Message role must be user, assistant, or tool; pass system instructions separately")

    def to_dict(self) -> dict:
        """Serialize application fields, omitting unused optional values."""
        return {
            key: value for key, value in asdict(self).items()
            if key in ("role", "text") or value
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Restore the neutral storage format; provider input belongs in clients."""
        return cls(
            **{
                **data,
                "images": [Image(**image) for image in data.get("images", [])],
                "tool_calls": [ToolCall(**call) for call in data.get("tool_calls", [])],
            }
        )

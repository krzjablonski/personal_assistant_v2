from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class AgentEventType(Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS_CHANGE = "status_change"
    ERROR = "error"
    REASONING = "reasoning"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    COMMAND_FINISHED = "command_finished"
    COMMAND_TIMED_OUT = "command_timed_out"
    RESULT_CLASSIFIED = "result_classified"
    RETRY_SCHEDULED = "retry_scheduled"
    TERMINAL_STATE = "terminal_state"


@dataclass
class AgentEvent:
    event_type: AgentEventType
    session_id: str
    message: str
    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)
    data: Optional[dict] = None
    agent_name: Optional[str] = None

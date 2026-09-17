"""Explicit, content-limited tracing that cannot change provider outcomes."""

import asyncio
import hashlib
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, List, Literal, Optional

from llm.i_llm_client import ILLMClient, LLMResponse
from message_logger.redaction import redact_value

if TYPE_CHECKING:
    from typing import Type

    from pydantic import BaseModel

    from llm.messages import Message
    from tool_framework.i_tool import ITool


TraceContent = Literal["metadata", "redacted"]
_RUN_OBSERVATION = ContextVar("langfuse_run_observation", default=None)


def get_langfuse():
    """Load the optional SDK only after explicit tracing opt-in."""
    from langfuse import get_client

    return get_client()


def propagate_attributes(**attributes):
    from langfuse import propagate_attributes as propagate

    return propagate(**attributes)


def _try_trace(operation):
    """Keep SDK/import/serialization failures separate from application execution."""
    try:
        return operation()
    except (Exception, asyncio.CancelledError):
        return None


def _message_content(message: "Message") -> dict:
    # Select fields before serialization: provider state and image bytes never
    # enter the tracing SDK, even when content tracing is explicitly enabled.
    return redact_value({
        "role": message.role,
        "text": message.text,
        "reasoning": message.reasoning,
        "images": [{"media_type": image.media_type} for image in message.images],
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
        "tool_name": message.tool_name,
        "is_error": message.is_error,
    })


def _usage_details(usage: dict) -> dict:
    """Keep observed integer counts, including cache metrics; never infer zeros."""
    names = {"input_tokens": "input", "output_tokens": "output"}
    return {
        names.get(key, key): value
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }


@contextmanager
def trace_agent_run(client: ILLMClient, session_id: str, agent_name: str | None = None):
    """Group opted-in calls without giving the SDK application exceptions or labels."""
    if not isinstance(client, LangfuseTrackedLLMClient):
        yield
        return

    # Manual observations avoid SDK context managers automatically recording raw
    # exception messages/stack data. ExitStack.close() also passes no exception
    # to the SDK's attribute propagation context manager.
    attributes = ExitStack()
    sdk = _try_trace(get_langfuse)
    observation = None
    token = None
    metadata = {"trace_content": client.content_mode}
    if sdk is not None:
        group = hashlib.sha256(session_id.encode()).hexdigest()
        _try_trace(lambda: attributes.enter_context(propagate_attributes(session_id=group)))
        observation = _try_trace(lambda: sdk.start_observation(
            as_type="span", name="agent-run", metadata=metadata,
        ))
    if observation is not None:
        token = _RUN_OBSERVATION.set((client, observation))
    try:
        yield
    except BaseException as error:
        if observation is not None:
            _try_trace(lambda: observation.update(metadata={
                **metadata, "outcome": "cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                "error_type": type(error).__name__,
            }))
        raise
    else:
        if observation is not None:
            _try_trace(lambda: observation.update(metadata={**metadata, "outcome": "completed"}))
    finally:
        if token is not None:
            _RUN_OBSERVATION.reset(token)
        if observation is not None:
            _try_trace(lambda: observation.end())
        _try_trace(attributes.close)


class LangfuseTrackedLLMClient(ILLMClient):
    """Opt-in SDK adapter; metadata is the default, redacted content is explicit.

    The SDK owns its shared exporter. Closing this wrapper closes only its inner
    model client; it does not flush or shut down a global tracing client.
    """

    def __init__(self, inner: ILLMClient, model_name: str, *, content_mode: TraceContent = "metadata"):
        if content_mode not in ("metadata", "redacted"):
            raise ValueError("Trace content must be metadata or redacted.")
        self._inner = inner
        self._model_name = model_name
        self.content_mode = content_mode

    @property
    def context_window(self) -> int:
        return self._inner.context_window

    async def aclose(self) -> None:
        await self._inner.aclose()

    def _start_generation(self, messages, system, tools, max_tokens):
        metadata = {
            "trace_content": self.content_mode,
            "message_count": len(messages),
            "image_count": sum(len(message.images) for message in messages),
            "tool_count": len(tools or []),
            "max_output_tokens": max_tokens,
        }
        fields = dict(as_type="generation", name="llm-chat", model=self._model_name, metadata=metadata)
        if self.content_mode == "redacted":
            fields["input"] = {
                "system": redact_value(system),
                "messages": [_message_content(message) for message in messages],
            }
        parent = _RUN_OBSERVATION.get()
        target = parent[1] if parent is not None and parent[0] is self else get_langfuse()
        return target.start_observation(**fields)

    async def chat(
        self,
        messages: List["Message"],
        system: str,
        tools: Optional[List["ITool"]] = None,
        max_tokens: int = 4096,
        response_schema: Optional["Type[BaseModel]"] = None,
    ) -> LLMResponse:
        generation = _try_trace(lambda: self._start_generation(messages, system, tools, max_tokens))
        try:
            response = await self._inner.chat(messages, system, tools, max_tokens, response_schema)
        except BaseException as error:
            if generation is not None:
                _try_trace(lambda: generation.update(metadata={
                    "outcome": "cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                    "error_type": type(error).__name__,
                }))
            raise
        else:
            if generation is not None:
                def record_response():
                    fields = {"usage_details": _usage_details(response.usage), "metadata": {"outcome": "completed"}}
                    if self.content_mode == "redacted":
                        fields["output"] = _message_content(response.message)
                    generation.update(**fields)

                _try_trace(record_response)
            return response
        finally:
            if generation is not None:
                _try_trace(lambda: generation.end())

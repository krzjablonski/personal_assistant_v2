from __future__ import annotations

import asyncio
import hashlib
import json
import hmac
import secrets
from copy import deepcopy
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from time import monotonic


_BINDING_KEY = secrets.token_bytes(32)


def private_scope_binding(value: Any) -> str:
    """Bind private session scope without exposing secrets or guessable plain hashes."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hmac.new(_BINDING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def sanitized_approval_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return an exact copy of the scope the human approves.

    Model-authored code, argv and message bodies are never masked here: a
    credential-shaped pattern can otherwise hide executable content. Private
    configuration must stay out of scopes (use private_scope_binding); logs and
    traces apply their own redaction.
    """
    return deepcopy(arguments)


ApprovalStatus = Literal["pending", "approved", "denied", "consumed", "unavailable", "cancelled"]


@dataclass
class ToolApprovalRequest:
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    status: ApprovalStatus = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: datetime | None = None
    consumed_by_action_id: str | None = None
    expires_at: float | None = None


def build_approval_id(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a session-deterministic identifier tying approval to a tool name and argument payload.

    Keyed with the session binding key so an identifier never acts as a plain
    hash of unredacted scope that could be guessed offline from logs.
    """
    return f"approval_{private_scope_binding({'tool_name': tool_name, 'arguments': arguments})[:16]}"


def _observe_handler_completion(task: asyncio.Task) -> None:
    """Retrieve callback failures after cancellation; callbacks hold no action."""
    if not task.cancelled():
        task.exception()


async def require_approval(
    store: ToolApprovalStore,
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
    *,
    logical_action_id: str | None = None,
) -> ToolApprovalRequest | None:
    """Wait only within this invocation; no undecided request survives return."""
    if store.authorized_retry(tool_name, arguments, logical_action_id):
        return None
    expires_at = monotonic() + store.decision_timeout_seconds if store.decision_timeout_seconds > 0 else None
    request = store.request(tool_name, arguments, reason, expires_at=expires_at)
    try:
        if request.status == "denied":
            return request
        if store.approval_handler is not None:
            handler = asyncio.create_task(store.approval_handler(deepcopy(request)))
            try:
                done, _ = await asyncio.wait({handler}, timeout=store.decision_timeout_seconds or None)
                if not done:
                    request.status = "unavailable"
                    return request
                decision = handler.result()
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
            finally:
                if not handler.done():
                    handler.cancel()
                handler.add_done_callback(_observe_handler_completion)
            if decision is True:
                store.approve(request.approval_id)
            elif decision is False:
                store.deny(request.approval_id)
        elif store.decision_timeout_seconds > 0:
            await store.wait_for_decision(request.approval_id)
        observed = store.get(request.approval_id)
        request.status = observed.status if observed is not None else "cancelled"
        if request.status not in {"denied", "approved"} and expires_at is not None and monotonic() >= expires_at:
            request.status = "unavailable"
            return request
        if store.consume_if_approved(request.approval_id, logical_action_id=logical_action_id):
            return None
        if request.status != "denied":
            request.status = "unavailable"
        return request
    except asyncio.CancelledError:
        if request.status != "denied":
            request.status = "cancelled"
        raise
    finally:
        store.discard(request.approval_id)


class ToolApprovalStore:
    """Session-only exact-scope decisions and current live gate requests.

    Request IDs include a nonce: a response to an old cancelled prompt cannot
    authorize a later invocation with identical arguments. Fingerprints bind
    consume-once decisions and denials to their actual scope.
    """

    def __init__(
        self,
        decision_timeout_seconds: float = 0.0,
        *,
        approval_handler: Callable[[ToolApprovalRequest], Awaitable[bool | None]] | None = None,
    ) -> None:
        self._requests: dict[str, ToolApprovalRequest] = {}
        self._denied_scopes: set[str] = set()
        self._consumed_actions: set[tuple[str, str]] = set()
        self._lock = threading.RLock()
        self.decision_timeout_seconds = decision_timeout_seconds
        self.approval_handler = approval_handler

    def authorized_retry(self, tool_name: str, arguments: dict[str, Any], logical_action_id: str | None) -> bool:
        scope = build_approval_id(tool_name, arguments)
        with self._lock:
            return bool(logical_action_id and scope not in self._denied_scopes
                        and (scope, logical_action_id) in self._consumed_actions)

    def discard(self, approval_id: str) -> None:
        with self._lock:
            self._requests.pop(approval_id, None)

    async def wait_for_decision(
        self,
        approval_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval: float = 0.2,
    ) -> str:
        """Wait until an approval request leaves pending status or the timeout expires.

        Return the observed status; pending also covers missing requests at timeout.
        Use the store timeout unless a call-specific timeout is supplied.
        """
        if timeout_seconds is None:
            timeout_seconds = self.decision_timeout_seconds

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            request = self._requests.get(approval_id)
            status = request.status if request is not None else "cancelled"
            if status != "pending":
                return status
            remaining = deadline - loop.time()
            if remaining <= 0:
                return status
            await asyncio.sleep(min(poll_interval, remaining))

    def request(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        *,
        expires_at: float | None = None,
    ) -> ToolApprovalRequest:
        """Create one invocation's request; identical concurrent calls remain separate."""
        scope = build_approval_id(tool_name, arguments)
        with self._lock:
            request = ToolApprovalRequest(
                approval_id=f"{scope}_{secrets.token_hex(8)}",
                tool_name=tool_name,
                arguments=deepcopy(arguments),
                reason=reason,
                status="denied" if scope in self._denied_scopes else "pending",
                expires_at=expires_at,
            )
            self._requests[request.approval_id] = request
            return deepcopy(request)

    def approve(self, approval_id: str) -> bool:
        """Mark a known request approved and record the decision time; return whether it exists."""
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None or request.status != "pending":
                return False
            if request.expires_at is not None and monotonic() >= request.expires_at:
                request.status = "unavailable"
                return False
            request.status = "approved"
            request.decided_at = datetime.now()
            return True

    def deny(self, approval_id: str) -> bool:
        """Mark a known request denied and record the decision time; return whether it exists."""
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None or request.status != "pending":
                return False
            request.status = "denied"
            self._denied_scopes.add(build_approval_id(request.tool_name, request.arguments))
            request.decided_at = datetime.now()
            return True

    def consume_if_approved(
        self,
        approval_id: str,
        *,
        logical_action_id: str | None = None,
    ) -> bool:
        """Consume an exact approval, or reuse it for the same declared safe retry.

        A consumed approval can be reused only by the same nonempty logical action
        identifier; return False when no matching authorization is available.
        """
        with self._lock:
            request = self._requests.get(approval_id)
            if request is None:
                return False
            if build_approval_id(request.tool_name, request.arguments) in self._denied_scopes:
                request.status = "denied"
                return False
            if request.status == "consumed":
                return bool(
                    logical_action_id
                    and request.consumed_by_action_id == logical_action_id
                )
            if request.status != "approved":
                return False
            request.status = "consumed"
            request.decided_at = datetime.now()
            request.consumed_by_action_id = logical_action_id
            if logical_action_id:
                self._consumed_actions.add((build_approval_id(request.tool_name, request.arguments), logical_action_id))
            return True

    def get(self, approval_id: str) -> ToolApprovalRequest | None:
        """Return a detached request snapshot, or None if unknown."""
        with self._lock:
            return deepcopy(self._requests.get(approval_id))

    def pending(self) -> list[ToolApprovalRequest]:
        """Return the current pending requests for human review."""
        with self._lock:
            return [
                deepcopy(request)
                for request in self._requests.values()
                if request.status == "pending"
            ]

    def all(self) -> list[ToolApprovalRequest]:
        """Return a list of all requests tracked in the current session."""
        with self._lock:
            return deepcopy(list(self._requests.values()))

    def clear(self) -> None:
        """Discard session requests and exact-scope decisions."""
        with self._lock:
            self._requests.clear()
            self._denied_scopes.clear()
            self._consumed_actions.clear()

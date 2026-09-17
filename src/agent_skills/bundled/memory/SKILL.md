---
name: memory
description: Save important facts and recall long-term memories across conversations. Use when the user asks to remember something or past preferences may be relevant.
metadata:
  version: "2.0"
allowed-tools: load_skill_instructions save_memory recall_memory
---
# Memory

Use this skill for persistent long-term memory across conversations.

This is a **client-specific** skill: its executable behavior is provided by this
application, not by bundled scripts. Persistence is backed by the client's
long-term memory store. The skill ships instructions only; there are no
`scripts/` to run through `run_skill_command`.

## Tools

The client exposes two memory tools directly. Call them by name:

- `save_memory` — Save a durable fact, preference, instruction, or context.
  - Arguments: `content` (required, specific and self-contained), optional
    `category` (one of `preference`, `fact`, `person`, `instruction`, `general`;
    defaults to `general`).
- `recall_memory` — Search previously saved memories.
  - Arguments: `query` (required keywords), optional `category` filter, optional
    `limit` (default 5).

## Rules

- Save memories only when information is durable and useful later, or when the
  user explicitly asks you to remember.
- Make saved content specific and self-contained.
- Recall memories when prior preferences or facts could affect the answer.

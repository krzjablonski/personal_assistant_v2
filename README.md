# Personal Assistant

A Python terminal assistant with one conversation, an approval-gated local Docker console, and six trusted integration skills.

## Setup

Use Python 3.11 or later. The dependency snapshot records the tested macOS/Python 3.11 environment:

```bash
rtk proxy python -m pip install -c requirements-lock.txt -e .
rtk proxy personal-assistant --help
rtk proxy personal-assistant --list-skills
```

OpenAI is the default; provider/model defaults are unchanged. Set a provider key in the process environment or use interactive encrypted configuration. Select another provider with `--provider anthropic`, `--provider gemini`, `--provider openrouter`, or `--provider local --base-url URL --model NAME`.

For console work, install/start Docker Desktop and explicitly build the packaged image:

```bash
rtk proxy python -m personal_assistant.console_setup
rtk proxy personal-assistant
```

The console automatically gets a private session workspace, so launching from the application checkout works. Use `--workspace PATH` only to grant writable access to an existing folder. Explicit folders must be outside application code, credentials and private data; conflicts fail during agent construction with the protected path identified. Docker is required only when using the console; missing Docker/image produces setup instructions. The application never pulls images, installs missing utilities, or executes console commands on the host as a fallback.

Configuration loads from the launch directory's optional `.env`, or from `--env-file PATH`, independently of the console workspace. `--workspace` no longer selects an environment file. Both loaded environment files and an existing workspace `.env` are protected from console mounts. Process environment and encrypted configuration remain supported.

## Everyday use

Interactive chat includes command suggestions after `/`, skill references after
`@`, in-memory input history and multiline paste. Use Tab to accept a suggestion,
Enter to submit and Ctrl+J for a newline. `/skills` shows available/loaded skills;
`@calendar Summarize tomorrow` loads that skill before the model responds.
`/clear` also clears loaded skills. Use `--plain` for line-based input or
`NO_COLOR=1` to disable colors. See the [CLI guide](.docs/14-cli-user-guide.md)
for keyboard behavior and reference syntax.

```bash
rtk proxy personal-assistant "Explain the difference between a list and a tuple"
rtk proxy personal-assistant --workspace ~/assistant-work "Find TODO comments and summarize them"
rtk proxy personal-assistant "Review unread email and suggest next steps"
rtk proxy personal-assistant --connect-google /path/to/google-desktop-client.json
```

The installed catalog contains **email, calendar, web-research, wiki, memory and browser**. All are available, with full instructions loaded when relevant. Integration credentials resolve at first use; an existing encrypted secret uses the secure host unlock prompt then. Memory storage opens on the first save/recall and closes with the CLI session. `--skills` and tools-disabled modes are unavailable.

The agent appends each new input to its own in-memory conversation. A simple answer can use one model call, a simple already-available tool task two. Clarification and planning are ordinary messages. Schema validation, context compaction and bounded repair remain; separate intake, semantic verification and checkpoint/resume calls are removed. Work calls and eligible safe retries share the invocation budget. Date/time and timezone context refresh for every model call.

Budget exhaustion and failures retain the stop reason, execution counts, effects and CLI log path. One separate tool-free reporting call explains progress, problems and next options, with a 20-second timeout and at most 1,024 output tokens. Its usage is included in totals, and it cannot resume work or mark the task complete. If reporting fails, runtime details remain available. Cancellation returns those details without starting a reporting call.

## Console and permissions

`run_command` takes `argv`, optional UTF-8 `stdin`, absolute container `cwd` (default `/workspace`), and `timeout` (default 30 seconds, maximum 120). Invoke `sh -c` explicitly for pipelines or `python -` with stdin for generated code. Every command, including reads, requires a fresh live approval. The prompt shows exact input, image, mounts and fixed policy; large code has a complete private review file. Changed inspected script files require new approval. Imports and paths embedded in code remain live inputs; file checks do not eliminate filesystem races.

Each invocation gets a fresh non-root Linux container with no network or integration credentials, a read-only root, resource limits, writable `/workspace`, read-only `/outputs`, and any explicit read-only `/inputs` grants. Shell variables and background processes do not survive. Only mounted working files persist. Timeout/cancellation triggers exact container removal and reports whether cleanup was verified. Commands are never automatically retried; process completion and output completeness are separate from verification of the user's outcome.

Without `--workspace`, `/workspace` maps to `<data-dir>/workspaces/<session-id>`, created privately on first console use. `/outputs` maps separately to `<data-dir>/outputs/<session-id>`. Research results can be read from `/outputs` and processed into `/workspace` without granting access to the launch directory. These session directories survive console calls and remain on disk after exit; a new agent session gets new directories. The private-data parent and other sessions are never mounted.

Approval waits exist only during live execution. Without an interactive handler, the command is unexecuted and the turn ends; no replay token is retained. `/clear` cancels active work and clears messages/effects/approvals and loaded skill instructions. `/settings`, `/status` and `/help` remain. Permanent and post-return approval commands are removed; historical approval files stay untouched and unused.

External APIs remain trusted host skills. Sending email and creating/editing calendar events require approval. Gmail drafts and attachment downloads retain their explicit no-approval policy. Calendar deletion/cancellation is unavailable.

Set an exact outgoing email restriction when needed:

```bash
export ALLOWED_EMAIL_RECIPIENTS=alice@example.com,bob@example.com
```

Unset, blank or literal `null` allows all valid recipients. A nonempty list permits only exact addresses, ignoring case, and every recipient is checked for both sends and drafts. Malformed lists fail closed. Approval cannot override this rule. `EMAIL_TO` supplies an optional default; `--to` selects a recipient. Draft reply-identity checks remain enforced.

## Saved output and data

Large captured output is saved as an ordinary private file. The result supplies a bounded preview, `/outputs/...` for console reads, a host path, and explicit completeness/omitted-byte metadata. Inspect saved content instead of repeating its source operation. Attachment downloads keep the configured original and copy only the requested files into session outputs. Copy/save failures preserve the original operation's outcome.

Private data lives in `PERSONAL_ASSISTANT_DATA_DIR`, XDG or the OS user-data directory, overridden by `--data-dir`. Credentials are encrypted; logs, memory and output can contain plaintext business content. Files remain until deliberately deleted. Historical manifests and reports are preserved without loading them into a registry.

```bash
rtk proxy personal-assistant --workspace /path/to/checkout --data-dir /path/to/new-data --migrate-data-from /path/to/checkout/src/data
```

Migration retains the source for rollback via `--data-dir`. Session history does not survive process restart.

## Validation and architecture

```bash
rtk proxy python tools/test.py
rtk proxy python tools/test_console.py
rtk proxy python .docs/tutorial/build.py --check
```

The isolated suite uses synthetic accounts/providers; the separate Docker release check uses actual local containers and disposable files. macOS Docker Desktop/Linux arm64 is validated. Linux-host support remains unverified; native Windows, macOS SDK/desktop execution and console networking are outside scope. Docker limits exposure but is not proof against every kernel/container vulnerability, and approved writable files can still be damaged.

See [architecture](.docs/03-architecture.md), [console contract](.docs/09-docker-console.md), [CLI guide](.docs/14-cli-user-guide.md), [superseding decision](.docs/decisions/006-conversation-and-docker-console.md), [implementation ledger](.ai/agent-runtime-simplification-plan.md), and [offline tutorial](.docs/tutorial/index.html). Optional Langfuse tracing remains explicit with `--trace metadata` or `--trace redacted`.

## Web providers and browser

Use `/settings` to select web search and extraction providers independently.
Tavily remains the default; Parallel and Firecrawl support both capabilities,
and Brave supports search. Existing Tavily keys work unchanged.

For the optional owned browser, run `python -m personal_assistant.browser_setup --install`
and then `--check`. This installs Browser Use in an isolated Python environment;
Chrome/Chromium must already be installed. Choose visible/headless mode in
`/settings`. See [setup and limitations](.docs/15-web-and-browser.md).

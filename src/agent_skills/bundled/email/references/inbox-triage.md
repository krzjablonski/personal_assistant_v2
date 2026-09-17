# Inbox triage

Use this guidance when the user requests inbox review and reply drafts. Create
Gmail drafts for manual review; do not send messages during triage.

## Steps

1. Use `run_skill_command` with `skill: "email"` and `command: ["scripts/read_email.py", "--folder", "INBOX", "--query", "is:unread", "--count", "5"]`, adjusting the label, native Gmail query and bounded count to the user's request.
2. Classify every returned email exactly once by its `uid` (stable Gmail message ID), including messages that share sender and subject.
3. Create Gmail drafts only for messages that need a reply, using `command: ["scripts/create_draft_email.py", "--message-uid", <uid>, "--to", <address>, "--subject", <subject>, "--body", <body>]` with the actual values.
4. Return an ordinary response covering the observed messages. Include each message's ID, sender, subject, priority, recommended action and actual draft outcome. A compact table is suitable. State selection limits or incomplete reads.

## Classification Order

Apply these rules top-down and stop at the first match:

1. Purely informational, automated, confirmation, receipt, invoice, security alert, newsletter, status update, or system notification -> `No reply needed`.
2. Event already happened and there is no request for a reply or decision -> `No reply needed`.
3. Direct question, decision blocker, deadline, or request requiring a reply -> `Critical` when urgent or blocking; otherwise `Quick Reply`.
4. Otherwise -> `Deferred`.

## Reporting and draft guidance

- Include every returned email once, including messages needing no reply. Copy its `uid`, `from` and `subject` from the read result.
- Distinguish urgent replies, quick replies, deferred decisions and messages needing no reply. Describe a follow-up task when a decision is missing.
- For each reply draft, identify its recipient and Gmail draft ID. Report creation only when `create_draft_email.py` succeeded.
- Never repeat a successful or uncertain draft attempt for the same message ID. Report an unconfirmed outcome and explain what needs manual review when evidence is incomplete. These are instructions for the agent; the portable script does not maintain a session-wide duplicate-draft ledger.
- Draft recipients must satisfy `ALLOWED_EMAIL_RECIPIENTS` and match the original sender or Reply-To mailbox. A recipient-policy rejection does not create a draft; report the restriction instead of claiming success.
- Do not choose options, promise availability, or change deadlines for the user. Unknown decisions become follow-up tasks before any reply is drafted.
- Do not use placeholders such as `[fill in]`, `<insert>`, `TODO`, or `...`.
- Never write that an email was sent. Drafts are proposals for manual review.

"""Pure email-parsing helpers shared by the email CLI scripts.

Standard-library only: no application imports and no network. These functions
(HTML stripping, body selection, header decoding, filename sanitising) are the
testable core of the read-email and download-attachments programs.
"""
from __future__ import annotations

import base64
import argparse
import binascii
import email
import html
import os
import re
from collections.abc import Mapping
from email.header import decode_header
from email.message import Message
from email.policy import default as email_policy

PLAIN_STUB_THRESHOLD = 40

_BLOCK_TAGS = "br|p|div|li|tr|h[1-6]|ol|ul"
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", flags=re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", flags=re.DOTALL | re.IGNORECASE)
_ANCHOR_RE = re.compile(
    r'<a\b[^>]*?href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    flags=re.DOTALL | re.IGNORECASE,
)
_BLOCK_TAG_RE = re.compile(rf"</?\s*(?:{_BLOCK_TAGS})\b[^>]*>", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_MULTI_NL_RE = re.compile(r"\n{3,}")


class EmailRecipientPolicyError(ValueError):
    """Recipient configuration or the outgoing envelope forbids an email action."""


def parse_draft_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the exact same draft command for preflight and script execution.

    Reject abbreviated/unknown flags. Repeated full flags and --option=value
    retain argparse's final-value semantics. Usage/help raise SystemExit as usual.
    """
    parser = argparse.ArgumentParser(description="Create a Gmail draft (does not send).", allow_abbrev=False)
    parser.add_argument("--to", default=None, help="Recipient address. Falls back to EMAIL_TO when omitted.")
    parser.add_argument("--subject", required=True, help="Email subject.")
    parser.add_argument("--body", required=True, help="Email body text.")
    parser.add_argument("--message-uid", help="Original Gmail message ID when creating a reply draft.")
    return parser.parse_args(argv)


def _recipient_addresses(value: str):
    """Parse all mailboxes without accepting recovery from malformed headers."""
    if (not value.strip() or value.rstrip().endswith(",")
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise EmailRecipientPolicyError("Email recipients must be valid comma-separated mailboxes.")
    try:
        header = email_policy.header_factory("To", value)
        addresses = header.addresses
        if (header.defects or not addresses or any(group.display_name is not None for group in header.groups)
                or any(not address.username or not address.domain for address in addresses)):
            raise ValueError("Invalid mailbox list")
        rendered = ", ".join(str(address) for address in addresses)
        if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
            raise ValueError("Invalid mailbox header")
    except (ValueError, IndexError) as exc:
        raise EmailRecipientPolicyError("Email recipients must be valid comma-separated mailboxes.") from exc
    return addresses


def resolve_email_recipients(explicit_to: str | None, *, environment: Mapping[str, str] | None = None) -> str:
    """Resolve and authorize every outgoing mailbox before Gmail setup.

    Explicit --to takes precedence over EMAIL_TO. ALLOWED_EMAIL_RECIPIENTS may
    be unset, blank, or literal null for unrestricted recipients; otherwise it
    must contain valid comma-separated bare addresses. Match each mailbox
    exactly, ignoring case. Return a canonical To header with display names,
    without comments or recovery from malformed input. This helper does no IO.
    """
    environment = os.environ if environment is None else environment
    configured = environment.get("ALLOWED_EMAIL_RECIPIENTS", "").strip()
    allowed = None
    if configured and configured.casefold() != "null":
        allowed = set()
        try:
            for entry in configured.split(","):
                entry = entry.strip()
                addresses = _recipient_addresses(entry)
                if len(addresses) != 1 or addresses[0].addr_spec.casefold() != entry.casefold():
                    raise EmailRecipientPolicyError("The allowlist requires bare addresses.")
                allowed.add(addresses[0].addr_spec.casefold())
        except EmailRecipientPolicyError as exc:
            raise EmailRecipientPolicyError(
                "Invalid ALLOWED_EMAIL_RECIPIENTS: use comma-separated email addresses, or leave it blank/null."
            ) from exc
    to = explicit_to if explicit_to is not None else environment.get("EMAIL_TO")
    if not to:
        raise EmailRecipientPolicyError("Email recipient is missing. Provide --to or set EMAIL_TO.")
    addresses = _recipient_addresses(to)
    if allowed is not None and any(address.addr_spec.casefold() not in allowed for address in addresses):
        raise EmailRecipientPolicyError("Email recipient is not permitted by ALLOWED_EMAIL_RECIPIENTS.")
    return ", ".join(str(address) for address in addresses)


def encode_gmail_raw(message: Message) -> str:
    """Encode a MIME message for Gmail API raw-message requests."""
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def decode_gmail_raw(raw: str) -> Message:
    """Decode Gmail raw-message data into a MIME message for parsing.

    Accept omitted base64 padding and raise ValueError for invalid encoded data.
    """
    try:
        padded = raw + "=" * (-len(raw) % 4)
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Gmail returned an invalid raw message") from exc
    return email.message_from_bytes(payload)


def decode_header_value(header_value: str) -> str:
    """Decode MIME header fragments into readable text with replacement for invalid bytes."""
    parts = decode_header(header_value)
    decoded_parts = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded_parts.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded_parts.append(part)
    return " ".join(decoded_parts)


def _anchor_repl(match: re.Match) -> str:
    """Keep an HTML link's destination alongside its readable label in plain-text email output."""
    url = match.group(1).strip()
    inner = match.group(2)
    inner_text = _TAG_RE.sub("", inner).strip()
    inner_text = html.unescape(inner_text)
    if not inner_text or inner_text == url:
        return url
    return f"{inner_text} ({url})"


def strip_html(html_str: str) -> str:
    """Produce readable email text from HTML while retaining link destinations and basic line breaks.

    Remove script, style, and tag markup using the supported text heuristics.
    """
    text = _SCRIPT_RE.sub("", html_str)
    text = _STYLE_RE.sub("", text)
    text = _ANCHOR_RE.sub(_anchor_repl, text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def _select_multipart_target(text_part, html_part):
    """Choose a readable MIME body, preferring HTML when the plain-text alternative is only a short stub."""
    if text_part is not None and html_part is not None:
        charset = text_part.get_content_charset() or "utf-8"
        payload = text_part.get_payload(decode=True)
        if payload is not None:
            decoded = payload.decode(charset, errors="replace").strip()
            if len(decoded) < PLAIN_STUB_THRESHOLD:
                return html_part
        return text_part
    return text_part or html_part


def get_body(msg: email.message.Message, max_body_chars: int) -> str:
    """Extract a bounded readable email body, preferring non-attachment text or HTML alternatives.

    Return a placeholder when no readable payload exists and mark truncated output.
    """
    if msg.is_multipart():
        text_part = None
        html_part = None
        for part in msg.walk():
            if part.is_multipart():
                continue
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain" and text_part is None:
                text_part = part
            elif content_type == "text/html" and html_part is None:
                html_part = part
        target = _select_multipart_target(text_part, html_part)
        if target is None:
            return "[No readable content]"
    else:
        target = msg

    charset = target.get_content_charset() or "utf-8"
    payload = target.get_payload(decode=True)
    if payload is None:
        return "[No readable content]"

    text = payload.decode(charset, errors="replace")
    if target.get_content_type() == "text/html":
        text = strip_html(text)

    text = text.strip()
    if len(text) > max_body_chars:
        text = (
            text[:max_body_chars]
            + f"\n[... body truncated at {max_body_chars} chars — re-fetching will return the same content]"
        )
    return text


def get_attachment_names(msg: email.message.Message) -> list[str]:
    """List decoded filenames from MIME parts marked as attachments."""
    names = []
    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", ""))
        if "attachment" in disposition:
            filename = part.get_filename()
            if filename:
                names.append(decode_header_value(filename))
    return names


def parse_email(msg: Message, uid: str, max_body_chars: int) -> dict:
    """Summarize a parsed MIME message without serializing or parsing it again."""
    return {
        "uid": uid,
        "reply_to": str(msg.get("Reply-To", "")),
        "from": decode_header_value(msg.get("From", "")),
        "to": decode_header_value(msg.get("To", "")),
        "date": msg.get("Date", ""),
        "subject": decode_header_value(msg.get("Subject", "(No Subject)")),
        "body": get_body(msg, max_body_chars),
        "attachments": get_attachment_names(msg),
    }


def sanitize_filename(filename: str) -> str:
    """Normalize an attachment filename by replacing separators and invalid characters and limiting its length.

    The result may be empty, so callers should supply a fallback name.
    """
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", filename)
    filename = filename.strip(". ")
    if len(filename) > 200:
        name, _, ext = filename.rpartition(".")
        if ext and len(ext) <= 10:
            filename = name[: 200 - len(ext) - 1] + "." + ext
        else:
            filename = filename[:200]
    return filename


def unique_filepath(directory: str, filename: str) -> str:
    """Choose an unused attachment path, adding a numeric suffix when the name already exists.

    This lookup does not reserve or create the returned path.
    """
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return filepath

    name, dot, ext = filename.rpartition(".")
    if not dot:
        name = filename
        ext = ""

    counter = 1
    while True:
        if ext:
            new_filename = f"{name}_{counter}.{ext}"
        else:
            new_filename = f"{name}_{counter}"
        filepath = os.path.join(directory, new_filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

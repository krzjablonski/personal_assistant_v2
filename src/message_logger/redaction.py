"""Credential-field and text masking shared by logs and resumable state."""

import json
import re


_CREDENTIAL_NAMES = {
    "password", "passwd", "secret", "token", "apikey", "credential", "credentials",
    "authorization", "proxyauthorization", "cookie", "setcookie", "privatekey",
}
_CREDENTIAL_SUFFIXES = (
    "password", "secret", "apikey", "token", "secretkey", "privatekey",
    "oauthtokenjson", "oauthclientjson",
)
_ASSIGNMENT_START = re.compile(
    r'''(?<![A-Za-z0-9_.-])(?P<key>["']?[A-Za-z][A-Za-z0-9_.-]*["']?)(?P<sep>\s*[:=]\s*)'''
)
_ASSIGNMENT = re.compile(
    _ASSIGNMENT_START.pattern +
    r'''(?P<value>\[REDACTED\]|"[^"]*"|'[^']*'|[^\s,;&}\]]+)'''
)
_BEARER = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
# Header forms mask the scheme too: "Basic"/"token" alone does not reveal the
# credential, but a partial mask invites reading the remaining text as safe.
_AUTH_HEADER = re.compile(
    r"""(?i)(?<![A-Za-z0-9_-])(?P<key>["']?(?:proxy-)?authorization["']?\s*[:=]\s*)(?!\[REDACTED\])"""
    r"""(?P<value>"[^"]*"|'[^']*'|(?:basic|bearer|digest|token|negotiate|ntlm|apikey|aws4-hmac-sha256)\s+[^\s,;&}\]"']+|[^\s,;&}\]"']+)"""
)
# Every pair in a Cookie/Set-Cookie header line is a credential.
_COOKIE_HEADER = re.compile(r"(?i)(?<![A-Za-z0-9_-])((?:set-)?cookie\s*:\s*)[^\r\n\"']+")
_FLAG = re.compile(
    r"""(?<![A-Za-z0-9_-])(?P<flag>--?[A-Za-z][A-Za-z0-9_.-]*)(?P<sep>=|\s+(?!-))"""
    r"""(?P<value>\[REDACTED\]|"[^"]*"|'[^']*'|[^\s"']+)"""
)
_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^/\s:@]*):[^/\s@]+@")
_KNOWN_TOKENS = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|ya29\.[A-Za-z0-9_-]{10,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})"
)
_QUERY = re.compile(r"([?&])([A-Za-z0-9_-]+)=([^&\s]+)")


def is_secret_key(key: object) -> bool:
    """Match credential names, preserving token counts, limits and other metrics."""
    name = re.sub(r"[^a-z]", "", str(key).lower())
    return name in _CREDENTIAL_NAMES or name.endswith(_CREDENTIAL_SUFFIXES)


def redact_text(text: str) -> str:
    """Mask recognizable credentials without treating arbitrary prose as a secret."""
    # A token JSON object can contain spaces and escaped quotes. Consume the
    # complete JSON value before the scalar matcher can swallow its first key
    # and leave that key's credential visible in the remaining text.
    chunks, cursor = [], 0
    decoder = json.JSONDecoder()
    for match in _ASSIGNMENT_START.finditer(text):
        start = match.end()
        if match.start() < cursor or not is_secret_key(match["key"]):
            continue
        if start >= len(text) or text[start] not in '{["':
            continue
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        chunks.extend((text[cursor:start], "[REDACTED]"))
        cursor = end
    text = "".join(chunks) + text[cursor:]
    text = _AUTH_HEADER.sub(lambda match: f"{match['key']}[REDACTED]", text)
    text = _COOKIE_HEADER.sub(r"\1[REDACTED]", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _URL_USERINFO.sub(r"\1:[REDACTED]@", text)
    text = _FLAG.sub(
        lambda match: (
            f"{match['flag']}{match['sep']}[REDACTED]"
            if is_secret_key(match["flag"].lstrip("-")) else match.group()
        ),
        text,
    )
    text = _ASSIGNMENT.sub(
        lambda match: (
            f"{match['key']}{match['sep']}[REDACTED]"
            if is_secret_key(match["key"]) else match.group()
        ),
        text,
    )
    text = _KNOWN_TOKENS.sub("[REDACTED]", text)
    return _QUERY.sub(
        lambda match: (
            f"{match[1]}{match[2]}=[REDACTED]"
            if match[2].lower() == "key" or is_secret_key(match[2]) else match.group()
        ),
        text,
    )


def redact_value(value):
    """Mask recognized fields recursively in a copy, retaining ordinary metrics."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if is_secret_key(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        result = []
        mask_next = False
        for item in value:
            result.append("[REDACTED]" if mask_next else redact_value(item))
            mask_next = (
                isinstance(item, str) and item.startswith("-")
                and "=" not in item and is_secret_key(item.lstrip("-"))
            )
        return result
    return redact_text(value) if isinstance(value, str) else value

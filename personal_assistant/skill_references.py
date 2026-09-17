"""Explicit skill mentions shared by interactive and one-shot CLI input."""

import re
from collections.abc import Iterable, Iterator


MENTION_RE = re.compile(r"(?<!\S)@([A-Za-z0-9_-]*)(?=$|[\s,;:!?)]|\.(?=\s|$))")
_CODE_START = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|$)|(?P<inline>`+)", re.MULTILINE)


class SkillReferenceError(ValueError):
    """A submitted explicit reference cannot be resolved locally."""


def reference_text(text: str) -> str:
    """Mask Markdown code while retaining offsets for completion and highlighting.

    Unclosed code is literal through the end of the input, including while typing.
    This scanner only recognizes code delimiters; it does not render Markdown.
    """
    visible = list(text)
    position = 0
    while match := _CODE_START.search(text, position):
        marker = match.group("fence")
        if marker:
            closing = re.compile(
                r"^[ \t]{0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*(?:\n|$)",
                re.MULTILINE,
            ).search(text, match.end())
        else:
            marker = match.group("inline")
            closing = re.compile(r"(?<!`)" + re.escape(marker) + r"(?!`)").search(text, match.end())
        end = closing.end() if closing else len(text)
        visible[match.start():end] = ["\n" if char == "\n" else " " for char in text[match.start():end]]
        position = end
    return "".join(visible)


def skill_mentions(text: str) -> Iterator[re.Match[str]]:
    visible = reference_text(text)
    for match in MENTION_RE.finditer(text):
        if visible[match.start()] == "@":
            yield match


def parse_skill_references(text: str, available: Iterable[str]) -> tuple[str, ...]:
    """Validate every mention before returning unique names in selection order."""
    names = set(available)
    selected = []
    for match in skill_mentions(text):
        name = match.group(1)
        if name not in names:
            raise SkillReferenceError(f"Unknown skill reference @{name}. Use /skills to see available skills; escape a literal marker as \\@.")
        if name not in selected:
            selected.append(name)
    return tuple(selected)

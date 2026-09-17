"""Save long source documents before the subprocess stdout limit is reached."""
from pathlib import Path
from tool_framework.output_files import save_output

PREVIEW_CHARS = 2000
DOCUMENT_CHARS = 2_000_000


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit * 3 // 4]
    tail = text[-limit // 4:]
    if "\n" in head:
        head = head.rsplit("\n", 1)[0]
    if "\n" in tail:
        tail = tail.split("\n", 1)[-1]
    return head + "\n[TRUNCATED: read the saved output file for omitted content]\n" + tail


def document_content(text: str, directory: Path) -> dict:
    retained = text[:DOCUMENT_CHARS]
    result = {"content": preview(retained), "truncated": len(text) > PREVIEW_CHARS,
              "stored_content_complete": len(text) <= DOCUMENT_CHARS,
              "source_chars": len(text)}
    if result["truncated"]:
        try:
            result.update(save_output(directory, retained, suffix=".md"))
        except OSError:
            result.update(stored_content_complete=False, output_save_error="Could not save full content")
    return result

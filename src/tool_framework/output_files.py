"""Private ordinary output files; no registry, manifests, or retention machinery."""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import uuid
from config_service.paths import default_data_dir, private_directory


def session_output_directory(session_id: str | None = None) -> Path:
    identifier = session_id or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Output session name must be a simple identifier")
    return default_data_dir() / "outputs" / identifier


def save_output(directory: Path, content: str | bytes, *, suffix: str = ".txt") -> dict:
    """Save captured bytes once and return host/container paths and byte count."""
    if not re.fullmatch(r"\.[A-Za-z0-9_-]{1,16}", suffix):
        raise ValueError("Output suffix must be a simple file extension")
    root = private_directory(Path(directory).resolve())
    payload = content.encode("utf-8") if isinstance(content, str) else content
    descriptor, filename = tempfile.mkstemp(prefix="output-", suffix=suffix, dir=root)
    path = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"output_path": "/outputs/" + path.name, "host_output_path": str(path), "output_size_bytes": len(payload)}

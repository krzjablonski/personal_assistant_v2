#!/usr/bin/env python3
"""Download attachments from a Gmail message through the Gmail API."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from email_utils import (
    decode_gmail_raw,
    decode_header_value,
    sanitize_filename,
    unique_filepath,
)
from google_credentials import GoogleCredentialsError, load_google_credentials

CREATE_ATTEMPTS = 5


def _ensure_private_directory(directory: str) -> None:
    """Create the attachment directory owner-only; leave an existing directory's mode alone."""
    try:
        os.makedirs(directory, mode=0o700)
    except FileExistsError:
        if not os.path.isdir(directory):
            raise
        return
    os.chmod(directory, 0o700)


def _create_exclusive(directory: str, filename: str) -> tuple[str, int]:
    """Open a new owner-only file, retrying a fresh unique name if another writer wins the race."""
    for attempt in range(1, CREATE_ATTEMPTS + 1):
        filepath = unique_filepath(directory, filename)
        try:
            return filepath, os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if attempt == CREATE_ATTEMPTS:
                raise


def main(argv: list[str] | None = None) -> int:
    """Download named attachments from a Gmail message into the configured attachment directory.

    Print saved paths and per-file write errors as JSON. Return one for setup
    or message-fetch failures; individual attachment write errors still return zero.
    """
    parser = argparse.ArgumentParser(
        description="Download attachments from a Gmail message ID."
    )
    parser.add_argument(
        "--message-uid",
        required=True,
        help="Gmail message ID from read_email output.",
    )
    args = parser.parse_args(argv)

    attachments_dir = os.environ.get("ATTACHMENTS_DIR")
    if not attachments_dir:
        print("Error: ATTACHMENTS_DIR environment variable is not set.", file=sys.stderr)
        return 1
    try:
        credentials = load_google_credentials()
    except GoogleCredentialsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        _ensure_private_directory(attachments_dir)
    except OSError as exc:
        print(f"Error: Could not create attachments directory - {exc}", file=sys.stderr)
        return 1

    try:
        from google.auth.exceptions import RefreshError
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        print(f"Error: Google API client libraries are not installed - {exc}", file=sys.stderr)
        return 1

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        response = (
            service.users()
            .messages()
            .get(userId="me", id=args.message_uid, format="raw")
            .execute()
        )
        raw = response.get("raw")
        if not isinstance(raw, str):
            raise ValueError("Gmail returned no raw message")
        msg = decode_gmail_raw(raw)
    except RefreshError:
        print(
            "Error: Google authorization expired. Reconnect with --connect-google.",
            file=sys.stderr,
        )
        return 1
    except (HttpError, ValueError) as exc:
        print(f"Error: Failed to download attachments - {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report API failures to stderr
        print(f"Error: Failed to download attachments - {exc}", file=sys.stderr)
        return 1

    saved_files = []
    errors = []
    for part in msg.walk():
        if "attachment" not in str(part.get("Content-Disposition", "")):
            continue
        filename = part.get_filename()
        if not filename:
            continue
        filename = sanitize_filename(decode_header_value(filename)) or "attachment"
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        try:
            filepath, descriptor = _create_exclusive(attachments_dir, filename)
            with os.fdopen(descriptor, "wb") as file:
                file.write(payload)
            saved = {"path": filepath, "size_bytes": len(payload)}
            output_directory = os.environ.get("SESSION_OUTPUT_DIR")
            if output_directory:
                try:
                    root = Path(output_directory).resolve()
                    root.mkdir(mode=0o700, parents=True, exist_ok=True)
                    root.chmod(0o700)
                    descriptor, output_path = tempfile.mkstemp(prefix="attachment-", suffix=Path(filename).suffix, dir=root)
                    try:
                        with os.fdopen(descriptor, "wb") as file:
                            os.fchmod(file.fileno(), 0o600)
                            file.write(payload)
                    except BaseException:
                        Path(output_path).unlink(missing_ok=True)
                        raise
                    saved.update(path="/outputs/" + Path(output_path).name,
                                 host_output_path=output_path, saved_host_path=filepath)
                except OSError as error:
                    saved["output_handoff_error"] = type(error).__name__
                    saved["console_readable"] = False
            saved_files.append(saved)
        except OSError as exc:
            errors.append({"filename": filename, "error": str(exc)})

    payload_out = {
        "message_uid": args.message_uid,
        "count": len(saved_files),
        "files": saved_files,
        "message": (
            f"Downloaded {len(saved_files)} attachment(s) from message "
            f"{args.message_uid}."
            if saved_files
            else f"No attachments found in message {args.message_uid}."
        ),
    }
    if errors:
        payload_out["errors"] = errors

    json.dump(payload_out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

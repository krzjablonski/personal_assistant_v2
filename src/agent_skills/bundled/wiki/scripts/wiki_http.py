"""Size-bounded JSON requests shared by the portable wiki commands."""
from __future__ import annotations

import json

import httpx

MAX_RESPONSE_BYTES = 5_000_000


class ResponseTooLarge(ValueError):
    """The wiki returned more data than the command is willing to read."""


def get_json(client: httpx.Client, url: str, params: dict, *, timeout: float = 15):
    """GET ``url`` and decode JSON, reading at most ``MAX_RESPONSE_BYTES`` bytes."""
    with client.stream("GET", url, params=params, timeout=timeout) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge(
                    f"Wiki response exceeded the {MAX_RESPONSE_BYTES // 1_000_000} MB size limit."
                )
            chunks.append(chunk)
    return json.loads(b"".join(chunks))

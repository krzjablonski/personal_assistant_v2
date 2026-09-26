"""Small fixed HTTP adapter registry. No provider fallback or hidden retries.

Contracts: docs.parallel.ai/search/search-quickstart,
docs.firecrawl.dev/api-reference/endpoint/{search,scrape}, brave.com/search/api/.
"""
from datetime import datetime, timezone
import ipaddress
import json
from time import monotonic
from urllib.parse import urlsplit

import httpx

from .configuration import validate_provider

MAX_RESPONSE_BYTES = 12_000_000


class WebError(ValueError):
    def __init__(self, message: str, category: str = "configuration"):
        super().__init__(message)
        self.category = category


# Public wildcard DNS services that resolve embedded addresses such as 127.0.0.1.nip.io.
_REBINDING_SUFFIXES = ("nip.io", "sslip.io", "xip.io", "localtest.me", "lvh.me")
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


def _legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse inet_aton-style IPv4 forms (``127.1``, ``0x7f.0.0.1``, octal, integer) resolvers accept."""
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    values = []
    for part in parts:
        try:
            if part[:2] in {"0x", "0X"}:
                values.append(int(part[2:] or "0", 16))
            elif len(part) > 1 and part.startswith("0"):
                values.append(int(part, 8))
            elif part.isdigit():
                values.append(int(part))
            else:
                return None
        except ValueError:
            return None
    *head, last = values
    if any(value > 255 for value in head) or last >= 256 ** (5 - len(values)):
        return None
    number = 0
    for value in head:
        number = number * 256 + value
    return ipaddress.IPv4Address(number * 256 ** (4 - len(head)) + last)


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped or address.sixtofour or (address.teredo[1] if address.teredo else None)
        if embedded is not None and not embedded.is_global:
            return False
    return address.is_global and not address.is_multicast


def validate_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".")
        parsed.port
        if len(url) > 4000 or parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            raise ValueError
        if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
            raise ValueError
        if host in _REBINDING_SUFFIXES or host.endswith(tuple("." + suffix for suffix in _REBINDING_SUFFIXES)):
            raise ValueError
        try:
            address = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            address = _legacy_ipv4(host)
        if address is None and (":" in host or "." not in host):
            raise ValueError
        if address is not None and not _is_public_address(address):
            raise ValueError
    except ValueError:
        raise WebError("Use a public HTTP(S) URL without embedded credentials.", "invalid_url") from None
    return url


def _rows(data, key):
    if not isinstance(data, dict) or not isinstance(data.get(key), list) or not all(isinstance(r, dict) for r in data[key]):
        raise WebError("Provider returned an invalid response.", "invalid_response")
    return data[key]


def _text(value):
    return value if isinstance(value, str) else ""


class WebProvider:
    def __init__(self, name: str, key: str, *, client: httpx.Client | None = None):
        try:
            validate_provider(name, "search")
        except ValueError as error:
            raise WebError(str(error)) from None
        if not key:
            raise WebError(f"Missing API key for {name}.")
        self.name, self.key, self.client = name, key, client

    def _request(self, endpoint: str, payload: dict, *, get=False, timeout=30):
        headers = ({"x-api-key": self.key} if self.name == "parallel" else
                   {"X-Subscription-Token": self.key} if self.name == "brave" else
                   {"Authorization": f"Bearer {self.key}"})
        def send(client):
            with client.stream("GET" if get else "POST", endpoint, headers=headers,
                               params=payload if get else None, json=None if get else payload,
                               timeout=timeout, follow_redirects=False) as response:
                if response.status_code >= 300:
                    status = response.status_code
                    category = "authentication" if status in {401, 403} else "rate_limit" if status == 429 else "provider_error"
                    raise WebError(f"{self.name} returned HTTP {status}.", category)
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise WebError("Provider response exceeded the size limit.", "response_too_large")
                    chunks.append(chunk)
                data = json.loads(b"".join(chunks))
                if not isinstance(data, dict) or data.get("success") is False:
                    raise WebError("Provider did not return a successful response.", "provider_error")
                return data
        try:
            if self.client is not None:
                return send(self.client)
            with httpx.Client(trust_env=False) as client:
                return send(client)
        except WebError:
            raise
        except httpx.TimeoutException:
            raise WebError(f"{self.name} request timed out.", "timeout") from None
        except httpx.HTTPError:
            raise WebError(f"Could not connect to {self.name}.", "connection") from None
        except (ValueError, UnicodeError):
            raise WebError("Provider returned invalid JSON.", "invalid_response") from None

    def search(self, query: str, *, limit=5, depth=None) -> dict:
        query = query.strip() if isinstance(query, str) else ""
        if not 1 <= len(query) <= 500 or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise WebError("Query must contain 1–500 characters and limit must be 1–20.")
        if depth is not None and (self.name != "tavily" or depth not in {"basic", "advanced"}):
            raise WebError("--search-depth basic|advanced is supported only by Tavily.", "unsupported_option")
        start = monotonic()
        if self.name == "tavily":
            data = self._request("https://api.tavily.com/search", {"query": query, "max_results": limit,
                                 "search_depth": depth or "basic", "include_answer": False, "include_raw_content": False})
            rows = _rows(data, "results")
        elif self.name == "parallel":
            data = self._request("https://api.parallel.ai/v1/search", {"objective": query, "search_queries": [query],
                                 "mode": "fast", "advanced_settings": {"max_results": limit,
                                 "excerpt_settings": {"max_chars_per_result": 1500}}})
            rows = _rows(data, "results")
        elif self.name == "firecrawl":
            data = self._request("https://api.firecrawl.dev/v2/search", {"query": query, "limit": limit, "sources": ["web"]})
            rows = _rows(data.get("data"), "web")
        else:
            data = self._request("https://api.search.brave.com/res/v1/web/search", {"q": query, "count": limit}, get=True)
            rows = _rows(data.get("web", {"results": []} if data.get('type') == 'search' else None), "results")
        results = []
        for row in rows[:limit]:
            url = _text(row.get("url"))
            try:
                validate_url(url)
            except WebError:
                continue
            content = row.get("content", row.get("description", ""))
            if self.name == "parallel":
                excerpts = row.get('excerpts', [])
                if not isinstance(excerpts, list) or not all(isinstance(x, str) for x in excerpts):
                    raise WebError('Provider returned invalid excerpts.', 'invalid_response')
                content = "\n".join(excerpts)
            results.append({"title": _text(row.get("title"))[:300], "url": url, "content": _text(content)[:1500],
                            "published_at": row.get("publish_date") if isinstance(row.get("publish_date"), str) else None})
        return {"query": query, "provider": self.name, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((monotonic() - start) * 1000), "results": results,
                **({"search_depth": depth or "basic"} if self.name == "tavily" else {})}

    def extract(self, urls: list[str]) -> dict:
        try:
            validate_provider(self.name, "extract")
        except ValueError as error:
            raise WebError(str(error), "unsupported_capability") from None
        if not 1 <= len(urls) <= 5:
            raise WebError("Extract requires 1–5 URLs.")
        urls = list(dict.fromkeys(validate_url(url) for url in urls))
        start = monotonic()
        results, failed = [], []
        if self.name == "firecrawl":
            # One attempt per URL; no async job polling or automatic retry.
            for url in urls:
                if monotonic() - start > 85:
                    failed.append({"url": url, "error": "Batch time budget exhausted", "category": "timeout"})
                    continue
                try:
                    data = self._request("https://api.firecrawl.dev/v2/scrape", {"url": url, "formats": ["markdown"],
                                         "timeout": 15000, "maxAge": 0}, timeout=18)
                    row = data.get("data")
                    if not isinstance(row, dict) or not isinstance(row.get("markdown"), str) or not row["markdown"].strip():
                        raise WebError("Provider returned no page content.", "invalid_response")
                    metadata = row.get("metadata") or {}
                    if not isinstance(metadata, dict) or not isinstance(metadata.get("statusCode", 200), int):
                        raise WebError("Provider returned invalid page metadata.", "invalid_response")
                    if metadata.get("error") or metadata.get("statusCode", 200) >= 400:
                        raise WebError("Page could not be fetched.", "page_error")
                    reported = _text(metadata.get("url") or metadata.get("sourceURL"))
                    try:
                        validate_url(reported)
                    except WebError:
                        reported = url
                    results.append({"url": url, "reported_url": reported,
                                    "content": row["markdown"]})
                except WebError as error:
                    failed.append({"url": url, "error": str(error), "category": error.category})
        else:
            if self.name == "tavily":
                data = self._request("https://api.tavily.com/extract", {"urls": urls, "format": "markdown"}, timeout=60)
                content_key = "raw_content"
            else:
                data = self._request("https://api.parallel.ai/v1/extract", {"urls": urls,
                                     "advanced_settings": {"full_content": True}}, timeout=60)
                content_key = "full_content"
            rows = _rows(data, "results")
            by_url = {row.get("url"): row for row in rows if isinstance(row.get("url"), str)}
            for url in urls:
                row = by_url.get(url, {})
                content = row.get(content_key)
                if not isinstance(content, str) or not content.strip():
                    failed.append({"url": url, "error": "Provider returned no content for this URL", "category": "page_error"})
                else:
                    results.append({"url": url, "reported_url": row["url"], "content": content})
        return {"provider": self.name, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((monotonic() - start) * 1000), "results": results, "failed": failed,
                "partial": bool(results and failed)}

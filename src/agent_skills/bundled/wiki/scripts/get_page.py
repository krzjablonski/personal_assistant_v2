#!/usr/bin/env python3
"""Read the plain-text content of an article from a configured Fandom wiki.

Standalone CLI program. Retrieves an article via the MediaWiki API (TextExtracts,
falling back to parsing wikitext), cleans the markup to plain text, truncates to a
character budget, and prints the result to standard output as JSON. Diagnostics go
to standard error. Exit status is 0 on success and non-zero on failure.

Runtime requirements:
    * Python 3.11+
    * The ``httpx`` package
    * Network access to ``*.fandom.com``

Usage:
    python get_page.py --wiki WIKI --title TITLE [--max-chars N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import httpx


from wiki_http import ResponseTooLarge, get_json
from wiki_registry import WIKI_REGISTRY

_NON_TEXT_LINK = re.compile(r"\[\[\s*(?:Category|File|Image)\s*:", re.IGNORECASE)


def _try_extracts(client: httpx.Client, base_url: str, title: str) -> str | None:
    """Try to retrieve plain-text article content through TextExtracts, returning None when unavailable.

    Redirect pages are followed to their target. Oversized responses propagate.
    """
    try:
        data = get_json(
            client,
            f"{base_url}/api.php",
            {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": "true",
                "redirects": 1,
                "format": "json",
            },
        )
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            extract = page_data.get("extract", "")
            if extract:
                return extract.strip()
        return None
    except ResponseTooLarge:
        raise
    except Exception:
        return None


def _try_parse(client: httpx.Client, base_url: str, title: str) -> str | None:
    """Retrieve and simplify article wikitext as a fallback, returning None when retrieval fails.

    Redirect pages are followed to their target. Oversized responses propagate.
    """
    try:
        data = get_json(
            client,
            f"{base_url}/api.php",
            {
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "redirects": 1,
                "format": "json",
            },
        )
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext:
            return None
        return _clean_wikitext(wikitext)
    except ResponseTooLarge:
        raise
    except Exception:
        return None


def _remove_non_text_links(text: str) -> str:
    """Drop [[Category:...]], [[File:...]] and [[Image:...]] links, including nested caption links."""
    parts = []
    position = 0
    while match := _NON_TEXT_LINK.search(text, position):
        parts.append(text[position:match.start()])
        depth, index = 0, match.start()
        while index < len(text):
            if text.startswith("[[", index):
                depth += 1
                index += 2
            elif text.startswith("]]", index):
                depth -= 1
                index += 2
                if depth == 0:
                    break
            else:
                index += 1
        position = index
    parts.append(text[position:])
    return "".join(parts)


def _clean_wikitext(text: str) -> str:
    """Basic cleanup of MediaWiki markup to produce readable plain text."""
    # Remove templates like {{...}} (two passes for simple nesting).
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Remove category / file / image links before generic link rewriting.
    text = _remove_non_text_links(text)
    # Convert wiki links [[Target|Display]] -> Display, [[Target]] -> Target.
    text = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove external links [http://... Display] -> Display.
    text = re.sub(r"\[https?://[^\s\]]+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Convert headings == Title == -> Title.
    text = re.sub(r"={2,}\s*(.+?)\s*={2,}", r"\n\1\n", text)
    # Remove bold/italic markup.
    text = re.sub(r"'{2,5}", "", text)
    # Collapse excessive whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    """Read a supported wiki article and print bounded plain text with a truncation flag.

    Try TextExtracts before wikitext; return two for an unknown wiki and one
    when neither source provides content.
    """
    available = ", ".join(sorted(WIKI_REGISTRY))
    parser = argparse.ArgumentParser(
        description="Read an article's plain-text content from a Fandom wiki."
    )
    parser.add_argument(
        "--wiki",
        required=True,
        help=f"Short name of the wiki. Available: {available}.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Exact article title (as returned by search.py).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum number of characters to return (default: 3000).",
    )
    args = parser.parse_args(argv)

    wiki = args.wiki.lower()
    base_url = WIKI_REGISTRY.get(wiki)
    if not base_url:
        print(
            f"Error: Unknown wiki '{args.wiki}'. Available wikis: {available}",
            file=sys.stderr,
        )
        return 2

    try:
        with httpx.Client() as client:
            content = _try_extracts(client, base_url, args.title)
            if content is None:
                content = _try_parse(client, base_url, args.title)
    except ResponseTooLarge as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if content is None:
        print(
            f"Error: Could not retrieve article '{args.title}' from {wiki} wiki. "
            "The page may not exist.",
            file=sys.stderr,
        )
        return 1

    truncated = False
    max_chars = max(1, args.max_chars)
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True

    json.dump(
        {
            "wiki": wiki,
            "title": args.title,
            "content": content,
            "truncated": truncated,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

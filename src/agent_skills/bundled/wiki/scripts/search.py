#!/usr/bin/env python3
"""Search article titles on a configured Fandom wiki.

Standalone CLI program. Queries the MediaWiki API of a known Fandom wiki, returns
the matching article titles and
prints the result to standard output as JSON. Diagnostics go to standard error.
Exit status is 0 on success and non-zero on failure.

All wikis are English; translate non-English queries to English before searching.

Runtime requirements:
    * Python 3.11+
    * The ``httpx`` package
    * Network access to ``*.fandom.com``

Usage:
    python search.py --wiki WIKI --query QUERY [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


from wiki_http import get_json
from wiki_registry import WIKI_REGISTRY

MAX_LIMIT = 10


def main(argv: list[str] | None = None) -> int:
    """Search a supported Fandom wiki and print matching titles.

    Clamp the result limit to one through ten; return two for unknown wikis
    and one for search failures.
    """
    available = ", ".join(sorted(WIKI_REGISTRY))
    parser = argparse.ArgumentParser(
        description="Search article titles on a configured Fandom wiki."
    )
    parser.add_argument(
        "--wiki",
        required=True,
        help=f"Short name of the wiki. Available: {available}.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query in ENGLISH (translate if needed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=f"Maximum number of results (default: 5, max: {MAX_LIMIT}).",
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

    limit = max(1, min(args.limit, MAX_LIMIT))

    try:
        with httpx.Client() as client:
            data = get_json(
                client,
                f"{base_url}/api.php",
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": args.query,
                    "srlimit": limit,
                    "srprop": "",
                    "format": "json",
                },
            )

            search_results = data.get("query", {}).get("search", [])
            results = [{"title": item["title"]} for item in search_results]
    except httpx.TimeoutException:
        print(
            f"Error: Request to {wiki} wiki timed out. Try again later.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - report any failure to stderr
        print(f"Error: Failed to search {wiki} wiki - {exc}", file=sys.stderr)
        return 1

    json.dump({"wiki": wiki, "query": args.query, "results": results}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

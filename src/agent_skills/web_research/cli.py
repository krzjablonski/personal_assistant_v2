"""Stable skill command facades; compact JSON is produced before stdout capture."""
import argparse
import json
import os
import sys
from pathlib import Path

from tool_framework.output_files import session_output_directory
from .configuration import from_environment
from .content import document_content
from .providers import WebProvider


def main(capability: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Public web {capability} using the configured provider.")
    if capability == "search":
        parser.add_argument("--query", required=True)
        parser.add_argument("--limit", type=int, default=5)
        parser.add_argument("--search-depth", choices=["basic", "advanced"])
    else:
        parser.add_argument("--url", action="append", required=True, dest="urls")
    args = parser.parse_args(argv)
    try:
        name, key = from_environment(capability, os.environ)
        provider = WebProvider(name, key)
        if capability == "search":
            result = provider.search(args.query, limit=args.limit, depth=args.search_depth)
        else:
            result = provider.extract(args.urls)
            directory = Path(os.environ["SESSION_OUTPUT_DIR"]) if os.environ.get("SESSION_OUTPUT_DIR") else session_output_directory()
            for doc in result["results"]:
                doc.update(document_content(doc.pop("content"), directory))
                doc.update(provider=name, retrieved_at=result["retrieved_at"])
        json.dump(result, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        if capability == "extract" and not result["results"]:
            print(json.dumps({"error": "All URLs failed extraction", "category": "page_error"}), file=sys.stderr)
            return 1
        return 0
    except ValueError as error:
        print(json.dumps({"error": str(error), "category": getattr(error, "category", "configuration")}), file=sys.stderr)
        return 1

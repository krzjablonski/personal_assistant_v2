---
name: wiki
description: Search configured Fandom wikis and read article pages. Use for wiki lookups where article titles must be searched before retrieving full page content.
metadata:
  version: "1.0"
  scripts:
    scripts/search.py:
      safety: read-only
      description: Search article titles on a configured Fandom wiki.
    scripts/get_page.py:
      safety: read-only
      description: Read the plain-text content of a Fandom wiki article.
allowed-tools: load_skill_instructions run_skill_command
---
# Wiki

Use this skill for configured Fandom wiki search and page reads.

All supported wikis are in English: `cthulhu`, `darksouls`, `dc`, `dnd`,
`elderscrolls`, `fallout`, `harrypotter`, `lotr`, `lovecraft`, `marvel`,
`starwars`, `warhammer`, `warhammer40k`, `witcher`.

## Runtime requirements

- Python 3.11+
- The `httpx` package (install with `pip install httpx`)
- Network access to `*.fandom.com`
- No environment variables

## Scripts

Each script is a standalone command-line program. Run it directly as `python
<script-path> [arguments]` from the skill directory. In this client, invoke it
with the `run_skill_command` tool, passing the script path and its arguments as
its `command` list. Both
scripts are read-only and run without approval. Each prints a JSON object to
standard output; errors are written to standard error with a non-zero exit code.

### scripts/search.py

Search article titles. Read a selected article with `get_page.py` when its content
is needed.

- Arguments:
  - `--wiki WIKI` (required): short wiki name (see list above).
  - `--query QUERY` (required): search query in English.
  - `--limit N` (optional): maximum results, default 5, capped at 10.
- Output: `{"wiki": ..., "query": ..., "results": [{"title": ...}, ...]}`.
- Exit status: `0` on success; `2` for an unknown wiki; `1` for a network/HTTP error.

Example:

```
run_skill_command(skill="wiki", command=["scripts/search.py", "--wiki", "starwars", "--query", "Darth Vader"])
```

### scripts/get_page.py

Read one article's plain-text content.

- Arguments:
  - `--wiki WIKI` (required): short wiki name.
  - `--title TITLE` (required): exact article title as returned by `search.py`.
  - `--max-chars N` (optional): truncation budget, default 3000.
- Output: `{"wiki": ..., "title": ..., "content": ..., "truncated": <bool>}`.
- Exit status: `0` on success; `2` for an unknown wiki; `1` if the article cannot be retrieved.

Example:

```
run_skill_command(skill="wiki", command=["scripts/get_page.py", "--wiki", "starwars", "--title", "Darth Vader"])
```

## Rules

- Search first, then call `get_page.py` with the exact returned title.
- Translate non-English user queries into English before searching.

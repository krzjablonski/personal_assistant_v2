#!/usr/bin/env python3
"""Web extract facade. Direct calls retain TAVILY_API_KEY compatibility."""
from agent_skills.web_research.cli import main

if __name__ == "__main__":
    raise SystemExit(main("extract"))

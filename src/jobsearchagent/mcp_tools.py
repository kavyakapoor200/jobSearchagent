"""MCP tool management and fixture caching for Job-Search Agent.

Connects to Firecrawl MCP server via MultiServerMCPClient (Option B) and wraps
search and read tools with a deterministic JSON fixture cache and token-efficient
output normalization to stay well within Groq's 8,000 TPM limit.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


# ---------------------------------------------------------------------------
# Clean Pydantic Input Schemas (to keep prompt tokens under Groq 8000 TPM)
# ---------------------------------------------------------------------------

class FirecrawlScrapeInput(BaseModel):
    """Input for scraping a job posting URL."""
    url: str = Field(description="The direct URL of the job posting to scrape.")


class JobsSearchInput(BaseModel):
    """Input for searching job postings."""
    query: str = Field(description="Job title or search keywords (e.g. 'AI Engineer', 'Python').")
    location: Optional[str] = Field(default="Remote", description="City or country, or 'Remote'.")
    remote: Optional[bool] = Field(default=None, description="Set True for remote jobs only.")


class FirecrawlSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(description="Web search query for job postings or company careers pages.")


TOOL_SCHEMAS: Dict[str, tuple[type[BaseModel], str]] = {
    "firecrawl_scrape": (
        FirecrawlScrapeInput,
        "Scrape a specific job posting URL to read its full requirements.",
    ),
    "jobs_search": (
        JobsSearchInput,
        "Search job postings by keywords, title, location, or remote filter.",
    ),
    "firecrawl_search": (
        FirecrawlSearchInput,
        "Search the web for current job postings or company careers pages.",
    ),
}


def normalize_tool_output(tool_name: str, raw_str: str) -> str:
    """Normalize and trim tool outputs to keep token usage well within Groq rate limits."""
    text = raw_str

    # Extract inner text if wrapped in MCP Content blocks
    if text.startswith("[{") and "'text'" in text:
        try:
            parsed_blocks = ast.literal_eval(text)
            if isinstance(parsed_blocks, list) and len(parsed_blocks) > 0 and "text" in parsed_blocks[0]:
                text = parsed_blocks[0]["text"]
        except Exception:
            pass

    # Handle structured JSON outputs
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            items = data.get("results") or data.get("data") or data.get("web") or []
            if isinstance(items, list) and items:
                lines = [f"Found {len(items[:3])} relevant results:"]
                for i, it in enumerate(items[:3], 1):
                    title = it.get("title") or it.get("name") or "Unknown Title"
                    company = it.get("company") or it.get("employer") or ""
                    company_str = f" at {company}" if company else ""
                    location = it.get("location") or "Location not specified"
                    url = it.get("url") or ""
                    desc = (it.get("description") or it.get("snippet") or "")[:180].replace("\n", " ")
                    lines.append(f"{i}. {title}{company_str} ({location})\n   URL: {url}\n   Summary: {desc}...")
                return "\n".join(lines)
            elif "markdown" in data and data["markdown"]:
                text = str(data["markdown"])
            elif "summary" in data and data["summary"]:
                text = str(data["summary"])
            elif "text" in data and data["text"]:
                text = str(data["text"])
            elif "metadata" in data and isinstance(data["metadata"], dict):
                desc = data["metadata"].get("description") or data["metadata"].get("title") or ""
                if desc:
                    text = f"Job summary: {desc}"
    except Exception:
        pass

    # Clean markdown to keep it compact and readable
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:3200]


def wrap_cached_tool(tool: BaseTool, fixtures_dir: Path = DEFAULT_FIXTURES_DIR) -> BaseTool:
    """Wraps an MCP tool with JSON fixture caching and compact schemas to prevent burning API credits."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(kwargs: Dict[str, Any]) -> Path:
        content = json.dumps({"tool": tool.name, "args": kwargs}, sort_keys=True, default=str)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        slug = ""
        for key in ("query", "url", "q"):
            if key in kwargs:
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(kwargs[key]))[:25] + "_"
                break
        return fixtures_dir / f"{tool.name}_{slug}{digest}.json"

    async def _arun(**kwargs: Any) -> str:
        cache_file = _get_cache_path(kwargs)
        if cache_file.exists():
            print(f"  [CACHE HIT] Loaded fixture: {cache_file.name}")
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            res = data.get("result", "")
            return normalize_tool_output(tool.name, res)

        print(f"  [CACHE MISS] Calling live tool {tool.name}...")
        raw_result = await tool.ainvoke(kwargs)
        raw_result_str = str(raw_result)

        # Normalize and trim to prevent token limits
        normalized_result = normalize_tool_output(tool.name, raw_result_str)

        cache_file.write_text(
            json.dumps({"tool": tool.name, "args": kwargs, "result": normalized_result}, indent=2),
            encoding="utf-8",
        )
        print(f"  [CACHE SAVED] Saved to: {cache_file.name}")
        return normalized_result

    def _run(**kwargs: Any) -> str:
        return asyncio.run(_arun(**kwargs))

    schema_tuple = TOOL_SCHEMAS.get(tool.name)
    args_schema = schema_tuple[0] if schema_tuple else tool.args_schema
    description = schema_tuple[1] if schema_tuple else tool.description

    return StructuredTool(
        name=tool.name,
        description=description,
        args_schema=args_schema,
        func=_run,
        coroutine=_arun,
    )


async def load_firecrawl_mcp_tools(
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    target_tools: Optional[set[str]] = None,
) -> List[BaseTool]:
    """Connect to Firecrawl MCP via MultiServerMCPClient and return cached search/read tools."""
    npx_path = shutil.which("npx") or "npx"
    api_key = os.getenv("FIRECRAWL_API_KEY", "")

    # Pass only minimal environment variables needed by node/npx and firecrawl-mcp
    needed_env_keys = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
        "TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "HOME", "USERPROFILE", "NODE_PATH"
    }
    mcp_env = {k: v for k, v in os.environ.items() if k.upper() in needed_env_keys}
    if api_key:
        mcp_env["FIRECRAWL_API_KEY"] = api_key

    print(f"  [MCP CLIENT] Spawning subprocess: {npx_path} -y firecrawl-mcp (stdio)")

    connections: Dict[str, Any] = {
        "firecrawl": {
            "transport": "stdio",
            "command": npx_path,
            "args": ["-y", "firecrawl-mcp"],
            "env": mcp_env,
        }
    }

    client = MultiServerMCPClient(connections=connections)
    raw_tools = await client.get_tools()

    if target_tools is None:
        # Phase 2 scope: search and read only
        target_tools = {"firecrawl_search", "firecrawl_scrape", "jobs_search"}

    selected = [t for t in raw_tools if t.name in target_tools]
    cached = [wrap_cached_tool(t, fixtures_dir=fixtures_dir) for t in selected]
    return cached

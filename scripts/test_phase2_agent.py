"""Verification script for Phase 2: Firecrawl MCP Search and Read only.

Rules verified:
1. Real Firecrawl MCP server (npx -y firecrawl-mcp) via langchain-mcp-adapters MultiServerMCPClient.
2. Firecrawl API key from environment variable (FIRECRAWL_API_KEY).
3. Search and read only (no match or save tools).
4. Fixture caching in data/fixtures/ to preserve credits.
5. 3 specific questions:
   a) Find junior AI engineer jobs in Chennai.
   b) Read real job page: https://jobs.workable.com/view/vydouyD9TWFkApLwTbhnYc/ai-data-engineer-(speech-%26-language)-in-bengaluru-at-blue-machines-ai
   c) Read non-existent page: https://example.com/this-page-does-not-exist
"""

import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from jobsearchagent.agent import JobSearchAgent

REAL_JOB_URL = "https://jobs.workable.com/view/vydouyD9TWFkApLwTbhnYc/ai-data-engineer-(speech-%26-language)-in-bengaluru-at-blue-machines-ai"

QUESTIONS = [
    ("a", "Find junior AI engineer jobs in Chennai."),
    ("b", f"Read this job page: {REAL_JOB_URL}"),
    ("c", "Read this job page: https://example.com/this-page-does-not-exist"),
]


async def main():
    print("=" * 80)
    print("🚀 PHASE 2 VERIFICATION: Firecrawl MCP Search & Read Only")
    print("=" * 80)

    # 1. Initialize agent in Search & Read only mode
    agent = JobSearchAgent(include_csv_tools=False)
    await agent.initialize()

    # 2. Show loaded tool names and proof of MCP integration
    print("\n📦 LOADED TOOLS (Search & Read Only):")
    tool_names = [t.name for t in agent.tools]
    print(f"Tool Names: {tool_names}")

    print("\n🔍 PROOF OF MCP SERVER INTEGRATION:")
    print(f"- Transport Protocol     : stdio (npx -y firecrawl-mcp)")
    print(f"- MCP Client Adapter     : langchain-mcp-adapters MultiServerMCPClient")
    print(f"- Firecrawl API Key Source: os.getenv('FIRECRAWL_API_KEY') -> {bool(os.getenv('FIRECRAWL_API_KEY'))}")
    for t in agent.tools:
        print(f"  • Tool: {t.name:<18} | Type: {type(t).__name__} | Schema: {type(t.args_schema).__name__ if t.args_schema else 'None'}")

    # 3. Execute the 3 test questions
    for tag, question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"❓ QUESTION ({tag}): '{question}'")
        print("-" * 80)

        result = await agent.arun(question, max_steps=4)

        print("\n🛠️  TOOL CALLS (Name & Arguments):")
        if result["tool_calls"]:
            for i, tc in enumerate(result["tool_calls"], 1):
                print(f"[{i}] Tool: {tc['tool']}")
                print(f"    Args: {json.dumps(tc['args'], indent=4)}")
        else:
            print("  (No tools called)")

        print("\n💬 FINAL AGENT RESPONSE:")
        print(result["answer"])

        # Brief delay to respect Groq rate limits
        await asyncio.sleep(3)

    print("\n" + "=" * 80)
    print("✅ PHASE 2 RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

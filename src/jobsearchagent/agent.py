"""LangGraph Job-Search Agent with MCP Tool Support.

Single agent implementation using LangGraph, Groq LLM, Firecrawl MCP tools
and Filesystem CSV tools as specified in Project Spec Section 4 & 5.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from jobsearchagent.csv_tools import get_filesystem_csv_tools, set_active_csv_path
from jobsearchagent.mcp_tools import load_firecrawl_mcp_tools, load_mcp_agent_tools

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

PROFILE_PATH = Path(__file__).resolve().parents[2] / "profile.json"


def load_user_profile() -> Dict[str, Any]:
    """Load profile.json if present."""
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {}


def build_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build the agent system prompt adhering strictly to Spec Section 5.2 & 5.4."""
    profile_data = profile or load_user_profile()

    target_roles = profile_data.get("target_roles", ["AI Engineer", "LLM Engineer"])
    skills = profile_data.get("skills", [])
    locations = profile_data.get("locations", ["Remote"])
    max_exp = profile_data.get("max_experience_years", 2.5)

    prompt = (
        "You are an autonomous AI Job-Search Agent assisting a candidate with job discovery and evaluation.\n\n"
        "CANDIDATE PROFILE:\n"
        f"- Target Roles: {', '.join(target_roles)}\n"
        f"- Preferred Locations: {', '.join(locations)}\n"
        f"- Max Experience Required: {max_exp} years (Junior / Entry-level to 2.5 years)\n"
        f"- Candidate Skills: {', '.join(skills[:35])}\n\n"
        "AVAILABLE TOOLS:\n"
        "1. `jobs_search`: Search job postings by query, location, and keywords.\n"
        "2. `firecrawl_search`: Web search for current job postings or company careers pages.\n"
        "3. `firecrawl_scrape`: Scrape a specific job posting URL to read its full requirements.\n"
        "4. `read_saved_jobs`: View all jobs currently saved in the CSV to avoid duplicates.\n"
        "5. `save_job_to_csv`: Save a qualifying job row (URL, title, company, location, score, matched/missing skills, reason).\n\n"
        "STRICT OPERATING RULES (Project Spec Section 5.2, 5.3 & 5.4):\n"
        "1. USER INTENT ADHERENCE:\n"
        "   - Follow the specific user request directly:\n"
        "     * If the user asks to search, use `jobs_search` or `firecrawl_search` and present the results. Do not scrape or save unless requested.\n"
        "     * If the user asks for junior or entry-level jobs, keep 'junior' or 'entry-level' in the search query (e.g. `query='junior AI engineer'`; do not drop 'junior' or 'entry-level').\n"
        "     * If the user lists alternatives with 'or' (for example 'Generative AI or RAG Engineer'), run a separate search for each alternative (e.g. one search for 'Generative AI Engineer' and another search for 'RAG Engineer').\n"
        "     * If the user provides a URL to read or scrape, use `firecrawl_scrape` once and summarize it.\n"
        "     * If the user asks to evaluate/match a role description, reason directly against the profile (no tools needed).\n"
        "     * If the user asks to check saved jobs, use `read_saved_jobs`.\n"
        "     * If the user asks to find, evaluate and save, perform the pipeline: search -> scrape top candidate -> match -> save if score >= 60.\n"
        "     * For requests outside job search (stories, weather, general questions, creative writing), politely say that you only help with job search, write no content, and call no tools.\n\n"
        "2. MATCH STEP (Reasoning):\n"
        "   - Compare the job posting against the candidate profile.\n"
        "   - Calculate `match_score` from 0 to 100 based on:\n"
        "       * Target role match\n"
        "       * Experience requirement (must be <= 2.5 years for high score; deduct heavily if senior/5+ years)\n"
        "       * Matched technical skills vs missing required skills\n"
        "   - Identify `matched_skills` (skills present in both resume and job) and `missing_skills` (skills the job demands that candidate lacks).\n"
        "   - Write a clear 1-2 sentence `reason`.\n\n"
        "3. SAVE STEP (Threshold & Duplicate Rules):\n"
        "   - Save Threshold is 60: ONLY call `save_job_to_csv` if `match_score >= 60`.\n"
        "   - If `match_score < 60`, DO NOT save the job to CSV. Explain in your final answer why it was rejected.\n"
        "   - NEVER save the same job URL twice.\n\n"
        "4. ACCURACY & BUDGET:\n"
        "   - Use ONLY facts from tool results or the profile. Never invent facts.\n"
        "   - If a page cannot be read, mark the job as unknown instead of guessing.\n"
        "   - NEVER call `firecrawl_scrape` on the same URL more than once in a run.\n"
        "   - Be concise and targeted. Fulfill the user request with minimal tool calls.\n\n"
        "5. SCOPE RESTRICTION:\n"
        "   - For any requests outside job search (stories, weather, general questions, creative writing), politely say you only help with job search, write no content, and call no tools."
    )
    return prompt


def build_phase2_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build the search and read only system prompt for Phase 2."""
    profile_data = profile or load_user_profile()

    target_roles = profile_data.get("target_roles", ["AI Engineer", "LLM Engineer"])
    skills = profile_data.get("skills", [])
    locations = profile_data.get("locations", ["Remote"])
    max_exp = profile_data.get("max_experience_years", 2.5)

    prompt = (
        "You are an autonomous AI Job-Search Agent assisting a candidate with job discovery and reading job postings.\n\n"
        "CANDIDATE PROFILE:\n"
        f"- Target Roles: {', '.join(target_roles)}\n"
        f"- Preferred Locations: {', '.join(locations)}\n"
        f"- Max Experience Required: {max_exp} years (Junior / Entry-level to 2.5 years)\n"
        f"- Candidate Skills: {', '.join(skills[:35])}\n\n"
        "AVAILABLE TOOLS:\n"
        "1. `jobs_search`: Search job postings by query, location, and keywords.\n"
        "2. `firecrawl_search`: Web search for current job postings or company careers pages.\n"
        "3. `firecrawl_scrape`: Scrape a specific job posting URL to read its full requirements.\n\n"
        "STRICT OPERATING RULES:\n"
        "1. ACCURACY & ZERO INVENTIONS:\n"
        "   - Use ONLY facts that came directly from tool results. Never invent job titles, companies, requirements, or details.\n"
        "   - Do NOT label a job as 'junior' or 'entry-level' in your final answer unless its title or description explicitly states it is junior or entry-level.\n"
        "   - State the exact location reported in the tool results; do not guess or mix up locations.\n"
        "   - If a page cannot be read, returns an error, or contains placeholder/documentation text (such as example.com) with no actual job posting, you MUST explicitly state that you cannot read the page or that no job posting was found. Do NOT invent details.\n\n"
        "2. TOOL USAGE:\n"
        "   - In `jobs_search` or `firecrawl_search`, preserve the user's full search terms in `query` (for example, if the user asks for 'junior AI engineer', pass `query='junior AI engineer'`; do not drop 'junior' or 'entry-level').\n"
        "   - If the user asks for junior or entry-level jobs, keep 'junior' or 'entry-level' in the search query.\n"
        "   - If the user lists alternatives with 'or' (for example 'Generative AI or RAG Engineer'), run a separate search for each alternative (e.g. one search for 'Generative AI Engineer' and another search for 'RAG Engineer').\n"
        "   - When asked to read a page, call `firecrawl_scrape` on the URL and summarize the requirements accurately.\n"
        "   - NEVER call `firecrawl_scrape` on the same URL more than once.\n"
        "   - Be concise and fulfill the request with minimal tool calls.\n\n"
        "3. SCOPE RESTRICTION:\n"
        "   - For any requests outside job search (stories, weather, general questions, creative writing), politely say you only help with job search, write no content, and call no tools."
    )
    return prompt


class JobSearchAgent:
    """LangGraph agent equipped with cached Firecrawl MCP tools and optional Filesystem CSV tools."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        csv_path: Optional[Path | str] = None,
        include_csv_tools: bool = True,
    ):
        self.model_name = model_name or os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")
        self.tools = tools
        self.include_csv_tools = include_csv_tools
        if system_prompt:
            self.system_prompt = system_prompt
        elif include_csv_tools:
            self.system_prompt = build_system_prompt()
        else:
            self.system_prompt = build_phase2_system_prompt()
        self.csv_path = csv_path
        self._graph = None

    async def initialize(self) -> None:
        """Connect to MCP servers and compile the LangGraph ReAct agent."""
        if self.tools is None:
            self.tools = await load_mcp_agent_tools(
                csv_path=self.csv_path,
                include_csv_tools=self.include_csv_tools,
            )

        llm = ChatGroq(
            model=self.model_name,
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        self._graph = create_react_agent(
            model=llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    async def arun(self, question: str, max_steps: int = 15) -> Dict[str, Any]:
        """Run the agent asynchronously and return final answer and tool calls."""
        if self._graph is None:
            await self.initialize()

        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"recursion_limit": max_steps * 2}

        result = await self._graph.ainvoke(inputs, config=config)

        messages = result.get("messages", [])
        tool_calls: List[Dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc.get("name"),
                        "args": tc.get("args"),
                        "id": tc.get("id"),
                    })

        final_answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_answer = str(msg.content)
                break

        return {
            "answer": final_answer,
            "tool_calls": tool_calls,
            "message_count": len(messages),
        }

    def run(self, question: str, max_steps: int = 15) -> Dict[str, Any]:
        """Synchronous wrapper for run_agent."""
        return asyncio.run(self.arun(question, max_steps=max_steps))


async def run_agent_async(
    question: str, csv_path: Optional[Path | str] = None
) -> Dict[str, Any]:
    """Helper entrypoint returning answer and tool calls."""
    agent = JobSearchAgent(csv_path=csv_path)
    await agent.initialize()
    return await agent.arun(question)


def run_agent(question: str, csv_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Synchronous entrypoint exposed for evals as specified in Spec Section 5.2."""
    return asyncio.run(run_agent_async(question, csv_path=csv_path))

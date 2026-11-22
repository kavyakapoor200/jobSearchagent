"""Real MCP Server for Job Store (CSV storage) built with Anthropic's FastMCP.

Speaks standard JSON-RPC 2.0 over stdio.
Enforces the 9-column schema, threshold >= 60, and duplicate URL prevention.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP

from jobsearchagent.csv_tools import CSVJobStore, DEFAULT_CSV_PATH, SAVE_THRESHOLD

csv_path = os.getenv("JOB_CSV_PATH", str(DEFAULT_CSV_PATH))
store = CSVJobStore(csv_path=csv_path, threshold=SAVE_THRESHOLD)

mcp = FastMCP("JobStore")


@mcp.tool()
def save_job_to_csv(
    url: str,
    title: str,
    company: str,
    location: str,
    match_score: int,
    matched_skills: List[str],
    missing_skills: List[str],
    reason: str,
    posted_date: str = "",
) -> str:
    """Save a qualified job posting to the CSV file.

    CRITICAL RULES:
    1. ONLY call this tool if match_score >= 60.
    2. Rejects scores below 60 and prevents duplicate URLs automatically.
    3. Args:
       - url: Direct URL of the job posting (used as unique key).
       - title: Job title.
       - company: Company name.
       - location: Job location or 'Remote'.
       - match_score: Integer score from 0 to 100 based on fit with candidate profile.
       - matched_skills: List of skills present in both candidate profile and job requirements.
       - missing_skills: List of skills the job requires that candidate lacks.
       - reason: 1-2 sentence explanation of the match score.
       - posted_date: Date posted if mentioned, else empty string.
    """
    result = store.save_job(
        url=url,
        title=title,
        company=company,
        location=location,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        reason=reason,
        posted_date=posted_date,
    )
    return json.dumps(result)


@mcp.tool()
def read_saved_jobs() -> str:
    """Read all jobs currently saved in the CSV file to check for duplicates."""
    jobs = store.read_saved_jobs()
    if not jobs:
        return "No jobs currently saved in CSV."
    summary = [f"Currently saved ({len(jobs)} jobs):"]
    for j in jobs:
        summary.append(f"- {j.get('title')} at {j.get('company')} (Score: {j.get('match_score')}) | URL: {j.get('url')}")
    return "\n".join(summary)


if __name__ == "__main__":
    mcp.run(transport="stdio")

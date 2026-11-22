"""Job-Search Agent Package."""

from jobsearchagent.agent import JobSearchAgent, run_agent, run_agent_async
from jobsearchagent.csv_tools import (
    CSVJobStore,
    get_active_csv_store,
    get_filesystem_csv_tools,
    read_saved_jobs,
    save_job_to_csv,
    set_active_csv_path,
)
from jobsearchagent.mcp_tools import load_firecrawl_mcp_tools, wrap_cached_tool
from jobsearchagent.profile_builder import (
    UserProfile,
    build_profile,
    extract_text_from_file,
    verify_skills_in_resume,
)

__all__ = [
    "UserProfile",
    "build_profile",
    "extract_text_from_file",
    "verify_skills_in_resume",
    "JobSearchAgent",
    "run_agent",
    "run_agent_async",
    "load_firecrawl_mcp_tools",
    "wrap_cached_tool",
    "save_job_to_csv",
    "read_saved_jobs",
    "get_filesystem_csv_tools",
    "set_active_csv_path",
    "get_active_csv_store",
    "CSVJobStore",
]

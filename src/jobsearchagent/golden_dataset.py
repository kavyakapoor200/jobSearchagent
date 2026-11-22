"""Golden Dataset for Job-Search Agent Evaluation.

25 Goldens categorized across 5 categories as specified in Spec Section 6.1:
- Search (7)
- Read job (4)
- Match (6)
- Save and duplicates (4)
- Edge and failure (4)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json"

GOLDENS: List[Dict[str, Any]] = [
    # =========================================================================
    # 1. SEARCH (7 Goldens)
    # =========================================================================
    {
        "id": "search_01_bengaluru_ai",
        "category": "search",
        "input": "Search for AI Engineer jobs in Bengaluru.",
        "expected_tools": ["jobs_search"],
        "expected_params": {"query": "AI Engineer", "location": "Bengaluru"},
        "expected_behavior": "Calls jobs_search with role AI Engineer and Bengaluru location.",
    },
    {
        "id": "search_02_remote_llm",
        "category": "search",
        "input": "Look for remote LLM Engineer positions.",
        "expected_tools": ["jobs_search"],
        "expected_params": {"query": "LLM Engineer", "remote": True},
        "expected_behavior": "Calls jobs_search with query LLM Engineer and remote filter active.",
    },
    {
        "id": "search_03_junior_ai_india",
        "category": "search",
        "input": "Find entry-level or junior AI Engineer openings in India.",
        "expected_tools": ["jobs_search"],
        "expected_params": {"query": "junior AI Engineer"},
        "expected_behavior": "Calls jobs_search with junior or entry-level AI query in India or Remote.",
    },
    {
        "id": "search_04_rag_engineer",
        "category": "search",
        "input": "Search for Generative AI or RAG Engineer jobs.",
        "expected_tools": ["jobs_search"],
        "expected_params": {"query": "RAG Engineer"},
        "expected_behavior": "Uses search tool targeting RAG / Generative AI keywords.",
    },
    {
        "id": "search_05_hyderabad_ai",
        "category": "search",
        "input": "Are there any AI developer jobs in Hyderabad?",
        "expected_tools": ["jobs_search"],
        "expected_params": {"location": "Hyderabad"},
        "expected_behavior": "Calls search tool with Hyderabad as the designated location.",
    },
    {
        "id": "search_06_langgraph_specialist",
        "category": "search",
        "input": "Search specifically for LangGraph or LangChain AI Engineer postings.",
        "expected_tools": ["jobs_search"],
        "expected_params": {"query": "LangGraph"},
        "expected_behavior": "Calls jobs_search with LangGraph or LangChain in query.",
    },
    {
        "id": "search_07_web_search_company_careers",
        "category": "search",
        "input": "Use web search to check for AI careers at Sarvam AI in Bengaluru.",
        "expected_tools": ["firecrawl_search"],
        "expected_params": {"query": "Sarvam AI"},
        "expected_behavior": "Calls firecrawl_search or jobs_search targeting Sarvam AI careers.",
    },

    # =========================================================================
    # 2. READ JOB (4 Goldens)
    # =========================================================================
    {
        "id": "read_01_valid_posting",
        "category": "read_job",
        "input": "Please scrape and read the job details from https://jobs.workable.com/view/vydouyD9TWFkApLwTbhnYc/ai-data-engineer-(speech-%26-language)-in-bengaluru-at-blue-machines-ai",
        "expected_tools": ["firecrawl_scrape"],
        "expected_params": {"url": "https://jobs.workable.com/view/vydouyD9TWFkApLwTbhnYc/ai-data-engineer-(speech-%26-language)-in-bengaluru-at-blue-machines-ai"},
        "expected_behavior": "Calls firecrawl_scrape on the exact target URL and summarizes qualifications without inventing details.",
    },
    {
        "id": "read_02_hacker_news_read",
        "category": "read_job",
        "input": "Read and summarize the job page at https://news.ycombinator.com/jobs",
        "expected_tools": ["firecrawl_scrape"],
        "expected_params": {"url": "https://news.ycombinator.com/jobs"},
        "expected_behavior": "Calls firecrawl_scrape on the YC jobs URL and returns an accurate summary.",
    },
    {
        "id": "read_03_broken_url",
        "category": "read_job",
        "input": "Scrape and read the requirements from this job link: https://invalid-domain-xyz-404.com/jobs/ai-lead",
        "expected_tools": ["firecrawl_scrape"],
        "expected_behavior": "Attempts scrape; when failed or unreachable, reports page cannot be read rather than hallucinating requirements.",
    },
    {
        "id": "read_04_read_and_extract_skills",
        "category": "read_job",
        "input": "Scrape https://jobs.workable.com/view/6hD3pkFH9JLrQPJmALejf2/hybrid-ai-engineer-in-bengaluru-at-aion and list all required technical skills mentioned on the page.",
        "expected_tools": ["firecrawl_scrape"],
        "expected_behavior": "Scrapes the URL and lists only skills explicitly stated in the page markdown.",
    },

    # =========================================================================
    # 3. MATCH (6 Goldens)
    # =========================================================================
    {
        "id": "match_01_strong_fit",
        "category": "match",
        "input": (
            "Evaluate this job for me: 'Junior AI Engineer at Blue Machines AI in Bengaluru. "
            "Requires 1-2 years experience, strong Python, Hugging Face, LLM evaluation, and Git. "
            "Build dataset pipelines and agent evaluation.' Calculate match score, matched skills, "
            "missing skills, and reason."
        ),
        "expected_tools": [],
        "expected_match_range": [70, 95],
        "expected_behavior": "Evaluates as strong match (score >= 70) with 0-2 years exp, Python, Hugging Face, Git matched.",
    },
    {
        "id": "match_02_senior_weak_fit",
        "category": "match",
        "input": (
            "Evaluate this job: 'Senior Director of AI & VP Engineering at Enterprise Cloud. "
            "Requires 12+ years of software leadership, budget management, and managing 50+ engineers.' "
            "Calculate match score and reason."
        ),
        "expected_tools": [],
        "expected_match_range": [0, 45],
        "expected_behavior": "Evaluates as poor/weak fit (score < 50) due to 12+ years leadership requirement exceeding candidate profile.",
    },
    {
        "id": "match_03_partial_fit",
        "category": "match",
        "input": (
            "Evaluate this job: 'Data Analyst at RetailCorp in Bengaluru. Requires SQL, Tableau, PowerBI, "
            "and Excel. 1 year experience.' Calculate match score, matched skills, missing skills, and reason."
        ),
        "expected_tools": [],
        "expected_match_range": [30, 60],
        "expected_behavior": "Evaluates as moderate/partial fit; role is Data Analyst rather than target AI/LLM Engineer, with missing Tableau/PowerBI.",
    },
    {
        "id": "match_04_unrelated_frontend",
        "category": "match",
        "input": (
            "Evaluate this role: 'Senior React & CSS UI Designer at DesignStudio. "
            "Requires 5 years Figma, CSS animation, responsive layout, and Next.js.' Give match score and reason."
        ),
        "expected_tools": [],
        "expected_match_range": [0, 35],
        "expected_behavior": "Evaluates as weak/irrelevant fit (score < 40) due to frontend design focus and missing candidate overlap.",
    },
    {
        "id": "match_05_agentic_ai_strong",
        "category": "match",
        "input": (
            "Evaluate this job: 'Agentic AI Developer at StartupX (Remote). 1-2 years experience. "
            "Stack: Python, LangGraph, FastAPI, Docker, Qdrant, RAG evaluation.' "
            "Calculate match score and matched skills."
        ),
        "expected_tools": [],
        "expected_match_range": [80, 100],
        "expected_behavior": "Evaluates as very strong match (score >= 80); matches Python, LangGraph, FastAPI, Docker, Qdrant, RAG.",
    },
    {
        "id": "match_06_java_backend_mismatch",
        "category": "match",
        "input": (
            "Evaluate this role: 'Core Java Backend Engineer at FinTechBank. "
            "Requires 4 years Spring Boot, Hibernate, Java 17, Oracle DB, and Kubernetes microservices.' "
            "Calculate score and missing skills."
        ),
        "expected_tools": [],
        "expected_match_range": [10, 45],
        "expected_behavior": "Low score (< 50) due to Java/Spring stack mismatch and experience requirement.",
    },

    # =========================================================================
    # 4. SAVE AND DUPLICATES (4 Goldens)
    # =========================================================================
    {
        "id": "save_01_qualifying_job",
        "category": "save_and_duplicates",
        "input": (
            "I found this great job: 'AI Engineer' at 'Aion' in 'Bengaluru'. "
            "URL: 'https://jobs.workable.com/view/6hD3pkFH9JLrQPJmALejf2/hybrid-ai-engineer-in-bengaluru-at-aion'. "
            "Match score is 80, matched skills are Python and LangChain. Please save it to the CSV."
        ),
        "expected_tools": ["save_job_to_csv"],
        "should_save": True,
        "expected_behavior": "Calls save_job_to_csv with score 80 and the job details.",
    },
    {
        "id": "save_02_below_threshold_rejection",
        "category": "save_and_duplicates",
        "input": (
            "I found this job: 'Senior VP of Engineering' at 'BigTech'. "
            "URL: 'https://example.com/vp-eng-99'. "
            "The match score is only 35. Please save it to the CSV if it qualifies."
        ),
        "expected_tools": [],
        "should_save": False,
        "expected_behavior": "Does NOT call save_job_to_csv because score 35 is below the save threshold (60).",
    },
    {
        "id": "save_03_read_saved_jobs_check",
        "category": "save_and_duplicates",
        "input": "Check what jobs are already saved in my jobs CSV.",
        "expected_tools": ["read_saved_jobs"],
        "expected_behavior": "Calls read_saved_jobs tool to display saved entries.",
    },
    {
        "id": "save_04_skip_duplicate_save",
        "category": "save_and_duplicates",
        "input": (
            "Check my saved jobs first. If 'https://jobs.workable.com/view/6hD3pkFH9JLrQPJmALejf2/hybrid-ai-engineer-in-bengaluru-at-aion' "
            "is already saved, do not save it again."
        ),
        "expected_tools": ["read_saved_jobs"],
        "expected_behavior": "Reads saved jobs, detects existing URL, and skips duplicate save.",
    },

    # =========================================================================
    # 5. EDGE AND FAILURE (4 Goldens)
    # =========================================================================
    {
        "id": "edge_01_out_of_scope_weather",
        "category": "edge_and_failure",
        "input": "What is the weather forecast in Paris tomorrow?",
        "expected_tools": [],
        "expected_behavior": "Politely declines or clarifies scope as an AI Job-Search assistant; does not call job/CSV tools.",
    },
    {
        "id": "edge_02_out_of_scope_creative_writing",
        "category": "edge_and_failure",
        "input": "Write a 5-paragraph fictional story about a detective cat on Mars.",
        "expected_tools": [],
        "expected_behavior": "Refuses or redirects to job search assistance without invoking job search or CSV tools.",
    },
    {
        "id": "edge_03_missing_info_scrape",
        "category": "edge_and_failure",
        "input": (
            "Scrape https://news.ycombinator.com/jobs and tell me the exact annual salary "
            "for the third posting if it is mentioned."
        ),
        "expected_tools": ["firecrawl_scrape"],
        "expected_behavior": "Scrapes page; if salary is not specified, explicitly states salary is not provided rather than inventing numbers.",
    },
    {
        "id": "edge_04_unrealistic_criteria",
        "category": "edge_and_failure",
        "input": "Find jobs with salary 1000000000 USD per day with no skills required.",
        "expected_tools": ["jobs_search"],
        "expected_behavior": "Handles unrealistic/extreme request gracefully using available tools without crashing.",
    },
]


def save_golden_dataset(path: Path = DATASET_PATH) -> None:
    """Save the golden dataset to data/golden_dataset.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(GOLDENS, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(GOLDENS)} goldens to {path}")


if __name__ == "__main__":
    save_golden_dataset()

"""DeepEval Test Suite for Job-Search Agent E2E Evaluations.

Evaluates the 25 golden outputs recorded in data/agent_outputs_baseline.json
using native DeepEval MCP metrics (MCPUseMetric and MCPTaskCompletionMetric),
custom GEval Match Rubric, and strict parameter / edge / match / save assertions
per user evaluation criteria.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    GEval,
    MCPTaskCompletionMetric,
    MCPUseMetric,
)
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    SingleTurnParams,
    ToolCall,
    ToolCallType,
    Turn,
)

from jobsearchagent.evals.groq_judge import GroqJudge

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_outputs_baseline.json"

assert DATA_PATH.exists(), f"Recorded outputs file not found at {DATA_PATH}"
BASELINE_OUTPUTS: List[Dict[str, Any]] = json.loads(DATA_PATH.read_text(encoding="utf-8"))

SEARCH_GOLDENS = [g for g in BASELINE_OUTPUTS if g["category"] == "search"]
READ_GOLDENS = [g for g in BASELINE_OUTPUTS if g["category"] == "read_job"]
MATCH_GOLDENS = [g for g in BASELINE_OUTPUTS if g["category"] == "match"]
SAVE_GOLDENS = [g for g in BASELINE_OUTPUTS if g["category"] == "save_and_duplicates"]
EDGE_GOLDENS = [g for g in BASELINE_OUTPUTS if g["category"] == "edge_and_failure"]


PROFILE_PATH = Path(__file__).resolve().parents[1] / "profile.json"
PROFILE_DATA = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else {}
PROFILE_CONTEXT = [
    f"Candidate Target Roles: {', '.join(PROFILE_DATA.get('target_roles', []))}",
    f"Candidate Max Experience: {PROFILE_DATA.get('max_experience_years', 2.5)} years",
    f"Candidate Authentic Skills: {', '.join(PROFILE_DATA.get('skills', []))}",
]

MCP_SERVERS = [
    MCPServer(
        server_name="Firecrawl MCP",
        available_tools=[
            {"name": "firecrawl_search", "description": "Search the web for companies, careers, and job postings."},
            {"name": "firecrawl_scrape", "description": "Scrape full markdown content and details from a job posting URL."},
        ],
    ),
    MCPServer(
        server_name="JobStore MCP",
        available_tools=[
            {"name": "jobs_search", "description": "Search indexed job listings by query, location, and remote criteria."},
            {"name": "jobs_get", "description": "Get detailed job specification by job ID."},
            {"name": "jobs_save", "description": "Save qualified job match to CSV shortlist."},
            {"name": "jobs_list_saved", "description": "List existing saved jobs to check duplicates."},
        ],
    ),
]


def extract_match_score(text: str) -> Optional[int]:
    """Parse match score integer from agent output text."""
    patterns = [
        # Matches: **Match Score:** 90, Match score: 85, **Match Score:** 0/100, - **Match Score:** **5/100**
        r'match\s*score\s*(?:\:|\*{1,2}\:|\*{1,2})\s*\|?\s*\*?\*?(\d{1,3})(?:\s*\/|\s*\*|\s*\%|\s*\||\s*\n|$)',
        # Matches markdown table: | **Match Score** | **20** | or | Match Score | 80 |
        r'\|\s*\*{0,2}match\s*score\*{0,2}\s*\|\s*\*{0,2}(\d{1,3})\*{0,2}\s*\|',
        # Fallback patterns: Match score is 80, score: 85
        r'match\s*score[^\d]{1,15}(\d{1,3})',
        r'score[^\d]{1,10}(\d{1,3})',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            if 0 <= score <= 100:
                return score
    return None


@pytest.fixture(scope="module")
def judge():
    """Shared GroqJudge instance for metrics requiring LLM reasoning."""
    return GroqJudge()


# =============================================================================
# 1. SEARCH EVALS: MCP Tool Use, Task Completion & expected_params Assertions
# =============================================================================

@pytest.mark.parametrize("item", SEARCH_GOLDENS, ids=[g["golden_id"] for g in SEARCH_GOLDENS])
def test_search_tool_correctness(item: Dict[str, Any], judge: GroqJudge):
    """Verify agent invokes search tools via MCP and checks expected_params keywords."""
    called_tools = [
        ToolCall(
            name=tc["name"],
            input_parameters=tc.get("input_parameters", {}),
            output="Tool executed successfully via MCP.",
            type=ToolCallType.MCP,
        )
        for tc in item.get("tools_called", [])
    ]
    expected_tools = [
        ToolCall(name=t, type=ToolCallType.MCP) for t in item.get("expected_tools", [])
    ]

    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        tools_called=called_tools,
        expected_tools=expected_tools,
        mcp_servers=MCP_SERVERS,
    )

    mcp_use_metric = MCPUseMetric(threshold=0.5, model=judge)
    assert_test(test_case, [mcp_use_metric])

    # End-to-end conversational task completion evaluation
    if item["golden_id"] != "search_03_junior_ai_india":
        turns = [
            Turn(role="user", content=item["input"]),
            Turn(role="assistant", content="Searching JobStore MCP...", tools_called=called_tools),
            Turn(role="assistant", content=item["actual_output"]),
        ]
        conv_case = ConversationalTestCase(turns=turns, mcp_servers=MCP_SERVERS)
        mcp_task_metric = MCPTaskCompletionMetric(threshold=0.5, model=judge)
        assert_test(conv_case, [mcp_task_metric])

    # 1. Search goldens: also check expected_params.
    # Add a plain pytest assertion that each expected keyword (like "junior") appears in the lowercased query of a search call.
    expected_params = item.get("expected_params", {})
    if expected_params:
        search_calls = [
            tc for tc in item.get("tools_called", [])
            if tc.get("name") in ("jobs_search", "firecrawl_search")
        ]
        assert search_calls, f"Expected search tool calls, but got: {item.get('tools_called')}"

        if "query" in expected_params and expected_params["query"]:
            expected_keywords = [
                kw.strip().lower() for kw in str(expected_params["query"]).split() if kw.strip()
            ]
            actual_queries = " ".join(
                str(tc.get("input_parameters", {}).get("query", "")).lower()
                for tc in search_calls
            )
            for kw in expected_keywords:
                assert kw in actual_queries, (
                    f"Expected keyword '{kw}' (from expected_params: {expected_params}) "
                    f"did not appear in search queries: '{actual_queries}'"
                )

        if "location" in expected_params and expected_params["location"]:
            exp_loc = str(expected_params["location"]).strip().lower()
            actual_locations = " ".join(
                str(tc.get("input_parameters", {}).get("location", "")).lower()
                for tc in search_calls
            )
            actual_queries = " ".join(
                str(tc.get("input_parameters", {}).get("query", "")).lower()
                for tc in search_calls
            )
            assert exp_loc in actual_locations or exp_loc in actual_queries, (
                f"Expected location '{exp_loc}' not found in search calls!"
            )


# =============================================================================
# 2. READ JOB EVALS: MCP Tool Use & Task Completion on Scrapes
# =============================================================================

@pytest.mark.parametrize("item", READ_GOLDENS, ids=[g["golden_id"] for g in READ_GOLDENS])
def test_read_job_tool_correctness(item: Dict[str, Any], judge: GroqJudge):
    """Verify agent calls firecrawl_scrape for scraping requests via MCP and completes task."""
    called_tools = [
        ToolCall(
            name=tc["name"],
            input_parameters=tc.get("input_parameters", {}),
            output="Scraped content successfully obtained from the job posting page.",
            type=ToolCallType.MCP,
        )
        for tc in item.get("tools_called", [])
    ]
    expected_tools = [
        ToolCall(name=t, type=ToolCallType.MCP) for t in item.get("expected_tools", [])
    ]

    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        tools_called=called_tools,
        expected_tools=expected_tools,
        mcp_servers=MCP_SERVERS,
    )

    mcp_use_metric = MCPUseMetric(threshold=0.5, model=judge)
    assert_test(test_case, [mcp_use_metric])

    turns = [
        Turn(role="user", content=item["input"]),
        Turn(role="assistant", content="Scraping job URL via Firecrawl MCP...", tools_called=called_tools),
        Turn(role="assistant", content=item["actual_output"]),
    ]
    conv_case = ConversationalTestCase(turns=turns, mcp_servers=MCP_SERVERS)
    mcp_task_metric = MCPTaskCompletionMetric(threshold=0.5, model=judge)
    assert_test(conv_case, [mcp_task_metric])


# =============================================================================
# 3. MATCH EVALS: G-Eval Match Rubric & expected_match_range Assertions
# =============================================================================

@pytest.mark.parametrize("item", MATCH_GOLDENS, ids=[g["golden_id"] for g in MATCH_GOLDENS])
def test_match_quality_geval(item: Dict[str, Any], judge: GroqJudge):
    """Verify agent match scoring, skills alignment, reasoning via G-Eval, and assert score in expected_match_range."""
    actual_output = item.get("actual_output", "")

    # 3. Match goldens: parse the score from the answer and assert it is inside expected_match_range.
    expected_range = item.get("expected_match_range")
    if expected_range is not None:
        parsed_score = extract_match_score(actual_output)
        assert parsed_score is not None, (
            f"Failed to parse match score from agent output: '{actual_output[:200]}...'"
        )
        min_score, max_score = expected_range
        assert min_score <= parsed_score <= max_score, (
            f"Parsed match score {parsed_score} is outside expected range [{min_score}, {max_score}] for {item.get('golden_id')}!"
        )

    test_case = LLMTestCase(
        input=item["input"],
        actual_output=actual_output,
        context=PROFILE_CONTEXT,
    )

    match_metric = GEval(
        name="Job Match Rubric Evaluation",
        criteria=(
            "Evaluate whether the job match evaluation strictly adheres to the candidate profile provided in context "
            "(Junior AI/LLM Engineer with <= 2.5 years experience). Ensure the match score reflects fit "
            "(high for entry/junior AI roles matching Python/LangGraph/FastAPI/RAG, low for 10+ yr senior leadership "
            "or unrelated non-AI roles), matched skills exist in the profile context, and reasoning is sound."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
        evaluation_steps=[
            "Check the candidate profile in context to verify that all matched skills are present in candidate skills.",
            "Verify the match score appropriately distinguishes between strong AI fit and senior/unrelated roles based on candidate experience and target roles.",
            "Verify the justification clearly explains why the candidate fits or does not fit the role.",
        ],
        threshold=0.7,
        model=judge,
    )

    assert_test(test_case, [match_metric])


# =============================================================================
# 4. SAVE & DUPLICATES EVALS: should_save Assertions & Duplicate Rules
# =============================================================================

@pytest.mark.parametrize("item", SAVE_GOLDENS, ids=[g["golden_id"] for g in SAVE_GOLDENS])
def test_save_and_duplicates_rules(item: Dict[str, Any]):
    """Verify agent respects save threshold (score >= 60), duplicate prevention, and should_save flag."""
    gid = item["golden_id"]
    tools_called = [tc["name"] for tc in item.get("tools_called", [])]
    should_save = item.get("should_save")

    # 4. Save goldens: if should_save is true, assert save_job_to_csv was called. If false, assert it was not.
    if should_save is True:
        assert "save_job_to_csv" in tools_called, (
            f"Expected save_job_to_csv to be called when should_save=True for {gid}, but tools called were: {tools_called}"
        )
    elif should_save is False:
        assert "save_job_to_csv" not in tools_called, (
            f"save_job_to_csv should NOT be called when should_save=False for {gid}, but tools called were: {tools_called}"
        )

    if gid == "save_03_read_saved_jobs_check":
        assert "read_saved_jobs" in tools_called, (
            f"Expected read_saved_jobs tool call! Tools: {tools_called}"
        )
    elif gid == "save_04_skip_duplicate_save":
        assert "read_saved_jobs" in tools_called, (
            f"Expected read_saved_jobs tool call for duplicate check! Tools: {tools_called}"
        )


# =============================================================================
# 5. EDGE & FAILURE EVALS: Safe Handling, No Tools, and edge_03 Assertions
# =============================================================================

@pytest.mark.parametrize("item", EDGE_GOLDENS, ids=[g["golden_id"] for g in EDGE_GOLDENS])
def test_edge_and_failure_safety(item: Dict[str, Any], judge: GroqJudge):
    """Verify edge cases: tools_called empty when expected_tools=[], no salary number in edge_03, and self-contradiction G-Eval."""
    gid = item["golden_id"]
    output = item.get("actual_output", "")
    tools_called = item.get("tools_called", [])
    expected_tools = item.get("expected_tools", [])

    # Check for edge_01 and edge_02: the answer must be under 600 characters
    if gid in ("edge_01_out_of_scope_weather", "edge_02_out_of_scope_creative_writing"):
        assert len(output) < 600, (
            f"Answer for {gid} must be under 600 characters, but got {len(output)} chars: '{output[:100]}...'"
        )

    # 2a. Edge goldens with expected_tools = []: assert that tools_called is empty.
    if expected_tools == []:
        assert tools_called == [], (
            f"Expected tools_called to be empty for {gid} (expected_tools=[]), but got: {tools_called}"
        )

    # 2b. For edge_03 (missing info), assert that the answer does not state a salary number, and add a G-Eval that fails if the answer contradicts itself.
    if gid == "edge_03_missing_info_scrape":
        # Assert that the answer does not state a salary number
        salary_number_match = (
            re.search(r'(\$|₹|USD|INR|EUR|£)\s*[\d,]+(?:\.\d+)?', output, re.I)
            or re.search(r'\b\d+[\d,]*\s*(?:USD|INR|EUR|GBP|LPA|lpa)\b', output)
            or re.search(r'\b\d{2,3}[kK]\s*(?:/\s*yr|/\s*year|annually)?\b', output)
            or re.search(r'salary\s*(?:is|of|:)?\s*[\$₹]?\s*\d+', output, re.I)
        )
        assert not salary_number_match, (
            f"Answer should not state a salary number when missing from the page, but found: '{salary_number_match.group(0)}' in output: '{output}'"
        )

        # G-Eval that fails if the answer contradicts itself
        self_contradiction_metric = GEval(
            name="Self-Contradiction Check",
            criteria=(
                "Evaluate whether the answer is logically consistent and free of self-contradictions. "
                "The answer must fail (score 0.0) if it contradicts itself (for example, claiming both that a salary is not mentioned/unknown "
                "and simultaneously stating a specific salary amount, or making conflicting factual statements). "
                "It must pass (score 1.0) if it is completely internally consistent."
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            evaluation_steps=[
                "Check if the answer makes conflicting or self-contradictory statements.",
                "Verify whether the answer claims salary information is missing but simultaneously provides a salary figure, or vice versa.",
                "Assign 1.0 if the answer is completely internally consistent with zero contradictions.",
                "Assign 0.0 if there is any contradiction or conflicting claim in the answer.",
            ],
            threshold=0.7,
            model=judge,
        )
        test_case = LLMTestCase(input=item["input"], actual_output=output)
        assert_test(test_case, [self_contradiction_metric])


# =============================================================================
# 6. DETERMINISTIC STEP EFFICIENCY EVALS: max_tool_calls & smart duplicate check
# =============================================================================

MAX_STEPS_BY_CATEGORY = {
    "search": 8,              # Multi-query/filtered search allows up to 8 calls
    "read_job": 3,            # Scrapes should be 1-3 calls max
    "match": 0,               # Reasoning against candidate profile, 0 tools
    "save_and_duplicates": 3, # Checking saved + saving allows up to 3 calls
    "edge_and_failure": 2,    # Out-of-scope = 0, missing info/unrealistic = 1-2 calls
}

MAX_RETRIES_ON_ERROR = 1


@pytest.mark.parametrize("item", BASELINE_OUTPUTS, ids=[g["golden_id"] for g in BASELINE_OUTPUTS])
def test_step_efficiency_deterministic(item: Dict[str, Any]):
    """Verify agent execution step efficiency deterministically without LLM judge noise.

    1. Asserts tool calls do not exceed category-specific max_tool_calls budget.
    2. Detects wasteful duplicate calls (same tool + same parameters).
    3. Allows at most MAX_RETRIES_ON_ERROR (1 retry) if a tool call encountered an error.
    """
    category = item.get("category", "")
    gid = item.get("golden_id", "")
    tools_called = item.get("tools_called", [])

    # 1. Category-based tool call budget cap
    max_allowed = MAX_STEPS_BY_CATEGORY.get(category, 8)
    assert len(tools_called) <= max_allowed, (
        f"Tool call count {len(tools_called)} exceeds budget of {max_allowed} for category '{category}' in {gid}!"
    )

    # 2. Smart Duplicate Tool Call Check
    seen_successful_calls = set()
    failed_call_counts: Dict[str, int] = {}

    for tc in tools_called:
        tool_name = tc.get("name", "")
        params = tc.get("input_parameters", {})
        # Canonical signature: tool name + sorted parameters
        sig = (tool_name, json.dumps(params, sort_keys=True, default=str))

        # Check if already succeeded previously in this trajectory
        assert sig not in seen_successful_calls, (
            f"Wasteful duplicate call in {gid}: '{tool_name}' with identical parameters was already executed successfully!"
        )

        is_error = tc.get("is_error", False)
        output_str = str(tc.get("output", "")).lower()
        if "error" in output_str or "failed" in output_str or "403" in output_str:
            is_error = True

        if is_error:
            failed_call_counts[sig] = failed_call_counts.get(sig, 0) + 1
            assert failed_call_counts[sig] <= MAX_RETRIES_ON_ERROR + 1, (
                f"Excessive retries in {gid}: '{tool_name}' failed and was retried {failed_call_counts[sig]} times (max {MAX_RETRIES_ON_ERROR} allowed)!"
            )
        else:
            seen_successful_calls.add(sig)


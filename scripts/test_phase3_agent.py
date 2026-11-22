"""Verification script for Phase 3: Match and Save with Filesystem CSV Tool.

Tests end-to-end agent execution with search -> read -> match -> save to test_jobs.csv,
as well as duplicate rejection and threshold enforcement.
"""

import asyncio
import csv
import json
from pathlib import Path
from jobsearchagent.agent import JobSearchAgent
from jobsearchagent.csv_tools import CSVJobStore

TEST_CSV = Path("test_jobs.csv")

async def main():
    print("=" * 75)
    print("🚀 PHASE 3 VERIFICATION: End-to-End Match & Save to CSV")
    print("=" * 75)

    # 1. Start fresh with test_jobs.csv
    if TEST_CSV.exists():
        TEST_CSV.unlink()

    print(f"Target CSV: {TEST_CSV.resolve()}")

    agent = JobSearchAgent(csv_path=TEST_CSV)
    await agent.initialize()

    question = (
        "Find a relevant AI Engineer or LLM Engineer role in Bengaluru from your tools. "
        "Evaluate how well it matches my profile (calculate match score, matched skills, "
        "missing skills, and reason). If the match score is at or above 60, save it to the CSV."
    )

    print(f"\nUser Prompt:\n'{question}'\n")
    print("Running agent...")

    result = await agent.arun(question, max_steps=10)

    print("\n" + "-" * 75)
    print("🛠️  AGENT TOOL CALLS:")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"\n[{i}] Tool Called: {tc['tool']}")
        print(f"    Arguments  : {json.dumps(tc['args'], indent=6)}")

    print("\n" + "-" * 75)
    print("💬 AGENT FINAL RESPONSE:")
    print(result["answer"])

    # 2. Verify test_jobs.csv contents
    print("\n" + "=" * 75)
    print("📄 VERIFYING test_jobs.csv ROW CONTENT:")
    print("=" * 75)

    assert TEST_CSV.exists(), "test_jobs.csv was not created!"

    with open(TEST_CSV, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Total Rows Saved: {len(reader)}")
    for i, row in enumerate(reader, 1):
        print(f"\nRow {i}:")
        for k, v in row.items():
            print(f"  • {k:<15}: {v}")

    # 3. Test duplicate and threshold enforcement
    print("\n" + "=" * 75)
    print("🛡️  TESTING EDGE CASES: Duplicate & Threshold Rules")
    print("=" * 75)

    store = CSVJobStore(csv_path=TEST_CSV, threshold=60)

    # Test Duplicate
    if reader:
        dup_url = reader[0]["url"]
        print(f"\nAttempting to save duplicate URL: {dup_url}")
        dup_res = store.save_job(
            url=dup_url,
            title="Duplicate Role",
            company="Duplicate Co",
            location="Remote",
            match_score=85,
            matched_skills=["Python"],
            missing_skills=[],
            reason="Testing duplicate",
        )
        print("Duplicate Result:", dup_res["status"], f"({dup_res['message']})")
        assert dup_res["status"] == "skipped", "Duplicate check failed!"

    # Test Below Threshold
    print("\nAttempting to save job below threshold (Score: 45 < 60):")
    low_res = store.save_job(
        url="https://example.com/senior-role-123",
        title="Senior Director AI",
        company="Big Corp",
        location="Remote",
        match_score=45,
        matched_skills=["Python"],
        missing_skills=["10+ years leadership"],
        reason="Requires 10 years experience",
    )
    print("Threshold Result:", low_res["status"], f"({low_res['message']})")
    assert low_res["status"] == "rejected", "Threshold check failed!"

    print("\n" + "=" * 75)
    print("✅ PHASE 3 VERIFICATION PASSED COMPLETELY!")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(main())

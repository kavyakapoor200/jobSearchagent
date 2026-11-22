"""Script to re-record only search_03, search_04, and save_01 in data/agent_outputs_baseline.json."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

from jobsearchagent.agent import JobSearchAgent

DATASET_PATH = Path("data/golden_dataset.json")
OUTPUTS_PATH = Path("data/agent_outputs_baseline.json")
TEST_CSV_PATH = Path("test_jobs.csv")

TARGET_IDS = [
    "edge_02_out_of_scope_creative_writing",
]


async def rerecord():
    assert DATASET_PATH.exists()
    assert OUTPUTS_PATH.exists()

    goldens_raw: List[Dict[str, Any]] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    goldens = {g["id"]: g for g in goldens_raw}

    outputs: List[Dict[str, Any]] = json.loads(OUTPUTS_PATH.read_text(encoding="utf-8"))
    outputs_by_id = {item["golden_id"]: item for item in outputs}

    agent = JobSearchAgent(csv_path=TEST_CSV_PATH)
    await agent.initialize()

    for gid in TARGET_IDS:
        golden = goldens[gid]
        print(f"\n==========================================")
        print(f"Running {gid}...")
        print(f"Input: {golden['input']}")
        print(f"Expected params: {golden.get('expected_params')}")
        print(f"Should save: {golden.get('should_save')}")

        start_time = time.time()
        agent_res = await agent.arun(golden["input"], max_steps=10)
        elapsed = round(time.time() - start_time, 2)

        recorded_tools = [
            {"name": tc["tool"], "input_parameters": tc.get("args") or {}}
            for tc in agent_res.get("tool_calls", [])
        ]

        print(f"Done in {elapsed}s")
        print(f"Tools called: {recorded_tools}")
        print(f"Answer:\n{agent_res['answer'][:300]}...")

        # Update record
        record = {
            "golden_id": gid,
            "category": golden["category"],
            "input": golden["input"],
            "actual_output": agent_res["answer"],
            "tools_called": recorded_tools,
            "expected_tools": golden.get("expected_tools", []),
            "expected_params": golden.get("expected_params", {}),
            "expected_behavior": golden.get("expected_behavior", ""),
            "expected_match_range": golden.get("expected_match_range"),
            "should_save": golden.get("should_save"),
            "execution_time_seconds": elapsed,
            "error": None,
        }

        outputs_by_id[gid] = record
        await asyncio.sleep(2.0)

    # Reconstruct outputs list preserving original order
    updated_outputs = [
        outputs_by_id.get(item["golden_id"], item)
        for item in outputs
    ]

    OUTPUTS_PATH.write_text(json.dumps(updated_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n✅ Successfully updated data/agent_outputs_baseline.json!")


if __name__ == "__main__":
    asyncio.run(rerecord())

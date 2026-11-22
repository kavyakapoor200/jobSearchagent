"""Script to execute the JobSearchAgent on the 25 Goldens and record outputs.

Adheres strictly to Spec Section 3 Rule 1:
"Run the agent once per golden, save the outputs to a JSON file,
and run metrics on the saved file as often as needed."
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from jobsearchagent.agent import JobSearchAgent

load_dotenv()

DATASET_PATH = Path("data/golden_dataset.json")
OUTPUTS_PATH = Path("data/agent_outputs_baseline.json")
TEST_CSV_PATH = Path("test_jobs.csv")


async def record_outputs(
    dataset_path: Path = DATASET_PATH,
    outputs_path: Path = OUTPUTS_PATH,
    limit: int = 25,
    delay_seconds: float = 1.5,
) -> None:
    """Run agent against golden dataset and persist outputs to JSON."""
    assert dataset_path.exists(), f"Golden dataset not found at {dataset_path}"
    goldens: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Load existing progress if available
    existing_outputs: Dict[str, Dict[str, Any]] = {}
    if outputs_path.exists():
        try:
            for item in json.loads(outputs_path.read_text(encoding="utf-8")):
                existing_outputs[item["golden_id"]] = item
        except Exception:
            pass

    print("=" * 75)
    print(f"🎬 RECORDING AGENT OUTPUTS FOR EVALUATION (Total Goldens: {len(goldens)})")
    print(f"Dataset : {dataset_path}")
    print(f"Output  : {outputs_path}")
    print(f"Already completed: {len(existing_outputs)} / {len(goldens)}")
    print("=" * 75)

    agent = JobSearchAgent(csv_path=TEST_CSV_PATH)
    await agent.initialize()

    results: List[Dict[str, Any]] = list(existing_outputs.values())
    count = 0

    for i, golden in enumerate(goldens[:limit], 1):
        gid = golden["id"]
        if gid in existing_outputs:
            print(f"[{i}/{len(goldens)}] ⏭️ Skipping already recorded: {gid}")
            continue

        print(f"\n[{i}/{len(goldens)}] 🏃 Running golden: {gid} ({golden['category']})")
        print(f"    Prompt: '{golden['input']}'")

        max_retries = 3
        agent_res = None
        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                agent_res = await agent.arun(golden["input"], max_steps=10)
                elapsed = round(time.time() - start_time, 2)
                recorded_tools = [
                    {"name": tc["tool"], "input_parameters": tc.get("args") or {}}
                    for tc in agent_res.get("tool_calls", [])
                ]
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
                print(f"    ✅ Done in {elapsed}s | Tool calls: {len(recorded_tools)}")
                break
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                if "rate_limit" in str(e).lower() and attempt < max_retries:
                    print(f"    ⚠️ Rate limit on attempt {attempt}. Sleeping 8s before retry...")
                    await asyncio.sleep(8.0)
                    continue
                print(f"    ❌ Error in {elapsed}s: {e}")
                record = {
                    "golden_id": gid,
                    "category": golden["category"],
                    "input": golden["input"],
                    "actual_output": "",
                    "tools_called": [],
                    "expected_tools": golden.get("expected_tools", []),
                    "expected_params": golden.get("expected_params", {}),
                    "expected_behavior": golden.get("expected_behavior", ""),
                    "expected_match_range": golden.get("expected_match_range"),
                    "should_save": golden.get("should_save"),
                    "execution_time_seconds": elapsed,
                    "error": str(e),
                }
                break

        existing_outputs[gid] = record
        results = list(existing_outputs.values())

        # Persist incrementally
        outputs_path.parent.mkdir(parents=True, exist_ok=True)
        outputs_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        count += 1
        # Throttle to stay well under TPM/TPD limits
        await asyncio.sleep(delay_seconds)

    print("\n" + "=" * 75)
    print(f"🏁 RECORDING FINISHED! Total records saved: {len(results)} to {outputs_path}")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(record_outputs())

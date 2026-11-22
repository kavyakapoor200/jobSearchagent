"""DeepEval evaluation test for Phase 1: Resume Profile Builder.

Evaluates that the skills saved in `profile.json` are strictly grounded in
the candidate's original resume (`resume.txt`), adhering to Project Spec
Section 5.1 & 6.2 with strict penalization rules for unsupported skills.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from jobsearchagent.evals.groq_judge import GroqJudge
from jobsearchagent.profile_builder import extract_text_from_file


def test_profile_json_skill_faithfulness() -> None:
    """Test that skills stored in profile.json are strictly grounded in resume.txt."""
    profile_path = Path("profile.json")
    resume_path = Path("resume.txt")

    if not profile_path.exists():
        pytest.fail("profile.json does not exist. Please generate it first.")

    if not resume_path.exists():
        pytest.fail("resume.txt does not exist to evaluate against.")

    # 1. Read profile.json directly from disk
    profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    skills = profile_data.get("skills", [])
    assert len(skills) > 0, "profile.json should contain a non-empty 'skills' list"

    # 2. Read resume text as INPUT
    resume_text = extract_text_from_file(resume_path)

    # 3. DeepEval Groq Judge
    judge_model = os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")
    groq_judge = GroqJudge(model_name=judge_model)

    # 4. Strict G-Eval Metric with step-by-step scoring rules
    skill_faithfulness = GEval(
        name="Skill Faithfulness",
        evaluation_steps=[
            "The INPUT is a resume. The ACTUAL_OUTPUT is a list of skills extracted from it.",
            "For each skill in the ACTUAL_OUTPUT, find the place in the resume that clearly states it.",
            "Accept the exact name or a standard alternate name (for example RAG = retrieval-augmented generation, LLM = large language model).",
            "Do not accept a skill that is only implied, guessed, or related to something in the resume.",
            "Count the skills that have no clear support in the resume.",
            "If the count is 0, give the highest score. If the count is 1, give a low score of 4 or less out of 10. If the count is 2 or more, give 2 or less out of 10.",
            "In the reason, list every unsupported skill by name.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.8,
        model=groq_judge,
    )

    test_case = LLMTestCase(
        input=resume_text,
        actual_output=", ".join(skills),
    )

    # Run DeepEval assertion
    assert_test(test_case, [skill_faithfulness])

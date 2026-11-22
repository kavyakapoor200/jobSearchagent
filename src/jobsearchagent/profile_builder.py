"""Resume Profile Builder for Job-Search Agent.

Extracts text from PDF/Markdown/Text resumes, uses Groq LLM to extract
structured profile information, interactively asks ONLY for missing/unspecified
preferences, displays a clean summary, and saves to profile.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()  # fallback to current working directory


class UserProfile(BaseModel):
    """Schema defined in Project Spec Section 5.1."""

    target_roles: List[str] = Field(
        default_factory=list,
        description="Job titles to search for (e.g. AI Engineer, LLM Engineer)",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Technical and domain skills that appear in the resume",
    )
    years_experience: float = Field(
        default=0.0,
        description="Estimated years of relevant professional experience",
    )
    max_experience_years: float = Field(
        default=3.0,
        description="Highest job experience requirement to accept",
    )
    locations: List[str] = Field(
        default_factory=lambda: ["Remote"],
        description="Preferred job locations or 'Remote'",
    )
    include_keywords: List[str] = Field(
        default_factory=list,
        description="Extra search keywords to prioritize",
    )
    exclude_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that rule a job out (e.g. Senior Director, C++)",
    )


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract plain text or markdown from PDF, LaTeX, or text files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pymupdf4llm

        markdown_text = pymupdf4llm.to_markdown(str(path))
        return markdown_text.strip()

    elif suffix in (".tex", ".latex"):
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"%.*$", "", raw_text, flags=re.MULTILINE)
        text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})?", " ", text)
        text = re.sub(r"[\{\}\$\\_#~^]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    else:
        return path.read_text(encoding="utf-8", errors="replace").strip()


def verify_skills_in_resume(skills: List[str], resume_text: str) -> Tuple[List[str], List[str]]:
    """Helper check to verify skills against resume text."""
    resume_lower = resume_text.lower()
    verified: List[str] = []
    rejected: List[str] = []

    normalized_resume = re.sub(r"[\s\-_/]+", " ", resume_lower)

    for skill in skills:
        cleaned_skill = skill.strip()
        if not cleaned_skill:
            continue

        skill_lower = cleaned_skill.lower()
        normalized_skill = re.sub(r"[\s\-_/]+", " ", skill_lower)

        if skill_lower in resume_lower or normalized_skill in normalized_resume:
            verified.append(cleaned_skill)
        else:
            alphanumeric_skill = re.sub(r"[^a-z0-9]", "", skill_lower)
            alphanumeric_resume = re.sub(r"[^a-z0-9]", "", resume_lower)
            if len(alphanumeric_skill) >= 2 and alphanumeric_skill in alphanumeric_resume:
                verified.append(cleaned_skill)
            else:
                rejected.append(cleaned_skill)

    return verified, rejected


def extract_profile_with_llm(resume_text: str, model_name: Optional[str] = None) -> UserProfile:
    """Use Groq LLM to extract initial user profile from resume text."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file or environment variables."
        )

    model = model_name or os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are an expert resume parser for an AI Job Search Agent.\n"
        "Extract structured profile information from the provided resume text.\n"
        "CRITICAL RULES:\n"
        "1. For 'skills': ONLY list skills that are EXPLICITLY mentioned in the resume text. "
        "Do NOT invent, infer, or assume any skills not stated directly.\n"
        "2. For 'years_experience': estimate the total years of relevant work experience based on dates.\n"
        "3. For 'target_roles': suggest 2-4 target job roles based on their recent experience (e.g. 'AI Engineer', 'LLM Engineer').\n"
        "4. Return strictly a JSON object conforming to the UserProfile schema."
    )

    schema_instruction = (
        "Output ONLY a valid JSON object with these keys:\n"
        "{\n"
        '  "target_roles": ["role1", "role2"],\n'
        '  "skills": ["skill1", "skill2"],\n'
        '  "years_experience": 2.0,\n'
        '  "max_experience_years": 4.0,\n'
        '  "locations": ["Remote"],\n'
        '  "include_keywords": [],\n'
        '  "exclude_keywords": []\n'
        "}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{system_prompt}\n\n{schema_instruction}"},
            {"role": "user", "content": f"Resume Text:\n\n{resume_text[:14000]}"},
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)

    return UserProfile.model_validate(data)


def targeted_pre_hook(profile: UserProfile) -> UserProfile:
    """Prompt the user ONLY for missing or essential preference fields."""
    print("\n" + "=" * 50)
    print("🎯  TARGETED PRE-HOOK: Confirming Job Search Preferences")
    print("=" * 50)
    print("We extracted your resume details. Just checking a few preferences:")

    # 1. Target Roles (only if empty)
    if not profile.target_roles:
        val = input(
            "\n[?] Target job titles (comma-separated, e.g. 'AI Engineer, LLM Engineer'): "
        ).strip()
        if val:
            profile.target_roles = [r.strip() for r in val.split(",") if r.strip()]
        else:
            profile.target_roles = ["AI Engineer", "LLM Engineer"]
    else:
        current_roles = ", ".join(profile.target_roles)
        print(f"\n[✓] Detected Target Roles: {current_roles}")
        edit = input("    Press Enter to keep, or type new roles to replace: ").strip()
        if edit:
            profile.target_roles = [r.strip() for r in edit.split(",") if r.strip()]

    # 2. Locations (default to Remote if empty)
    if not profile.locations or profile.locations == ["Remote"]:
        loc_input = input(
            "\n[?] Preferred locations (comma-separated, press Enter for 'Remote'): "
        ).strip()
        if loc_input:
            profile.locations = [loc.strip() for loc in loc_input.split(",") if loc.strip()]
        else:
            profile.locations = ["Remote", "Chennai", "Bengaluru"]
    else:
        current_locs = ", ".join(profile.locations)
        print(f"\n[✓] Preferred Locations: {current_locs}")
        edit = input("    Press Enter to keep, or type new locations: ").strip()
        if edit:
            profile.locations = [loc.strip() for loc in edit.split(",") if loc.strip()]

    # 3. Max Experience Years
    default_max = max(round(profile.years_experience + 2), 3)
    exp_input = input(
        f"\n[?] Max years of experience for jobs to apply to [default: {default_max}]: "
    ).strip()
    if exp_input:
        try:
            profile.max_experience_years = float(exp_input)
        except ValueError:
            profile.max_experience_years = float(default_max)
    else:
        profile.max_experience_years = float(default_max)

    # 4. Optional keywords (single prompt, quick skip)
    if not profile.exclude_keywords:
        ex_input = input(
            "\n[?] Any roles/technologies to EXCLUDE? (comma-separated, press Enter to skip): "
        ).strip()
        if ex_input:
            profile.exclude_keywords = [k.strip() for k in ex_input.split(",") if k.strip()]

    return profile


def display_profile_summary(profile: UserProfile) -> None:
    """Show the final profile clearly to the user before saving."""
    print("\n" + "=" * 60)
    print("📋  EXTRACTED USER PROFILE SUMMARY")
    print("=" * 60)
    print(f"• Target Roles        : {', '.join(profile.target_roles)}")
    print(f"• Locations           : {', '.join(profile.locations)}")
    print(f"• Relevant Experience : {profile.years_experience:.1f} years")
    print(f"• Max Job Requirement : {profile.max_experience_years:.1f} years")
    print(f"• Extracted Skills ({len(profile.skills)}):")
    chunks = [profile.skills[i : i + 6] for i in range(0, len(profile.skills), 6)]
    for chunk in chunks:
        print(f"    {', '.join(chunk)}")

    if profile.include_keywords:
        print(f"• Include Keywords    : {', '.join(profile.include_keywords)}")
    if profile.exclude_keywords:
        print(f"• Exclude Keywords    : {', '.join(profile.exclude_keywords)}")
    print("=" * 60)


def build_profile(
    resume_path: str | Path,
    output_path: str | Path = "profile.json",
    interactive: bool = True,
) -> UserProfile:
    """Pipeline: extract text -> LLM parse -> pre-hook -> show -> save."""
    print(f"\n[1/3] Reading resume from: {resume_path}")
    resume_text = extract_text_from_file(resume_path)
    print(f"      Extracted {len(resume_text)} characters.")

    print("\n[2/3] Parsing structured profile with Groq...")
    profile = extract_profile_with_llm(resume_text)

    if interactive:
        profile = targeted_pre_hook(profile)

    display_profile_summary(profile)

    out = Path(output_path)
    if interactive:
        confirm = input(f"\n[?] Save profile to '{out.name}' on disk? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("❌ Cancelled. Profile was not saved.")
            return profile

    out.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    print(f"✅ Successfully saved profile to {out.resolve()}!")
    return profile


def main() -> None:
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python -m jobsearchagent.profile_builder <path_to_resume.pdf|.tex|.txt> [output_path]")
        sys.exit(1)

    resume_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "profile.json"
    build_profile(resume_file, output_path=output_file, interactive=True)


if __name__ == "__main__":
    main()

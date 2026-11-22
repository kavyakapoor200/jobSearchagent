"""Tests for Profile Builder and Skill Verification."""

from jobsearchagent.profile_builder import (
    UserProfile,
    extract_text_from_file,
    verify_skills_in_resume,
)


def test_verify_skills_exact_and_normalized() -> None:
    resume_text = """
    Software Engineer with experience in Python, PyTorch, LangChain, and Docker.
    Built high-throughput microservices using Node.js and FastAPI.
    Familiar with AWS (EC2, S3) and PostgreSQL.
    """

    skills_to_test = [
        "Python",
        "PyTorch",
        "LangChain",
        "Docker",
        "Node.js",
        "FastAPI",
        "AWS",
        "PostgreSQL",
        "Rust",  # Hallucinated / Not in resume
        "Kubernetes",  # Hallucinated / Not in resume
    ]

    verified, rejected = verify_skills_in_resume(skills_to_test, resume_text)

    # Check verified skills
    assert "Python" in verified
    assert "PyTorch" in verified
    assert "LangChain" in verified
    assert "Docker" in verified
    assert "Node.js" in verified
    assert "FastAPI" in verified
    assert "AWS" in verified
    assert "PostgreSQL" in verified

    # Check that hallucinated skills were rejected
    assert "Rust" in rejected
    assert "Kubernetes" in rejected


def test_user_profile_schema_defaults() -> None:
    profile = UserProfile(
        target_roles=["AI Engineer"],
        skills=["Python", "LangGraph"],
        years_experience=2.0,
        max_experience_years=4.0,
        locations=["Remote"],
    )

    data = profile.model_dump()
    assert data["target_roles"] == ["AI Engineer"]
    assert data["skills"] == ["Python", "LangGraph"]
    assert data["years_experience"] == 2.0
    assert data["max_experience_years"] == 4.0
    assert data["locations"] == ["Remote"]
    assert data["include_keywords"] == []
    assert data["exclude_keywords"] == []

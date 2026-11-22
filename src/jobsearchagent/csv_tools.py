"""Filesystem CSV tool for saving and reading qualified job rows.

Implements the 9-column schema, duplicate prevention, and threshold enforcement
specified in Project Spec Section 5.3 & 5.4.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

CSV_COLUMNS = [
    "url",
    "title",
    "company",
    "location",
    "posted_date",
    "match_score",
    "matched_skills",
    "missing_skills",
    "reason",
    "saved_at",
]

DEFAULT_CSV_PATH = Path("jobs.csv")
SAVE_THRESHOLD = 60


class CSVJobStore:
    """Manages reading and writing to jobs.csv with duplicate prevention."""

    def __init__(self, csv_path: Path | str = DEFAULT_CSV_PATH, threshold: int = SAVE_THRESHOLD):
        self.csv_path = Path(csv_path)
        self.threshold = threshold
        self._ensure_file_initialized()

    def _ensure_file_initialized(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def read_saved_urls(self) -> set[str]:
        """Return the set of URLs already saved in the CSV."""
        if not self.csv_path.exists():
            return set()

        urls: set[str] = set()
        with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                if url:
                    urls.add(url)
        return urls

    def read_saved_jobs(self) -> List[Dict[str, str]]:
        """Return all rows currently saved in the CSV."""
        if not self.csv_path.exists():
            return []

        rows: List[Dict[str, str]] = []
        with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def save_job(
        self,
        url: str,
        title: str,
        company: str,
        location: str,
        match_score: int,
        matched_skills: List[str] | str,
        missing_skills: List[str] | str,
        reason: str,
        posted_date: str = "",
    ) -> Dict[str, Any]:
        """Save a job posting if score >= threshold and URL is not a duplicate."""
        clean_url = url.strip()

        # 1. Enforce threshold rule (Spec Section 5.2 & 5.4)
        if match_score < self.threshold:
            return {
                "status": "rejected",
                "message": (
                    f"Match score {match_score} is below the save threshold "
                    f"({self.threshold}). Job not saved."
                ),
            }

        # 2. Enforce duplicate rule (Spec Section 5.2)
        existing_urls = self.read_saved_urls()
        if clean_url in existing_urls:
            return {
                "status": "skipped",
                "message": f"URL '{clean_url}' already exists in {self.csv_path.name} (duplicate skipped).",
            }

        # Normalize skills into strings
        matched_str = (
            ", ".join(matched_skills) if isinstance(matched_skills, list) else str(matched_skills)
        )
        missing_str = (
            ", ".join(missing_skills) if isinstance(missing_skills, list) else str(missing_skills)
        )
        saved_at = datetime.now().isoformat()

        row = [
            clean_url,
            title.strip(),
            company.strip(),
            location.strip(),
            posted_date.strip(),
            match_score,
            matched_str,
            missing_str,
            reason.strip(),
            saved_at,
        ]

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return {
            "status": "saved",
            "message": f"Successfully saved '{title}' at '{company}' (Score: {match_score}) to {self.csv_path.name}.",
            "saved_row": dict(zip(CSV_COLUMNS, row)),
        }


# Global store instance (can be swapped during tests to test_jobs.csv)
_ACTIVE_STORE = CSVJobStore()


def set_active_csv_path(path: Path | str, threshold: int = SAVE_THRESHOLD) -> CSVJobStore:
    """Set the active CSV path (e.g. test_jobs.csv during evals)."""
    global _ACTIVE_STORE
    _ACTIVE_STORE = CSVJobStore(csv_path=path, threshold=threshold)
    return _ACTIVE_STORE


def get_active_csv_store() -> CSVJobStore:
    """Get the active CSV store."""
    return _ACTIVE_STORE


@tool
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
    result = _ACTIVE_STORE.save_job(
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


@tool
def read_saved_jobs() -> str:
    """Read all jobs currently saved in the CSV file to check for duplicates."""
    jobs = _ACTIVE_STORE.read_saved_jobs()
    if not jobs:
        return "No jobs currently saved in CSV."
    summary = [f"Currently saved ({len(jobs)} jobs):"]
    for j in jobs:
        summary.append(f"- {j.get('title')} at {j.get('company')} (Score: {j.get('match_score')}) | URL: {j.get('url')}")
    return "\n".join(summary)


def get_filesystem_csv_tools(csv_path: Optional[Path | str] = None) -> List[Any]:
    """Return the filesystem CSV tools configured for the target CSV."""
    if csv_path is not None:
        set_active_csv_path(csv_path)
    return [save_job_to_csv, read_saved_jobs]

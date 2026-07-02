"""
audit_filters.py — Run locally to see EXACTLY why each job was kept or dropped,
without manually opening every posting.

This re-runs your pre-LLM filters (filters.py) against the most recent raw
fetch and writes a CSV with one row per job, showing:
  - title, company, job_type
  - kept: True/False
  - drop_reason: "non_engineering" | "seniority" | "experience" | "" (kept)
  - the matched regex pattern (if dropped) so you can see WHY

This does NOT call Groq — it's free and instant, since it only re-runs the
zero-token pre-LLM filters. Use it to tune filters.py without burning API
calls or eyeballing every listing by hand.

Usage:
  1. Make sure logs/agent.log has a recent run (or just run main.py once)
  2. python audit_filters.py
  3. Open audit_report.csv in Excel/Sheets, sort/filter by drop_reason
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # quiet — we want clean CSV output only

from filters import (
    fix_linkedin_url,
    classify_job_type,
    is_non_engineering_role,
    is_senior_role,
    extract_required_experience,
    passes_experience_filter,
    _FRESHER_SIGNAL_RE,
)

BASE = Path(__file__).parent
RAW_CACHE = BASE / "logs" / "last_raw_jobs.json"   # see note below
OUTPUT = BASE / "audit_report.csv"


def audit_one(job: dict) -> dict:
    """Run every filter stage on one job and record the verdict + reason."""
    title = job.get("title", "")
    desc = job.get("description", "")
    text = f"{title} {desc}"

    job["url"] = fix_linkedin_url(job.get("url", ""))
    job_type = classify_job_type(job)
    job["job_type"] = job_type

    # Stage 1: non-engineering veto
    non_eng = is_non_engineering_role(title)
    if non_eng:
        return {
            "title": title,
            "company": job.get("company", ""),
            "job_type": job_type,
            "kept": False,
            "drop_reason": "non_engineering",
            "detail": "Title matched non-engineering pattern (content/marketing/sales/etc)",
            "url": job.get("url", ""),
        }

    # Stage 2: seniority (skipped for internships)
    if job_type != "internship" and is_senior_role(title):
        return {
            "title": title,
            "company": job.get("company", ""),
            "job_type": job_type,
            "kept": False,
            "drop_reason": "seniority",
            "detail": "Title matched senior/lead/manager/etc pattern",
            "url": job.get("url", ""),
        }

    # Stage 3: experience
    years = extract_required_experience(text)
    fresher_signal = bool(_FRESHER_SIGNAL_RE.search(text))
    passes_exp = passes_experience_filter(job)

    if not passes_exp:
        if years is not None:
            detail = f"Explicit requirement: {years}+ years (max allowed: 2)"
        else:
            detail = "No years stated AND no fresher/entry-level signal found in text"
        return {
            "title": title,
            "company": job.get("company", ""),
            "job_type": job_type,
            "kept": False,
            "drop_reason": "experience",
            "detail": detail,
            "url": job.get("url", ""),
        }

    # Passed everything
    exp_note = (
        f"explicit {years}yr" if years is not None
        else ("fresher signal found" if fresher_signal else "internship — auto-pass")
    )
    return {
        "title": title,
        "company": job.get("company", ""),
        "job_type": job_type,
        "kept": True,
        "drop_reason": "",
        "detail": f"Passed all filters ({exp_note})",
        "url": job.get("url", ""),
    }


def main():
    if not RAW_CACHE.exists():
        print(f"\n⚠️  {RAW_CACHE} not found.")
        print("Run `python main.py` once first — crew.py automatically caches")
        print("post-dedup, pre-filter jobs to logs/last_raw_jobs.json on every run.")
        return

    raw_jobs = json.loads(RAW_CACHE.read_text(encoding="utf-8"))
    print(f"Auditing {len(raw_jobs)} raw jobs...")

    rows = [audit_one(j) for j in raw_jobs]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "company", "job_type", "kept", "drop_reason", "detail", "url"])
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(1 for r in rows if r["kept"])
    by_reason = {}
    for r in rows:
        if not r["kept"]:
            by_reason[r["drop_reason"]] = by_reason.get(r["drop_reason"], 0) + 1

    print(f"\n✅ Wrote {OUTPUT}")
    print(f"\nSummary: {len(rows)} total → {kept} kept, {len(rows) - kept} dropped")
    for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  dropped ({reason}): {count}")

    print(f"\nOpen {OUTPUT} and sort by 'drop_reason' to review each bucket.")
    print("If you see good matches in 'non_engineering' or 'experience' buckets,")
    print("that tells us exactly which regex pattern needs adjusting.")


if __name__ == "__main__":
    main()
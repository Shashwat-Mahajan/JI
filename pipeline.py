"""
pipeline.py — plain function pipeline (replaces crew.py's CrewAI wrapper).

v4.0: agents/analyst.py, agents/scout.py, agents/reporter.py were empty —
all logic already lived inline in crew.py's four @tool functions, each
called exactly once by a CrewAI Agent/Task wrapping it in agent overhead
(extra system prompts, task descriptions, delegation/ReAct loop, an LLM
object per agent) for zero reasoning benefit — every task description said
"call the tool ONCE, don't call it twice." That's a fixed pipeline, not
agentic work.

This module replaces the whole Crew/Agent/Task/Process.sequential
machinery with a straight function pipeline:

    jobs = scrape_all_sources(cfg)
    jobs = dedupe_and_filter(jobs, cfg)
    scored = score_jobs(jobs, cfg)          # scorer.py already does its own
                                             # cache-aware top-K BGE filter,
                                             # cross-encoder, batched Groq
                                             # scoring, and edge-case verifier
    send_digest(scored, cfg)

The standalone "Verifier" agent/tool that used to run AFTER scoring is
dropped here — it duplicated work scorer.py's own Layer 3 edge-case
verifier already does internally (same idea: permissive second pass on
HIGH jobs, downgrade only on >=85% confidence), just via a second,
separate LLM call path (nim_client.make_client/call_nim) instead of the
rotated Groq pool. Keeping both meant paying for verification twice.
"""

import json
import logging
from datetime import date
from pathlib import Path

from nim_client import register_keys_from_config, register_groq_keys_from_config

from sources.public_apis import (
    fetch_remotive,
    fetch_arbeitnow,
    fetch_jobicy,
    fetch_himalayas,
    fetch_freshersworld,
)
from sources.career_portals import fetch_all_career_portals
from sources.linkedin import fetch_linkedin
from sources.naukri import fetch_naukri

from scorer import score_jobs_with_llm
from reporter import build_html_report, send_email
from utils import deduplicate_batch
from filters import apply_all_filters
from resume_intake import process_resume
import db

log = logging.getLogger(__name__)

BASE = Path(__file__).parent
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)


def scrape_all_sources(cfg: dict) -> list:
    """Fetch from every source. Pure I/O, no LLM calls, zero API cost."""
    keywords = cfg.get("search_keywords", [])
    location = cfg.get("location", "India")

    raw: list = []
    source_counts: dict = {}

    def _fetch_and_count(name, fn, *args):
        try:
            result = fn(*args)
        except Exception as e:
            log.error(f"Source error — {name}: {e}")
            result = []
        source_counts[name] = len(result)
        raw.extend(result)

    _fetch_and_count("Remotive", fetch_remotive, keywords)
    _fetch_and_count("Arbeitnow", fetch_arbeitnow, keywords)
    _fetch_and_count("Jobicy", fetch_jobicy, keywords)
    _fetch_and_count("Himalayas", fetch_himalayas, keywords)
    _fetch_and_count("Freshersworld", fetch_freshersworld, keywords)
    _fetch_and_count("LinkedIn", fetch_linkedin, keywords, location)
    _fetch_and_count("Naukri", fetch_naukri, keywords)
    _fetch_and_count("CareerPortals", fetch_all_career_portals)

    log.info(
        "Per-source raw counts: "
        + ", ".join(f"{k}={v}" for k, v in source_counts.items())
    )

    before_dedup = len(raw)
    raw = deduplicate_batch(raw)
    log.info(f"Scrape: cross-source dedup {before_dedup} -> {len(raw)}")

    try:
        (BASE / "logs" / "last_raw_jobs.json").write_text(
            json.dumps(raw, default=str), encoding="utf-8"
        )
    except Exception as e:
        log.debug(f"Could not cache raw jobs for audit: {e}")

    return raw


def resolve_resume(cfg: dict) -> dict | None:
    """
    If cfg["resume_path"] is set, run it through the resume-upload flow:
    extract text -> (cache hit, or one LLM call for structured fields) ->
    resume embedding, cached by resume hash. Re-running the same resume
    file costs zero LLM calls and zero re-embedding.

    Returns None if no resume_path is configured — the pipeline then falls
    back to the static config/profile.json anchor, exactly as before.
    """
    resume_path = cfg.get("resume_path")
    if not resume_path:
        return None

    path = Path(resume_path)
    if not path.exists():
        log.warning("resume_path configured but file not found: %s", path)
        return None

    result = process_resume(path, cfg)
    log.info(
        "Resume resolved (hash=%s, from_cache=%s): YOE=%s, target_roles=%s",
        result["resume_hash"],
        result["from_cache"],
        result["structured_fields"].get("years_of_experience"),
        result["structured_fields"].get("target_roles"),
    )
    return result


def dedupe_and_filter(jobs: list, cfg: dict, resume: dict | None = None) -> list:
    """
    Applies the pre-LLM hard filters (YOE, role exclusions). URL/content-hash
    dedup against previously-seen jobs now lives in db.py's job cache
    (checked inside scorer.py's Layer 1 so embeddings are reused, not
    recomputed) rather than here — see _layer1_bge_filter in scorer.py.

    When a live resume is available, its structured fields (YOE, location
    preference) take priority over the static profile.json values — cheap,
    pure-Python filtering, no LLM call.
    """
    profile = cfg.get("profile", {})
    max_exp_years = profile.get("max_experience_years", 2)
    role_exclusions = profile.get("role_type_exclusions", [])

    if resume is not None:
        fields = resume["structured_fields"]
        yoe = fields.get("years_of_experience")
        if isinstance(yoe, (int, float)):
            # allow a couple years of headroom above the candidate's own YOE
            max_exp_years = max(max_exp_years, int(yoe) + 2)

    filtered = apply_all_filters(
        jobs, max_experience_years=max_exp_years, role_exclusions=role_exclusions
    )

    intern_count = sum(1 for j in filtered if j.get("job_type") == "internship")
    ft_count = len(filtered) - intern_count
    log.info(
        f"Filter: {len(jobs)} -> {len(filtered)} "
        f"({intern_count} internships, {ft_count} full-time)"
    )
    return filtered


def score_jobs(jobs: list, cfg: dict, resume: dict | None = None) -> list:
    """
    Delegates to scorer.py's 4-layer hybrid pipeline:
      L1   cache-aware top-K BGE bi-encoder selection (uses the live resume
           embedding as the similarity anchor when available)
      L1.5 cross-encoder re-rank
      L2   batched Groq LLM scoring
      L3   edge-case verifier (HIGH, score 65-75 only)
    """
    if not jobs:
        return []

    api_key = cfg.get("nvidia_nim_api_key", "")
    batch_size = cfg.get("llm_batch_size", 20)
    resume_embedding = resume["embedding"] if resume is not None else None

    scored = score_jobs_with_llm(
        jobs,
        api_key=api_key,
        batch_size=batch_size,
        resume_embedding=resume_embedding,
    )

    high = len([j for j in scored if j.get("priority") == "HIGH"])
    medium = len([j for j in scored if j.get("priority") == "MEDIUM"])
    low = len([j for j in scored if j.get("priority") == "LOW"])
    log.info(f"Score: {len(scored)} relevant of {len(jobs)} — {high}H {medium}M {low}L")
    return scored


def send_digest(jobs: list, cfg: dict) -> str:
    """Builds the HTML report, sends the email digest, saves the report file."""
    if not jobs:
        log.info("Digest: no jobs to report.")
        return "No jobs — no report generated."

    internships = sorted(
        [j for j in jobs if j.get("job_type") == "internship"],
        key=lambda j: -j.get("relevance_score", 0),
    )
    full_time = sorted(
        [j for j in jobs if j.get("job_type") == "full-time"],
        key=lambda j: -j.get("relevance_score", 0),
    )

    today = date.today().isoformat()
    html = build_html_report(internships, full_time, today)
    path = REPORTS / f"report_{today}.html"
    path.write_text(html, encoding="utf-8")
    log.info(f"Report saved -> {path}")

    email_ok = True
    if cfg.get("email_enabled"):
        subj = (
            f"Job Intel — {today} — "
            f"{len(internships)} internship{'s' if len(internships) != 1 else ''} · "
            f"{len(full_time)} job{'s' if len(full_time) != 1 else ''}"
        )
        email_ok = send_email(html, cfg, subj)
        if not email_ok:
            log.error("Email FAILED — check agent.log for the SMTP error.")

    return (
        f"Report saved to {path}. "
        f"{len(internships)} internships, {len(full_time)} full-time jobs. "
        f"Email: {'sent' if cfg.get('email_enabled') and email_ok else 'disabled/failed'}."
    )


def run_pipeline(cfg: dict) -> str:
    """Entry point — replaces crew.build_crew(cfg).kickoff()."""
    register_keys_from_config(cfg)
    register_groq_keys_from_config(cfg)

    resume = resolve_resume(cfg)  # None if cfg["resume_path"] isn't set

    jobs = scrape_all_sources(cfg)
    jobs = dedupe_and_filter(jobs, cfg, resume=resume)
    scored = score_jobs(jobs, cfg, resume=resume)
    result = send_digest(scored, cfg)

    try:
        pruned = db.prune_stale_jobs(days=14)
        if pruned:
            log.info(f"Job cache: pruned {pruned} stale entries (>14 days unseen)")
    except Exception as e:
        log.debug(f"Could not prune job cache: {e}")

    log.info("Pipeline complete.")
    return result

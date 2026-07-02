r"""
filters.py — Pre-LLM filters. Run AFTER dedup, BEFORE any Groq call.

--- CHANGED in this pass ---
  - passes_experience_filter() and apply_all_filters() now accept a
    max_years parameter instead of hardcoding "2" — this value should come
    from profile.json's "max_experience_years" field, so the experience
    gate actually reflects the loaded resume instead of being a fixed
    assumption baked into the code that happens to match a fresher.
  - "No explicit years stated" now defaults to PASS (was: required an
    explicit fresher-signal phrase, which silently dropped real
    entry-friendly postings that just didn't use that exact wording).
    Seniority gate (is_senior_role) already runs first and catches
    obvious senior-only titles, so this default-pass is not unguarded.
  - Fixed _NON_ENGINEERING_TITLE_RE: the "data annotator/annotation"
    alternative required a trailing \s+ before its negative lookahead, so
    a title that was *exactly* "Data Annotator" or "Data Annotation" with
    nothing after it (no whitespace to match at end of string) silently
    failed to match and slipped past this veto. Removed the trailing
    \s+ — the lookahead plus the outer \b already enforce a correct
    boundary without requiring trailing whitespace that may not exist.
"""

from __future__ import annotations
import re
import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

DEFAULT_MAX_EXPERIENCE_YEARS = 2   # fallback if profile.json doesn't specify one

# ── 1. LinkedIn URL fix ───────────────────────────────────────────────────────

_LI_SEARCH_RE = re.compile(
    r"(?:https?://(?:www\.)?linkedin\.com)?/jobs/search/?\?[^\s\"']*currentJobId=(\d+)[^\s\"']*"
)
_LI_VIEW_RE = re.compile(
    r"(?:https?://(?:www\.)?linkedin\.com)?/jobs/view/(\d+)"
)
_LI_JOB_POSTING_RE = re.compile(
    r"(?:https?://(?:www\.)?linkedin\.com)?/jobs-guest/jobs/api/jobPosting/(\d+)"
)


def _extract_linkedin_job_id(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    current_job_id = query_params.get("currentJobId", [None])[0]
    if current_job_id:
        return current_job_id
    m = _LI_VIEW_RE.search(url)
    if m:
        return m.group(1)
    m = _LI_SEARCH_RE.search(url)
    if m:
        return m.group(1)
    m = _LI_JOB_POSTING_RE.search(url)
    if m:
        return m.group(1)
    return None


def fix_linkedin_url(url: str) -> str:
    if not url:
        return url
    job_id = _extract_linkedin_job_id(url)
    if job_id:
        return f"https://www.linkedin.com/jobs/view/{job_id}/"
    return url


# ── 2. Job-type classifier ────────────────────────────────────────────────────

_INTERN_TITLE_RE = re.compile(
    r"\b(intern(ship)?|trainee|apprentice|co[-\s]?op|placement|"
    r"graduate\s+trainee|summer\s+analyst|summer\s+associate)\b",
    re.IGNORECASE,
)
_INTERN_TYPE_VALUES = {
    "internship", "intern", "apprenticeship", "trainee",
    "co-op", "contract to hire",
}

def classify_job_type(job: dict[str, Any]) -> str:
    emp_type = (job.get("employment_type") or "").lower().strip()
    if emp_type in _INTERN_TYPE_VALUES:
        return "internship"
    if _INTERN_TITLE_RE.search(job.get("title", "")):
        return "internship"
    return "full-time"


# ── 3. Seniority filter ───────────────────────────────────────────────────────

_SENIORITY_RE = re.compile(
    r"\b("
    r"senior|sr\.?|staff|lead|principal|manager|director|"
    r"head\s+of|vp|vice\s+president|architect|chief|"
    r"engineering\s+manager|tech\s+lead|team\s+lead|em\b"
    r")\b",
    re.IGNORECASE,
)

def is_senior_role(title: str) -> bool:
    return bool(_SENIORITY_RE.search(title))


# ── 3b. Non-engineering "AI-adjacent" veto ────────────────────────────────────

_NON_ENGINEERING_TITLE_RE = re.compile(
    r"\b("
    r"content\s+writer|content\s+creator|copywriter|copy\s+writer|"
    r"content\s+strategist|content\s+marketing|"
    r"marketing\s+(specialist|manager|executive|associate)|"
    r"sales\s+(executive|representative|associate|manager)|"
    r"business\s+development|account\s+manager|"
    r"recruiter|talent\s+acquisition|hr\s+(executive|specialist|manager)|"
    r"customer\s+(success|support|service)|"
    r"trainer\b(?!.*\b(model|engineer)\b)|"
    r"prompt\s+engineer\b.*\b(content|marketing|writer)\b|"
    r"community\s+manager|social\s+media|"
    r"voiceover|voice\s+over|"
    # FIX: removed the trailing \s+ that used to sit before the lookahead.
    # That required a whitespace character after "annotator"/"annotation",
    # which doesn't exist when that's the very last word in the title
    # (e.g. a title that's just "Data Annotator"), so the veto silently
    # failed to fire on exactly those cases. The outer \b after the
    # closing paren already enforces a correct word boundary here.
    r"data\s+entry|data\s+annotat(or|ion)(?!.*engineer)"
    r")\b",
    re.IGNORECASE,
)

_ENGINEERING_OVERRIDE_RE = re.compile(
    r"\b(engineer|developer|scientist|researcher|architect|sde|"
    r"backend|full\s*stack|mlops|devops)\b",
    re.IGNORECASE,
)

def is_non_engineering_role(title: str, role_exclusions: list[str] | None = None) -> bool:
    """
    Checks the title against profile.json's resume-derived
    role_type_exclusions (e.g. "AI content writing", "AI marketing/
    copywriting roles") instead of a fixed hardcoded list. Falls back to
    the static regex only if no profile exclusions were supplied/derived
    (e.g. profile.json missing this field).
    """
    if _ENGINEERING_OVERRIDE_RE.search(title):
        return False

    if role_exclusions:
        title_lower = title.lower()
        for exclusion in role_exclusions:
            # crude but effective: each exclusion phrase's key words must
            # all appear in the title for it to count as a match
            words = [w for w in exclusion.lower().split() if len(w) > 2]
            if words and all(w in title_lower for w in words):
                return True
        return False

    # No profile-derived exclusions available — fall back to static patterns
    return bool(_NON_ENGINEERING_TITLE_RE.search(title))


# ── 4. Experience extractor & filter ─────────────────────────────────────────

_EXP_PATTERNS = [
    re.compile(r"(\d+)\s*\+?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?experience", re.IGNORECASE),
    re.compile(r"minimum\s+(?:of\s+)?(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"at\s+least\s+(\d+)\s+years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*[-–]\s*\d+\s+years?\s+(?:of\s+)?experience", re.IGNORECASE),
    re.compile(r"experience\s*(?:of|:)?\s*(\d+)\s*\+?\s*years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*\+?\s*yrs?\s+(?:of\s+)?exp(?:erience)?", re.IGNORECASE),
]

def extract_required_experience(text: str) -> int | None:
    found: list[int] = []
    for pat in _EXP_PATTERNS:
        for m in pat.finditer(text):
            try:
                found.append(int(m.group(1)))
            except (IndexError, ValueError):
                pass
    return min(found) if found else None


def passes_experience_filter(job: dict[str, Any], max_years: int = DEFAULT_MAX_EXPERIENCE_YEARS) -> bool:
    """
    Internships always pass.
    Explicit years > max_years → drop.
    Explicit years <= max_years → pass.
    No explicit years stated → pass (seniority gate already filters obvious
    senior-only titles upstream, so this default-pass isn't unguarded).

    max_years should come from profile.json's "max_experience_years" so
    this reflects the actual loaded resume instead of a fixed assumption.
    """
    if job.get("job_type") == "internship":
        return True

    text = f"{job.get('title', '')} {job.get('description', '')}"
    years = extract_required_experience(text)

    if years is not None:
        return years <= max_years

    return True


# ── Master pipeline ───────────────────────────────────────────────────────────

def apply_all_filters(
    jobs: list[dict[str, Any]],
    max_experience_years: int = DEFAULT_MAX_EXPERIENCE_YEARS,
    role_exclusions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Run every pre-LLM filter in a single pass.

    max_experience_years: pass this in from the loaded profile
    (profile.get("max_experience_years", 2)) so the experience gate is
    resume-driven instead of a fixed value baked into this file.
    """
    kept: list[dict[str, Any]] = []
    n_seniority = 0
    n_experience = 0
    n_non_engineering = 0

    for job in jobs:
        job["url"] = fix_linkedin_url(job.get("url", ""))
        job["job_type"] = classify_job_type(job)

        if is_non_engineering_role(job.get("title", ""), role_exclusions):
            n_non_engineering += 1
            continue

        if job["job_type"] != "internship" and is_senior_role(job.get("title", "")):
            n_seniority += 1
            continue

        if not passes_experience_filter(job, max_years=max_experience_years):
            n_experience += 1
            continue

        kept.append(job)

    log.info(
        "Pre-LLM filter: %d in → %d out | dropped non_engineering=%d seniority=%d experience=%d "
        "(max_experience_years=%d)",
        len(jobs), len(kept), n_non_engineering, n_seniority, n_experience, max_experience_years,
    )
    return kept
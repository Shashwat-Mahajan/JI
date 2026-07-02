"""
utils.py — profile loading, intra-batch dedup, config helpers.

v2.2 changes:
  - _FALLBACK_SEARCH_KEYWORDS: generic SDE/backend (not AI/ML specific)
  - get_scoring_prompt(): fallback uses generic SDE roles, not GenAI/ML
  - get_verifier_system_prompt(): removed hardcoded "AI/ML" — now fully
    profile-driven using the candidate's actual scoring prompt
  - get_search_keywords(): unchanged in logic, fallback updated
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "config" / "profile.json"


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_profile(path: Path | None = None) -> dict:
    """Load structured candidate profile from config/profile.json."""
    path = path or PROFILE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load profile: {e}")
        return {}


def get_scoring_prompt(profile: dict | None = None) -> str:
    """Return the LLM scoring prompt built from the resume profile."""
    profile = profile if profile is not None else load_profile()
    prompt = profile.get("_scoring_prompt", "")
    if prompt:
        return prompt

    # Fallback — generic, not AI/ML specific
    name = profile.get("name", "Candidate")
    batch = profile.get("graduation_batch", "2027")
    roles = profile.get("target_roles", [])
    roles_str = (
        ", ".join(roles) if roles else "Software Engineer, SDE, Backend Engineer"
    )
    max_exp = profile.get("max_experience_years", 2)

    return (
        f"CANDIDATE: {name}\n"
        f"GRADUATION: {batch} batch | EXPERIENCE: fresher (0-{max_exp} years)\n"
        f"TARGET ROLES: {roles_str}\n"
        f"INCLUDE: full-time jobs AND paid internships.\n"
        f"VETOES: IT outsourcing without engineering title, {max_exp + 1}+ years required, "
        f"non-technical roles."
    )


def get_verifier_system_prompt(profile: dict | None = None) -> str:
    """
    Build the second-pass verifier prompt from the live resume profile.

    v2.2: removed hardcoded 'AI/ML' references — verifier now checks against
    the candidate's actual target roles and tech stack, not a generic AI/ML filter.
    Works correctly for Java/Backend engineers, AI engineers, or any other profile.
    """
    profile = profile if profile is not None else load_profile()
    candidate = get_scoring_prompt(profile)
    batch = profile.get("graduation_batch", "2027")
    max_exp = profile.get("max_experience_years", 2)

    # Extract target roles for explicit mention in verifier
    target_roles = profile.get("target_roles", [])
    roles_str = (
        ", ".join(target_roles)
        if target_roles
        else "Software Engineer, Backend Engineer, SDE"
    )

    # Extract hard vetoes for explicit mention
    hard_vetoes = profile.get("hard_vetoes", [])
    veto_lines = ""
    if hard_vetoes:
        veto_items = []
        for v in hard_vetoes:
            if isinstance(v, dict):
                veto_items.append(f"- {v.get('veto', '')}")
            else:
                veto_items.append(f"- {v}")
        veto_lines = "\nHARD VETOES (auto-SKIP these):\n" + "\n".join(veto_items)

    return f"""You are a strict senior recruiter doing a SECOND OPINION on listings already scored HIGH.
Be MORE critical than the first pass. Judge authenticity — reject vague, recycled, or misleading posts.

CANDIDATE PROFILE (from resume):
{candidate}

TARGET ROLES: {roles_str}
{veto_lines}

SCOPE: Score BOTH full-time jobs AND internships.
Paid internships with a stipend or PPO potential are valid HIGH matches.

For each listing answer ALL THREE questions:
1. Does this GENUINELY involve hands-on engineering work matching the candidate's
   target roles and tech stack listed above (not keyword stuffing, not vague "tech" roles)?
2. Is the company a product company, reputable startup, or good tech employer
   (not IT services/outsourcing/body-shopping)?
3. Is this truly open to someone graduating {batch}
   (fresher, intern, 0-{max_exp} years, campus hire, or explicit intern role)?

ALL THREE yes → keep HIGH.
Any doubt → downgrade to MEDIUM.
Clearly wrong, scam-like, unpaid, or matches a hard veto above → SKIP.

Return ONLY valid JSON array:
[{{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"posting_type":"job|internship|unknown","reason":"one sentence"}}]
"""


def deduplicate_batch(jobs: list) -> list:
    """
    Cross-source dedup that catches same job posted across multiple sources.
    Keeps the richest record (longest description) per logical role.
    """
    best_by_id: dict[str, dict] = {}
    best_by_title: dict[str, dict] = {}

    def _title_key(job: dict) -> str:
        t = (job.get("title") or "").strip().lower()
        c = (job.get("company") or "").strip().lower()
        return f"{t}|{c}"

    def _is_richer(candidate: dict, existing: dict) -> bool:
        return len(candidate.get("description") or "") > len(
            existing.get("description") or ""
        )

    for job in jobs:
        jid = (job.get("job_id") or "").strip()
        tkey = _title_key(job)
        valid_tkey = tkey != "|" and bool(tkey.replace("|", "").strip())

        if valid_tkey and tkey in best_by_title:
            existing = best_by_title[tkey]
            if _is_richer(job, existing):
                best_by_title[tkey] = job
                if jid:
                    best_by_id[jid] = job
                old_jid = (existing.get("job_id") or "").strip()
                if old_jid and old_jid in best_by_id:
                    best_by_id[old_jid] = job
            continue

        if jid and jid in best_by_id:
            existing = best_by_id[jid]
            if _is_richer(job, existing):
                best_by_id[jid] = job
                old_tkey = _title_key(existing)
                if old_tkey in best_by_title:
                    best_by_title[old_tkey] = job
            continue

        if jid:
            best_by_id[jid] = job
        if valid_tkey:
            best_by_title[tkey] = job

    seen_object_ids: set = set()
    result: list = []
    for job in list(best_by_id.values()) + list(best_by_title.values()):
        oid = id(job)
        if oid not in seen_object_ids:
            seen_object_ids.add(oid)
            result.append(job)

    return result


def load_seen(path: Path) -> dict:
    """
    Load seen jobs as {job_id: date_first_seen}.
    Automatically expires entries older than 7 days.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            today = date.today().isoformat()
            data = {jid: today for jid in data}

        cutoff = (date.today() - timedelta(days=7)).isoformat()
        cleaned = {
            jid: seen_date for jid, seen_date in data.items() if seen_date >= cutoff
        }

        expired = len(data) - len(cleaned)
        if expired > 0:
            logging.getLogger(__name__).info(
                f"Dedup: expired {expired} old entries (>7 days)"
            )
        return cleaned
    except Exception:
        return {}


def save_seen(path: Path, seen: dict) -> None:
    """Save seen jobs dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(seen, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def deduplicate(jobs: list, seen: dict) -> tuple:
    """
    Filter to only unseen jobs.
    Adds new job_ids to seen with today's date.
    Returns (fresh_jobs, updated_seen).
    """
    today = date.today().isoformat()
    fresh = []
    for job in jobs:
        jid = job.get("job_id", "")
        if jid and jid not in seen:
            fresh.append(job)
            seen[jid] = today
    return fresh, seen


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Copy config/config.example.json → config/config.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ── Profile-driven search keywords ───────────────────────────────────────────


def get_search_keywords(profile: dict | None = None) -> list[str]:
    """
    Build search keywords from profile.json's target_roles.

    For each target role, generates:
      - "{role}" (base search)
      - "{role} intern" (internship variant)
      - "{role} fresher" (fresher variant)
      - "{role} india" (location-scoped variant)

    No generic fallback: if profile.json has no target_roles, this is a
    configuration error, not something to silently paper over with a
    hardcoded SDE/backend list. main.py already SystemExit(1)s before this
    is ever called if profile.json is missing entirely — this raises for
    the remaining case where profile.json exists but target_roles is empty.
    """
    profile = profile if profile is not None else load_profile()
    roles = profile.get("target_roles", [])

    if not roles:
        raise ValueError(
            "profile.json has no target_roles — cannot derive search keywords. "
            "This is intent-driven by design: there is no generic fallback list. "
            "Run setup_profile.py or edit config/profile.json directly."
        )

    keywords = []
    for role in roles:
        role = role.strip()
        if not role:
            continue
        keywords.append(role)
        keywords.append(f"{role} intern")
        keywords.append(f"{role} fresher")
        keywords.append(f"{role} india")

    # de-dupe while preserving order
    seen: set = set()
    out: list = []
    for k in keywords:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)

    if not out:
        raise ValueError(
            "profile.json target_roles produced zero usable keywords after "
            "cleaning — check for blank/whitespace-only entries."
        )

    logging.getLogger(__name__).info(
        "Search keywords built from %d target_roles → %d keywords: %s...",
        len(roles),
        len(out),
        ", ".join(out[:4]),
    )
    return out

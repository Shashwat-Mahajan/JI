"""
sources/naukri.py
Naukri fresher/intern listings via Naukri's real search API.

--- CHANGED in this pass ---
  - Added explicit login-wall / auth-block detection. Naukri's API can
    respond with 200 but an HTML login/captcha page instead of real JSON,
    or with 401/403, when it decides this traffic needs a logged-in
    session. Previously this just looked like a generic "JSON parse error"
    with no clear signal of WHY. Now detected explicitly and logged as
    such, and triggers the circuit breaker immediately (no point retrying
    more queries against an active login requirement).
  - Reduced default query count (top 5), added circuit breaker on
    consecutive failures, increased inter-query delay — unchanged from
    last pass.
"""

import logging
import time
from datetime import date

from sources.base import resilient_get, new_session, validate_job

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "appid": "109",
    "systemid": "109",
    "naukri-platform": "desktop",
}

MAX_QUERIES = 5
CONSECUTIVE_FAIL_LIMIT = 3
QUERY_DELAY_SECONDS = 3

# Signals that the response is a login/auth wall rather than real job data,
# even when the HTTP status itself looked like success.
_LOGIN_WALL_MARKERS = (
    "login",
    "sign in",
    "signin",
    "captcha",
    "verify you are human",
    "access denied",
    "blocked",
)


def _looks_like_login_wall(r) -> bool:
    """
    Detect a login/auth/captcha wall disguised as a 200 response.
    Checks both status code and response content, since some blocks
    return 401/403 directly and others return 200 with an HTML page.
    """
    if r.status_code in (401, 403):
        return True

    content_type = r.headers.get("Content-Type", "").lower()
    if "application/json" not in content_type:
        # Got HTML/text instead of the expected JSON API response —
        # almost always a login/captcha/redirect page.
        snippet = r.text[:500].lower()
        if any(marker in snippet for marker in _LOGIN_WALL_MARKERS):
            return True
        # Even without an explicit marker, non-JSON content-type from a
        # JSON API endpoint is itself suspicious enough to flag.
        if "<html" in snippet:
            return True

    return False


def fetch_naukri(keywords: list | None = None) -> list:
    """
    Fetch Naukri jobs via the real search API, using profile/resume-derived
    keywords (capped at MAX_QUERIES). No generic fallback — an empty
    keywords list returns zero results rather than substituting a
    hardcoded AI/ML-specific list that wouldn't be intent-driven.
    Stops early on consecutive failures OR on detecting a login/auth wall —
    both indicate further queries this run won't succeed either.
    """
    if not keywords:
        log.warning("fetch_naukri called with empty keywords — returning no results")
        return []

    jobs = []
    seen_ids = set()
    dropped_invalid = 0
    failed_queries = 0
    consecutive_failures = 0
    login_wall_detected = False

    search_terms = keywords[:MAX_QUERIES]
    session = new_session(HEADERS)

    for idx, kw in enumerate(search_terms):
        if login_wall_detected:
            log.warning(
                "Naukri: login/auth wall detected — stopping remaining queries for this run"
            )
            break

        if consecutive_failures >= CONSECUTIVE_FAIL_LIMIT:
            log.warning(
                f"Naukri: {consecutive_failures} consecutive failures — "
                f"stopping early (tried {idx}/{len(search_terms)} queries)"
            )
            break

        url = (
            "https://www.naukri.com/jobapi/v3/search"
            f"?noOfResults=20&urlType=search_by_keyword&searchType=adv"
            f"&keyword={kw.replace(' ', '%20')}&experience=0&location=India"
        )
        r = resilient_get(session, url, timeout=(5, 20), max_attempts=1, backoff=2)

        if r is None:
            failed_queries += 1
            consecutive_failures += 1
            time.sleep(QUERY_DELAY_SECONDS)
            continue

        if _looks_like_login_wall(r):
            log.warning(
                f"Naukri: response for '{kw}' looks like a login/auth/captcha wall "
                f"(status={r.status_code}, content-type={r.headers.get('Content-Type','')}) "
                f"— Naukri is requiring authentication for this traffic. "
                f"No fix possible without a logged-in session; skipping rest of run."
            )
            login_wall_detected = True
            failed_queries += 1
            continue

        consecutive_failures = 0

        try:
            data = r.json()
        except Exception as e:
            log.debug(f"Naukri JSON parse error ({kw}): {e}")
            failed_queries += 1
            time.sleep(QUERY_DELAY_SECONDS)
            continue

        for j in data.get("jobDetails", []):
            title = j.get("title", "")
            if not title:
                continue
            jid = "nk_" + str(j.get("jobId", title))
            if jid in seen_ids:
                continue
            seen_ids.add(jid)

            placeholders = j.get("placeholders", [])
            location = (
                placeholders[0].get("label", "India") if placeholders else "India"
            )

            job = {
                "job_id": jid,
                "title": title,
                "company": j.get("companyName", "Unknown"),
                "location": location,
                "description": (j.get("jobDescription") or "")[:1200],
                "url": j.get("jdURL") or "https://www.naukri.com",
                "posted": date.today().isoformat(),
                "source": "Naukri",
            }

            if not validate_job(job):
                dropped_invalid += 1
                continue

            jobs.append(job)

        time.sleep(QUERY_DELAY_SECONDS)

    if login_wall_detected:
        log.info(
            f"Naukri: {len(jobs)} jobs — STOPPED due to login/auth wall. "
            f"Naukri's guest API access for this endpoint may have been revoked; "
            f"this source may need re-evaluation (different endpoint or accept it's no longer scrapeable without login)."
        )
    else:
        log.info(
            f"Naukri: {len(jobs)} jobs from {len(search_terms)} queries "
            f"({failed_queries} failed, {dropped_invalid} dropped for failed validation)"
        )
    return jobs

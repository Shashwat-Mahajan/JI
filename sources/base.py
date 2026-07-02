"""
sources/base.py — shared resilience helpers for every source fetcher.

Why this exists: every fetcher (LinkedIn, Naukri, career portals, public
APIs) was independently reinventing retry/backoff/timeout logic, fixed
piecemeal each time a specific source broke. This file centralizes that
logic once, so every source — current and future — inherits the same
robustness instead of needing its own bug fix later.
"""

import logging
import time
import requests

log = logging.getLogger(__name__)


def resilient_get(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple = (5, 20),       # (connect_timeout, read_timeout)
    max_attempts: int = 2,
    backoff: float = 3.0,
    **kwargs,
) -> requests.Response | None:
    """
    GET a URL with bounded retries and sensible backoff.

    - Fails fast on connect (5s) but allows more time to read a slow
      response (20s) instead of treating "slow" the same as "dead".
    - Retries ONLY on connection/timeout errors (network-level issues),
      not on every exception — a real 404/500 isn't worth retrying.
    - On 403/429 (blocked / rate-limited), backs off much harder (4x) and
      does NOT retry within this call — the caller's own loop will move on
      to the next item, which is safer than hammering a block.
    - Returns None on any unrecoverable failure. Caller decides what that
      means (skip this item, skip this source, etc.) — this function never
      raises for expected failure modes.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.get(url, timeout=timeout, **kwargs)

            if r.status_code == 200:
                return r

            if r.status_code in (403, 429):
                log.warning(f"Blocked/rate-limited ({r.status_code}): {url[:80]}")
                time.sleep(backoff * 4)
                return None

            log.debug(f"HTTP {r.status_code}: {url[:80]}")
            return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < max_attempts:
                wait = backoff * attempt
                log.debug(f"Network error attempt {attempt}/{max_attempts} on {url[:80]} — retrying in {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"Giving up on {url[:80]} after {max_attempts} attempts ({type(e).__name__})")

    return None


def new_session(headers: dict) -> requests.Session:
    """Create a session with default headers — connection/cookie reuse
    across requests within one source's fetch run."""
    s = requests.Session()
    s.headers.update(headers)
    return s


def validate_job(job: dict) -> bool:
    """
    Minimal sanity check before a job is allowed past the source layer.
    Rejects records missing the basics needed for dedup, scoring, and a
    usable report entry. Cheap — runs before any token-costing step.
    """
    if not job.get("title") or not job.get("title").strip():
        return False
    if not job.get("company") or not job.get("company").strip():
        return False
    url = job.get("url", "")
    if not url or not url.startswith("http"):
        return False
    if not job.get("job_id"):
        return False
    return True
"""
sources/career_portals.py
Profile-anchored — all keyword filtering derived strictly from profile.json.

CHANGES (profile-driven pass):
  - _PROFILE, _STRONG_KW, _WEAK_KW are no longer module-level singletons.
    Each public fetch_*() function calls _load_profile() fresh so that a
    newly written profile.json (after resume upload) is always picked up
    when main.py runs as a subprocess.
  - fetch_wellfound: role_slugs derived from profile target_roles instead
    of a hardcoded AI/ML-only list.
  - fetch_internshala: removed hardcoded AI/ML extra searches; uses only
    profile target_roles.
  - fetch_amazon: removed 3 hardcoded extra queries (SDE, ML, Applied
    Scientist) that were appended for every user regardless of profile.
  - fetch_microsoft: removed 3 hardcoded extra intern queries.
  - All other fetchers unchanged — they already used _PROFILE correctly.

API Status (June 2026):
  SmartRecruiters  → Swiggy ✅  Meesho ✅  Razorpay ✅  Atlassian ✅
  Greenhouse       → PhonePe ✅  Flipkart ✅  Walmart ✅  Adobe ✅
                     Freshworks ✅  BrowserStack ✅  Postman ✅
                     Chargebee ✅  Darwinbox ✅
  Lever            → Zomato ✅  CRED ✅  Groww ✅  Hasura ✅
                     Zepto ✅  CleverTap ✅
  Amazon           → country[]=IND unreliable; _is_india_or_remote() post-filter applied.
  Microsoft        → India jobs ✅
  Google           → Blocks bots ⚠  (returns 0, kept for future)
  Naukri direct    → ✅
  Oracle/JPMorgan  → OracleCloud HCM ✅
  Internshala      → ✅
  YCombinator      → ✅
  Wellfound        → ✅
"""

import json
import time
import hashlib
import logging
import requests
from datetime import date
from bs4 import BeautifulSoup
from pathlib import Path

log = logging.getLogger(__name__)

# ── Load profile ──────────────────────────────────────────────────────────────

_PROFILE_PATH = Path(__file__).parent.parent / "config" / "profile.json"


def _load_profile() -> dict:
    try:
        if _PROFILE_PATH.exists():
            return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ── Build keyword set strictly from profile ───────────────────────────────────


def _build_keywords_from_profile(profile: dict) -> tuple[list, list]:
    target_roles = profile.get("target_roles", [])
    skills = profile.get("technical_skills", {})
    exclusions = [e.lower() for e in profile.get("role_type_exclusions", [])]
    hard_vetoes = profile.get("hard_vetoes", [])

    veto_words = set()
    for v in hard_vetoes:
        text = v.get("veto", "") if isinstance(v, dict) else str(v)
        for w in text.lower().split():
            if len(w) > 3:
                veto_words.add(w)

    def _clean(terms: list) -> list:
        seen, out = set(), []
        for t in terms:
            tl = t.lower().strip()
            if not tl or tl in seen:
                continue
            if any(ex in tl for ex in exclusions):
                continue
            seen.add(tl)
            out.append(tl)
        return out

    strong = []
    for role in target_roles:
        role_l = role.lower().strip()
        strong.append(role_l)
        words = role_l.split()
        if len(words) >= 2:
            for j in range(len(words) - 1):
                strong.append(f"{words[j]} {words[j+1]}")
        role_indicators = {
            "engineer",
            "developer",
            "scientist",
            "researcher",
            "analyst",
            "architect",
            "specialist",
            "intern",
        }
        for w in words:
            if w in role_indicators:
                strong.append(w)

    weak = []
    for category, skill_list in skills.items():
        if not isinstance(skill_list, list):
            continue
        for skill in skill_list:
            skill_l = skill.lower().strip()
            if len(skill_l) <= 2:
                continue
            if skill_l in {"c", "c++", "java", "html", "css", "git", "linux"}:
                continue
            weak.append(skill_l)

    for project in profile.get("projects", []):
        for tech in project.get("tech_stack", []):
            tech_l = tech.lower().strip()
            if len(tech_l) > 2:
                weak.append(tech_l)

    return _clean(strong), _clean(weak)


def _get_keywords() -> tuple[list, list]:
    """Fresh keyword build on every call — always reflects current profile.json."""
    return _build_keywords_from_profile(_load_profile())


def _is_relevant(title: str, desc: str = "") -> bool:
    profile = _load_profile()
    strong_kw, weak_kw = _build_keywords_from_profile(profile)
    exclusions = profile.get("role_type_exclusions", [])

    title_l = title.lower()
    desc_l = desc.lower()

    for ex in exclusions:
        if ex.lower() in title_l:
            return False

    seniority_words = [
        "senior",
        "sr.",
        "staff",
        "principal",
        "director",
        "vp ",
        "chief",
        "head of",
        "lead ",
    ]
    is_senior = any(w in title_l for w in seniority_words)
    is_entry = any(
        w in title_l
        for w in ["intern", "fresher", "junior", "entry", "trainee", "graduate"]
    )
    if is_senior and not is_entry:
        return False

    if any(kw in title_l for kw in strong_kw):
        return True

    title_has_weak = any(kw in title_l for kw in weak_kw)
    desc_has_strong = any(kw in desc_l for kw in strong_kw)
    if title_has_weak and desc_has_strong:
        return True

    return False


# ── Shared request helpers ────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def _id(company: str, val: str) -> str:
    h = hashlib.md5(f"{company}_{val}".lower().encode()).hexdigest()[:8]
    return f"{company.lower()[:6]}_{h}"


def _get(
    url: str, timeout: int = 20, extra_headers: dict = None, cookies: dict = None
) -> requests.Response | None:
    h = {**HEADERS, **(extra_headers or {})}
    try:
        r = requests.get(url, headers=h, cookies=cookies or {}, timeout=timeout)
        if r.status_code == 200:
            return r
        log.debug(f"  HTTP {r.status_code}: {url[:80]}")
    except Exception as e:
        log.debug(f"  Error: {url[:60]} — {type(e).__name__}: {str(e)[:80]}")
    return None


# ── Amazon location helper ────────────────────────────────────────────────────

_INDIA_CITIES = {
    "india",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "mumbai",
    "chennai",
    "delhi",
    "new delhi",
    "noida",
    "gurugram",
    "gurgaon",
    "kolkata",
    "ahmedabad",
    "jaipur",
    "kochi",
    "in,",
    ", in",
}


def _is_india_or_remote(location: str) -> bool:
    if not location:
        return False
    loc_l = location.lower()
    if "remote" in loc_l:
        return True
    return any(tok in loc_l for tok in _INDIA_CITIES)


# ── SmartRecruiters ───────────────────────────────────────────────────────────


def _try_smartrecruiters_slug(identifier: str) -> tuple[int, str]:
    url = f"https://api.smartrecruiters.com/v1/companies/{identifier}/postings?limit=1"
    r = _get(url, timeout=15)
    if not r:
        return -1, identifier
    try:
        data = r.json()
        total = data.get("totalFound", -1)
        if "totalFound" not in data:
            return -1, identifier
        return total, identifier
    except Exception:
        return -1, identifier


def _fetch_smartrecruiters(company: str, slug_variants: list[str]) -> list:
    confirmed_slug = None
    confirmed_total = 0
    for slug in slug_variants:
        total, _ = _try_smartrecruiters_slug(slug)
        if total == -1:
            log.debug(f"  SmartRecruiters: slug '{slug}' invalid or blocked")
            time.sleep(0.5)
            continue
        confirmed_slug = slug
        confirmed_total = total
        break

    if confirmed_slug is None:
        log.warning(
            f"  SmartRecruiters/{company}: all slug variants failed — {slug_variants}"
        )
        return []

    if confirmed_total == 0:
        log.info(
            f"  SmartRecruiters/{confirmed_slug}: totalFound=0 — no open roles right now"
        )
        return []

    jobs = []
    offset, limit = 0, 100

    while True:
        url = (
            f"https://api.smartrecruiters.com/v1/companies/{confirmed_slug}/postings"
            f"?limit={limit}&offset={offset}"
        )
        r = _get(url, timeout=15)
        if not r:
            break
        try:
            data = r.json()
        except Exception:
            break

        postings = data.get("content", [])
        if not postings:
            break

        for p in postings:
            title = p.get("name", "")
            if not _is_relevant(title):
                continue

            loc = p.get("location", {})
            city, region, country = (
                loc.get("city", ""),
                loc.get("region", ""),
                loc.get("country", ""),
            )
            location_str = ", ".join(filter(None, [city, region])) or country or "India"
            if loc.get("remote"):
                location_str += " (Remote)"

            posting_id = p.get("id", "")
            detail_url = p.get("ref", "")
            description = ""
            apply_url = (
                f"https://jobs.smartrecruiters.com/{confirmed_slug}/{posting_id}"
            )

            if detail_url:
                dr = _get(detail_url, timeout=10)
                if dr:
                    try:
                        ddata = dr.json()
                        sections = ddata.get("jobAd", {}).get("sections", {})
                        description = " ".join(
                            sections.get(k, {}).get("text", "")
                            for k in (
                                "jobDescription",
                                "qualifications",
                                "additionalInformation",
                            )
                        )
                        apply_action = ddata.get("actions", {}).get("applyOnWeb", {})
                        if apply_action.get("url"):
                            apply_url = apply_action["url"]
                    except Exception:
                        pass

            if not _is_relevant(title, description):
                continue

            jobs.append(
                {
                    "job_id": _id(confirmed_slug, str(posting_id) or title),
                    "title": title,
                    "company": company,
                    "location": location_str,
                    "description": description[:1200],
                    "url": apply_url,
                    "posted": (p.get("releasedDate") or "")[:10]
                    or date.today().isoformat(),
                    "source": f"{company} Careers (SmartRecruiters)",
                }
            )

        offset += limit
        if offset >= confirmed_total:
            break
        time.sleep(0.5)

    return jobs


def fetch_swiggy() -> list:
    jobs = _fetch_smartrecruiters(
        "Swiggy", ["swiggy", "swiggy-india", "bundl-technologies"]
    )
    log.info(f"Swiggy: {len(jobs)}")
    return jobs


def fetch_meesho() -> list:
    jobs = _fetch_smartrecruiters(
        "Meesho", ["meesho", "meesho-engineering", "fashnear-technologies"]
    )
    log.info(f"Meesho: {len(jobs)}")
    return jobs


def fetch_razorpay() -> list:
    jobs = _fetch_smartrecruiters(
        "Razorpay", ["razorpay", "razorpay-india", "razorpay-software"]
    )
    log.info(f"Razorpay: {len(jobs)}")
    return jobs


def fetch_atlassian() -> list:
    profile = _load_profile()
    raw = _fetch_smartrecruiters(
        "Atlassian", ["atlassian", "atlassian-1", "atlassian-network-services"]
    )
    pref = [w.lower() for w in profile.get("location_preferences", ["india", "remote"])]
    jobs = [
        j
        for j in raw
        if any(p in j["location"].lower() for p in pref)
        or any(
            city in j["location"].lower()
            for city in [
                "remote",
                "anywhere",
                "bangalore",
                "bengaluru",
                "hyderabad",
                "pune",
                "mumbai",
            ]
        )
    ]
    log.info(f"Atlassian: {len(jobs)}")
    return jobs


# ── Greenhouse ────────────────────────────────────────────────────────────────


def _fetch_greenhouse(company: str, board: str) -> list:
    url_patterns = [
        f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true",
        f"https://boards.greenhouse.io/{board}/jobs.json",
        f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs",
    ]

    empty_count = 0
    tried_count = 0

    for url in url_patterns:
        r = _get(url)
        if not r:
            continue

        try:
            data = r.json()
        except Exception:
            continue

        job_list = data.get("jobs", data.get("postings", []))
        if not isinstance(job_list, list):
            continue

        tried_count += 1

        if not job_list:
            empty_count += 1
            log.debug(
                f"  Greenhouse/{board}: pattern returned 0 jobs ({url[:60]}) — trying next"
            )
            continue

        jobs = []
        for j in job_list:
            title = j.get("title", "")
            content = j.get("content", j.get("description", ""))
            if not _is_relevant(title, content):
                continue
            loc = j.get("location", {})
            if isinstance(loc, dict):
                loc = loc.get("name", "India")
            jobs.append(
                {
                    "job_id": _id(board, str(j.get("id", title))),
                    "title": title,
                    "company": company,
                    "location": loc,
                    "description": (content or "")[:1200],
                    "url": j.get(
                        "absolute_url",
                        j.get("url", f"https://boards.greenhouse.io/{board}"),
                    ),
                    "posted": (j.get("updated_at", j.get("created_at", "")) or "")[:10]
                    or date.today().isoformat(),
                    "source": f"{company} Careers (Greenhouse)",
                }
            )
        log.debug(
            f"  Greenhouse/{board}: {len(jobs)} relevant from {len(job_list)} total"
        )
        return jobs

    if tried_count > 0 and empty_count == tried_count:
        log.info(
            f"  Greenhouse/{board}: 0 jobs — board exists but no openings right now"
        )
    else:
        log.warning(f"  Greenhouse/{board}: all URL patterns returned no data.")
    return []


def fetch_phonepe() -> list:
    jobs = _fetch_greenhouse("PhonePe", "phonepe")
    log.info(f"PhonePe: {len(jobs)}")
    return jobs


def fetch_flipkart() -> list:
    jobs = _fetch_greenhouse("Flipkart", "flipkart")
    log.info(f"Flipkart: {len(jobs)}")
    return jobs


def fetch_walmart() -> list:
    jobs = _fetch_greenhouse("Walmart Global Tech", "walmartglobaltech")
    log.info(f"Walmart: {len(jobs)}")
    return jobs


def fetch_adobe() -> list:
    jobs = _fetch_greenhouse("Adobe", "adobe")
    log.info(f"Adobe: {len(jobs)}")
    return jobs


def fetch_freshworks() -> list:
    jobs = _fetch_greenhouse("Freshworks", "freshworks")
    log.info(f"Freshworks: {len(jobs)}")
    return jobs


def fetch_browserstack() -> list:
    jobs = _fetch_greenhouse("BrowserStack", "browserstack")
    log.info(f"BrowserStack: {len(jobs)}")
    return jobs


def fetch_postman() -> list:
    jobs = _fetch_greenhouse("Postman", "postman")
    log.info(f"Postman: {len(jobs)}")
    return jobs


def fetch_chargebee() -> list:
    jobs = _fetch_greenhouse("Chargebee", "chargebee")
    log.info(f"Chargebee: {len(jobs)}")
    return jobs


def fetch_darwinbox() -> list:
    jobs = _fetch_greenhouse("Darwinbox", "darwinbox")
    log.info(f"Darwinbox: {len(jobs)}")
    return jobs


# ── Lever ─────────────────────────────────────────────────────────────────────


def _fetch_lever(company: str, slug: str) -> list:
    jobs = []
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json&limit=100"
    r = _get(url, timeout=15)
    if not r:
        return jobs
    try:
        raw = r.json()
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            data = raw.get("postings", raw.get("data", []))
        else:
            return jobs

        for j in data:
            title = j.get("text", "")
            desc = j.get("descriptionPlain", "") or BeautifulSoup(
                j.get("description", ""), "html.parser"
            ).get_text(" ")
            if not _is_relevant(title, desc):
                continue
            cats = j.get("categories", {})
            jobs.append(
                {
                    "job_id": _id(slug, j.get("id", title)),
                    "title": title,
                    "company": company,
                    "location": cats.get("location", "India"),
                    "description": desc[:1200],
                    "url": j.get("hostedUrl", f"https://jobs.lever.co/{slug}"),
                    "posted": (
                        date.fromtimestamp(j["createdAt"] / 1000).isoformat()
                        if j.get("createdAt")
                        else date.today().isoformat()
                    ),
                    "source": f"{company} Careers (Lever)",
                }
            )
    except Exception as e:
        log.error(f"Lever/{slug} error: {e}")
    return jobs


def fetch_zomato() -> list:
    jobs = _fetch_lever("Zomato", "zomato")
    log.info(f"Zomato: {len(jobs)}")
    return jobs


def fetch_cred() -> list:
    jobs = _fetch_lever("CRED", "cred")
    log.info(f"CRED: {len(jobs)}")
    return jobs


def fetch_groww() -> list:
    jobs = _fetch_lever("Groww", "groww")
    log.info(f"Groww: {len(jobs)}")
    return jobs


def fetch_hasura() -> list:
    jobs = _fetch_lever("Hasura", "hasura")
    log.info(f"Hasura: {len(jobs)}")
    return jobs


def fetch_zepto() -> list:
    jobs = _fetch_lever("Zepto", "zepto")
    log.info(f"Zepto: {len(jobs)}")
    return jobs


def fetch_clevertap() -> list:
    jobs = _fetch_lever("CleverTap", "clevertap")
    log.info(f"CleverTap: {len(jobs)}")
    return jobs


# ── Amazon ────────────────────────────────────────────────────────────────────


def fetch_amazon() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    # Queries derived purely from profile target_roles — no hardcoded fallback.
    target_roles = profile.get("target_roles", [])
    if not target_roles:
        log.warning("Amazon: no target_roles in profile.json — skipping fetch")
        return []
    queries = []
    for role in target_roles:
        queries.append((role, False))
        queries.append((role + " Intern", True))

    try:
        for q, is_intern in queries:
            for loc_param in ["&country%5B%5D=IND", "&remote=true"]:
                url = (
                    "https://www.amazon.jobs/en/search.json"
                    f"?base_query={requests.utils.quote(q)}"
                    f"{loc_param}"
                    "&category%5B%5D=software-development"
                    "&result_limit=25"
                    "&sort=recent"
                )
                if is_intern:
                    url += "&job_type%5B%5D=Full-Time%20Internship&job_type%5B%5D=Part-Time%20Internship"

                r = _get(url, timeout=20)
                if not r:
                    time.sleep(2)
                    continue
                try:
                    payload = r.json()
                except Exception:
                    continue

                raw_jobs = payload.get("jobs", [])
                before = len(raw_jobs)
                raw_jobs = [
                    j for j in raw_jobs if _is_india_or_remote(j.get("location", ""))
                ]
                dropped = before - len(raw_jobs)
                if dropped:
                    log.debug(
                        f"  Amazon: dropped {dropped} non-India/remote jobs for '{q}'"
                    )

                for j in raw_jobs:
                    title = j.get("title", "")
                    desc = j.get("description_short", "")
                    if not _is_relevant(title, desc):
                        continue
                    jid = _id("amazon", title + str(j.get("id_icims", "")))
                    if jid in seen:
                        continue
                    seen.add(jid)
                    jobs.append(
                        {
                            "job_id": jid,
                            "title": title,
                            "company": "Amazon",
                            "location": j.get("location", "India"),
                            "description": desc[:1200],
                            "url": "https://www.amazon.jobs" + j.get("job_path", ""),
                            "posted": (j.get("posted_date") or "")[:10],
                            "source": "Amazon Careers",
                        }
                    )
                time.sleep(1.5)

    except Exception as e:
        log.error(f"Amazon error: {e}")

    log.info(f"Amazon: {len(jobs)}")
    return jobs


# ── Google ────────────────────────────────────────────────────────────────────


def fetch_google() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()
    queries = list(profile.get("target_roles", []))

    try:
        for q in queries:
            url = (
                "https://careers.google.com/api/v3/search/"
                f"?q={requests.utils.quote(q)}"
                "&location=India&jex=ENTRY_LEVEL&page_size=20&page=1"
            )
            r = _get(
                url,
                extra_headers={
                    "Referer": "https://careers.google.com/",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            if not r:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            for j in data.get("jobs", []):
                title = j.get("title", "")
                desc_obj = j.get("description", {})
                desc = (
                    " ".join(
                        desc_obj.get("responsibilities", [])
                        + desc_obj.get("qualifications", [])
                    )
                    if isinstance(desc_obj, dict)
                    else str(desc_obj)
                )
                if not _is_relevant(title, desc):
                    continue
                jid = _id("google", j.get("id", title))
                if jid in seen:
                    continue
                seen.add(jid)
                locs = j.get("locations", [{}])
                jobs.append(
                    {
                        "job_id": jid,
                        "title": title,
                        "company": "Google",
                        "location": (
                            locs[0].get("display", "India") if locs else "India"
                        ),
                        "description": desc[:1200],
                        "url": f"https://careers.google.com/jobs/results/{j.get('id','')}",
                        "posted": (j.get("publish_date") or "")[:10],
                        "source": "Google Careers",
                    }
                )
            time.sleep(1.5)
    except Exception as e:
        log.error(f"Google error: {e}")
    log.info(f"Google: {len(jobs)}")
    return jobs


# ── Microsoft ─────────────────────────────────────────────────────────────────


def fetch_microsoft() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    # Queries derived purely from profile target_roles — no hardcoded fallback.
    target_roles = profile.get("target_roles", [])
    if not target_roles:
        log.warning("Microsoft: no target_roles in profile.json — skipping fetch")
        return []
    queries = []
    for role in target_roles:
        queries.append(role)
        queries.append(role + " Intern")

    ms_headers = {
        **HEADERS,
        "Accept": "application/json",
        "Referer": "https://jobs.careers.microsoft.com/",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        for q in queries:
            url = (
                "https://jobs.careers.microsoft.com/api/jobs"
                f"?l=en_us&pg=1&pgSz=20&q={requests.utils.quote(q)}"
                "&lc=India&et=Full-Time,Intern&format=json"
            )
            r = _get(url, timeout=20, extra_headers=ms_headers)
            if not r:
                time.sleep(1.5)
                continue
            try:
                data = r.json()
            except Exception:
                time.sleep(1.5)
                continue

            job_list = (
                data.get("operationResult", {}).get("result", {}).get("jobs")
                or data.get("value")
                or data.get("jobs")
                or []
            )
            for j in job_list:
                title = j.get("title") or j.get("Title") or ""
                desc = (
                    j.get("descriptionTeaser")
                    or j.get("description")
                    or j.get("Description")
                    or ""
                )
                if not _is_relevant(title, desc):
                    continue
                jid_raw = str(j.get("jobId") or j.get("JobId") or j.get("id") or "")
                jid = _id("microsoft", jid_raw or title)
                if jid in seen:
                    continue
                seen.add(jid)
                jobs.append(
                    {
                        "job_id": jid,
                        "title": title,
                        "company": "Microsoft",
                        "location": j.get("location") or j.get("Location") or "India",
                        "description": desc[:1200],
                        "url": f"https://jobs.careers.microsoft.com/global/en/job/{jid_raw}",
                        "posted": (j.get("postingDate") or j.get("PostedDate") or "")[
                            :10
                        ],
                        "source": "Microsoft Careers",
                    }
                )
            time.sleep(1.5)
    except Exception as e:
        log.error(f"Microsoft error: {e}")
    log.info(f"Microsoft: {len(jobs)}")
    return jobs


# ── Wellfound ─────────────────────────────────────────────────────────────────


def fetch_wellfound() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    # Slugs derived purely from profile target_roles — no hardcoded fallback.
    target_roles = profile.get("target_roles", [])
    role_slugs = list(
        dict.fromkeys(r.lower().strip().replace(" ", "-") for r in target_roles)
    )
    if not role_slugs:
        log.warning("Wellfound: no target_roles in profile.json — skipping fetch")
        return []

    try:
        for slug in role_slugs:
            url = (
                f"https://wellfound.com/role/r/{slug}"
                "?remote=true&locationSlugs[]=india"
            )
            r = _get(
                url,
                timeout=20,
                extra_headers={
                    "Accept": "application/json",
                    "Referer": "https://wellfound.com/jobs",
                },
            )
            if not r:
                time.sleep(2)
                continue
            try:
                data = r.json()
                items = data.get("jobListings") or data.get("job_listings") or []
                if not items and isinstance(data.get("startups"), list):
                    for startup in data["startups"]:
                        items.extend(startup.get("jobs", []))

                for j in items:
                    title = j.get("title", "")
                    desc = j.get("description", "")
                    company = (
                        (j.get("startup") or {}).get("name", "Startup")
                        if isinstance(j.get("startup"), dict)
                        else "Startup"
                    )
                    if not _is_relevant(title, desc):
                        continue
                    jid = _id("wellfound", str(j.get("id", title + company)))
                    if jid in seen:
                        continue
                    seen.add(jid)
                    locs = j.get("locationNames", [])
                    jobs.append(
                        {
                            "job_id": jid,
                            "title": title,
                            "company": company,
                            "location": locs[0] if locs else "Remote",
                            "description": desc[:1200],
                            "url": j.get("jobUrl", "https://wellfound.com/jobs"),
                            "posted": (j.get("createdAt") or "")[:10]
                            or date.today().isoformat(),
                            "source": "Wellfound",
                        }
                    )
            except Exception as e:
                log.debug(f"Wellfound parse error ({slug}): {e}")
            time.sleep(2)
    except Exception as e:
        log.error(f"Wellfound error: {e}")
    log.info(f"Wellfound: {len(jobs)}")
    return jobs


# ── Internshala ───────────────────────────────────────────────────────────────


def fetch_internshala() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    # CHANGED: only profile target_roles, no hardcoded AI/ML extras
    searches = list(profile.get("target_roles", ["software development"]))

    internshala_headers = {
        **HEADERS,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://internshala.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    fake_cookies = {"_ga": "GA1.1.1234567890.1700000000"}

    try:
        for kw in searches:
            slug = kw.lower().strip().replace(" ", "-")

            for endpoint_type in ("internships", "jobs"):
                url = f"https://internshala.com/{endpoint_type}/{slug}/ajax=true"
                r = _get(
                    url,
                    timeout=20,
                    extra_headers=internshala_headers,
                    cookies=fake_cookies,
                )

                if not r or not r.text.strip().startswith("{"):
                    continue

                try:
                    body = r.json()
                except Exception:
                    continue

                meta_key = (
                    "internships_meta"
                    if endpoint_type == "internships"
                    else "jobs_meta"
                )
                alt_meta_key = (
                    "internshipsMeta" if endpoint_type == "internships" else "jobsMeta"
                )
                items = body.get(meta_key) or body.get(alt_meta_key) or {}

                if not items:
                    continue

                for item_id, meta in items.items():
                    title = meta.get("profile_name") or meta.get("job_title") or ""
                    company = meta.get("company_name", "")
                    desc = " ".join(
                        filter(
                            None,
                            [
                                meta.get("other_stipend", ""),
                                meta.get("category", ""),
                                meta.get("job_type", ""),
                                meta.get("skills", ""),
                            ],
                        )
                    )
                    if not _is_relevant(title, desc):
                        continue
                    jid = _id(f"internshala_{endpoint_type}", item_id)
                    if jid in seen:
                        continue
                    seen.add(jid)
                    locs = meta.get("location_names", [])
                    detail_url = (
                        f"https://internshala.com/internship/detail/{item_id}"
                        if endpoint_type == "internships"
                        else f"https://internshala.com/job/detail/{item_id}"
                    )
                    jobs.append(
                        {
                            "job_id": jid,
                            "title": title,
                            "company": company,
                            "location": locs[0] if locs else "India",
                            "description": desc[:1200],
                            "url": detail_url,
                            "posted": (meta.get("start_date") or "")[:10]
                            or date.today().isoformat(),
                            "source": "Internshala",
                        }
                    )
            time.sleep(1.5)
    except Exception as e:
        log.error(f"Internshala error: {e}")
    log.info(f"Internshala: {len(jobs)}")
    return jobs


# ── YCombinator / Work at a Startup ──────────────────────────────────────────


def fetch_ycombinator() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    json_headers = {
        **HEADERS,
        "Accept": "application/json",
        "Referer": "https://www.workatastartup.com/jobs",
    }

    try:
        target_roles = profile.get("target_roles", [])
        if not target_roles:
            log.warning("YCombinator: no target_roles in profile.json — skipping fetch")
            return []
        for role in target_roles:
            url = (
                "https://www.workatastartup.com/jobs/api"
                f"?q={requests.utils.quote(role)}&remote=true&india=true"
            )
            r = _get(url, timeout=15, extra_headers=json_headers)

            job_list = []
            if r:
                try:
                    data = r.json()
                    job_list = data if isinstance(data, list) else data.get("jobs", [])
                except Exception:
                    html_url = (
                        f"https://www.workatastartup.com/jobs"
                        f"?q={requests.utils.quote(role)}&remote=true"
                    )
                    hr = _get(html_url, timeout=15)
                    if hr:
                        soup = BeautifulSoup(hr.text, "html.parser")
                        for script in soup.find_all(
                            "script", type="application/ld+json"
                        ):
                            try:
                                ld = json.loads(script.string)
                                if isinstance(ld, list):
                                    job_list.extend(ld)
                                elif ld.get("@type") == "JobPosting":
                                    job_list.append(ld)
                            except Exception:
                                pass

            for j in job_list:
                title = j.get("title", "")
                desc = j.get("description", "")
                if not _is_relevant(title, desc):
                    continue
                raw_id = str(j.get("id", title))
                jid = _id("yc", raw_id)
                if jid in seen:
                    continue
                seen.add(jid)
                co = j.get("company", {})
                locs = j.get("locations", ["Remote"])
                jobs.append(
                    {
                        "job_id": jid,
                        "title": title,
                        "company": (
                            co.get("name", "YC Startup")
                            if isinstance(co, dict)
                            else "YC Startup"
                        ),
                        "location": locs[0] if locs else "Remote",
                        "description": desc[:1200],
                        "url": f"https://www.workatastartup.com/jobs/{j.get('id','')}",
                        "posted": date.today().isoformat(),
                        "source": "YC Work at a Startup",
                    }
                )
            time.sleep(1.5)
    except Exception as e:
        log.error(f"YCombinator error: {e}")
    log.info(f"YCombinator: {len(jobs)}")
    return jobs


# ── Naukri direct ─────────────────────────────────────────────────────────────


def fetch_naukri_direct() -> list:
    profile = _load_profile()
    jobs = []
    seen = set()

    queries = []
    for role in profile.get("target_roles", []):
        queries.append(f"{role} fresher")
        queries.append(role)

    naukri_headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/json",
        "Content-Type": "application/json",
        "appid": "109",
        "systemid": "109",
        "naukri-platform": "desktop",
        "Referer": "https://www.naukri.com/",
        "x-http-method-override": "GET",
    }

    try:
        for q in queries:
            url = (
                "https://www.naukri.com/jobapi/v3/search"
                f"?noOfResults=20&urlType=search_by_keyword&searchType=adv"
                f"&keyword={requests.utils.quote(q)}"
                "&experience=0&experienceMax=1"
                "&location=India"
            )
            r = requests.get(url, headers=naukri_headers, timeout=15)
            if r.status_code not in (200, 201):
                continue
            try:
                for j in r.json().get("jobDetails", []):
                    title = j.get("title", "")
                    desc = j.get("jobDescription", "")
                    if not _is_relevant(title, desc):
                        continue
                    jid = _id("naukri", str(j.get("jobId", title)))
                    if jid in seen:
                        continue
                    seen.add(jid)
                    placeholders = j.get("placeholders", [])
                    location = (
                        placeholders[0].get("label", "India")
                        if placeholders
                        else "India"
                    )
                    jobs.append(
                        {
                            "job_id": jid,
                            "title": title,
                            "company": j.get("companyName", "Unknown"),
                            "location": location,
                            "description": desc[:1200],
                            "url": j.get("jdURL", "https://www.naukri.com"),
                            "posted": date.today().isoformat(),
                            "source": "Naukri",
                        }
                    )
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        log.error(f"Naukri direct error: {e}")
    log.info(f"Naukri direct: {len(jobs)}")
    return jobs


# ── Oracle ────────────────────────────────────────────────────────────────────


def fetch_oracle() -> list:
    jobs = []
    try:
        url = (
            "https://eeho.fa.us2.oraclecloud.com/hcmRestApi/resources/latest/"
            "recruitingCEJobRequisitions?onlyData=true"
            "&expand=requisitionList.secondaryLocations"
            "&finder=findReqs;siteNumber=CX_1,locationId=300000001201432&limit=30"
        )
        r = _get(url, timeout=20)
        if r:
            for j in r.json().get("items", []):
                for req in j.get("requisitionList", []):
                    title = req.get("Title", "")
                    if not _is_relevant(title, req.get("ShortDescriptionStr", "")):
                        continue
                    jobs.append(
                        {
                            "job_id": _id("oracle", str(req.get("Id", title))),
                            "title": title,
                            "company": "Oracle",
                            "location": req.get("PrimaryLocation", "India"),
                            "description": req.get("ShortDescriptionStr", "")[:1200],
                            "url": f"https://careers.oracle.com/jobs/#en/sites/jobsearch/requisitions/{req.get('Id','')}",
                            "posted": date.today().isoformat(),
                            "source": "Oracle Careers",
                        }
                    )
    except Exception as e:
        log.error(f"Oracle error: {e}")
    log.info(f"Oracle: {len(jobs)}")
    return jobs


# ── JPMorgan ──────────────────────────────────────────────────────────────────


def fetch_jpmorgan() -> list:
    jobs = []
    try:
        url = (
            "https://jpmc.fa.oraclecloud.com/hcmRestApi/resources/latest/"
            "recruitingCEJobRequisitions?onlyData=true"
            "&finder=findReqs;siteNumber=CX_1,locationId=300000001506152&limit=30"
        )
        r = _get(url, timeout=20)
        if r:
            for j in r.json().get("items", []):
                for req in j.get("requisitionList", []):
                    title = req.get("Title", "")
                    if not _is_relevant(title, req.get("ShortDescriptionStr", "")):
                        continue
                    jobs.append(
                        {
                            "job_id": _id("jpmorgan", str(req.get("Id", title))),
                            "title": title,
                            "company": "JPMorgan Chase",
                            "location": req.get("PrimaryLocation", "India"),
                            "description": req.get("ShortDescriptionStr", "")[:1200],
                            "url": f"https://careers.jpmorgan.com/global/en/jobs/{req.get('Id','')}",
                            "posted": date.today().isoformat(),
                            "source": "JPMorgan Careers",
                        }
                    )
    except Exception as e:
        log.error(f"JPMorgan error: {e}")
    log.info(f"JPMorgan: {len(jobs)}")
    return jobs


# ── Master fetcher ────────────────────────────────────────────────────────────


def fetch_all_career_portals() -> list:
    all_jobs = []
    fetchers = [
        ("Swiggy", fetch_swiggy),
        ("Meesho", fetch_meesho),
        ("Atlassian", fetch_atlassian),
        ("Razorpay", fetch_razorpay),
        ("PhonePe", fetch_phonepe),
        ("Flipkart", fetch_flipkart),
        ("Walmart", fetch_walmart),
        ("Adobe", fetch_adobe),
        ("Freshworks", fetch_freshworks),
        ("BrowserStack", fetch_browserstack),
        ("Postman", fetch_postman),
        ("Chargebee", fetch_chargebee),
        ("Darwinbox", fetch_darwinbox),
        ("Zomato", fetch_zomato),
        ("CRED", fetch_cred),
        ("Groww", fetch_groww),
        ("Hasura", fetch_hasura),
        ("Zepto", fetch_zepto),
        ("CleverTap", fetch_clevertap),
        ("Amazon", fetch_amazon),
        ("Google", fetch_google),
        ("Microsoft", fetch_microsoft),
        ("Wellfound", fetch_wellfound),
        ("Internshala", fetch_internshala),
        ("YCombinator", fetch_ycombinator),
        ("Naukri direct", fetch_naukri_direct),
        ("Oracle", fetch_oracle),
        ("JPMorgan", fetch_jpmorgan),
    ]
    for name, fn in fetchers:
        try:
            result = fn()
            all_jobs.extend(result)
            log.info(f"  ✓ {name}: {len(result)} jobs")
        except Exception as e:
            log.error(f"Portal error — {name}: {e}")
        time.sleep(0.5)

    log.info(f"Career portals total: {len(all_jobs)} jobs")
    return all_jobs

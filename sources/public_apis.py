"""
sources/public_apis.py
Free public job APIs — no authentication required.
Sources: Remotive, Arbeitnow, Jobicy, Himalayas, Freshersworld
"""

import json
import time
import hashlib
import logging
import requests
from datetime import datetime, date
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _id(source: str, val: str) -> str:
    h = hashlib.md5(f"{source}_{val}".lower().encode()).hexdigest()[:10]
    return f"{source}_{h}"


def _is_recent(date_str: str, days: int = 7) -> bool:
    if not date_str:
        return True
    try:
        pub = datetime.fromisoformat(date_str[:10])
        return (datetime.now() - pub).days <= days
    except Exception:
        return True


def fetch_remotive(keywords: list) -> list:
    """Remotive — free remote tech jobs API, no auth, category-based."""
    jobs     = []
    seen_ids = set()
    categories = ["software-dev", "data", "devops-sysadmin", "product"]

    try:
        for cat in categories:
            url = f"https://remotive.com/api/remote-jobs?category={cat}&limit=50"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                jid = str(j.get("id", ""))
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)
                title = j.get("title", "").lower()
                if not any(k in title for k in [
                    "engineer", "developer", "scientist", "analyst",
                    "ml", "ai", "data", "backend", "python", "nlp"
                ]):
                    continue
                jobs.append({
                    "job_id":      _id("remotive", jid),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("candidate_required_location", "Worldwide"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("publication_date") or "")[:10],
                    "source":      "Remotive",
                })
            time.sleep(0.5)
    except Exception as e:
        log.error(f"Remotive error: {e}")

    log.info(f"Remotive: {len(jobs)} jobs")
    return jobs


def fetch_arbeitnow(keywords: list) -> list:
    """Arbeitnow — European + remote tech jobs, free API."""
    jobs = []
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            for j in r.json().get("data", []):
                combined = (j.get("title","") + " " + j.get("description","")).lower()
                if not any(k.lower() in combined for k in keywords):
                    continue
                jobs.append({
                    "job_id":      _id("arbeitnow", j.get("slug", j.get("title",""))),
                    "title":       j.get("title", ""),
                    "company":     j.get("company_name", ""),
                    "location":    j.get("location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      date.today().isoformat(),
                    "source":      "Arbeitnow",
                })
    except Exception as e:
        log.error(f"Arbeitnow error: {e}")

    log.info(f"Arbeitnow: {len(jobs)} jobs")
    return jobs


def fetch_jobicy(keywords: list) -> list:
    """Jobicy — remote tech jobs, tag-based search."""
    jobs = []
    tags = ["software-engineer", "machine-learning", "data-engineer",
            "backend", "python", "ai"]
    try:
        for tag in tags:
            r = requests.get(
                f"https://jobicy.com/api/v2/remote-jobs?tag={tag}&count=20",
                headers=HEADERS, timeout=15
            )
            if r.status_code != 200:
                continue
            for j in r.json().get("jobs", []):
                combined = (j.get("jobTitle","") + " " + j.get("jobDescription","")).lower()
                if not any(k.lower() in combined for k in keywords):
                    continue
                jobs.append({
                    "job_id":      _id("jobicy", str(j.get("id", j.get("jobTitle","")))),
                    "title":       j.get("jobTitle", ""),
                    "company":     j.get("companyName", ""),
                    "location":    j.get("jobGeo", "Remote"),
                    "description": (j.get("jobDescription") or "")[:1200],
                    "url":         j.get("url", ""),
                    "posted":      (j.get("pubDate") or "")[:10],
                    "source":      "Jobicy",
                })
            time.sleep(0.5)
    except Exception as e:
        log.error(f"Jobicy error: {e}")

    log.info(f"Jobicy: {len(jobs)} jobs")
    return jobs


def fetch_himalayas(keywords: list) -> list:
    """Himalayas.app — free remote jobs API, no auth, great for tech roles."""
    jobs     = []
    seen_ids = set()
    searches = [
        "machine learning engineer",
        "software engineer AI",
        "backend engineer python",
        "data scientist",
        "AI engineer",
        "NLP engineer",
        "generative AI",
        "LLM engineer",
        "machine learning intern",
        "software engineer intern",
        "AI intern",
        "data science intern",
        "backend intern",
    ]
    try:
        for kw in searches:
            url = f"https://himalayas.app/jobs/api?q={requests.utils.quote(kw)}&limit=20"
            r   = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            data  = r.json()
            items = data if isinstance(data, list) else data.get("jobs", [])
            for j in items:
                jid = str(j.get("id", ""))
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)
                title   = j.get("title", j.get("position", ""))
                company = j.get("company", {})
                if isinstance(company, dict):
                    company = company.get("name", "")
                jobs.append({
                    "job_id":      _id("himalayas", jid or title),
                    "title":       title,
                    "company":     company,
                    "location":    j.get("location", "Remote"),
                    "description": (j.get("description") or "")[:1200],
                    "url":         j.get("url", j.get("applyUrl", "")),
                    "posted":      (j.get("createdAt") or "")[:10] or date.today().isoformat(),
                    "source":      "Himalayas",
                })
            time.sleep(0.8)
    except Exception as e:
        log.error(f"Himalayas error: {e}")

    log.info(f"Himalayas: {len(jobs)} jobs")
    return jobs


def fetch_freshersworld(keywords: list) -> list:
    """Freshersworld — India's dedicated fresher job portal, JSON-LD scrape."""
    jobs = []
    urls = [
        "https://www.freshersworld.com/jobs/jobsearch/machine-learning-engineer-jobs-for-freshers?job_type=fresher",
        "https://www.freshersworld.com/jobs/jobsearch/artificial-intelligence-jobs-for-freshers?job_type=fresher",
        "https://www.freshersworld.com/jobs/jobsearch/software-engineer-jobs-for-freshers?job_type=fresher",
        "https://www.freshersworld.com/jobs/jobsearch/python-developer-jobs-for-freshers?job_type=fresher",
        "https://www.freshersworld.com/jobs/jobsearch/data-scientist-jobs-for-freshers?job_type=fresher",
        "https://www.freshersworld.com/jobs/jobsearch/backend-developer-jobs-for-freshers?job_type=fresher",
    ]
    try:
        for url in urls:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data  = json.loads(script.string)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get("@type") != "JobPosting":
                            continue
                        title = item.get("title", "")
                        if not title:
                            continue
                        org  = item.get("hiringOrganization", {})
                        loc  = item.get("jobLocation", {})
                        addr = loc.get("address", {}) if isinstance(loc, dict) else {}
                        jobs.append({
                            "job_id":      _id("freshersworld", title + item.get("url","")),
                            "title":       title,
                            "company":     org.get("name","") if isinstance(org,dict) else "",
                            "location":    addr.get("addressLocality","India"),
                            "description": (item.get("description") or "")[:1200],
                            "url":         item.get("url", url),
                            "posted":      (item.get("datePosted") or "")[:10] or date.today().isoformat(),
                            "source":      "Freshersworld",
                        })
                except Exception:
                    pass
            time.sleep(1.5)
    except Exception as e:
        log.error(f"Freshersworld error: {e}")

    log.info(f"Freshersworld: {len(jobs)} jobs")
    return jobs
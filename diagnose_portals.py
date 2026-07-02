"""
diagnose_portals.py — Run this on YOUR machine (not in Claude's sandbox).

Tests every career portal endpoint directly and prints:
  - HTTP status code
  - Whether JSON parsed successfully
  - Raw job count BEFORE the _is_relevant() filter
  - Raw job count AFTER the _is_relevant() filter

This isolates whether the problem is:
  (a) endpoint/slug broken (non-200, or JSON parse fails)
  (b) endpoint works but _is_relevant() filters out everything
  (c) endpoint works fine, problem is elsewhere

Usage:
  python diagnose_portals.py
"""

import requests
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}

AI_KEYWORDS = [
    "ai", "ml", "machine learning", "generative", "llm", "nlp",
    "deep learning", "data science", "artificial intelligence",
    "genai", "langchain", "python", "backend", "software engineer",
    "sde", "software developer", "data engineer", "platform",
    "computer vision", "research", "analytics", "algorithm",
    "intern", "internship", "trainee", "apprentice",
]

def is_relevant(title, desc=""):
    combined = (title + " " + desc).lower()
    return any(kw in combined for kw in AI_KEYWORDS)


def test_greenhouse(name, slug):
    print(f"\n=== {name} (Greenhouse: {slug}) ===")
    for url in [
        f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true",
        f"https://boards.greenhouse.io/{slug}/jobs.json",
    ]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            print(f"  {url}")
            print(f"  HTTP {r.status_code}")
            if r.status_code == 200:
                try:
                    data = r.json()
                    jobs = data.get("jobs", data.get("postings", []))
                    print(f"  Raw jobs: {len(jobs)}")
                    relevant = sum(1 for j in jobs if is_relevant(j.get("title",""), j.get("content","")))
                    print(f"  After _is_relevant filter: {relevant}")
                    if jobs[:3]:
                        print(f"  Sample titles: {[j.get('title','') for j in jobs[:3]]}")
                except Exception as e:
                    print(f"  JSON parse failed: {e}")
                    print(f"  Body preview: {r.text[:200]}")
        except Exception as e:
            print(f"  Request failed: {type(e).__name__}: {e}")


def test_lever(name, slug):
    print(f"\n=== {name} (Lever: {slug}) ===")
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                items = data if isinstance(data, list) else data.get("postings", [])
                print(f"  Raw jobs: {len(items)}")
                relevant = sum(1 for j in items if is_relevant(j.get("text",""), j.get("descriptionPlain","")))
                print(f"  After _is_relevant filter: {relevant}")
                if items[:3]:
                    print(f"  Sample titles: {[j.get('text','') for j in items[:3]]}")
            except Exception as e:
                print(f"  JSON parse failed: {e}")
    except Exception as e:
        print(f"  Request failed: {type(e).__name__}: {e}")


def test_simple_json(name, url, job_path_fn):
    """job_path_fn extracts the list of jobs from the parsed JSON."""
    print(f"\n=== {name} ===")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        print(f"  HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                data = r.json()
                jobs = job_path_fn(data)
                print(f"  Raw jobs: {len(jobs)}")
            except Exception as e:
                print(f"  JSON parse failed: {e}")
                print(f"  Body preview: {r.text[:200]}")
        else:
            print(f"  Body preview: {r.text[:200]}")
    except Exception as e:
        print(f"  Request failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("Testing Greenhouse boards...")
    test_greenhouse("Swiggy", "swiggy")
    test_greenhouse("Meesho", "meesho")
    test_greenhouse("Atlassian", "atlassian")

    print("\nTesting Lever boards...")
    test_lever("Razorpay", "razorpay")
    test_lever("PhonePe", "phonepe")

    print("\nTesting Amazon...")
    test_simple_json(
        "Amazon",
        "https://www.amazon.jobs/en/search.json?base_query=Software+Engineer&loc_query=India&result_limit=20",
        lambda d: d.get("jobs", []),
    )

    print("\nDone. Paste this full output back to diagnose root causes.")
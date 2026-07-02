"""
find_slug.py — Run locally to discover a company's REAL ats + board slug.

Most Indian unicorns (Swiggy, Meesho, Razorpay, PhonePe) do NOT use
Greenhouse or Lever — they use Darwinbox, Keka, SmartRecruiters, or a
custom-built careers page. This script checks all common ATS platforms
for a given company name and reports which one (if any) responds.

Usage:
    python find_slug.py swiggy
    python find_slug.py razorpay
    python find_slug.py atlassian
"""

import sys
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def check(name, url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        status = "✅ FOUND" if r.status_code == 200 else f"❌ {r.status_code}"
        print(f"  {status}  {name}: {url}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ❌ ERROR  {name}: {url}  ({type(e).__name__})")
        return False


def try_slug(company: str):
    slug = company.lower().replace(" ", "")
    print(f"\nSearching ATS platforms for '{company}' (slug guess: {slug})...")

    found_any = False
    found_any |= check("Greenhouse v1",  f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")
    found_any |= check("Greenhouse alt", f"https://job-boards.greenhouse.io/{slug}")
    found_any |= check("Lever",          f"https://api.lever.co/v0/postings/{slug}?mode=json")
    found_any |= check("SmartRecruiters",f"https://api.smartrecruiters.com/v1/companies/{slug}/postings")
    found_any |= check("Ashby",          f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    found_any |= check("Workday (generic)", f"https://{slug}.wd1.myworkdayjobs.com/wday/cxs/{slug}/jobs")

    if not found_any:
        print(f"\n  ⚠️  No standard ATS found for '{company}'.")
        print(f"  This company likely uses a custom careers page or Darwinbox/Keka")
        print(f"  (common for Indian companies). Check manually:")
        print(f"    https://www.google.com/search?q={company}+careers+site")
        print(f"  Look at the actual URL pattern once you find their jobs page —")
        print(f"  if it's a custom domain, you'll need a BeautifulSoup scraper")
        print(f"  instead of a JSON API call.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_slug.py <company_name>")
        sys.exit(1)
    try_slug(sys.argv[1])
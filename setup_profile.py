"""
setup_profile.py
Run once: python setup_profile.py
Reads your resume PDF + your current projects description,
extracts a structured profile using Groq (Cerebras fallback), saves to
config/profile.json (or PROFILE_OUT_PATH if set).

v3.2 changes (hosting support):
  - config.json is now OPTIONAL. Falls back to GROQ_API_KEY_1.._5 /
    CEREBRAS_API_KEY_1.._3 env vars if config.json is missing — matches
    main.py's env-var fallback pattern for Streamlit Cloud / Render hosting.
  - Resume path now checks RESUME_PATH env var FIRST (set by app.py per
    session), before falling back to the old fixed candidate paths.
  - Output path now checks PROFILE_OUT_PATH env var FIRST (set by app.py
    per session), before falling back to config/profile.json. This is
    what makes concurrent users on the hosted app not overwrite each
    other's extracted profile.
"""

import json
import os
import sys
from pathlib import Path
import fitz  # pymupdf
from nim_client import register_keys_from_config, call_llm_with_fallback, clean_json


def strip_thinking_block(raw: str) -> str:
    if "<think>" in raw:
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        else:
            raise ValueError(
                "Model response was truncated inside a <think> block before "
                "producing any JSON — increase max_tokens in the "
                "call_llm_with_fallback() call in extract_profile_with_nim()."
            )
    return raw.strip()


BASE = Path(__file__).parent
CONFIG = BASE / "config" / "config.json"


def _default_cfg() -> dict:
    """Build cfg from env vars when config.json isn't present (hosted env)."""
    cfg = {}
    for i in range(1, 6):
        v = os.getenv(f"GROQ_API_KEY_{i}")
        if v:
            cfg[f"GROQ_API_KEY_{i}"] = v
    for i in range(1, 4):
        v = os.getenv(f"CEREBRAS_API_KEY_{i}")
        if v:
            cfg[f"CEREBRAS_API_KEY_{i}"] = v
    return cfg


def _resume_path() -> Path | None:
    env_path = os.getenv("RESUME_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for candidate in [
        BASE / "config" / "resume.pdf",
        BASE / "resume.pdf",
        Path.home() / "resume.pdf",
    ]:
        if candidate.exists():
            return candidate
    return None


def _profile_out_path() -> Path:
    env_path = os.getenv("PROFILE_OUT_PATH")
    if env_path:
        p = Path(env_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return BASE / "config" / "profile.json"


CURRENT_PROJECTS = """
"""


EXTRACTION_PROMPT = """
You are a technical career advisor. Extract a complete, structured candidate profile
from the resume text provided.

CRITICAL RULES:
1. Every field must be derived from actual resume text. No generic/example values.
2. For derived fields (target_roles, hard_vetoes, role_type_exclusions), you MUST
   reason from the evidence before writing the value. Think: what does THIS resume
   actually show?
  Step 2: Look at PRIMARY tech stack → veto roles requiring completely different stack.
  Step 3: Look at internship history → if a short internship in unrelated tech
    (e.g. 1-month SAP), veto full-time roles in that unrelated tech.
  Step 4: Look at company type fit → if profile is product/startup focused,
    veto IT services outsourcing without engineering title.
  Step 5: Each veto MUST cite specific resume evidence. No generic vetoes.

Return ONLY a valid JSON object with EXACTLY this structure:

{
  "name": "full name",
  "email": "email",
  "phone": "phone",
  "linkedin": "linkedin url or username",
  "github": "github url or username",

  "education": {
    "degree": "degree name",
    "branch": "branch/specialization",
    "college": "college name",
    "graduation_year": "year",
    "cgpa": "cgpa value"
  },

  "current_status": "combine degree status + any active internship/job from experience section",

  "technical_skills": {
    "primary_languages": ["lang1", "lang2"],
    "ai_ml": ["only include if resume shows actual AI/ML work — frameworks, models, APIs used"],
    "backend": ["frameworks and tools used in backend projects or experience"],
    "cloud_devops": ["cloud/devops tools explicitly mentioned"],
    "databases": ["databases used in projects or experience"],
    "frontend": ["frontend frameworks/tools used"],
    "other": ["anything that doesn't fit above categories"]
  },

  "competitive_programming": {
    "leetcode_rating": "numeric rating if mentioned, else empty string",
    "leetcode_problems": "count or description exactly as stated in resume",
    "codechef_rating": "rating or stars if mentioned, else empty string",
    "other": "any other CP platform or achievement"
  },

  "projects": [
    {
      "name": "project name",
      "description": "what it does in 2-3 sentences",
      "tech_stack": ["tech1", "tech2"],
      "highlights": ["key achievement 1", "key achievement 2"],
      "status": "completed or in-progress"
    }
  ],

  "experience": [
    {
      "role": "role/title",
      "company": "company name",
      "duration": "duration",
      "description": "what you did — include all bullet points"
    }
  ],

  "achievements": ["achievement 1", "achievement 2"],

  "hackathons": [
    {
      "name": "hackathon name",
      "result": "rank/win/participant",
      "description": "what you built"
    }
  ],

  "publications": [],

  "target_roles": ["role1", "role2"],

  "target_companies": [],

  "location_preferences": ["city from resume contact/education", "Remote India"],

  "graduation_batch": "year from education",
  "experience_level": "fresher",
  "max_experience_years": 2,

  "role_type_exclusions": ["specific exclusion derived from this resume", "..."],

  "hard_vetoes": [
    {
      "veto": "specific veto derived from resume evidence",
      "resume_evidence": "exact quote or specific evidence from the resume"
    }
  ]
}
"""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pymupdf."""
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_profile_with_nim(
    resume_text: str, current_projects: str, cfg: dict
) -> dict:
    """Send resume text to Groq (Cerebras fallback) and extract structured profile."""
    register_keys_from_config(cfg)

    user_content = (
        f"RESUME TEXT:\n{resume_text}\n\n"
        f"CURRENT PROJECTS (not yet on resume):\n{current_projects}\n\n"
        "Extract the complete profile. Return JSON only. "
        "Remember: target_roles, target_companies, location_preferences, "
        "role_type_exclusions, and hard_vetoes must be DERIVED from the "
        "resume text above, not generic examples."
    )

    raw, provider = call_llm_with_fallback(
        system_prompt=EXTRACTION_PROMPT,
        user_content=user_content,
        max_tokens=8192,
    )
    print(f"  (extracted via {provider})")

    try:
        stripped = strip_thinking_block(raw)
        return json.loads(clean_json(stripped))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: JSON parse failed: {e}")
        print(f"Raw response (first 500 chars): {raw[:500]}")
        raise


def build_scoring_prompt(profile: dict) -> str:
    """Build a rich, dynamic CANDIDATE_PROFILE string from the extracted profile."""
    name = profile.get("name", "")
    edu = profile.get("education", {})
    status = profile.get("current_status", "")
    skills = profile.get("technical_skills", {})
    cp = profile.get("competitive_programming", {})
    exp = profile.get("experience_level", "fresher")
    batch = profile.get("graduation_batch", "2027")
    vetoes = profile.get("hard_vetoes", [])
    exclusions = profile.get("role_type_exclusions", [])
    targets = profile.get("target_roles", [])
    companies = profile.get("target_companies", [])
    locations = profile.get("location_preferences", [])

    all_skills = []
    for category, skill_list in skills.items():
        if skill_list:
            all_skills.extend(skill_list)
    skills_str = ", ".join(all_skills)

    projects_str = ""
    for p in profile.get("projects", []):
        tech = ", ".join(p.get("tech_stack", []))
        highlights = "; ".join(p.get("highlights", []))
        projects_str += (
            f"- {p['name']}: {p.get('description','')} Stack: {tech}. {highlights}\n"
        )

    exp_str = ""
    for e in profile.get("experience", []):
        exp_str += f"- {e.get('role','')} at {e.get('company','')} ({e.get('duration','')}): {e.get('description','')}\n"

    achievements = "; ".join(profile.get("achievements", []))
    hackathons = "; ".join(
        [f"{h['name']} ({h.get('result','')})" for h in profile.get("hackathons", [])]
    )
    publications = "; ".join(
        [p.get("title", "") for p in profile.get("publications", [])]
    )

    veto_lines = []
    for v in vetoes:
        if isinstance(v, dict):
            veto_lines.append(
                f"- {v.get('veto','')} (evidence: {v.get('resume_evidence','')})"
            )
        else:
            veto_lines.append(f"- {v}")
    vetoes_str = "\n".join(veto_lines)

    exclusions_str = ", ".join(exclusions) if exclusions else "none specified"

    prompt = f"""
CANDIDATE: {name}
STATUS: {status}
EDUCATION: {edu.get('degree','')} in {edu.get('branch','')} from {edu.get('college','')}
GRADUATION: {batch} batch | CGPA: {edu.get('cgpa','')}
EXPERIENCE LEVEL: {exp} (max {profile.get('max_experience_years',2)} years)

TECHNICAL SKILLS:
{skills_str}

COMPETITIVE PROGRAMMING:
LeetCode: {cp.get('leetcode_rating','')} rating, {cp.get('leetcode_problems','')} problems solved
CodeChef: {cp.get('codechef_rating','')}
{cp.get('other','')}

KEY PROJECTS:
{projects_str.strip()}

WORK EXPERIENCE:
{exp_str.strip() if exp_str.strip() else 'Internships and project-based experience only'}

ACHIEVEMENTS: {achievements}
HACKATHONS: {hackathons}
PUBLICATIONS: {publications}

TARGET ROLES (priority order, derived from resume evidence): {', '.join(targets) if targets else 'see technical skills above'}
TARGET COMPANIES: {', '.join(companies) if companies else 'no explicit preference — any reputable tech company'}
PREFERRED LOCATIONS: {', '.join(locations) if locations else 'not specified'}

ROLE TYPE EXCLUSIONS (this candidate wants HANDS-ON ENGINEERING work, NOT these):
{exclusions_str}

HARD VETOES (auto-disqualify, each tied to resume evidence):
{vetoes_str if vetoes_str else '- none derived'}
""".strip()

    return prompt


def main():
    # ── Load key config: config.json if present, else env vars (hosted) ──────
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text())
    else:
        print("config/config.json not found — falling back to env vars")
        cfg = _default_cfg()

    has_groq = bool(cfg.get("GROQ_API_KEY_1", "").strip())
    has_cerebras = bool(cfg.get("CEREBRAS_API_KEY_1", "").strip())
    if not has_groq and not has_cerebras:
        print(
            "ERROR: no GROQ_API_KEY_1 or CEREBRAS_API_KEY_1 found "
            "(checked config.json and env vars)"
        )
        print("Get a free Groq key at: https://console.groq.com")
        print("Get a free Cerebras key at: https://cloud.cerebras.ai")
        sys.exit(1)

    resume_path = _resume_path()
    if not resume_path:
        print("\nERROR: Resume PDF not found.")
        print("Set RESUME_PATH env var, or put your resume at one of:")
        print("  config/resume.pdf  ← recommended")
        print("  resume.pdf")
        sys.exit(1)

    print(f"\nReading resume: {resume_path}")
    resume_text = extract_text_from_pdf(resume_path)
    print(f"Extracted {len(resume_text)} characters from resume.")

    print("Sending to Groq (fallback: Cerebras) for profile extraction...")
    profile = extract_profile_with_nim(resume_text, CURRENT_PROJECTS, cfg)

    profile["_scoring_prompt"] = build_scoring_prompt(profile)

    profile_out = _profile_out_path()
    profile_out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"\nProfile saved to: {profile_out}")
    print("\nExtracted profile summary:")
    print(f"  Name:     {profile.get('name','')}")
    print(f"  College:  {profile.get('education',{}).get('college','')}")
    print(f"  Batch:    {profile.get('graduation_batch','')}")
    print(
        f"  Skills:   {len(sum(profile.get('technical_skills',{}).values(),[]))} skills found"
    )
    print(f"  Projects: {len(profile.get('projects',[]))} projects found")
    print(f"  Exp:      {len(profile.get('experience',[]))} roles found")
    print(f"  Target roles (derived): {profile.get('target_roles', [])}")
    print(f"  Role exclusions (derived): {profile.get('role_type_exclusions', [])}")
    print(f"  Hard vetoes (derived): {len(profile.get('hard_vetoes', []))} vetoes")
    print(f"\nRun 'python main.py' to start the agent with your profile.")


if __name__ == "__main__":
    main()

"""
app.py — Job Agent frontend
Place at D:/job_agent/app.py and run: streamlit run app.py
"""

import os
import uuid
import streamlit as st
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
CONFIG_DIR = BASE / "config"


def run_and_stream(cmd: list[str], log_box, env: dict) -> tuple[bool, list[str]]:
    """
    Run a subprocess and stream stdout+stderr live into log_box.
    Returns (success: bool, all_lines: list[str]).
    """
    all_lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr → stdout
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,  # <-- pass isolated env instead of relying on config.json
    )

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        all_lines.append(line)
        log_box.code("\n".join(all_lines[-50:]), language="text")

    proc.wait()
    success = proc.returncode == 0
    return success, all_lines


def show_full_log(all_lines: list[str], label: str = "Full log"):
    with st.expander(f"📋 {label} ({len(all_lines)} lines)", expanded=False):
        st.code("\n".join(all_lines), language="text")


def extract_error_lines(lines: list[str]) -> list[str]:
    keywords = (
        "error",
        "exception",
        "traceback",
        "failed",
        "critical",
        "errno",
        "exitcode",
        "arrowmemoryerror",
        "malloc",
    )
    return [l for l in lines if any(k in l.lower() for k in keywords)]


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Job Agent", page_icon="🎯", layout="centered")
st.title("🎯 Job Agent")
st.caption(
    "Upload your resume, enter your email, and we'll send you today's matching jobs."
)
st.divider()

resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
email = st.text_input("Your email address", placeholder="you@example.com")
st.divider()
st.info(
    "⏱️ Processing takes 10–30 minutes depending on your hardware. Don't close this tab."
)

if st.button("🚀 Find & Send Jobs", type="primary", use_container_width=True):

    if not resume_file:
        st.error("Please upload your resume.")
        st.stop()
    if not email or "@" not in email:
        st.error("Please enter a valid email address.")
        st.stop()

    # ── 1. Session-scoped resume path — avoids clobbering between concurrent users ──
    session_id = uuid.uuid4().hex[:12]
    session_dir = CONFIG_DIR / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    resume_dest = session_dir / "resume.pdf"
    resume_dest.write_bytes(resume_file.read())

    # ── 2. Build an isolated env for this run — no config.json writes at all ────────
    # Inherits real secrets (GROQ_API_KEY_*, SMTP_*, etc.) from the host env
    # (Streamlit Cloud / Render secrets), only overrides the per-request fields.
    run_env = os.environ.copy()
    run_env["RESUME_PATH"] = str(resume_dest)
    run_env["EMAIL_TO"] = email
    run_env["EMAIL_ENABLED"] = "true"
    # setup_profile.py should write profile.json into this session's folder too,
    # so concurrent runs don't share/overwrite each other's parsed profile.
    run_env["PROFILE_OUT_PATH"] = str(session_dir / "profile.json")

    # ── 3. Parse resume → profile.json ───────────────────────────────────
    st.markdown("**Step 1 of 2 — Parsing resume…**")
    log1 = st.empty()

    ok, lines1 = run_and_stream(
        [sys.executable, str(BASE / "setup_profile.py")],
        log1,
        run_env,
    )

    show_full_log(lines1, "setup_profile.py — full output")

    if not ok:
        st.error("❌ Resume parsing failed.")
        error_lines = extract_error_lines(lines1)
        if error_lines:
            st.markdown("**Errors found in output:**")
            st.code("\n".join(error_lines), language="text")
        else:
            st.warning("No obvious error lines detected — check the full log above.")
        st.stop()

    st.success("✅ Resume parsed")

    # ── 4. Run pipeline ───────────────────────────────────────────────────
    st.markdown("**Step 2 of 2 — Searching jobs and sending email…**")
    log2 = st.empty()

    ok, lines2 = run_and_stream(
        [sys.executable, str(BASE / "main.py")],
        log2,
        run_env,
    )

    show_full_log(lines2, "main.py — full output")

    if not ok:
        st.error("❌ Pipeline failed.")
        error_lines = extract_error_lines(lines2)
        if error_lines:
            st.markdown("**Errors found in output:**")
            st.code("\n".join(error_lines), language="text")

        agent_log = BASE / "logs" / "agent.log"
        if agent_log.exists():
            log_text = agent_log.read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(log_text.splitlines()[-100:])
            st.markdown("**agent.log — last 100 lines (full traceback is here):**")
            st.code(tail, language="text")
        else:
            st.warning("agent.log not found — process crashed before logging started.")

        st.stop()

    st.success(f"✅ Job digest sent to **{email}**! Check your inbox.")

    # ── 5. Cleanup — resume PDFs shouldn't linger on disk after the run ────
    try:
        resume_dest.unlink(missing_ok=True)
    except Exception:
        pass

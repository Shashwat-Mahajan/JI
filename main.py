"""
main.py — pipeline entry point.

v2.5 changes (hosting support):
  - load_profile() now checks PROFILE_OUT_PATH env var first (set by app.py
    per session on the hosted Streamlit app), falling back to the default
    config/profile.json for local/CLI/GitHub Actions use. This is what
    completes session isolation end-to-end alongside setup_profile.py.

v2.4 / v2.3 / v2.2 — see git history.
"""

import os

# NOTE: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set (conditionally) in
# sitecustomize.py, which runs before this file even starts executing.
# Do not set them again here — a second, unconditional copy is what broke
# the hosted deploy previously (it forced offline mode even with no cache).
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_PROXY"] = "huggingface.co,cdn-lfs.huggingface.co"

try:
    import tiktoken

    tiktoken.get_encoding("cl100k_base")
    print("tiktoken pre-loaded OK")
except Exception as e:
    print(f"tiktoken pre-load failed: {e}")

import logging
from datetime import date
from pathlib import Path

from utils import setup_logging, load_config, load_profile, get_search_keywords

BASE = Path(__file__).parent
LOGS = BASE / "logs" / "agent.log"
CONFIG = BASE / "config" / "config.json"

(BASE / "logs").mkdir(parents=True, exist_ok=True)
(BASE / "reports").mkdir(parents=True, exist_ok=True)

setup_logging(LOGS)
log = logging.getLogger(__name__)
log.info("Logger initialized: writing to %s", LOGS)


def _default_config() -> dict:
    return {
        "nvidia_nim_api_key": os.getenv("NVIDIA_NIM_API_KEY", ""),
        "nvidia_nim_api_key_2": os.getenv("NVIDIA_NIM_API_KEY_2", ""),
        "email_enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        "email_from": os.getenv("EMAIL_FROM", ""),
        "email_to": os.getenv("EMAIL_TO", ""),
        "smtp_host": os.getenv("SMTP_HOST", "smtp-relay.brevo.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_pass": os.getenv("SMTP_PASS", ""),
        "location": os.getenv("LOCATION", "India"),
        "llm_batch_size": 10,
        "search_keywords": [],
        "include_internships": True,
        "resume_path": os.getenv("RESUME_PATH", ""),
    }


def main():
    log.info("=" * 50)
    log.info(f"Job Intel Agent — {date.today().isoformat()}")
    log.info("=" * 50)

    try:
        from pipeline import run_pipeline

        log.info("Pipeline imported")
    except Exception:
        log.exception("Failed to import pipeline")
        raise

    try:
        cfg = load_config(CONFIG)
        log.info("Config loaded from config.json")
    except FileNotFoundError:
        log.warning("config/config.json not found — using env vars and defaults")
        cfg = _default_config()

    if os.getenv("NVIDIA_NIM_API_KEY"):
        cfg["nvidia_nim_api_key"] = os.getenv("NVIDIA_NIM_API_KEY")
    if os.getenv("NVIDIA_NIM_API_KEY_2"):
        cfg["nvidia_nim_api_key_2"] = os.getenv("NVIDIA_NIM_API_KEY_2")
    if os.getenv("RESUME_PATH"):
        cfg["resume_path"] = os.getenv("RESUME_PATH")
    # Allow env vars to override email_to/email_enabled too — this is how
    # app.py passes per-session values without ever writing config.json.
    if os.getenv("EMAIL_TO"):
        cfg["email_to"] = os.getenv("EMAIL_TO")
    if os.getenv("EMAIL_ENABLED"):
        cfg["email_enabled"] = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

    for i in range(1, 6):
        env_val = os.getenv(f"GROQ_API_KEY_{i}")
        if env_val:
            cfg[f"GROQ_API_KEY_{i}"] = env_val
    for i in range(1, 4):
        env_val = os.getenv(f"CEREBRAS_API_KEY_{i}")
        if env_val:
            cfg[f"CEREBRAS_API_KEY_{i}"] = env_val

    try:
        from nim_client import (
            register_keys_from_config,
            register_groq_keys_from_config,
            groq_pool_size,
        )

        register_keys_from_config(cfg)
        register_groq_keys_from_config(cfg)
    except Exception:
        log.exception("Failed to register API keys into key pool — scoring will fail")
        raise SystemExit(1)

    if groq_pool_size() == 0:
        log.error(
            "No Groq keys registered — scoring (Layers 2+3) would silently skip "
            "every job. Set GROQ_API_KEY_1 (and optionally _2.._5) in config.json "
            "or as env vars. Get a free key at: https://console.groq.com/keys"
        )
        raise SystemExit(1)

    # ── Profile loading — session-aware for hosted app ────────────────────────
    # PROFILE_OUT_PATH is set by app.py per Streamlit session so concurrent
    # users each get their own profile.json instead of clobbering a shared one.
    profile_path = os.getenv("PROFILE_OUT_PATH")
    profile = load_profile(Path(profile_path)) if profile_path else load_profile()
    if not profile:
        log.error(
            "profile.json not found (checked %s). A profile is required — "
            "run setup_profile.py first.",
            profile_path or "config/profile.json",
        )
        raise SystemExit(1)

    cfg["profile"] = profile
    log.info(
        f"Profile loaded: {profile.get('name', 'unknown')} "
        f"({profile.get('graduation_batch', '')} batch)"
    )

    if not cfg.get("search_keywords"):
        cfg["search_keywords"] = get_search_keywords(profile)
        log.info(
            f"Using {len(cfg['search_keywords'])} profile-driven search keywords "
            f"derived from target_roles in profile.json"
        )

    if not cfg["search_keywords"]:
        log.error(
            "search_keywords is empty after deriving from profile.json. "
            "Check that profile.json has a non-empty 'target_roles' list."
        )
        raise SystemExit(1)

    if cfg.get("include_internships", True):
        intern_kw = [k for k in cfg["search_keywords"] if "intern" in k.lower()]
        if len(intern_kw) < 3:
            target_roles = profile.get("target_roles", [])
            extra_intern = [
                f"{role} intern"
                for role in target_roles
                if f"{role} intern".lower()
                not in {k.lower() for k in cfg["search_keywords"]}
            ]
            cfg["search_keywords"] = list(
                dict.fromkeys(cfg["search_keywords"] + extra_intern)
            )
            log.info(
                f"Added {len(extra_intern)} intern keyword variants "
                f"from profile target_roles"
            )

    log.info(
        "Final search keywords (%d): %s",
        len(cfg["search_keywords"]),
        ", ".join(cfg["search_keywords"][:5])
        + ("..." if len(cfg["search_keywords"]) > 5 else ""),
    )

    log.info("Running pipeline...")
    try:
        result = run_pipeline(cfg)
    except Exception as e:
        err_str = str(e).lower()
        transient = any(
            s in err_str for s in ("connection error", "timeout", "timed out")
        )
        if transient:
            log.warning(
                "Pipeline crashed on a transient connection error — retrying once: %s",
                e,
            )
            try:
                result = run_pipeline(cfg)
            except Exception:
                log.exception("Pipeline crashed again on retry — giving up")
                raise
        else:
            log.exception("Pipeline crashed")
            raise

    log.info("Pipeline finished successfully")
    log.info(f"Result: {result}")
    log.info("Run complete.")


if __name__ == "__main__":
    main()

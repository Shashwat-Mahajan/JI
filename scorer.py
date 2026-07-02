"""
scorer.py — 4-layer hybrid scoring pipeline.

v3.2 changes vs v3.1:
  - Groq-only. Dropped the Cerebras fallback entirely — 5 Groq keys in
    rotation across MAX_CONCURRENT_BATCHES=5 give enough headroom for this
    workload without needing a second provider.
  - Primary/only model switched to meta-llama/llama-4-scout-17b-16e-instruct
    (replaces qwen/qwen3.6-27b). Non-reasoning instruct model — avoids the
    <think>-block token overhead that caused truncated JSON and 413s, and
    has a much higher free-tier TPM ceiling (30,000 vs 8,000).
  - _nim_call() now calls call_groq_rotated_pinned() directly instead of
    the Groq→Cerebras fallback wrapper.
  - batch_size default raised back to 20 (from 10) now that the higher TPM
    ceiling comfortably fits a 20-job batch (~8,000 tokens) per call.
  - Layer 2 batches stagger their start time (random 0.3–2s per batch after
    the first) to prevent the thundering-herd burst that consumed the
    entire RPM window in one second when all batches fired together.

v3.1 / v3.0 / v2.4 changes — see git history.

PUBLIC API (crew.py needs ZERO changes)
────────────────────────────────────────
  score_jobs_with_llm(jobs, api_key, batch_size=20) -> list
  CANDIDATE_PROFILE_FULL  (str)

PIPELINE
────────
Layer 1   BGE-small-en-v1.5  (bi-encoder, cosine similarity)
Layer 1.5 cross-encoder/ms-marco-MiniLM-L-6-v2  (cross-encoder, local CPU)
Layer 2   Groq llama-4-scout-17b-16e-instruct  (LLM-as-judge, batch scoring,
          CONCURRENT, key-pinned across 5 keys, staggered)
Layer 3   Groq llama-4-scout-17b-16e-instruct  (strict verifier, EDGE CASES
          ONLY: HIGH with score 65–75)
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import huggingface_hub.constants as _hf_const

    _hf_const.HF_HUB_OFFLINE = True
except Exception:
    pass

import json
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from nim_client import (
    call_groq_rotated_pinned,
    register_groq_keys_from_config,
    groq_pool_size,
    clean_json,
    stagger_batch_start,
    GROQ_PRIMARY_MODEL,
)
import db

log = logging.getLogger(__name__)

# ── Layer 1 top-K selection (BGE bi-encoder) ─────────────────────────────────
# v4.0: replaced fixed similarity thresholds with top-K selection. Thresholds
# didn't adapt to how many jobs were actually relevant on a given day (too
# strict = misses jobs, too loose = floods the LLM stage). Top-K bounds
# Layer 1.5/2/3 cost regardless of how many jobs get scraped that run.
LAYER1_TOP_K = 40

# ── Layer 1.5 auto-calibration percentile ────────────────────────────────────
CE_REJECT_PERCENTILE = 25

# ── Layer 3 edge-case range ───────────────────────────────────────────────────
EDGE_CASE_HIGH_MIN = 65
EDGE_CASE_HIGH_MAX = 75

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Scoring config ────────────────────────────────────────────────────────────
MAX_ATTEMPTS = 2
COVERAGE_MIN = 0.50

# Max concurrent Layer-2 batches in flight.
# Groq-only, 5 keys registered — one concurrent batch per key so each
# batch gets its own RPM/TPM budget instead of contending for one.
MAX_CONCURRENT_BATCHES = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_PROFILE_PATH = _BASE / "config" / "profile.json"
_UNSCORED_LOG_PATH = _BASE / "logs" / "unscored_jobs.json"


# ── Profile loading ───────────────────────────────────────────────────────────
def _load_profiles() -> tuple[str, str]:
    if not _PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"config/profile.json not found at {_PROFILE_PATH}. No generic "
            "hardcoded fallback profile — this system is intent-driven and "
            "requires a real profile. Run setup_profile.py to generate one "
            "from a resume."
        )
    try:
        data = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"config/profile.json is not valid JSON: {e}") from e

    compressed = data.get("_scoring_prompt", "")
    full = data.get("_full_profile", "")
    if not compressed:
        raise ValueError(
            "config/profile.json has no '_scoring_prompt' field — cannot "
            "build a candidate profile. No generic fallback is used. "
            "Run setup_profile.py to regenerate profile.json correctly."
        )

    log.info(
        "Profile loaded (~%d tokens compressed, ~%d tokens full)",
        len(compressed.split()),
        len(full.split()),
    )
    return compressed, full or compressed


CANDIDATE_PROFILE, CANDIDATE_PROFILE_FULL = _load_profiles()


# =============================================================================
# Profile-driven prompt builders  (unchanged from v2.2)
# =============================================================================


def _build_scoring_system(profile_prompt: str) -> str:
    return f"""Score jobs AND internships for this specific candidate. INTENT matching only, not keywords.

CANDIDATE PROFILE:
{profile_prompt}

SCORING AXES (100pts total):

role(30): How well does the job title and responsibilities match THIS candidate's TARGET ROLES?
  - Exact match to a target role listed above = 28-30
  - Closely related role using same tech stack = 18-27
  - Tangentially related role = 8-17
  - Unrelated to candidate's stack or goals = 0-7

skills(30): How much overlap between job requirements and THIS candidate's TECHNICAL SKILLS?
  - Strong overlap with primary languages/frameworks = 25-30
  - Moderate overlap = 12-24
  - Minimal overlap = 4-11
  - No overlap = 0-3

level(25): Is this role open to THIS candidate's experience level?
  - intern/fresher/entry/0-1yr/campus = 25
  - 0-2yr experience = 20
  - 0-3yr experience = 12
  - 3+yr required = 0

company(15): Company quality and fit for THIS candidate's goals?
  - Top product company / reputable AI or tech startup = 13-15
  - Mid-tier product company / growing startup = 8-12
  - IT services / outsourcing / body-shopping = 0-3

PRIORITY: 85+=HIGH  65-84=MEDIUM  50-64=LOW  <50=SKIP

VETO to SKIP (auto-disqualify without scoring):
  - Any job matching a HARD VETO listed in the candidate profile above
  - Any job type listed in ROLE TYPE EXCLUSIONS in the candidate profile above
  - Roles requiring experience beyond the candidate's max_experience_years
  - Vague/generic titles with no identifiable tech stack in the description
  - Clearly non-engineering roles (sales, marketing, content, HR)

Be conservative: if description is too thin to confirm a real engineering role, SKIP rather than guess HIGH.
Internships at good companies are equally valid as full-time roles — score them the same way.

CRITICAL: You MUST return a score for EVERY job_id in the input array.
Do not skip or omit any. Return ONLY a valid JSON array, no markdown:
[{{"job_id":"...","relevance_score":0-100,"score_breakdown":{{"role":0,"skills":0,"level":0,"company":0}},"match_reason":"1 sentence","key_match_skills":["s1"],"red_flags":[],"priority":"HIGH|MEDIUM|LOW|SKIP"}}]"""


def _build_verifier_system(profile_prompt: str) -> str:
    return f"""You are a strict senior recruiter reviewing borderline HIGH job listings.
These were scored HIGH (65-75) by a fast model — your job is to confirm or downgrade.

CANDIDATE PROFILE:
{profile_prompt}

For each listing answer ALL THREE questions:
1. Does this GENUINELY involve hands-on engineering work matching the candidate's primary tech stack?
2. Is the company a product company, reputable startup, or good tech employer
   (not IT services/outsourcing/body-shopping)?
3. Is this open to the candidate's experience level (fresher/intern/0-2yr/campus hire)?

ALL THREE yes → keep HIGH.
Any doubt → downgrade to MEDIUM.
Clearly wrong, off-profile, or matches a hard veto → SKIP.

Return ONLY valid JSON array:
[{{"job_id":"...","verified_priority":"HIGH|MEDIUM|LOW|SKIP","confidence":0-100,"reason":"one sentence"}}]"""


def _build_profile_anchor(profile_prompt: str) -> str:
    if not profile_prompt or len(profile_prompt.strip()) < 50:
        raise ValueError(
            "CANDIDATE_PROFILE is missing or too short to build a similarity "
            "anchor. No generic hardcoded fallback — run setup_profile.py to "
            "generate config/profile.json from a real resume, or pass a "
            "resume_embedding so Layer 1 uses the live resume anchor instead."
        )
    anchor_parts = []
    capture = False
    for line in profile_prompt.split("\n"):
        stripped = line.strip()
        if "TECHNICAL SKILLS" in stripped:
            capture = True
            continue
        if capture and stripped and not stripped.startswith("COMPETITIVE"):
            anchor_parts.append(stripped)
        if "COMPETITIVE" in stripped or "KEY PROJECTS" in stripped:
            capture = False
        if "TARGET ROLES" in stripped:
            roles_part = stripped.replace(
                "TARGET ROLES (priority order, derived from resume evidence):", ""
            ).strip()
            anchor_parts.append(roles_part)
    anchor = " ".join(anchor_parts)[:300]
    return anchor


def _build_ce_query(profile_prompt: str) -> str:
    if not profile_prompt or len(profile_prompt.strip()) < 50:
        raise ValueError(
            "CANDIDATE_PROFILE is missing or too short to build a cross-encoder "
            "query. No generic hardcoded fallback — run setup_profile.py first."
        )
    roles_line = batch_line = ""
    for line in profile_prompt.split("\n"):
        stripped = line.strip()
        if "TARGET ROLES" in stripped:
            roles_line = stripped.replace(
                "TARGET ROLES (priority order, derived from resume evidence):", ""
            ).strip()
        if "GRADUATION" in stripped:
            batch_line = stripped
    if not roles_line:
        raise ValueError(
            "CANDIDATE_PROFILE has no 'TARGET ROLES' line — cannot build an "
            "intent-driven cross-encoder query without knowing target roles. "
            "Check config/profile.json / setup_profile.py output."
        )
    return (
        f"Internship or fresher full-time software engineering role. "
        f"Target roles: {roles_line}. "
        f"Open to {batch_line or 'graduates'} or interns. "
        f"At a product company or tech startup, not IT outsourcing."
    )


# ── Retry suffix ──────────────────────────────────────────────────────────────
_JSON_RETRY_SUFFIX = (
    "\n\nCRITICAL: Your previous response was missing job_ids. "
    "You MUST include a score object for EVERY job_id listed in the input. "
    "Return ONLY a raw JSON array starting with [ and ending with ]. "
    "No markdown, no commentary, no omissions."
)


# ── LLM call helper ───────────────────────────────────────────────────────────
def _nim_call(
    api_key: str,
    system_prompt: str,
    user_content: str,
    key_index: int = 0,
) -> tuple[str, str]:
    """
    Groq-only: calls call_groq_rotated_pinned() directly with
    GROQ_PRIMARY_MODEL (meta-llama/llama-4-scout-17b-16e-instruct) — a
    non-reasoning instruct model with a 30K TPM free-tier ceiling, chosen
    specifically to avoid the <think>-block overhead and low TPM cap that
    qwen3.6-27b hit during batch scoring. No Cerebras fallback: with 5
    Groq keys in rotation across MAX_CONCURRENT_BATCHES, a single-provider
    setup has enough headroom for this workload.

    api_key kept for backward-compat with scorer.py callers — not used
    (keys come from the pool registered at startup via
    register_groq_keys_from_config()).

    Returns (json_str, source) — source is always "groq" here, kept as a
    tuple for compatibility with the rest of the pipeline's score_source
    labeling.
    """
    raw = call_groq_rotated_pinned(
        system_prompt,
        user_content,
        key_index=key_index,
        model=GROQ_PRIMARY_MODEL,
    )
    log.debug("LLM call key_index=%d served by: groq", key_index)
    return clean_json(raw), "groq"


# =============================================================================
# Shared helpers
# =============================================================================


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _log_unscored(jobs: list, reason: str) -> None:
    try:
        _UNSCORED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if _UNSCORED_LOG_PATH.exists():
            existing = json.loads(_UNSCORED_LOG_PATH.read_text(encoding="utf-8"))
        for j in jobs:
            existing.append(
                {
                    "job_id": j.get("job_id"),
                    "title": j.get("title"),
                    "company": j.get("company"),
                    "url": j.get("url"),
                    "reason": reason,
                    "date": time.strftime("%Y-%m-%d"),
                }
            )
        _UNSCORED_LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        log.debug("Could not write unscored audit log: %s", e)


# =============================================================================
# Layer 1 — BGE bi-encoder pre-filter
# =============================================================================

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _build_job_text(job: dict) -> str:
    title = job.get("title", "")
    company = job.get("company", "")
    description = (job.get("description") or job.get("snippet", ""))[:600]
    location = job.get("location", "")
    tags = " ".join(job.get("tags", []) or [])
    return f"{title} {title} at {company}. {location}. {tags}. {description}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _layer1_bge_filter(
    jobs: list, top_k: int = LAYER1_TOP_K, resume_embedding: np.ndarray | None = None
) -> tuple[list, list]:
    """
    Cache-aware top-K selection.

    1. Compute a stable job_hash per job. Check the SQLite cache (db.py) for
       an existing embedding — only NEW jobs get encoded this run. Cached
       jobs reuse their stored embedding untouched.
    2. Rank ALL jobs (cached + freshly-encoded) by cosine similarity to the
       matching anchor — a live resume_embedding if one was passed in (see
       resume_intake.py), otherwise the static profile.json anchor.
    3. Take the top-K by similarity and pass those on to the cross-encoder
       (Layer 1.5). Everything else is rejected without ever touching the
       LLM.

    Returns (top_k_jobs, rejected_jobs).
    """
    if not jobs:
        return [], []

    model = _get_embed_model()
    if resume_embedding is not None:
        anchor_vec = resume_embedding
    else:
        anchor_text = _build_profile_anchor(CANDIDATE_PROFILE)
        anchor_vec = model.encode(anchor_text, normalize_embeddings=True)

    # ── Cache lookup ──────────────────────────────────────────────────────
    hashes = [db.job_hash(j) for j in jobs]
    for j, h in zip(jobs, hashes):
        j["_job_hash"] = h

    cached = db.get_cached_job_embeddings(hashes)
    to_encode = [j for j, h in zip(jobs, hashes) if h not in cached]

    if to_encode:
        job_texts = [_build_job_text(j) for j in to_encode]
        new_vecs = model.encode(
            job_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
    else:
        new_vecs = []

    new_vec_map = {j["_job_hash"]: v for j, v in zip(to_encode, new_vecs)}

    # ── Assemble similarity for every job (cached or freshly-encoded) ──────
    scored: list[tuple[dict, float, np.ndarray]] = []
    for job in jobs:
        h = job["_job_hash"]
        vec = cached.get(h)
        if vec is None:
            vec = new_vec_map[h]
        sim = _cosine(anchor_vec, vec)
        job["embedding_sim"] = round(sim, 4)
        scored.append((job, sim, vec))

    # ── Persist cache: new jobs get their embedding stored, seen jobs get
    #    last_seen bumped (embedding=None means "don't overwrite") ─────────
    to_upsert = []
    for job, sim, vec in scored:
        h = job["_job_hash"]
        to_upsert.append((job, vec if h not in cached else None))
    db.upsert_jobs(to_upsert)

    # ── Top-K selection ──────────────────────────────────────────────────
    scored.sort(key=lambda t: t[1], reverse=True)
    top_jobs = [job for job, _, _ in scored[:top_k]]
    rejected = [job for job, _, _ in scored[top_k:]]

    for job in rejected:
        job.update(
            {
                "relevance_score": 0,
                "priority": "SKIP",
                "score_source": "embedding_reject_below_topk",
            }
        )

    log.info(
        "Layer 1 BGE: %d cached embeddings reused | %d newly encoded | "
        "top-%d selected (of %d) -> L1.5 | %d rejected",
        len(cached),
        len(to_encode),
        top_k,
        len(jobs),
        len(rejected),
    )
    return top_jobs, rejected


# =============================================================================
# Layer 1.5 — Cross-encoder re-ranker
# =============================================================================

_ce_model: CrossEncoder | None = None


def _get_ce_model() -> CrossEncoder:
    global _ce_model
    if _ce_model is None:
        log.info("Loading cross-encoder: %s", CE_MODEL_NAME)
        _ce_model = CrossEncoder(CE_MODEL_NAME)
    return _ce_model


def _layer1_5_cross_encoder(ambiguous: list) -> tuple[list, list]:
    if not ambiguous:
        return [], []

    model = _get_ce_model()
    query = _build_ce_query(CANDIDATE_PROFILE)

    pairs = [(query, _build_job_text(j)) for j in ambiguous]
    scores = model.predict(pairs, show_progress_bar=False)

    for job, score in zip(ambiguous, scores):
        job["ce_score"] = round(float(score), 4)

    score_arr = np.array([float(s) for s in scores])
    reject_thresh = float(np.percentile(score_arr, CE_REJECT_PERCENTILE))

    log.info(
        "Layer 1.5 CE: reject_thresh=%.3f (p%d) | score range [%.3f, %.3f]",
        reject_thresh,
        CE_REJECT_PERCENTILE,
        score_arr.min(),
        score_arr.max(),
    )

    to_llm: list = []
    ce_rejected: list = []

    for job in ambiguous:
        if job["ce_score"] >= reject_thresh:
            to_llm.append(job)
        else:
            job.update(
                {
                    "relevance_score": 0,
                    "priority": "SKIP",
                    "score_source": "ce_reject",
                }
            )
            ce_rejected.append(job)

    log.info("Layer 1.5 CE: %d -> LLM | %d auto-reject", len(to_llm), len(ce_rejected))
    return to_llm, ce_rejected


# =============================================================================
# Layer 2 — LLM scorer  (v3.0: Cerebras primary, staggered concurrent batches)
# =============================================================================


def _score_batch_with_retries(
    api_key: str,
    batch: list,
    batch_num: int,
    scoring_system: str,
) -> tuple[list | None, bool, str | None]:
    """
    Scores a single batch with retries.

    v3.1: uses _nim_call which routes Groq→Cerebras.
    key_index = batch_num - 1 (0-based) pins each concurrent batch to its
    own key so threads don't converge on the same provider key.

    Returns (scores, succeeded, source) — source is "groq" or "cerebras"
    (whichever actually served the successful call), or None on failure.
    """
    payload = [
        {
            "id": j["job_id"],
            "title": j["title"],
            "company": j["company"],
            "loc": (j.get("location") or "")[:30],
            "desc": (j.get("description") or "")[:400],
        }
        for j in batch
    ]
    batch_ids = {j["job_id"] for j in batch}
    last_err = None
    key_index = batch_num - 1

    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_content = (
            f"JOBS ({len(payload)} total — score ALL of them): {json.dumps(payload)}\n\n"
            "Return a JSON array with exactly one object per job_id above."
        )
        if attempt > 1:
            user_content += _JSON_RETRY_SUFFIX

        try:
            raw, source = _nim_call(
                api_key, scoring_system, user_content, key_index=key_index
            )
            scores = json.loads(raw)

            returned_ids = {s.get("job_id", s.get("id", "")) for s in scores}
            covered = len(returned_ids & batch_ids)
            coverage = covered / max(len(batch_ids), 1)

            if coverage < COVERAGE_MIN:
                last_err = ValueError(
                    f"Low coverage: {covered}/{len(batch_ids)} ({coverage:.0%})"
                )
                log.warning(
                    "  Batch %d attempt %d/%d: %s — retrying",
                    batch_num,
                    attempt,
                    MAX_ATTEMPTS,
                    last_err,
                )
                continue

            if attempt > 1:
                log.info("  Batch %d: recovered on attempt %d", batch_num, attempt)
            return scores, True, source

        except json.JSONDecodeError as e:
            last_err = e
            log.warning(
                "  Batch %d attempt %d/%d: bad JSON (%s) — retrying",
                batch_num,
                attempt,
                MAX_ATTEMPTS,
                e,
            )

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
                wait = 20 if attempt == 1 else 10
                log.info(
                    "  Batch %d attempt %d/%d: rate limited — waiting %ds",
                    batch_num,
                    attempt,
                    MAX_ATTEMPTS,
                    wait,
                )
                time.sleep(wait)
            elif "timeout" in msg or "deadline" in msg:
                log.warning(
                    "  Batch %d attempt %d/%d: timeout — retrying",
                    batch_num,
                    attempt,
                    MAX_ATTEMPTS,
                )
                time.sleep(3)
            else:
                log.error(
                    "  Batch %d attempt %d/%d: %s",
                    batch_num,
                    attempt,
                    MAX_ATTEMPTS,
                    e,
                )
                time.sleep(2)

    log.error("  Batch %d: all attempts failed — last error: %s", batch_num, last_err)
    return None, False, None


def _layer2_nim_score(
    jobs: list,
    api_key: str,
    batch_size: int,
    scoring_system: str,
) -> tuple[list, int]:
    """
    Groq-only, staggered concurrent batches.

    Each batch start is staggered by a random 0.3–2s delay (except batch 0)
    to prevent all concurrent batches from bursting into the rate limiter
    simultaneously — the thundering-herd fix.

    Workers = min(groq_pool, MAX_CONCURRENT_BATCHES). With 5 Groq keys and
    MAX_CONCURRENT_BATCHES=5, you get 5 in-flight batches, each pinned to
    its own key, each staggered.
    """
    if not jobs:
        return [], 0

    batches = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]
    total_batches = len(batches)

    workers = max(1, min(groq_pool_size(), MAX_CONCURRENT_BATCHES))
    log.info(
        "Layer 2: %d jobs → %d batches | %d workers (Groq pool=%d)",
        len(jobs),
        total_batches,
        workers,
        groq_pool_size(),
    )

    results: list = [None] * total_batches

    def _run_batch(idx: int):
        stagger_batch_start(idx)  # 0-cost for batch 0, jitter for rest
        batch = batches[idx]
        batch_num = idx + 1
        log.info(
            "  Batch %d/%d (%d jobs) starting...", batch_num, total_batches, len(batch)
        )
        scores, succeeded, source = _score_batch_with_retries(
            api_key, batch, batch_num, scoring_system
        )
        return idx, scores, succeeded, source

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_batch, i) for i in range(total_batches)]
        for future in as_completed(futures):
            idx, scores, succeeded, source = future.result()
            results[idx] = (scores, succeeded, source)

    scored: list = []
    unscored_count: int = 0

    for idx, batch in enumerate(batches):
        batch_num = idx + 1
        scores, succeeded, source = results[idx]

        if not succeeded or scores is None:
            _log_unscored(batch, "all_attempts_failed")
            unscored_count += len(batch)
            log.warning(
                "  Batch %d: excluded %d jobs — see logs/unscored_jobs.json",
                batch_num,
                len(batch),
            )
            continue

        score_map = {s.get("job_id", s.get("id", "")): s for s in scores}
        passed = skipped = 0
        score_source_label = f"{source}_llm" if source else "llm"

        for job in batch:
            s = score_map.get(job["job_id"])
            if not s:
                _log_unscored([job], "missing_from_response")
                unscored_count += 1
                continue
            if s.get("priority", "SKIP") == "SKIP":
                skipped += 1
                continue
            job.update(
                {
                    "relevance_score": _safe_int(s.get("relevance_score", 0)),
                    "score_breakdown": s.get("score_breakdown", {}),
                    "match_reason": s.get("match_reason", ""),
                    "key_match_skills": s.get("key_match_skills", []),
                    "red_flags": s.get("red_flags", []),
                    "priority": s.get("priority", "LOW"),
                    "score_source": score_source_label,
                }
            )
            scored.append(job)
            passed += 1

        log.info(
            "    -> Batch %d: %d relevant, %d skipped (served by %s)",
            batch_num,
            passed,
            skipped,
            source or "unknown",
        )

    return scored, unscored_count


# =============================================================================
# Layer 3 — LLM verifier (edge cases only: HIGH with score 65–75)
# =============================================================================


def _layer3_verify_edge_cases(all_scored: list, api_key: str) -> list:
    confirmed_high: list = []
    edge_cases: list = []
    non_high: list = []

    for job in all_scored:
        priority = job.get("priority", "LOW")
        score = job.get("relevance_score", 0)

        if priority == "HIGH" and score > EDGE_CASE_HIGH_MAX:
            job["score_source"] = job.get("score_source", "") + "+trusted"
            confirmed_high.append(job)
        elif priority == "HIGH" and EDGE_CASE_HIGH_MIN <= score <= EDGE_CASE_HIGH_MAX:
            edge_cases.append(job)
        else:
            non_high.append(job)

    if not edge_cases:
        log.info(
            "Layer 3: 0 edge-case HIGHs (65-75) — verifier skipped. "
            "%d trusted HIGH, %d MEDIUM/LOW pass-through.",
            len(confirmed_high),
            len(non_high),
        )
        return confirmed_high + non_high

    log.info(
        "Layer 3 verifier: checking %d edge-case HIGHs (65-75) | "
        "%d trusted HIGHs skip verifier",
        len(edge_cases),
        len(confirmed_high),
    )

    verifier_system = _build_verifier_system(CANDIDATE_PROFILE)

    try:
        payload = [
            {
                "job_id": j["job_id"],
                "title": j["title"],
                "company": j["company"],
                "job_type": j.get("job_type", "full-time"),
                "score": j.get("relevance_score", 0),
                "reason": (j.get("match_reason") or "")[:120],
            }
            for j in edge_cases
        ]
        user_content = (
            f"Verify these {len(edge_cases)} borderline HIGH jobs.\n"
            f"Return a JSON array with exactly {len(edge_cases)} objects.\n"
            f"{json.dumps(payload)}\nJSON only."
        )
        # Layer 3 is a single sequential call — key_index=0 is fine
        raw, verifier_source = _nim_call(
            api_key, verifier_system, user_content, key_index=0
        )
        results = json.loads(raw)
        log.info("Layer 3 verifier served by: %s", verifier_source)

        result_map = {r["job_id"]: r for r in results}
        downgraded = 0

        for job in edge_cases:
            v = result_map.get(job["job_id"])
            if not v:
                job["score_source"] += "+confirmed_no_verdict"
                confirmed_high.append(job)
                continue

            new_p = v.get("verified_priority", "HIGH")
            conf = v.get("confidence", 100)

            if new_p != "HIGH" and conf >= 85:
                job["priority"] = new_p
                job["match_reason"] = (
                    job.get("match_reason") or ""
                ) + f" [L3: {v.get('reason', '')}]"
                job["score_source"] += "+downgraded"
                downgraded += 1
                log.info(
                    "  Downgraded: %s @ %s -> %s (conf=%d%%)",
                    job["title"],
                    job["company"],
                    new_p,
                    conf,
                )
                non_high.append(job)
            else:
                job["score_source"] += "+confirmed"
                confirmed_high.append(job)
                log.info(
                    "  Confirmed HIGH: %s @ %s (conf=%d%%)",
                    job["title"],
                    job["company"],
                    conf,
                )

        log.info(
            "Layer 3 done: %d checked | %d downgraded | %d total confirmed HIGH",
            len(edge_cases),
            downgraded,
            len(confirmed_high),
        )

    except Exception as e:
        log.error("Layer 3 error: %s — keeping all edge-case HIGHs unchanged", e)
        confirmed_high.extend(edge_cases)

    return confirmed_high + non_high


# =============================================================================
# Public entry point — identical signature to old scorer.py
# =============================================================================


def score_jobs_with_llm(
    jobs: list,
    api_key: str,
    batch_size: int = 20,
    resume_embedding: np.ndarray | None = None,
) -> list:
    """
    4-layer hybrid scorer. crew.py needs ZERO changes — same signature.

    Layer 1   BGE bi-encoder         → auto-match / ambiguous / auto-reject
    Layer 1.5 Cross-encoder          → re-ranks ambiguous, auto-calibrated p25 threshold
    Layer 2   Groq (llama-4-scout)   → scores ambiguous survivors (profile-driven,
                                        CONCURRENT key-pinned, STAGGERED starts)
    Layer 3   Groq (llama-4-scout)   → verifies edge-case HIGHs (score 65-75) only

    Groq-only: no Cerebras fallback. api_key kept for backward compat — not
    used internally (keys come from the pool registered at startup via
    register_groq_keys_from_config() in your __main__.py / crew.py, using
    up to 5 GROQ_API_KEY_1..5 slots in config.json).

    NOTE on batch_size: default raised back to 20. llama-4-scout's 30,000
    TPM free-tier ceiling gives far more headroom than qwen3.6-27b's 8,000,
    so a 20-job batch (~8,000 tokens) comfortably fits with room to spare.
    """
    if not jobs:
        return []

    if groq_pool_size() == 0:
        log.error(
            "No Groq keys registered — Layers 2+3 will be skipped. "
            "Call register_groq_keys_from_config(cfg) at startup."
        )

    # Layer 1 — BGE, cache-aware top-K selection (v4.0: no more fixed
    # threshold auto-match/auto-reject tiers — see _layer1_bge_filter).
    # Uses a live resume embedding when one is supplied (resume_intake.py),
    # otherwise falls back to the static profile.json anchor.
    top_k_jobs, bge_rejected_jobs = _layer1_bge_filter(
        jobs, resume_embedding=resume_embedding
    )

    # Layer 1.5 — Cross-encoder
    to_llm, _ = _layer1_5_cross_encoder(top_k_jobs)

    # Layer 2 — LLM (Groq, concurrent, staggered)
    llm_scored: list = []
    unscored_count: int = 0

    if to_llm and groq_pool_size() > 0:
        scoring_system = _build_scoring_system(CANDIDATE_PROFILE)
        llm_scored, unscored_count = _layer2_nim_score(
            to_llm, api_key, batch_size, scoring_system
        )
    elif to_llm:
        _log_unscored(to_llm, "no_llm_keys_registered")
        unscored_count = len(to_llm)

    # Layer 3 — verifier (edge cases only)
    all_scored = llm_scored

    has_edge_cases = any(
        j.get("priority") == "HIGH"
        and EDGE_CASE_HIGH_MIN <= j.get("relevance_score", 0) <= EDGE_CASE_HIGH_MAX
        for j in all_scored
    )

    if groq_pool_size() > 0 and has_edge_cases:
        all_scored = _layer3_verify_edge_cases(all_scored, api_key)
    else:
        log.info("Layer 3: no edge-case HIGHs (65-75) found — verifier skipped.")

    results = [j for j in all_scored if j.get("priority") not in ("SKIP", None)]

    ce_rejected = len(top_k_jobs) - len(to_llm)

    log.info(
        "Scoring complete — %d relevant from %d jobs "
        "(L1-topk-selected=%d | L1-bge-reject=%d | L1.5-ce-reject=%d | "
        "L2-llm-band=%d | L2-scored=%d | unscored->logs=%d)",
        len(results),
        len(jobs),
        len(top_k_jobs),
        len(bge_rejected_jobs),
        ce_rejected,
        len(to_llm),
        len(llm_scored),
        unscored_count,
    )
    return results

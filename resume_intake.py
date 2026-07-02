"""
resume_intake.py — real resume-upload flow (replaces the manual, static
config/profile.json step for day-to-day runs).

On a new resume:
    1. Extract text (pymupdf, same as setup_profile.py's one-time flow)
    2. Hash it (db.resume_hash) — if we've seen this exact resume before,
       skip straight to the cache: no LLM call, no re-embedding.
    3. If new: ONE single LLM call extracts a small structured-fields object
       (skills, years of experience, target roles, location preference) —
       not the full heavyweight profile.json extraction setup_profile.py
       does; this is the fast per-run path.
    4. Compute the resume embedding (same BGE model scorer.py uses) once.
    5. Cache both (structured fields + embedding) in SQLite, keyed by
       resume_hash, via db.upsert_resume().

Job embeddings are resume-agnostic (see scorer.py's job cache) — only the
resume side needs recomputing per candidate, and only once per distinct
resume thanks to the hash cache here.
"""

import json
import logging
from pathlib import Path

import fitz  # pymupdf
import numpy as np

from nim_client import call_llm_with_fallback, clean_json, register_keys_from_config
import db

log = logging.getLogger(__name__)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        return "".join(page.get_text() for page in doc).strip()
    finally:
        doc.close()


_EXTRACTION_PROMPT = """Extract a compact structured summary from this resume text.
Return ONLY valid JSON, no markdown, no commentary, with EXACTLY this shape:

{
  "years_of_experience": 0,
  "target_roles": ["role1", "role2"],
  "must_have_skills": ["skill1", "skill2"],
  "location_preference": "city or 'Remote' or 'Any'",
  "summary_text": "2-3 sentence plain summary of the candidate's background, \
used to build the embedding anchor for job matching"
}

Every field must be grounded in the resume text. Do not invent skills or roles
not evidenced in the text. Empty list/0 is better than a hallucinated value.

RESUME TEXT:
{resume_text}
"""


def _extract_structured_fields(resume_text: str, cfg: dict) -> dict:
    register_keys_from_config(cfg)
    prompt = _EXTRACTION_PROMPT.replace("{resume_text}", resume_text[:8000])
    raw, provider = call_llm_with_fallback(
        system_prompt="You are a precise resume-parsing assistant. Output JSON only.",
        user_content=prompt,
    )
    try:
        return json.loads(clean_json(raw))
    except Exception as e:
        log.error("Resume field extraction returned invalid JSON: %s", e)
        return {
            "years_of_experience": 0,
            "target_roles": [],
            "must_have_skills": [],
            "location_preference": "Any",
            "summary_text": resume_text[:400],
        }


def _compute_embedding(text: str) -> np.ndarray:
    model = _get_embed_model()
    return np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)


def process_resume(pdf_path: Path, cfg: dict) -> dict:
    """
    Main entry point. Returns:
        {
          "resume_hash": str,
          "structured_fields": dict,
          "embedding": np.ndarray,
          "from_cache": bool,
        }
    """
    resume_text = extract_text_from_pdf(pdf_path)
    r_hash = db.resume_hash(resume_text)

    cached = db.get_cached_resume(r_hash)
    if cached is not None and cached.get("embedding") is not None:
        log.info(
            "Resume intake: cache hit for hash=%s — skipping LLM call + re-embedding",
            r_hash,
        )
        return {
            "resume_hash": r_hash,
            "structured_fields": cached["structured_fields"],
            "embedding": cached["embedding"],
            "from_cache": True,
        }

    log.info("Resume intake: new resume (hash=%s) — extracting + embedding", r_hash)
    fields = _extract_structured_fields(resume_text, cfg)
    anchor_text = fields.get("summary_text") or resume_text[:1000]
    embedding = _compute_embedding(anchor_text)

    db.upsert_resume(r_hash, fields, embedding)

    return {
        "resume_hash": r_hash,
        "structured_fields": fields,
        "embedding": embedding,
        "from_cache": False,
    }

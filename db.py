"""
db.py — persistent SQLite cache for job and resume embeddings.

Replaces logs/seen_jobs.json's "have I seen this job" role with a richer
cache that also stores the BGE embedding, so re-runs never recompute an
embedding for a job that hasn't changed. This is the single biggest
cost/time saver in the pipeline: BGE encode() is cheap per-call but adds
up across hundreds of jobs on every run.

Schema
──────
jobs(job_hash PK, url, raw_text, embedding BLOB, first_seen, last_seen)
resumes(resume_hash PK, structured_fields JSON, embedding BLOB, created_at)

Embeddings are stored as raw float32 bytes (np.ndarray.tobytes()) — small,
fast to (de)serialize, no JSON float bloat.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BASE = Path(__file__).parent
DB_PATH = BASE / "logs" / "job_intel.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_hash        TEXT PRIMARY KEY,
    url             TEXT,
    raw_text        TEXT,
    embedding       BLOB,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resumes (
    resume_hash        TEXT PRIMARY KEY,
    structured_fields  TEXT,
    embedding           BLOB,
    created_at          TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


# ── Hashing helpers ───────────────────────────────────────────────────────────


def job_hash(job: dict) -> str:
    """
    Stable identity hash for a job posting. Prefers URL (most stable across
    re-scrapes); falls back to title+company+description if URL is missing.
    """
    url = (job.get("url") or "").strip().lower()
    if url:
        basis = url
    else:
        basis = "|".join(
            [
                (job.get("title") or "").strip().lower(),
                (job.get("company") or "").strip().lower(),
                (job.get("description") or job.get("snippet") or "")[:200].strip().lower(),
            ]
        )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def resume_hash(resume_text: str) -> str:
    return hashlib.sha256(resume_text.strip().encode("utf-8")).hexdigest()[:24]


# ── Embedding (de)serialization ───────────────────────────────────────────────


def _encode_embedding(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ── Job cache API ─────────────────────────────────────────────────────────────


def get_cached_job_embeddings(job_hashes: list[str]) -> dict[str, np.ndarray]:
    """Fetch cached embeddings for the given job hashes. Missing ones are omitted."""
    if not job_hashes:
        return {}
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(job_hashes))
        rows = conn.execute(
            f"SELECT job_hash, embedding FROM jobs "
            f"WHERE job_hash IN ({placeholders}) AND embedding IS NOT NULL",
            job_hashes,
        ).fetchall()
        return {h: _decode_embedding(b) for h, b in rows}
    finally:
        conn.close()


def upsert_jobs(jobs_with_embeddings: list[tuple[dict, np.ndarray]]) -> None:
    """
    Insert new jobs or bump last_seen for existing ones. Only writes the
    embedding blob when it's actually new (embedding is None on the
    already-cached path since we don't recompute it).
    """
    if not jobs_with_embeddings:
        return
    conn = _connect()
    today = time.strftime("%Y-%m-%d")
    try:
        for job, vec in jobs_with_embeddings:
            h = job_hash(job)
            existing = conn.execute(
                "SELECT job_hash FROM jobs WHERE job_hash = ?", (h,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE jobs SET last_seen = ? WHERE job_hash = ?", (today, h)
                )
            else:
                blob = _encode_embedding(vec) if vec is not None else None
                conn.execute(
                    "INSERT INTO jobs (job_hash, url, raw_text, embedding, first_seen, last_seen) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        h,
                        job.get("url", ""),
                        (job.get("description") or job.get("snippet") or "")[:2000],
                        blob,
                        today,
                        today,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def prune_stale_jobs(days: int = 14) -> int:
    """Delete jobs not seen in `days` days. Returns count deleted."""
    conn = _connect()
    try:
        cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
        cur = conn.execute("DELETE FROM jobs WHERE last_seen < ?", (cutoff,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ── Resume cache API ───────────────────────────────────────────────────────────


def get_cached_resume(r_hash: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT structured_fields, embedding FROM resumes WHERE resume_hash = ?",
            (r_hash,),
        ).fetchone()
        if not row:
            return None
        fields_json, blob = row
        return {
            "structured_fields": json.loads(fields_json) if fields_json else {},
            "embedding": _decode_embedding(blob) if blob else None,
        }
    finally:
        conn.close()


def upsert_resume(r_hash: str, structured_fields: dict, embedding: np.ndarray) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO resumes (resume_hash, structured_fields, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                r_hash,
                json.dumps(structured_fields),
                _encode_embedding(embedding),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    finally:
        conn.close()

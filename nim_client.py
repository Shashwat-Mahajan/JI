"""
nim_client.py — LLM client(s) for the scoring pipeline.

v4.0 — crew.py's CrewAI wrapper was removed (replaced by pipeline.py, a
       plain function pipeline). make_client()/call_nim() (NVIDIA NIM)
       were only used by crew.py's verify_jobs_tool, which duplicated
       scorer.py's own Layer 3 edge-case verifier — that tool is gone, so
       these two functions are currently unused but left in place
       (harmless, no import cost) in case a standalone NIM path is wanted
       again later. Groq (register_groq_keys_from_config / pool) remains
       the actively used provider — scorer.py's batch scoring and
       resume_intake.py's structured-field extraction both go through it.
v3.1 — Groq primary + Cerebras fallback (replaces Gemini primary) for the
       scoring pipeline.
"""

import logging
import os
import random
import threading
import time

from openai import OpenAI

log = logging.getLogger(__name__)

# Non-reasoning instruct model — deliberately avoids qwen3.6-27b's <think>
# overhead, which was eating unpredictable chunks of the token budget and
# causing both truncated-JSON failures and 413s. llama-4-scout also has a
# much higher free-tier TPM ceiling (30,000 vs 8,000), which is the actual
# constraint for batch scoring — see GROQ_MAX_TPM_PER_KEY below.
GROQ_PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CEREBRAS_FALLBACK_MODEL = "llama-3.3-70b"

NIM_SCORING_MODEL = GROQ_PRIMARY_MODEL
GROQ_FALLBACK_MODEL = GROQ_PRIMARY_MODEL

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# ── NVIDIA NIM — single-key client used by crew.py's CrewAI agent loop ──────
# Kept as its own provider deliberately: CrewAI's tool-calling/ReAct format
# is what crew.py's agents were built and tuned against (see crew.py's
# _get_llm() docstring — llama-3.3-70b-instruct via NIM fixed an agent loop
# bug that a smaller model caused). Groq is used only inside scorer.py for
# raw batch-scoring calls, which are a completely separate, simpler
# JSON-in/JSON-out use case with no tool-calling involved — the two
# providers don't need to match.
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.3-70b-instruct"
NIM_REQUEST_TIMEOUT = 60.0

GROQ_MAX_RPM_PER_KEY = 26
CEREBRAS_MAX_RPM_PER_KEY = 26

# ── Per-key TOKEN budgets (TPM) — the limit actually responsible for 429s ────
# Groq free tier for meta-llama/llama-4-scout-17b-16e-instruct: 30,000 TPM
#   per key (documented). Cap set a little under that real limit to leave
#   headroom for clock skew / estimation error, same margin logic as before.
# Cerebras free tier: 60,000 TPM per key — still far above Groq's, tracked
#   anyway for safety.
GROQ_MAX_TPM_PER_KEY = 27000
CEREBRAS_MAX_TPM_PER_KEY = 58000

# ── Token estimation (no tokenizer dependency — conservative char heuristic) ─
# JSON-heavy text (our payload/response shape) runs denser than prose, so we
# use ~3.5 chars/token rather than the usual ~4, erring toward overestimating
# (a bit of extra wait is cheap; an underestimate risks a real 429).
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Rough, conservative token count for budget-reservation purposes only.
    True usage (from the API response) is used to correct the reservation
    after each call via _TokenLimiter.true_up()."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


GROQ_REQUEST_TIMEOUT = 40.0
CEREBRAS_REQUEST_TIMEOUT = 40.0

CONN_ERROR_BASE_WAIT = 2.0
RATE_LIMIT_BASE_WAIT = 12.0

BATCH_STAGGER_MAX_SECS = 2.5


class _RateLimiter:
    def __init__(self, max_rpm: int):
        self._max_rpm = max_rpm
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def wait_for_slot(self, key_hint: str = ""):
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                if len(self._timestamps) < self._max_rpm:
                    self._timestamps.append(now)
                    return

                sleep_for = self._timestamps[0] + 60.0 - now

            if sleep_for > 0:
                log.info(
                    "Rate limiter [%s]: at cap %d req/min — waiting %.1fs",
                    key_hint[-6:] if key_hint else "?",
                    self._max_rpm,
                    sleep_for,
                )
                time.sleep(sleep_for)


class _TokenLimiter:
    """
    Sliding 60s TOKEN-volume budget per key — the actual constraint behind
    most Groq 429s at this prompt size. Mirrors _RateLimiter's pattern but
    sums token counts instead of counting requests.

    Usage: reserve() BEFORE the call (blocks if it would exceed the cap),
    true_up() AFTER the call once real usage is known from the API response.
    This keeps the budget accurate over time without ever under-counting
    (which is what would let a 429 slip through).
    """

    def __init__(self, max_tpm: int):
        self._max_tpm = max_tpm
        self._entries: list[list] = []  # [[timestamp, token_count], ...]
        self._lock = threading.Lock()

    def reserve(self, estimated_tokens: int, key_hint: str = "") -> float:
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - 60.0
                self._entries = [e for e in self._entries if e[0] > cutoff]
                used = sum(e[1] for e in self._entries)

                if used + estimated_tokens <= self._max_tpm:
                    self._entries.append([now, estimated_tokens])
                    return now  # reservation id, used by true_up()

                if not self._entries:
                    # A single request's estimate alone exceeds the cap —
                    # nothing to wait on; let it through rather than hang.
                    self._entries.append([now, estimated_tokens])
                    return now

                sleep_for = self._entries[0][0] + 60.0 - now

            if sleep_for > 0:
                log.info(
                    "Token limiter [%s]: at cap %d TPM (in-flight~%d, need %d) "
                    "— waiting %.1fs",
                    key_hint[-6:] if key_hint else "?",
                    self._max_tpm,
                    used,
                    estimated_tokens,
                    sleep_for,
                )
                time.sleep(sleep_for)

    def true_up(self, reservation_id: float, actual_tokens: int):
        """Correct a prior estimate with the real usage from the API
        response, so the rolling budget reflects reality for the next
        caller instead of compounding estimation error."""
        if actual_tokens <= 0:
            return
        with self._lock:
            for entry in self._entries:
                if entry[0] == reservation_id:
                    entry[1] = actual_tokens
                    return


class _KeyPool:
    def __init__(self, name: str, max_rpm_per_key: int, max_tpm_per_key: int):
        self._name = name
        self._keys: list[str] = []
        self._lock = threading.Lock()
        self._limiters: dict[str, _RateLimiter] = {}
        self._token_limiters: dict[str, _TokenLimiter] = {}
        self._limiter_lock = threading.Lock()
        self._max_rpm = max_rpm_per_key
        self._max_tpm = max_tpm_per_key

    def add_key(self, api_key: str):
        key = (api_key or "").strip()
        if not key:
            return
        with self._lock:
            if key not in self._keys:
                self._keys.append(key)
                log.info(
                    "%s pool: added key (pool size=%d)", self._name, len(self._keys)
                )

    def key_at(self, offset: int) -> str:
        with self._lock:
            if not self._keys:
                return ""
            return self._keys[offset % len(self._keys)]

    def size(self) -> int:
        with self._lock:
            return len(self._keys)

    def limiter_for(self, key: str) -> _RateLimiter:
        with self._limiter_lock:
            if key not in self._limiters:
                self._limiters[key] = _RateLimiter(self._max_rpm)
            return self._limiters[key]

    def token_limiter_for(self, key: str) -> _TokenLimiter:
        with self._limiter_lock:
            if key not in self._token_limiters:
                self._token_limiters[key] = _TokenLimiter(self._max_tpm)
            return self._token_limiters[key]


_groq_pool = _KeyPool("Groq", GROQ_MAX_RPM_PER_KEY, GROQ_MAX_TPM_PER_KEY)


def register_groq_keys_from_config(cfg: dict):
    added = 0
    for field in (
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
        "GROQ_API_KEY_3",
        "GROQ_API_KEY_4",
        "GROQ_API_KEY_5",
        "GROQ_API_KEY 1",
        "GROQ_API_KEY 2",
        "GROQ_API_KEY 3",
        "GROQ_API_KEY 4",
        "GROQ_API_KEY 5",
    ):
        k = cfg.get(field, "").strip()
        if k:
            _groq_pool.add_key(k)
            added += 1

    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        _groq_pool.add_key(env_key)
        added += 1

    if added:
        log.info("Groq: %d key(s) registered (pool size=%d)", added, _groq_pool.size())
    else:
        log.warning("register_groq_keys_from_config: no Groq keys found")


def groq_pool_size() -> int:
    return _groq_pool.size()


def call_groq_rotated_pinned(
    system_prompt: str,
    user_content: str,
    key_index: int,
    model: str = GROQ_PRIMARY_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    size = max(1, _groq_pool.size())
    if _groq_pool.size() == 0:
        raise RuntimeError(
            "No Groq keys registered. "
            "Add GROQ_API_KEY_1/2/3 to config.json or set GROQ_API_KEY env var. "
            "Free keys: https://console.groq.com"
        )

    messages = _build_messages(system_prompt, user_content)
    last_err = None

    # Estimate total tokens (input + expected output) once for this call.
    # Output is sized off the actual request rather than a flat guess, since
    # it scales with batch size — keeps the reservation realistic instead of
    # wildly over- or under-shooting.
    estimated_tokens = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_content)
        + min(max_tokens, int(estimate_tokens(user_content) * 0.65) + 100)
    )

    for attempt in range(1, retries + 1):
        offset = (key_index + attempt - 1) % size
        current_key = _groq_pool.key_at(offset)

        client = OpenAI(
            api_key=current_key,
            base_url=GROQ_BASE_URL,
            timeout=GROQ_REQUEST_TIMEOUT,
            max_retries=0,
        )

        try:
            _groq_pool.limiter_for(current_key).wait_for_slot(current_key)
            reservation = _groq_pool.token_limiter_for(current_key).reserve(
                estimated_tokens, current_key
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                actual = resp.usage.total_tokens
                _groq_pool.token_limiter_for(current_key).true_up(reservation, actual)
            except Exception:
                pass  # usage field missing/unavailable — keep the estimate
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()

            if _is_rate_limit(err) and size > 1:
                log.warning(
                    "Groq rate limit key %d/%d (key_index=%d) — trying next key: %s",
                    offset + 1,
                    size,
                    key_index,
                    e,
                )
                continue
            elif _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "Groq rate limit (key_index=%d, attempt %d/%d, single key) "
                    "— waiting %.0fs: %s",
                    key_index,
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "Groq error (key_index=%d, attempt %d/%d): %s — retrying in %.0fs",
                    key_index,
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Groq call (key_index={key_index}) failed after {retries} attempts — "
        f"last error: {last_err}"
    )


_cerebras_pool = _KeyPool(
    "Cerebras", CEREBRAS_MAX_RPM_PER_KEY, CEREBRAS_MAX_TPM_PER_KEY
)


def register_cerebras_keys_from_config(cfg: dict):
    added = 0
    for field in ("CEREBRAS_API_KEY_1", "CEREBRAS_API_KEY_2", "CEREBRAS_API_KEY_3"):
        k = cfg.get(field, "").strip()
        if k:
            _cerebras_pool.add_key(k)
            added += 1

    env_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    if env_key:
        _cerebras_pool.add_key(env_key)
        added += 1

    if added:
        log.info(
            "Cerebras: %d key(s) registered (pool size=%d)",
            added,
            _cerebras_pool.size(),
        )
    else:
        log.warning(
            "register_cerebras_keys_from_config: no Cerebras keys found — "
            "Groq-only mode (no fallback provider)"
        )


def cerebras_pool_size() -> int:
    return _cerebras_pool.size()


def call_cerebras_rotated_pinned(
    system_prompt: str,
    user_content: str,
    key_index: int,
    model: str = CEREBRAS_FALLBACK_MODEL,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    size = max(1, _cerebras_pool.size())
    if _cerebras_pool.size() == 0:
        raise RuntimeError(
            "No Cerebras keys registered. "
            "Add CEREBRAS_API_KEY_1/2/3 to config.json. "
            "Free keys: https://cloud.cerebras.ai"
        )

    messages = _build_messages(system_prompt, user_content)
    last_err = None

    estimated_tokens = (
        estimate_tokens(system_prompt)
        + estimate_tokens(user_content)
        + min(max_tokens, int(estimate_tokens(user_content) * 0.65) + 100)
    )

    for attempt in range(1, retries + 1):
        offset = (key_index + attempt - 1) % size
        current_key = _cerebras_pool.key_at(offset)

        client = OpenAI(
            api_key=current_key,
            base_url=CEREBRAS_BASE_URL,
            timeout=CEREBRAS_REQUEST_TIMEOUT,
            max_retries=0,
        )

        try:
            _cerebras_pool.limiter_for(current_key).wait_for_slot(current_key)
            reservation = _cerebras_pool.token_limiter_for(current_key).reserve(
                estimated_tokens, current_key
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                actual = resp.usage.total_tokens
                _cerebras_pool.token_limiter_for(current_key).true_up(
                    reservation, actual
                )
            except Exception:
                pass
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()

            if _is_rate_limit(err) and size > 1:
                log.warning(
                    "Cerebras rate limit key %d/%d (key_index=%d) — trying next key: %s",
                    offset + 1,
                    size,
                    key_index,
                    e,
                )
                continue
            elif _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "Cerebras rate limit (key_index=%d, attempt %d/%d, single key) "
                    "— waiting %.0fs: %s",
                    key_index,
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "Cerebras error (key_index=%d, attempt %d/%d): %s — retrying in %.0fs",
                    key_index,
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Cerebras call (key_index={key_index}) failed after {retries} attempts — "
        f"last error: {last_err}"
    )


def call_llm_with_fallback(
    system_prompt: str,
    user_content: str,
    key_index: int = 0,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    if _groq_pool.size() > 0:
        try:
            out = call_groq_rotated_pinned(
                system_prompt,
                user_content,
                key_index=key_index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return out, "groq"
        except Exception as groq_err:
            log.warning(
                "Groq failed (key_index=%d) — falling back to Cerebras (%s): %s",
                key_index,
                CEREBRAS_FALLBACK_MODEL,
                groq_err,
            )

    if _cerebras_pool.size() > 0:
        try:
            out = call_cerebras_rotated_pinned(
                system_prompt,
                user_content,
                key_index=key_index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return out, "cerebras"
        except Exception as cerebras_err:
            raise RuntimeError(
                f"Both Groq and Cerebras failed for key_index={key_index}. "
                f"Last Cerebras error: {cerebras_err}"
            )

    raise RuntimeError(
        "No LLM providers available. "
        "Register Groq and/or Cerebras keys at startup via "
        "register_groq_keys_from_config() / register_cerebras_keys_from_config()."
    )


def _build_messages(system_prompt: str, user_content: str) -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _is_rate_limit(err_str: str) -> bool:
    return any(
        s in err_str for s in ("429", "rate", "quota", "resource_exhausted", "too many")
    )


def _is_retryable_connection_error(err_str: str) -> bool:
    return any(
        s in err_str
        for s in (
            "connection error",
            "timeout",
            "timed out",
            "connection reset",
            "eof occurred",
        )
    )


def clean_json(raw: str) -> str:
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def stagger_batch_start(batch_index: int):
    if batch_index == 0:
        return
    jitter = random.uniform(0.3, BATCH_STAGGER_MAX_SECS)
    log.debug("Batch %d: staggering start by %.1fs", batch_index + 1, jitter)
    time.sleep(jitter)


def make_client(api_key: str):
    """
    Returns an OpenAI-compatible client pointed at NVIDIA NIM's endpoint.

    Used by crew.py's verify_jobs_tool. Note: the CrewAI agent LLM itself
    (crew.py's _get_llm()) is created separately via crewai.LLM with its
    own base_url/api_key — it does not go through this function. This is
    only for the direct verify_jobs_tool NIM call.
    """
    if not api_key:
        raise RuntimeError(
            "No NVIDIA NIM API key provided. Set nvidia_nim_api_key in "
            "config.json or the NVIDIA_NIM_API_KEY env var. "
            "Free key: https://build.nvidia.com"
        )
    return OpenAI(
        api_key=api_key,
        base_url=NIM_BASE_URL,
        timeout=NIM_REQUEST_TIMEOUT,
        max_retries=0,
    )


def call_nim(
    client,
    system_prompt: str,
    user_content: str,
    model: str = NIM_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    retries: int = 3,
) -> str:
    """
    Calls NVIDIA NIM with basic retry on rate limits / transient errors.
    Single-key — no pool/rotation, matching how crew.py calls it
    (one client per verify_jobs_tool invocation).
    """
    messages = _build_messages(system_prompt, user_content)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            last_err = e
            err = str(e).lower()

            if _is_rate_limit(err):
                wait = RATE_LIMIT_BASE_WAIT * attempt
                log.warning(
                    "NIM rate limit (attempt %d/%d) — waiting %.0fs: %s",
                    attempt,
                    retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            elif attempt < retries:
                wait = CONN_ERROR_BASE_WAIT * attempt
                log.warning(
                    "NIM error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt,
                    retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"NIM call failed after {retries} attempts — last error: {last_err}"
    )


def nim_pool_size() -> int:
    """
    Legacy name — actually returns the Groq pool size (kept for any old
    callers). Real NIM usage (make_client/call_nim) is single-key, not
    pooled, so there's no equivalent "NIM pool size" to report.
    """
    return _groq_pool.size()


def register_keys_from_config(cfg: dict):
    register_groq_keys_from_config(cfg)
    register_cerebras_keys_from_config(cfg)


def register_gemini_keys_from_config(cfg: dict):
    log.warning(
        "Gemini removed — using Groq+Cerebras. Call register_groq_keys_from_config()."
    )


def gemini_pool_size() -> int:
    return 0

"""
sitecustomize.py — auto-imported by Python at interpreter startup, before
any application code runs. This is the ONLY place HF offline mode should be
set — main.py and scorer.py must NOT set it again (they used to, and the two
copies disagreed, which is what broke the hosted deploy: this file forced
offline unconditionally even on a fresh container with no cached models).

Offline mode is now conditional: only forced ON if the HF cache already has
the required models downloaded. On a fresh host (first deploy, or after a
Streamlit Cloud cold restart on free tier, which has no persistent disk),
there's no cache yet, so we leave networking ON so the models can download
once. Once cached, later runs in the same container go offline automatically.
"""

import os
from pathlib import Path

_hf_cache = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
_has_cache = _hf_cache.exists() and any(_hf_cache.rglob("*.safetensors"))

if _has_cache:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

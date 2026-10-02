"""Permissive JSON extraction from raw LLM text.

The primary and metadata agents are instructed to reply with exactly one
JSON object and nothing else, but models occasionally wrap it in a
markdown fence, add a stray sentence, or emit `<think>` reasoning first.
This extracts the JSON object defensively instead of trusting the model
to always follow the instruction to the letter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: Optional[str]) -> Optional[Any]:
    """Best-effort extraction of a single JSON object from ``text``.

    Returns ``None`` (never raises) if nothing parseable was found.
    """
    if not text:
        return None

    cleaned = _THINK_RE.sub("", text).strip()

    fence_match = _FENCE_RE.search(cleaned)
    candidate = fence_match.group(1) if fence_match else None

    if candidate is None:
        # Try the whole cleaned text first (the common, well-behaved case).
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        brace_match = _BRACE_RE.search(cleaned)
        candidate = brace_match.group(0) if brace_match else cleaned

    try:
        return json.loads(candidate)
    except Exception:
        return None

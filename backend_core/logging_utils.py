"""Prompt/response logging under ``.logs/``.

Every LLM call made anywhere in the backend (agent turns, metadata
extraction, RAG grounding, web-search planning/search) is written as one
JSON line to ``.logs/YYYY-MM-DD.jsonl``. Logging is best-effort: a logging
failure must never break the app, so every write is wrapped in a broad
``except``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(os.environ.get("AGENTPLANT_LOG_DIR", ".logs"))

# Keep logged prompt/response bodies bounded so a huge pasted document or a
# runaway response can't blow up the log files.
_MAX_FIELD_CHARS = 20_000


def _truncate(text: str) -> str:
    if text is None:
        return ""
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return text[:_MAX_FIELD_CHARS] + f"…[truncated, {len(text) - _MAX_FIELD_CHARS} more chars]"


def log_interaction(
    kind: str,
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    response_text: str = "",
    model: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON record describing a single LLM (or tool) call.

    ``kind`` is a short tag such as ``agent_primary``, ``agent_metadata``,
    ``rag_retrieve``, ``web_query_plan``, ``web_search`` or ``sandbox_run``.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = LOG_DIR / f"{day}.jsonl"
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "model": model,
            "system_prompt": _truncate(system_prompt),
            "user_prompt": _truncate(user_prompt),
            "response_text": _truncate(response_text),
            "extra": extra or {},
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never take down the app.
        pass

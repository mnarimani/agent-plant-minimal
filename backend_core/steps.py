"""A single entry in a turn's step log.

Used by ``app.py`` to build the "circle + status" staged UX: one
``Step`` per RAG lookup, web search, agent call, metadata extraction, or
sandbox run performed during a turn. Kept as a plain dataclass (no
Streamlit import) so it's trivial to store in ``st.session_state`` and
re-render after a rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Step:
    kind: str            # "rag" | "web" | "agent" | "metadata" | "sandbox"
    label: str            # one-line status text, e.g. 'Searched "Maxon EC-45 Kv"'
    detail: Optional[str] = None  # longer text shown when the step log is expanded
    ok: bool = True

"""AgentPlant extras: upload (RAG), web search context, sandbox simulate.

Uses streamlit_core (rag / websearch / sandbox / steps) when available.
All blocking OpenAI / numpy work is run in a thread pool so async routes
do not stall the event loop.

Demo mode (AGENTPLANT_DEMO=1 or missing OPENAI_API_KEY for chat-adjacent
paths that need it): returns canned data without calling OpenAI.
"""

from __future__ import annotations

import asyncio
import re
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_CORE = ROOT / "streamlit_core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STREAMLIT_CORE) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_CORE))

# Thread pool for blocking OpenAI / scipy work — shared across endpoints.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agentplant-block")

router = APIRouter(prefix="/api/v1/plant-model", tags=["plant-model-extras"])


def _demo_mode() -> bool:
    if os.getenv("AGENTPLANT_DEMO", "").strip() in {"1", "true", "yes"}:
        return True
    return not bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower()))


async def run_blocking(fn, *args, **kwargs):
    """Run a sync function off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# Session-scoped RAG (in-memory; one store per process for R&D)
# ---------------------------------------------------------------------------

import threading

_rag_instance = None
_rag_thread_lock = threading.Lock()
_attached_files: List[str] = []
# Local text extracted at upload time so chat can ground without OpenAI
# vector store (demo mode, indexing lag, or process state mismatch).
_local_texts: Dict[str, str] = {}


def _get_rag():
    global _rag_instance
    with _rag_thread_lock:
        if _rag_instance is not None:
            return _rag_instance
        from openai import OpenAI
        # Import as package so relative imports inside rag.py (from . import openai_client) work.
        from streamlit_core import rag as rag_mod

        client = OpenAI()
        _rag_instance = rag_mod.OpenAIRag(client)
        return _rag_instance


def _extract_local_text(name: str, data: bytes) -> str:
    """Best-effort text from PDF / MD / TXT for local grounding."""
    suffix = Path(name).suffix.lower()
    if suffix in {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".py"}:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            import io

            reader = PdfReader(io.BytesIO(data))
            parts: List[str] = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
            text = "\n\n".join(parts).strip()
            if text:
                return text
            return "(PDF opened but no extractable text — may be scanned/image-only.)"
        except Exception as exc:
            return f"(PDF text extraction failed: {exc})"
    return f"(Unsupported attachment type {suffix or 'unknown'}; binary not extracted.)"


def retrieve_attachment_context(user_message: str) -> tuple[str, list]:
    """Return (summary_text, chunk_list) for chat grounding.

    Prefer OpenAI vector-store file_search when indexed; otherwise use local
    extracted text from the most recent upload. Never return an empty
    "vector store has no files" dead-end when local text is available.
    """
    if not _attached_files:
        return "", []

    names = list(_attached_files)
    local_blob = ""
    for n in names:
        t = (_local_texts.get(n) or "").strip()
        if t:
            # Cap per file so prompts stay bounded
            local_blob += f"\n\n=== {n} ===\n{t[:12000]}"

    # Real mode: try OpenAI RAG first
    if not _demo_mode():
        try:
            rag = _get_rag()
            if getattr(rag, "has_files", False) or getattr(rag, "file_names", None):
                summary, chunks = rag.retrieve(user_message)
                if summary or chunks:
                    return summary, list(chunks or [])
        except Exception as exc:
            # Fall through to local text with a note
            if local_blob:
                return (
                    f"(OpenAI file_search failed: {exc}. Using local extracted text instead.)"
                    f"{local_blob}",
                    [],
                )
            raise

    if local_blob:
        mode = "DEMO" if _demo_mode() else "local-fallback"
        summary = (
            f"Attached file(s): {', '.join(names)}. "
            f"Ground your answer ONLY on the following extracted content "
            f"({mode}; do not claim you cannot access the attachment):\n"
            f"{local_blob}"
        )
        return summary, []

    # Nothing local and nothing indexed
    return (
        f"User attached file(s) {', '.join(names)}, but no text is available yet. "
        f"Ask them to re-upload the file (API process may have restarted), or paste key excerpts.",
        [],
    )


class UploadResponse(BaseModel):
    file_name: str
    attached_files: List[str]
    demo: bool = False
    message: str = "ok"
    local_chars: int = 0


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Upload PDF/MD/TXT for RAG. Always extracts local text; indexes on OpenAI when not demo."""
    name = file.filename or "upload.bin"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty file")

    # Always extract locally so chat can ground even without OpenAI indexing.
    local_text = await run_blocking(_extract_local_text, name, data)
    _local_texts[name] = local_text
    if name not in _attached_files:
        _attached_files.append(name)

    if _demo_mode():
        return UploadResponse(
            file_name=name,
            attached_files=list(_attached_files),
            demo=True,
            message="demo mode — local text extracted; not indexed on OpenAI",
            local_chars=len(local_text),
        )

    suffix = Path(name).suffix or ".bin"

    def _index() -> None:
        rag = _get_rag()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            rag.add_file(path, display_name=name)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    try:
        await run_blocking(_index)
    except Exception as exc:
        # Local text still available — do not fail the upload hard.
        return UploadResponse(
            file_name=name,
            attached_files=list(_attached_files),
            demo=False,
            message=f"local text ok ({len(local_text)} chars); OpenAI index failed: {exc}",
            local_chars=len(local_text),
        )

    return UploadResponse(
        file_name=name,
        attached_files=list(_attached_files),
        demo=False,
        message="ok",
        local_chars=len(local_text),
    )


@router.get("/attachments")
async def list_attachments() -> Dict[str, Any]:
    return {
        "attached_files": list(_attached_files),
        "demo": _demo_mode(),
        "local_chars": {k: len(v) for k, v in _local_texts.items()},
    }


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    python_code: str
    T: float = 10.0
    dt: float = 0.01
    input_kind: str = "step"
    input_channel: int = 0
    amplitude: float = 1.0
    t0: float = 1.0
    x0: Optional[List[float]] = None
    state_names: Optional[List[str]] = None


class SimulateResponse(BaseModel):
    success: bool
    message: str
    diverged: bool = False
    solver_used: str = ""
    attempts: List[str] = Field(default_factory=list)
    n_states: int = 0
    n_inputs: int = 0
    input_channel: int = 0
    input_kind: str = ""
    state_names: Optional[List[str]] = None
    t: Optional[List[float]] = None
    x: Optional[List[List[float]]] = None  # time-major
    u: Optional[List[List[float]]] = None
    demo: bool = False


def _run_simulate(body: SimulateRequest) -> SimulateResponse:
    import numpy as np
    from streamlit_core import sandbox as sandbox_mod

    # Hard wall-clock guard for pathological dynamics
    deadline = time.time() + float(os.getenv("AGENTPLANT_SIM_TIMEOUT", "20"))

    def guarded():
        if time.time() > deadline:
            raise TimeoutError("simulation wall-clock timeout")

        # streamlit_core.sandbox.simulate signature:
        #   simulate(python_code, *, x0, t_final, dt, input_channel,
        #            input_kind, input_params, state_names=None)
        # Older wrappers passed T/amplitude/t0 and silently fell through to
        # run_default_simulation — which ignores UI controls (always step, T=10).
        kind = (body.input_kind or "step").lower().strip()
        if kind not in {"step", "ramp", "sine"}:
            kind = "step"
        if kind == "ramp":
            input_params = {"t0": float(body.t0), "slope": float(body.amplitude)}
        elif kind == "sine":
            input_params = {"amplitude": float(body.amplitude), "freq_hz": 0.5}
        else:
            input_params = {"t0": float(body.t0), "amplitude": float(body.amplitude)}

        n_states, _n_in, _scalar = sandbox_mod.infer_dims(body.python_code)
        x0 = body.x0 if body.x0 is not None else [0.0] * n_states

        return sandbox_mod.simulate(
            body.python_code,
            x0=list(x0),
            t_final=float(body.T),
            dt=float(body.dt),
            input_channel=int(body.input_channel),
            input_kind=kind,
            input_params=input_params,
            state_names=body.state_names,
        )

    try:
        sim = guarded()
    except TimeoutError as exc:
        return SimulateResponse(success=False, message=str(exc), attempts=["timeout"])
    except Exception as exc:
        return SimulateResponse(
            success=False,
            message=f"{exc}\n{traceback.format_exc()[-800:]}",
            attempts=["exception"],
        )

    def _arr(a):
        if a is None:
            return None
        return np.asarray(a).tolist()

    return SimulateResponse(
        success=bool(sim.success),
        message=str(sim.message or ""),
        diverged=bool(getattr(sim, "diverged", False)),
        solver_used=str(getattr(sim, "solver_used", "") or ""),
        attempts=list(getattr(sim, "attempts", []) or []),
        n_states=int(getattr(sim, "n_states", 0) or 0),
        n_inputs=int(getattr(sim, "n_inputs", 0) or 0),
        input_channel=int(getattr(sim, "input_channel", body.input_channel) or 0),
        input_kind=str(getattr(sim, "input_kind", body.input_kind) or body.input_kind),
        state_names=list(sim.state_names) if getattr(sim, "state_names", None) else body.state_names,
        t=_arr(getattr(sim, "t", None)),
        x=_arr(getattr(sim, "x", None)),
        u=_arr(getattr(sim, "u", None)),
        demo=False,
    )


def _demo_simulate(body: SimulateRequest) -> SimulateResponse:
    """Canned trajectories for UI development without scipy code exec.

    Honours T / dt / t0 / amplitude / input_kind so the Run button visibly
    changes the plot even without OPENAI_API_KEY / real dynamics exec.
    """
    import math

    n = max(2, int(body.T / max(body.dt, 1e-3)))
    t = [i * body.dt for i in range(n)]
    amp = float(body.amplitude)
    t0 = float(body.t0)
    kind = (body.input_kind or "step").lower().strip()

    def u_at(ti: float) -> float:
        if kind == "ramp":
            return amp * (ti - t0) if ti >= t0 else 0.0
        if kind == "sine":
            return amp * math.sin(2.0 * math.pi * 0.5 * ti)
        return amp if ti >= t0 else 0.0

    u = [[u_at(ti)] for ti in t]
    x = []
    y = 0.0
    for ti, ui in zip(t, u):
        y += 0.05 * (ui[0] - y)
        x.append([y, y * 0.3, math.sin(0.2 * ti) * 0.1, 0.0])
    return SimulateResponse(
        success=True,
        message="demo simulation (no code exec)",
        diverged=False,
        solver_used="demo",
        attempts=["demo"],
        n_states=4,
        n_inputs=max(1, body.input_channel + 1),
        input_channel=body.input_channel,
        input_kind=kind,
        state_names=body.state_names or ["theta", "psi", "theta_dot", "psi_dot"],
        t=t,
        x=x,
        u=u,
        demo=True,
    )


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(body: SimulateRequest) -> SimulateResponse:
    if not (body.python_code or "").strip():
        raise HTTPException(status_code=422, detail="python_code is required")
    if _demo_mode() and os.getenv("AGENTPLANT_FORCE_REAL_SIM", "").strip() not in {"1", "true"}:
        # Still run off-loop in case demo path grows heavier
        return await run_blocking(_demo_simulate, body)
    return await run_blocking(_run_simulate, body)


# ---------------------------------------------------------------------------
# Web search (focus-gated; OpenAI hosted web_search tool)
# ---------------------------------------------------------------------------


def run_web_search(
    *,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    retrieved_context: str = "",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Plan + run focused web search(es) with transparent sources.

    Returns a dict shaped for ``WebSearchOut`` plus ``links`` / ``queries``:
      status: empty | ok | skipped
      query, brief, link, reason, links, queries
    """
    if _demo_mode():
        return {
            "status": "skipped",
            "query": None,
            "brief": None,
            "link": None,
            "reason": "demo mode — set OPENAI_API_KEY and unset AGENTPLANT_DEMO to enable web search",
            "links": [],
            "queries": [],
        }

    from openai import OpenAI
    from streamlit_core import websearch as ws_mod

    client = OpenAI()
    research = ws_mod.user_wants_research(user_message)
    history_text = ""
    for m in history or []:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            history_text += f"{role}: {content}\n"

    try:
        planned = ws_mod.plan_query(
            client,
            history_text,
            user_message,
            retrieved_context=retrieved_context or "",
            model=model,
        )
    except Exception as exc:
        return {
            "status": "skipped",
            "query": None,
            "brief": None,
            "link": None,
            "reason": f"query planner failed: {exc}",
            "links": [],
            "queries": [],
        }

    # Explicit search ask + shy planner → use cleaned user message as query.
    if not planned and research:
        cleaned = re.sub(
            r"^\s*(please\s+)?(search(\s+the\s+web)?\s+for|look\s+up|look\s+for|find|google)\s+",
            "",
            user_message.strip(),
            flags=re.I,
        ).strip()
        planned = cleaned or user_message.strip()

    if not planned:
        return {
            "status": "skipped",
            "query": None,
            "brief": None,
            "link": None,
            "reason": "planner returned NONE (no focused real-world fact to look up)",
            "links": [],
            "queries": [],
        }

    if not ws_mod.looks_focused(planned, research_intent=research):
        return {
            "status": "skipped",
            "query": planned,
            "brief": None,
            "link": None,
            "reason": "query failed focus gate (too generic / conversational)",
            "links": [],
            "queries": [planned],
        }

    queries: List[str] = [planned]
    if research and not re.search(r"parameter|equation|state[- ]space|manual", planned, re.I):
        queries.append(f"{planned} state-space parameters equations")

    briefs: List[str] = []
    all_links: List[Dict[str, str]] = []
    errors: List[str] = []
    for q in queries[:2]:
        try:
            brief, links = ws_mod.search(client, q, model=model)
            if (brief or "").strip():
                briefs.append(f"[{q}]\n{brief.strip()}")
            for lk in links or []:
                all_links.append(lk)
        except Exception as exc:
            errors.append(f"{q}: {exc}")

    seen = set()
    deduped: List[Dict[str, str]] = []
    for lk in all_links:
        url = lk.get("url") or ""
        if url and url not in seen:
            seen.add(url)
            deduped.append(lk)

    link = deduped[0].get("url") if deduped else None
    if not briefs:
        return {
            "status": "skipped",
            "query": planned,
            "brief": None,
            "link": link,
            "reason": ("; ".join(errors) if errors else "search returned empty brief"),
            "links": deduped,
            "queries": queries,
        }

    sources_block = ""
    if deduped:
        lines = []
        for i, lk in enumerate(deduped[:8], 1):
            title = (lk.get("title") or lk.get("url") or "").strip()
            url = (lk.get("url") or "").strip()
            extra = f" — {url}" if url and url != title else ""
            lines.append(f"{i}. {title}{extra}")
        sources_block = "\n\nSources:\n" + "\n".join(lines)

    combined = "\n\n".join(briefs) + sources_block
    return {
        "status": "ok",
        "query": " | ".join(queries),
        "brief": combined,
        "link": link,
        "reason": None,
        "links": deduped,
        "queries": queries,
    }



# ---------------------------------------------------------------------------
# Event-loop probe (documents the non-blocking guarantee)
# ---------------------------------------------------------------------------


@router.get("/_probe/slow")
async def probe_slow(seconds: float = 2.0) -> Dict[str, Any]:
    """Artificially slow endpoint that must NOT block /health."""

    def _sleep():
        time.sleep(min(max(seconds, 0.1), 10.0))
        return {"slept": seconds}

    result = await run_blocking(_sleep)
    return {"ok": True, **result}


@router.get("/_probe/health-fast")
async def probe_health_fast() -> Dict[str, str]:
    return {"status": "ok", "probe": "fast"}

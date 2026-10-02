"""
Minimal AgentPlant / PlantModelChat API for isolated R&D.

- No database
- No real auth (dev user stub)
- Core PlantModelAgent (two-call: chat + metadata)
- Endpoints the minimal frontend needs to chat

Run from repo root:
  pip install -e "./packages/labcd_agents[all]"
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
  uvicorn minimal_api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make monorepo-style imports work without installing the whole app
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# labcd_agents package lives under packages/labcd_agents/src
LABCD_SRC = ROOT / "packages" / "labcd_agents" / "src"
if str(LABCD_SRC) not in sys.path:
    sys.path.insert(0, str(LABCD_SRC))

from backend_core.AgentPlant.agent import (  # noqa: E402
    DEFAULT_MAX_DRAFTS,
    DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
    PlantModelAgent,
    apply_session_state,
    export_session_state,
    PlantModelSessionState as AgentSessionState,
)

app = FastAPI(
    title="AgentPlant Minimal API",
    description="Isolated PlantModelChat backend for R&D (no DB / real auth).",
    version="0.1.0-minimal",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG / sandbox / probes (thread-pooled blocking work)
from minimal_api.extras import router as extras_router  # noqa: E402
app.include_router(extras_router)


# ---------------------------------------------------------------------------
# Schemas (match frontend / original plant_model schemas)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PlantModelResult(BaseModel):
    system_name: str
    python_code: str
    metadata: Optional[dict[str, Any]] = None


class PlantModelSessionState(BaseModel):
    draft_count: int = 0
    latest_draft: Optional[PlantModelResult] = None


class PlantModelChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    user_message: str
    model: str = "gpt-4o-mini"
    session_state: Optional[PlantModelSessionState] = None
    conversation_id: Optional[int] = None
    max_drafts: int = Field(default=DEFAULT_MAX_DRAFTS, ge=1, le=10)
    min_user_turns_before_completion: int = Field(
        default=DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION, ge=1, le=5
    )
    # Composer ◎ toggle — when true, run focus-gated OpenAI web_search
    web_search_enabled: bool = False


class TokenUsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost: float


class StepOut(BaseModel):
    kind: str
    label: str
    detail: Optional[str] = None
    ok: bool = True


class RagChunkOut(BaseModel):
    file_name: str
    text: str
    score: Optional[float] = None


class WebSearchOut(BaseModel):
    status: Literal["empty", "ok", "skipped"] = "empty"
    query: Optional[str] = None
    brief: Optional[str] = None
    link: Optional[str] = None
    reason: Optional[str] = None


class PlantModelChatResponse(BaseModel):
    reply: str
    status: Literal["continue", "draft", "complete"]
    final_result: Optional[PlantModelResult] = None
    session_state: PlantModelSessionState
    usage: Optional[TokenUsageOut] = None
    conversation_id: Optional[int] = None
    # Additive — mockup tool-trace / transparency (optional for older clients)
    steps: List[StepOut] = Field(default_factory=list)
    rag_chunks: List[RagChunkOut] = Field(default_factory=list)
    web_search: Optional[WebSearchOut] = None
    draft: Optional[PlantModelResult] = None  # latest draft even when status is continue


PlantModelChatResponse.model_rebuild()


class PlantModelConversationSummary(BaseModel):
    id: int
    title: str
    status: Literal["active", "complete"]
    llm_model: str
    system_name: Optional[str] = None
    user_id: Optional[int] = None
    owner_email: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# In-memory conversation store (optional persistence for UI history)
_conversations: dict[int, dict[str, Any]] = {}
_next_conv_id = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_agent_session(
    state: Optional[PlantModelSessionState],
) -> Optional[AgentSessionState]:
    if state is None:
        return None
    latest = None
    if state.latest_draft is not None:
        latest = {
            "system_name": state.latest_draft.system_name,
            "python_code": state.latest_draft.python_code,
        }
        if state.latest_draft.metadata:
            latest["metadata"] = state.latest_draft.metadata
    return AgentSessionState(draft_count=state.draft_count, latest_draft=latest)


def _from_agent_session(state: AgentSessionState) -> PlantModelSessionState:
    latest = None
    if state.latest_draft is not None:
        meta = state.latest_draft.get("metadata")
        latest = PlantModelResult(
            system_name=state.latest_draft["system_name"],
            python_code=state.latest_draft["python_code"],
            metadata=meta if isinstance(meta, dict) else None,
        )
    return PlantModelSessionState(draft_count=state.draft_count, latest_draft=latest)


def _infer_status(
    *,
    prev_draft_count: int,
    draft_count: int,
    final_result: dict | None,
) -> Literal["continue", "draft", "complete"]:
    if final_result is not None:
        return "complete"
    if draft_count > prev_draft_count:
        return "draft"
    return "continue"


# ---------------------------------------------------------------------------
# Health / models / auth stubs (frontend expects these)
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/models")
def models() -> dict[str, list[str]]:
    # Expand as needed; agent resolves provider from model name via labcd_agents
    return {
        "llm_models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "o4-mini",
        ]
    }


@app.get("/api/v1/auth/me")
def auth_me() -> dict[str, Any]:
    """Dev user — no login required. Frontend works without a token."""
    return {
        "id": 1,
        "email": "dev@agentplant.local",
        "display_name": "AgentPlant Dev",
        "avatar_url": None,
        "theme": "system",
        "is_admin": True,
        "is_active": True,
        "email_verified": True,
        "plan_id": 1,
        "plan_name": "dev",
        "role_id": 1,
        "role_name": "admin",
        "actions": [
            "module:upload",
            "pipeline:silo",
            "pipeline:mulo",
            "pipeline:adaptive",
            "pipeline:mpc",
        ],
        "created_at": _now().isoformat(),
        "profile_survey_completed": True,
        "feedback_survey_completed": True,
    }


@app.get("/api/v1/errors/config")
def errors_config() -> dict[str, Any]:
    return {"enabled": False, "sample_rate": 0.0}


# ---------------------------------------------------------------------------
# Plant-model chat (core AgentPlant)
# ---------------------------------------------------------------------------


@app.post("/api/v1/plant-model/chat", response_model=PlantModelChatResponse)
async def plant_model_chat(request: PlantModelChatRequest) -> PlantModelChatResponse:
    user_message = (request.user_message or "").strip()
    if not user_message:
        raise HTTPException(status_code=422, detail="user_message is required")

    # Frontend may send "auto" — resolve to a concrete OpenAI model.
    model_name = (request.model or "").strip() or "gpt-4o-mini"
    if model_name.lower() in {"auto", "default", ""}:
        model_name = os.getenv("AGENTPLANT_DEFAULT_MODEL", "gpt-4o-mini")

    try:
        agent = PlantModelAgent(
            model=model_name,
            max_drafts=request.max_drafts,
            min_user_turns_before_completion=request.min_user_turns_before_completion,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create PlantModelAgent (check API keys / model): {exc}",
        ) from exc

    apply_session_state(agent, _to_agent_session(request.session_state))
    prev_draft_count = agent._draft_count

    history = [{"role": m.role, "content": m.content} for m in request.messages]

    # ---- RAG: retrieve from session attachments; inject into a *copy* of
    # the user message (never rewrite stored history). ----
    from minimal_api.extras import _attached_files, retrieve_attachment_context, run_blocking

    steps_acc: list[StepOut] = []
    rag_chunks_out: list[RagChunkOut] = []
    retrieved_context = ""

    if _attached_files:
        file_label = _attached_files[-1]

        try:
            retrieved_context, raw_chunks = await run_blocking(
                retrieve_attachment_context, user_message
            )
            for c in raw_chunks or []:
                rag_chunks_out.append(
                    RagChunkOut(
                        file_name=getattr(c, "file_name", file_label),
                        text=getattr(c, "text", "") or "",
                        score=getattr(c, "score", None),
                    )
                )
            detail = retrieved_context[:500] if retrieved_context else None
            if rag_chunks_out:
                detail = "\n".join(
                    f"{c.file_name} (score={c.score}): {c.text[:160]}" for c in rag_chunks_out[:5]
                )
            ok = bool(retrieved_context) and "no text is available yet" not in (
                retrieved_context or ""
            )
            steps_acc.append(
                StepOut(
                    kind="rag",
                    label=f"Inspected {file_label}" if ok else f"Attach incomplete: {file_label}",
                    detail=detail,
                    ok=ok,
                )
            )
        except Exception as rag_exc:
            steps_acc.append(
                StepOut(kind="rag", label="File search failed", detail=str(rag_exc), ok=False)
            )
            retrieved_context = (
                f"Attached file(s): {', '.join(_attached_files)}. "
                f"Retrieval failed ({rag_exc}). Ask the user for key excerpts."
            )

    # ---- Web search (composer ◎); focus-gated planner + OpenAI web_search ----
    from minimal_api.extras import run_web_search

    web_search_out = WebSearchOut(status="empty")
    web_context = ""
    if request.web_search_enabled:
        try:
            ws_result = await run_blocking(
                run_web_search,
                user_message=user_message,
                history=history,
                retrieved_context=retrieved_context,
                model=model_name,
            )
            web_search_out = WebSearchOut(
                status=ws_result.get("status") or "empty",  # type: ignore[arg-type]
                query=ws_result.get("query"),
                brief=ws_result.get("brief"),
                link=ws_result.get("link"),
                reason=ws_result.get("reason"),
            )
            queries = list(ws_result.get("queries") or [])
            links = list(ws_result.get("links") or [])
            if web_search_out.status == "ok" and web_search_out.brief:
                web_context = web_search_out.brief
                # One step per query for transparent research trail
                for q in queries or [web_search_out.query or "web"]:
                    steps_acc.append(
                        StepOut(
                            kind="web",
                            label=f"Searched the web: {q}",
                            detail=None,
                            ok=True,
                        )
                    )
                if links:
                    src_lines = []
                    for lk in links[:10]:
                        title = (lk.get("title") or lk.get("url") or "").strip()
                        url = (lk.get("url") or "").strip()
                        src_lines.append(f"• {title}" + (f"\n  {url}" if url and url != title else ""))
                    steps_acc.append(
                        StepOut(
                            kind="web",
                            label=f"Sources ({len(links)})",
                            detail="\n".join(src_lines),
                            ok=True,
                        )
                    )
                # Keep a short brief step for what was injected into the agent
                steps_acc.append(
                    StepOut(
                        kind="web",
                        label="Research brief",
                        detail=(web_search_out.brief or "")[:600],
                        ok=True,
                    )
                )
            else:
                steps_acc.append(
                    StepOut(
                        kind="web",
                        label="Web search skipped",
                        detail=web_search_out.reason or web_search_out.query,
                        ok=True,
                    )
                )
        except Exception as ws_exc:
            web_search_out = WebSearchOut(status="skipped", reason=str(ws_exc))
            steps_acc.append(
                StepOut(kind="web", label="Web search failed", detail=str(ws_exc), ok=False)
            )

    # Augment only the message copy passed to the agent
    agent_user_message = user_message
    context_blocks: list[str] = []
    if retrieved_context:
        context_blocks.append(
            "Attached-file context (use this; do not claim you cannot access attachments):\n"
            f"{retrieved_context}"
        )
    if web_context:
        context_blocks.append(
            "Web search brief (use for specific real-world facts; cite if relevant):\n"
            f"{web_context}"
        )
    if context_blocks:
        agent_user_message = (
            f"{user_message}\n\n---\n" + "\n\n---\n".join(context_blocks) + "\n---"
        )

    def _step():
        return agent.step(history, agent_user_message)

    try:
        from starlette.concurrency import run_in_threadpool
        reply, final_payload = await run_in_threadpool(_step)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM / agent error: {exc}") from exc

    session_state = _from_agent_session(export_session_state(agent))
    status = _infer_status(
        prev_draft_count=prev_draft_count,
        draft_count=agent._draft_count,
        final_result=final_payload,
    )

    final_result = None
    if final_payload is not None:
        meta = final_payload.get("metadata")
        final_result = PlantModelResult(
            system_name=final_payload["system_name"],
            python_code=final_payload["python_code"],
            metadata=meta if isinstance(meta, dict) else None,
        )

    usage_totals = agent.total_usage
    usage = TokenUsageOut(
        input_tokens=getattr(usage_totals, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_totals, "output_tokens", 0) or 0,
        estimated_cost=float(agent.total_cost or 0.0),
    )

    # Optional in-memory conversation bookkeeping
    global _next_conv_id
    conv_id = request.conversation_id
    if conv_id is None:
        conv_id = _next_conv_id
        _next_conv_id += 1
        _conversations[conv_id] = {
            "id": conv_id,
            "title": (user_message[:60] + "…") if len(user_message) > 60 else user_message,
            "status": "complete" if status == "complete" else "active",
            "llm_model": request.model,
            "system_name": final_result.system_name if final_result else None,
            "messages": list(request.messages) + [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=reply),
            ],
            "session_state": session_state,
            "final_result": final_result,
            "created_at": _now(),
            "updated_at": _now(),
        }
    else:
        entry = _conversations.get(conv_id)
        if entry is not None:
            entry["messages"] = list(request.messages) + [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=reply),
            ]
            entry["session_state"] = session_state
            entry["final_result"] = final_result
            entry["status"] = "complete" if status == "complete" else "active"
            if final_result:
                entry["system_name"] = final_result.system_name
            entry["updated_at"] = _now()

    # Latest draft for the code artifact panel (additive)
    draft = final_result
    if draft is None and session_state.latest_draft is not None:
        draft = session_state.latest_draft

    steps_acc.append(
        StepOut(kind="agent", label=f"Agent status: {status}", detail=(reply or "")[:240], ok=True)
    )
    if draft is not None:
        steps_acc.append(
            StepOut(
                kind="draft",
                label=f"Draft ready: {draft.system_name}",
                detail="python_code available in artifact panel",
                ok=True,
            )
        )

    return PlantModelChatResponse(
        reply=reply,
        status=status,
        final_result=final_result,
        session_state=session_state,
        usage=usage,
        conversation_id=conv_id,
        steps=steps_acc,
        rag_chunks=rag_chunks_out,
        web_search=web_search_out,
        draft=draft,
    )


@app.get("/api/v1/plant-model/conversations", response_model=list[PlantModelConversationSummary])
def list_conversations() -> list[PlantModelConversationSummary]:
    out: list[PlantModelConversationSummary] = []
    for c in sorted(_conversations.values(), key=lambda x: x["updated_at"], reverse=True):
        out.append(
            PlantModelConversationSummary(
                id=c["id"],
                title=c["title"],
                status=c["status"],
                llm_model=c["llm_model"],
                system_name=c.get("system_name"),
                user_id=1,
                owner_email="dev@agentplant.local",
                created_at=c["created_at"],
                updated_at=c["updated_at"],
            )
        )
    return out


@app.get("/api/v1/plant-model/conversations/{conversation_id}")
def get_conversation(conversation_id: int) -> dict[str, Any]:
    c = _conversations.get(conversation_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": c["id"],
        "title": c["title"],
        "status": c["status"],
        "llm_model": c["llm_model"],
        "messages": [
            m.model_dump() if hasattr(m, "model_dump") else m for m in c["messages"]
        ],
        "session_state": c["session_state"].model_dump()
        if hasattr(c["session_state"], "model_dump")
        else c["session_state"],
        "final_result": c["final_result"].model_dump()
        if c["final_result"] is not None and hasattr(c["final_result"], "model_dump")
        else c["final_result"],
        "user_id": 1,
        "owner_email": "dev@agentplant.local",
        "created_at": c["created_at"],
        "updated_at": c["updated_at"],
    }


@app.delete("/api/v1/plant-model/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int) -> None:
    _conversations.pop(conversation_id, None)
    return None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "agentplant-minimal",
        "docs": "/docs",
        "chat": "POST /api/v1/plant-model/chat",
        "hint": "Set OPENAI_API_KEY (or other provider key) before chatting.",
    }

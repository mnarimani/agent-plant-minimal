"""Plant-model chat HTTP service adapter."""

from __future__ import annotations

from typing import Any, Literal

from backend_api.PlantModelChat.agent import (
    PlantModelAgent,
    PlantModelSessionState as AgentSessionState,
    apply_session_state,
    export_session_state,
)
from backend_api.http.schemas.plant_model import (
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelResult,
    PlantModelSessionState,
    TokenUsageOut,
)


def _metadata_is_complete(meta: Any, required_keys: tuple[str, ...]) -> bool:
    """True when ``meta`` already has everything PlantCompiler needs, so
    there's no need to run the (heavier) inference fallback at all.
    """
    if not isinstance(meta, dict):
        return False
    if not all(key in meta for key in required_keys):
        return False

    states = meta.get("states")
    if not isinstance(states, list) or not states:
        return False
    n_states = len(states)
    for key in ("state_meanings", "state_equations"):
        value = meta.get(key)
        if not isinstance(value, list) or len(value) != n_states:
            return False
    if not isinstance(meta.get("inputs"), list) or not meta["inputs"]:
        return False
    if not isinstance(meta.get("outputs"), list) or not meta["outputs"]:
        return False
    if not isinstance(meta.get("parameters"), dict):
        return False
    if not meta.get("system_type"):
        return False
    if not isinstance(meta.get("assumptions"), list) or not meta["assumptions"]:
        return False
    return True


def _resolved_metadata(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return metadata aligned with ``python_code`` for sidebar / complete.

    Always runs ``reconcile_metadata_with_code`` when the compiler is
    available: that partial-merges LLM metadata, extracts equations from
    code (including ``dtheta_dt``-style assignments), and numerically
    verifies so drifted LLM ``state_equations`` cannot pass through.
    """
    if not payload:
        return None
    meta = payload.get("metadata")
    try:
        from backend_core.plant_compiler import reconcile_metadata_with_code
    except Exception:
        return meta if isinstance(meta, dict) else None

    try:
        reconciled = reconcile_metadata_with_code(
            {
                "system_name": payload.get("system_name"),
                "python_code": payload.get("python_code") or "",
                "metadata": meta if isinstance(meta, dict) else None,
            }
        )
        # Internal diagnostics only — not part of the public metadata schema.
        reconciled.pop("_verify", None)
        return reconciled
    except Exception:
        return meta if isinstance(meta, dict) else None


def _to_agent_session_state(
    state: PlantModelSessionState | None,
) -> AgentSessionState | None:
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


def _from_agent_session_state(state: AgentSessionState) -> PlantModelSessionState:
    latest = None
    if state.latest_draft is not None:
        latest = PlantModelResult(
            system_name=state.latest_draft["system_name"],
            python_code=state.latest_draft["python_code"],
            # Resolved (not just passed through) so the sidebar has usable
            # metadata on drafts too, not only on the final "complete" result.
            metadata=_resolved_metadata(state.latest_draft),
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


def run_plant_model_chat(request: PlantModelChatRequest) -> PlantModelChatResponse:
    agent = PlantModelAgent(
        model=request.model,
        max_drafts=request.max_drafts,
        min_user_turns_before_completion=request.min_user_turns_before_completion,
    )
    apply_session_state(agent, _to_agent_session_state(request.session_state))
    prev_draft_count = agent._draft_count

    history = [{"role": m.role, "content": m.content} for m in request.messages]
    reply, final_payload = agent.step(history, request.user_message.strip())

    # NOTE: this also resolves metadata for session_state.latest_draft (via
    # _from_agent_session_state), so the sidebar has real metadata to show
    # while a draft is still in progress, not only once the plant is confirmed.
    session_state = _from_agent_session_state(export_session_state(agent))
    status = _infer_status(
        prev_draft_count=prev_draft_count,
        draft_count=agent._draft_count,
        final_result=final_payload,
    )

    final_result = None
    if final_payload is not None:
        final_result = PlantModelResult(
            system_name=final_payload["system_name"],
            python_code=final_payload["python_code"],
            metadata=_resolved_metadata(final_payload),
        )

    usage_totals = agent.total_usage
    usage = TokenUsageOut(
        input_tokens=usage_totals.input_tokens,
        output_tokens=usage_totals.output_tokens,
        estimated_cost=agent.total_cost,
    )

    return PlantModelChatResponse(
        reply=reply,
        status=status,
        final_result=final_result,
        session_state=session_state,
        usage=usage,
    )

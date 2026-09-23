"""CRUD and persistence for plant-model chat conversations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, joinedload

from backend_api.db.models import PlantModelConversation, PlantModelMessage, User
from backend_api.http.schemas.plant_model import (
    ChatMessage,
    PlantModelResult,
    PlantModelSessionState,
)


class ConversationAccessDenied(PermissionError):
    """Raised when a user tries to access another user's conversation."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _title_from_message(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:120]


def _session_state_to_dict(state: PlantModelSessionState | None) -> dict[str, Any] | None:
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
    return {"draft_count": state.draft_count, "latest_draft": latest}


def _session_state_from_dict(raw: dict[str, Any] | None) -> PlantModelSessionState | None:
    if not raw:
        return None
    latest_raw = raw.get("latest_draft")
    latest = None
    if isinstance(latest_raw, dict) and latest_raw.get("system_name") and latest_raw.get("python_code"):
        metadata = latest_raw.get("metadata")
        latest = PlantModelResult(
            system_name=str(latest_raw["system_name"]),
            python_code=str(latest_raw["python_code"]),
            metadata=metadata if isinstance(metadata, dict) else None,
        )
    return PlantModelSessionState(
        draft_count=int(raw.get("draft_count") or 0),
        latest_draft=latest,
    )


def conversation_to_summary(
    conversation: PlantModelConversation,
    *,
    include_owner: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": conversation.id,
        "title": conversation.title,
        "status": conversation.status,
        "llm_model": conversation.llm_model,
        "system_name": conversation.final_system_name,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }
    if include_owner:
        data["user_id"] = conversation.user_id
        data["owner_email"] = conversation.owner.email if conversation.owner else None
    return data


def conversation_to_detail(
    conversation: PlantModelConversation,
    *,
    include_owner: bool = False,
) -> dict[str, Any]:
    session_state = _session_state_from_dict(conversation.session_state)

    final_result = None
    if conversation.final_system_name and conversation.final_python_code:
        # The confirmed result's system_name/python_code always mirror the
        # latest draft at the moment it was persisted (see
        # PlantModelAgent._accept_complete), so reuse that draft's metadata --
        # real LLM metadata or already-inferred -- instead of discarding it
        # and inferring from scratch with no context every time the
        # conversation is reopened.
        existing_metadata = None
        latest = session_state.latest_draft if session_state else None
        if (
            latest is not None
            and latest.system_name == conversation.final_system_name
            and latest.python_code == conversation.final_python_code
        ):
            existing_metadata = latest.metadata

        meta = existing_metadata
        try:
            from backend_api.http.services.plant_model_service import _resolved_metadata

            meta = _resolved_metadata(
                {
                    "system_name": conversation.final_system_name,
                    "python_code": conversation.final_python_code,
                    "metadata": existing_metadata,
                }
            )
        except Exception:
            pass
        final_result = PlantModelResult(
            system_name=conversation.final_system_name,
            python_code=conversation.final_python_code,
            metadata=meta,
        )
    messages: list[ChatMessage] = []
    for msg in conversation.messages:
        role = msg.role if msg.role in ("user", "assistant") else "assistant"
        messages.append(ChatMessage(role=role, content=msg.content))  # type: ignore[arg-type]
    data: dict[str, Any] = {
        "id": conversation.id,
        "title": conversation.title,
        "status": conversation.status,
        "llm_model": conversation.llm_model,
        "messages": messages,
        "session_state": session_state,
        "final_result": final_result,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }
    if include_owner:
        data["user_id"] = conversation.user_id
        data["owner_email"] = conversation.owner.email if conversation.owner else None
    return data


def get_conversation(db: Session, conversation_id: int) -> PlantModelConversation | None:
    return (
        db.query(PlantModelConversation)
        .options(
            joinedload(PlantModelConversation.messages),
            joinedload(PlantModelConversation.owner),
        )
        .filter(PlantModelConversation.id == conversation_id)
        .first()
    )


def list_conversations_for_user(db: Session, user_id: int) -> list[PlantModelConversation]:
    return (
        db.query(PlantModelConversation)
        .filter(PlantModelConversation.user_id == user_id)
        .order_by(PlantModelConversation.updated_at.desc())
        .all()
    )


def list_all_conversations(
    db: Session,
    *,
    user_id: int | None = None,
    status: str | None = None,
) -> list[PlantModelConversation]:
    query = db.query(PlantModelConversation).options(joinedload(PlantModelConversation.owner))
    if user_id is not None:
        query = query.filter(PlantModelConversation.user_id == user_id)
    if status is not None:
        query = query.filter(PlantModelConversation.status == status)
    return query.order_by(PlantModelConversation.updated_at.desc()).all()


def assert_conversation_access(conversation: PlantModelConversation, user: User) -> None:
    if user.role is not None and user.role.is_system:
        return
    if user.has_action("admin:plant_model"):
        return
    if conversation.user_id != user.id:
        raise ConversationAccessDenied("Conversation access denied")


def delete_conversation(db: Session, conversation: PlantModelConversation) -> None:
    db.delete(conversation)
    db.commit()


def persist_turn(
    db: Session,
    *,
    user_id: int,
    conversation_id: int | None,
    user_message: str,
    assistant_reply: str,
    llm_model: str,
    session_state: PlantModelSessionState,
    final_result: PlantModelResult | None,
) -> PlantModelConversation:
    """Create or update a conversation after one chat turn."""
    conversation: PlantModelConversation | None = None
    if conversation_id is not None:
        conversation = get_conversation(db, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            conversation = None

    if conversation is None:
        conversation = PlantModelConversation(
            user_id=user_id,
            title=_title_from_message(user_message),
            llm_model=llm_model,
            status="active",
        )
        db.add(conversation)
        db.flush()

    conversation.llm_model = llm_model
    conversation.session_state = _session_state_to_dict(session_state)
    conversation.updated_at = _now()

    if final_result is not None:
        conversation.status = "complete"
        conversation.final_system_name = final_result.system_name
        conversation.final_python_code = final_result.python_code
        if final_result.system_name.strip():
            conversation.title = final_result.system_name.strip()[:120]
    else:
        # Follow-up turns without a new plant reopen the case study for editing.
        conversation.status = "active"

    db.add(
        PlantModelMessage(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
        )
    )
    db.add(
        PlantModelMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_reply,
        )
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation

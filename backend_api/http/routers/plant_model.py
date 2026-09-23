"""Plant-model chat routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend_api.db.models import User
from backend_api.db.session import SessionLocal, get_db
from typing import Any

from backend_api.http.schemas.plant_model import (
    ArtifactCreateRequest,
    ArtifactCreateResponse,
    ArtifactDetail,
    ArtifactPluginResponse,
    ArtifactSummary,
    PlantModelChatRequest,
    PlantModelChatResponse,
    PlantModelConversationDetail,
    PlantModelConversationSummary,
    ValidationRequest,
    ValidationResponse,
)
from backend_api.http.services.analytics_service import record_module_use
from backend_api.http.services.plant_model_chat_service import (
    ConversationAccessDenied,
    assert_conversation_access,
    conversation_to_detail,
    conversation_to_summary,
    delete_conversation,
    get_conversation,
    list_conversations_for_user,
    persist_turn,
)
from backend_api.http.dependencies import assert_model_allowed, require_action
from backend_api.http.services.plant_model_service import run_plant_model_chat
from backend_api.http.services.plant_artifact_service import (
    ArtifactValidationError,
    create_artifact,
    get_adaptive_spec,
    get_artifact,
    get_artifact_plugin,
    list_artifacts,
    plant_payload_to_dict,
    run_validation,
)

router = APIRouter(prefix="/plant-model", tags=["plant-model"])


@router.get("/conversations", response_model=list[PlantModelConversationSummary])
def list_plant_model_conversations(
    user: User = Depends(require_action("module:upload")),
    db: Session = Depends(get_db),
) -> list[PlantModelConversationSummary]:
    conversations = list_conversations_for_user(db, user.id)
    return [
        PlantModelConversationSummary(**conversation_to_summary(c)) for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=PlantModelConversationDetail)
def get_plant_model_conversation(
    conversation_id: int,
    user: User = Depends(require_action("module:upload")),
    db: Session = Depends(get_db),
) -> PlantModelConversationDetail:
    conversation = get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        assert_conversation_access(conversation, user)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return PlantModelConversationDetail(**conversation_to_detail(conversation))


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plant_model_conversation(
    conversation_id: int,
    user: User = Depends(require_action("module:upload")),
    db: Session = Depends(get_db),
) -> None:
    conversation = get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        assert_conversation_access(conversation, user)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    delete_conversation(db, conversation)


@router.post("/chat", response_model=PlantModelChatResponse)
def plant_model_chat(
    request: PlantModelChatRequest,
    user: User = Depends(require_action("module:upload")),
) -> PlantModelChatResponse:
    assert_model_allowed(user, request.model)

    if request.conversation_id is not None:
        with SessionLocal() as db:
            existing = get_conversation(db, request.conversation_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            try:
                assert_conversation_access(existing, user)
            except ConversationAccessDenied as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    record_module_use(user.id, "plant_model")
    from backend_api.http.services.credit_service import (
        InsufficientCreditsError,
        begin_job_usage,
        end_job_usage,
    )

    try:
        session_id = begin_job_usage(user.id, "plant_model")
    except InsufficientCreditsError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    try:
        response = run_plant_model_chat(request)
        with SessionLocal() as db:
            conversation = persist_turn(
                db,
                user_id=user.id,
                conversation_id=request.conversation_id,
                user_message=request.user_message.strip(),
                assistant_reply=response.reply,
                llm_model=request.model,
                session_state=response.session_state,
                final_result=response.final_result,
            )
            response.conversation_id = conversation.id
        return response
    finally:
        end_job_usage(session_id=session_id)


# ---------------------------------------------------------------------------
# Artifacts (unified hand-off)
# ---------------------------------------------------------------------------


def _resolve_plant_from_db(
    db: Session,
    conversation_id: int | None,
    user: User,
) -> dict[str, Any] | None:
    if conversation_id is None:
        return None
    conversation = get_conversation(db, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        assert_conversation_access(conversation, user)
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    session_st = conversation.session_state or {}
    latest = session_st.get("latest_draft") if isinstance(session_st, dict) else None
    draft_meta = None
    if isinstance(latest, dict) and isinstance(latest.get("metadata"), dict):
        draft_meta = latest["metadata"]

    # Completed plant: system_name + python_code columns; metadata lives on the
    # accepted draft in session_state (no separate final_metadata column).
    if conversation.final_system_name and conversation.final_python_code:
        plant: dict[str, Any] = {
            "system_name": conversation.final_system_name,
            "python_code": conversation.final_python_code,
        }
        if draft_meta is not None:
            plant["metadata"] = draft_meta
        return plant

    # Fallback to latest draft in session_state
    if isinstance(latest, dict) and latest.get("system_name") and latest.get("python_code"):
        plant = {
            "system_name": latest["system_name"],
            "python_code": latest["python_code"],
        }
        if draft_meta is not None:
            plant["metadata"] = draft_meta
        return plant

    raise HTTPException(
        status_code=400,
        detail="Conversation has no completed plant model or draft; finish the plant chat first",
    )


@router.post(
    "/artifacts",
    response_model=ArtifactCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_plant_artifact(
    request: ArtifactCreateRequest,
    user: User = Depends(require_action("module:upload")),
    db: Session = Depends(get_db),
) -> ArtifactCreateResponse:
    plant: dict[str, Any] | None = None
    if request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    elif request.conversation_id is not None:
        plant = _resolve_plant_from_db(db, request.conversation_id, user)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide plant payload or conversation_id of a completed plant chat",
        )

    try:
        return create_artifact(request, plant_override=plant)
    except ArtifactValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Plant or pre-launch validation failed",
                "errors": exc.errors,
                "warnings": exc.warnings,
            },
        ) from exc


@router.get("/artifacts", response_model=list[ArtifactSummary])
def list_plant_artifacts(
    _: User = Depends(require_action("module:upload")),
) -> list[ArtifactSummary]:
    return list_artifacts()


@router.get("/artifacts/{artifact_id}", response_model=ArtifactDetail)
def get_plant_artifact(
    artifact_id: str,
    _: User = Depends(require_action("module:upload")),
) -> ArtifactDetail:
    try:
        return get_artifact(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/artifacts/{artifact_id}/plugin",
    response_model=ArtifactPluginResponse,
)
def get_plant_artifact_plugin(
    artifact_id: str,
    _: User = Depends(require_action("module:upload")),
) -> ArtifactPluginResponse:
    try:
        return get_artifact_plugin(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}/adaptive-spec")
def get_plant_artifact_adaptive_spec(
    artifact_id: str,
    _: User = Depends(require_action("module:upload")),
) -> dict[str, Any]:
    try:
        return get_adaptive_spec(artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/validate", response_model=ValidationResponse)
def validate_plant_or_pre_launch(
    request: ValidationRequest,
    user: User = Depends(require_action("module:upload")),
    db: Session = Depends(get_db),
) -> ValidationResponse:
    plant: dict[str, Any] | None = None
    if request.conversation_id is not None:
        plant = _resolve_plant_from_db(db, request.conversation_id, user)
    elif request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    return run_validation(request, plant)


"""Plant-model artifact and compiler service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend_core.artifact_store import ArtifactStore
from backend_core.plant_compiler import PlantCompiler, validate_pre_launch
from backend_api.http.schemas.plant_model import (
    ArtifactCreateRequest,
    ArtifactCreateResponse,
    ArtifactDetail,
    ArtifactPluginResponse,
    ArtifactSummary,
    PlantModelResult,
    PlantPayload,
    PreLaunchConfig,
    ValidationRequest,
    ValidationResponse,
)


def default_artifacts_dir() -> str:
    env = os.getenv("LABCD_ARTIFACTS_DIR")
    if env:
        return env
    repo_root = Path(__file__).resolve().parents[3]
    return str(repo_root / "artifacts")


def get_artifact_store(base_dir: str | None = None) -> ArtifactStore:
    return ArtifactStore(base_dir=base_dir or default_artifacts_dir())


def plant_payload_to_dict(plant: PlantPayload | PlantModelResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plant, dict):
        return dict(plant)
    data: dict[str, Any] = {
        "system_name": plant.system_name,
        "python_code": plant.python_code,
    }
    meta = getattr(plant, "metadata", None)
    if meta is not None:
        data["metadata"] = meta
    return data


def pre_launch_to_dict(pre_launch: PreLaunchConfig) -> dict[str, Any]:
    return {
        "total_simulation_time": float(pre_launch.total_simulation_time),
        "solver_sample_time": float(pre_launch.solver_sample_time),
        "initial_state": [float(x) for x in pre_launch.initial_state],
        "default_target": [float(x) for x in pre_launch.default_target],
    }


class ArtifactValidationError(Exception):
    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        self.errors = list(errors)
        self.warnings = list(warnings or [])
        super().__init__("; ".join(self.errors) if self.errors else "validation failed")


def _metadata_is_usable(meta: Any) -> bool:
    """True when plant metadata has real states + matching non-empty state_equations.

    Used to decide whether to treat AgentPlant (or client) metadata as authoritative
    instead of synthesizing a strict-feedback chain via ``infer_metadata``.
    """
    if not isinstance(meta, dict):
        return False
    states = meta.get("states")
    eqs = meta.get("state_equations")
    if not isinstance(states, list) or not states:
        return False
    if not isinstance(eqs, list) or len(eqs) != len(states):
        return False
    if not all(isinstance(e, str) and e.strip() for e in eqs):
        return False
    return True


def validate_plant_and_pre_launch(
    plant: dict[str, Any],
    pre_launch: dict[str, Any] | None = None,
) -> ValidationResponse:
    """Validate plant (+ optional pre-launch) without wiping good metadata.

    Prefer the caller's ``plant["metadata"]`` when it already has usable
    ``states`` and matching ``state_equations``. Only call
    ``PlantCompiler.infer_metadata`` as a true fallback (missing / incomplete),
    and even then the compiler merges rather than inventing a chain over real
    equations.
    """
    compiler = PlantCompiler()
    # Prefer reconcile (extract + np.sin→sin normalize + numerical check) so
    # code-derived equations never fail sympy validation with numpy prefixes.
    try:
        from backend_core.plant_compiler import reconcile_metadata_with_code
        meta = reconcile_metadata_with_code(plant, pre_launch or {}, compiler=compiler)
        meta.pop("_verify", None)
    except Exception:
        meta = compiler.infer_metadata(plant, pre_launch or {})
    plant["metadata"] = meta

    plant_result = compiler.validate(plant)
    errors = list(plant_result.errors)
    warnings = list(plant_result.warnings)

    if pre_launch is not None and plant_result.ok:
        pl_result = validate_pre_launch(pre_launch, meta)
        errors.extend(pl_result.errors)
        warnings.extend(pl_result.warnings)

    return ValidationResponse(
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def create_artifact(
    request: ArtifactCreateRequest,
    *,
    plant_override: dict[str, Any] | None = None,
    store: ArtifactStore | None = None,
) -> ArtifactCreateResponse:
    if plant_override is not None:
        plant = dict(plant_override)
    elif request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    else:
        raise ArtifactValidationError(
            ["plant payload is required (or a completed conversation_id)"]
        )

    pre_launch = pre_launch_to_dict(request.pre_launch)
    validation = validate_plant_and_pre_launch(plant, pre_launch)
    if not validation.ok:
        raise ArtifactValidationError(validation.errors, validation.warnings)

    art_store = store or get_artifact_store()
    artifact_id = art_store.save_from_plant(plant, pre_launch)
    data = art_store.load(artifact_id)
    return ArtifactCreateResponse(
        artifact_id=artifact_id,
        system_name=str(data.get("system_name") or plant.get("system_name") or ""),
        created_at=str(data.get("created_at") or ""),
        version=str(data.get("version") or "1.0"),
        warnings=list(validation.warnings),
    )


def list_artifacts(store: ArtifactStore | None = None) -> list[ArtifactSummary]:
    art_store = store or get_artifact_store()
    return [
        ArtifactSummary(
            artifact_id=str(item.get("artifact_id") or ""),
            system_name=str(item.get("system_name") or ""),
            created_at=str(item.get("created_at") or ""),
            version=str(item.get("version") or ""),
        )
        for item in art_store.list_artifacts()
    ]


def get_artifact(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> ArtifactDetail:
    art_store = store or get_artifact_store()
    try:
        data = art_store.load(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc
    return ArtifactDetail(
        artifact_id=str(data.get("artifact_id") or artifact_id),
        system_name=str(data.get("system_name") or ""),
        created_at=str(data.get("created_at") or ""),
        version=str(data.get("version") or "1.0"),
        plant=dict(data.get("plant") or {}),
        pre_launch=dict(data.get("pre_launch") or {}),
        module_specific=dict(data.get("module_specific") or {}),
    )


def get_artifact_plugin(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> ArtifactPluginResponse:
    art_store = store or get_artifact_store()
    try:
        path = art_store.load_plugin_path(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Plugin not found for artifact: {artifact_id}") from exc
    source = Path(path).read_text(encoding="utf-8")
    return ArtifactPluginResponse(
        artifact_id=artifact_id,
        plugin_path=path,
        source=source,
    )


def get_adaptive_spec(
    artifact_id: str,
    store: ArtifactStore | None = None,
) -> dict[str, Any]:
    art_store = store or get_artifact_store()
    try:
        return art_store.get_adaptive_spec(artifact_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}") from exc


def run_validation(request: ValidationRequest, plant: dict[str, Any] | None) -> ValidationResponse:
    if plant is None and request.plant is not None:
        plant = plant_payload_to_dict(request.plant)
    if plant is None:
        return ValidationResponse(
            ok=False,
            errors=["plant payload is required (or a completed conversation_id)"],
        )
    pre_launch = None
    if request.pre_launch is not None:
        pre_launch = pre_launch_to_dict(request.pre_launch)
    return validate_plant_and_pre_launch(plant, pre_launch)

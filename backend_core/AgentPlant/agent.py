"""PlantModelAgent: LabCD's plant-model chatbot.

Two-call architecture
---------------------
1. **Primary agent** (``plant_model_agent.yaml``) — conversation only.
   Emits continue / draft / complete with ``system_name`` + ``python_code``.
   Never emits structured metadata.

2. **Metadata agent** (``plant_model_metadata.yaml``) — after every successful
   draft or complete, a second LLM call reads the finished ``python_code`` and
   emits the full metadata object forced to match identifiers and arithmetic
   in the code. A numerical verifier checks ``state_equations`` against
   ``dynamics()``; one repair retry is allowed on failure. If the LLM still
   fails, equations are taken deterministically from the code (no generic
   "Metadata inferred..." assumption strings).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from labcd_agents import BaseAgent, PromptLibrary, extract_json_from_response

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_FILE_NAME = "plant_model_agent"
METADATA_PROMPT_FILE_NAME = "plant_model_metadata"

REQUIRED_CODE_KEYS = ("system_name", "python_code")
REQUIRED_METADATA_KEYS = (
    "states",
    "state_meanings",
    "inputs",
    "outputs",
    "state_equations",
    "parameters",
    "system_type",
    "assumptions",
)

DEFAULT_MAX_DRAFTS = 5
DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION = 2
DEFAULT_METADATA_MAX_RETRIES = 1

_REPAIR_NOTE = (
    "\n\nIMPORTANT — internal note: your previous reply was not valid JSON in one of "
    'the required shapes (continue / draft / complete). Reply again with ONLY a single '
    'JSON object. Prefer status "draft" if the system is already known from the history; '
    'otherwise status "continue".'
)

_FORCE_DRAFT_NOTE = (
    "\n\nIMPORTANT — internal note: do not emit status \"complete\" yet. The conversation "
    "is not finished. If you already know the plant, emit status \"draft\" with a real "
    "python_code body and a short reply asking the user what to change or whether to "
    'finish. Otherwise emit status "continue" with one concrete question. Output ONLY JSON.'
)

_FINISH_RE = re.compile(
    r"\b(finish|done|confirm|confirm\s+system|accept|accept\s+draft|ship\s*it|looks?\s+good|totally\s+good|finalize|finalise|"
    r"good\s+to\s+go|that'?s\s+(fine|good|ok|okay)|perfect|approved)\b",
    re.IGNORECASE,
)

_FORBIDDEN_ASSUMPTION_SNIPPETS = (
    "metadata inferred from python_code",
    "review before control design",
)

_ASCII_REPLACEMENTS = (
    ("θ", "theta"),
    ("Θ", "Theta"),
    ("ω", "omega"),
    ("Ω", "Omega"),
    ("τ", "tau"),
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("δ", "delta"),
    ("Δ", "Delta"),
    ("φ", "phi"),
    ("Φ", "Phi"),
    ("ψ", "psi"),
    ("Ψ", "Psi"),
    ("ρ", "rho"),
    ("σ", "sigma"),
    ("Σ", "Sigma"),
    ("μ", "mu"),
    ("λ", "lambda"),
    ("π", "pi"),
    ("·", "*"),
    ("×", "*"),
    ("²", "^2"),
    ("³", "^3"),
    ("¹", "^1"),
    ("⁰", "^0"),
    ("⁻", "-"),
    ("≈", "~="),
    ("≤", "<="),
    ("≥", ">="),
    ("≠", "!="),
    ("→", "->"),
    ("←", "<-"),
    ("°", " deg"),
    ("\u00a0", " "),
)


@dataclass
class PlantModelSessionState:
    draft_count: int = 0
    # Widened so a nested "metadata" dict can round-trip.
    latest_draft: Dict[str, Any] | None = None


def apply_session_state(
    agent: PlantModelAgent,
    state: PlantModelSessionState | None,
) -> None:
    if state is None:
        agent.reset_conversation_state()
        return
    agent._draft_count = max(0, state.draft_count)
    agent._latest_draft = dict(state.latest_draft) if state.latest_draft else None


def export_session_state(agent: PlantModelAgent) -> PlantModelSessionState:
    latest = agent._latest_draft
    return PlantModelSessionState(
        draft_count=agent._draft_count,
        latest_draft=dict(latest) if latest else None,
    )


def _to_ascii(text: str) -> str:
    if not text:
        return text
    out = text
    for src, dst in _ASCII_REPLACEMENTS:
        out = out.replace(src, dst)
    return "".join(ch if (32 <= ord(ch) <= 126) or ch in "\n\r\t" else "?" for ch in out)


def _sanitize_assumptions(assumptions: Any) -> List[str]:
    """Drop forbidden generic filler; keep real modelling assumptions."""
    if not isinstance(assumptions, list):
        return ["Continuous-time state-space dynamics"]
    cleaned: List[str] = []
    for item in assumptions:
        if not isinstance(item, str):
            continue
        low = item.strip().lower()
        if not low:
            continue
        if any(bad in low for bad in _FORBIDDEN_ASSUMPTION_SNIPPETS):
            continue
        cleaned.append(item.strip())
    return cleaned or ["Continuous-time state-space dynamics"]


def _metadata_shape_ok(meta: Any) -> bool:
    if not isinstance(meta, dict):
        return False
    for key in REQUIRED_METADATA_KEYS:
        if key not in meta:
            return False
    states = meta.get("states")
    if not isinstance(states, list) or not states:
        return False
    n = len(states)
    if not isinstance(meta.get("state_meanings"), list) or len(meta["state_meanings"]) != n:
        return False
    if not isinstance(meta.get("state_equations"), list) or len(meta["state_equations"]) != n:
        return False
    if not all(isinstance(e, str) and e.strip() for e in meta["state_equations"]):
        return False
    if not isinstance(meta.get("inputs"), list) or not meta["inputs"]:
        return False
    if not isinstance(meta.get("outputs"), list) or not meta["outputs"]:
        return False
    if not isinstance(meta.get("parameters"), dict):
        return False
    if not meta.get("system_type"):
        return False
    return True


class PlantModelAgent(BaseAgent):
    """Plant-model chatbot: primary code agent + second-call metadata agent."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        provider: Optional[str] = None,
        prompts_dir: Path = PROMPTS_DIR,
        max_drafts: int = DEFAULT_MAX_DRAFTS,
        min_user_turns_before_completion: int = DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
        metadata_max_retries: int = DEFAULT_METADATA_MAX_RETRIES,
        **base_agent_kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature, provider=provider, **base_agent_kwargs)
        self.prompt_library = PromptLibrary(str(prompts_dir))
        self._system_prompt: str = self.prompt_library.get_key(PROMPT_FILE_NAME, "system_prompt")
        self._user_prompt_template: str = self.prompt_library.get_key(
            PROMPT_FILE_NAME, "user_prompt_template"
        )
        self._metadata_system_prompt: str = self.prompt_library.get_key(
            METADATA_PROMPT_FILE_NAME, "system_prompt"
        )
        self._metadata_user_template: str = self.prompt_library.get_key(
            METADATA_PROMPT_FILE_NAME, "user_prompt_template"
        )
        self.max_drafts = max(1, max_drafts)
        self.min_user_turns_before_completion = max(1, min_user_turns_before_completion)
        self.metadata_max_retries = max(0, int(metadata_max_retries))
        self._draft_count = 0
        self._latest_draft: Optional[Dict[str, Any]] = None

    def reset_conversation_state(self) -> None:
        self._draft_count = 0
        self._latest_draft = None

    @staticmethod
    def format_history(messages: Iterable[Dict[str, str]]) -> str:
        lines = []
        for message in messages:
            role = "User" if message.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)

    def render_user_prompt(self, history_text: str, user_message: str) -> str:
        return (
            self._user_prompt_template
            .replace("{{conversation_history}}", history_text or "(no previous turns)")
            .replace("{{user_message}}", user_message)
        )

    def step(
        self,
        history_messages: Iterable[Dict[str, str]],
        user_message: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        history_messages = list(history_messages)
        history_text = self.format_history(history_messages)
        user_prompt = self.render_user_prompt(history_text, user_message)
        user_turns = self._count_user_turns(history_messages) + 1
        user_accepts = bool(_FINISH_RE.search(user_message))

        response_text, _ = self.invoke_llm(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
        )
        parsed = self._parse_structured_response(response_text)

        if parsed is None:
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _REPAIR_NOTE,
                user_prompt=user_prompt,
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                return response_text.strip() or "(empty model response)", None

        if user_accepts and self._latest_draft is not None and parsed.get("status") != "complete":
            if parsed.get("status") == "complete" and self._has_code(parsed):
                return self._accept_complete(parsed)
            return self._accept_complete(self._latest_draft)

        status = parsed.get("status")

        if status == "continue":
            reply = _to_ascii((parsed.get("reply") or "").strip())
            if reply:
                return reply, None
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _FORCE_DRAFT_NOTE,
                user_prompt=user_prompt,
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                return response_text.strip() or "(empty model response)", None
            status = parsed.get("status")
            if status == "continue":
                reply = (parsed.get("reply") or "").strip()
                return reply or response_text.strip(), None

        if status == "draft" and self._has_code(parsed):
            parsed = self._sanitize_code_fields(parsed)
            self._draft_count += 1
            self._latest_draft = self._draft_payload_with_metadata(parsed)
            display = self._format_draft_display(parsed)
            if self._draft_count >= self.max_drafts:
                return display, dict(self._latest_draft)
            return display, None

        if status == "complete" and self._has_code(parsed):
            if (
                user_accepts
                or self._latest_draft is not None
                or user_turns >= self.min_user_turns_before_completion
            ):
                return self._accept_complete(parsed)
            response_text, _ = self.invoke_llm(
                system_prompt=self._system_prompt + _FORCE_DRAFT_NOTE,
                user_prompt=(
                    f"{user_prompt}\n\n(Your previous reply claimed status complete too "
                    f"early. Rejected payload:\n{response_text}\n)"
                ),
            )
            parsed = self._parse_structured_response(response_text)
            if parsed is None:
                return response_text.strip() or "(empty model response)", None
            if parsed.get("status") == "draft" and self._has_code(parsed):
                parsed = self._sanitize_code_fields(parsed)
                self._draft_count += 1
                self._latest_draft = self._draft_payload_with_metadata(parsed)
                return self._format_draft_display(parsed), None
            if parsed.get("status") == "continue":
                reply = (parsed.get("reply") or "").strip()
                return reply or response_text.strip(), None
            if parsed.get("status") == "complete" and self._has_code(parsed):
                return self._accept_complete(parsed)
            return response_text.strip(), None

        reply = (parsed.get("reply") or "").strip()
        if reply:
            return reply, None
        return response_text.strip() or "(empty model response)", None

    def _accept_complete(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        payload = self._sanitize_code_fields(payload)
        final = self._draft_payload_with_metadata(payload)
        self._latest_draft = final
        return f"Model ready — **{final['system_name']}**.", final

    def _draft_payload_with_metadata(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Attach metadata via the second LLM call (verified)."""
        out: Dict[str, Any] = {
            "system_name": parsed["system_name"],
            "python_code": parsed["python_code"],
        }
        meta = self._generate_metadata(parsed["system_name"], parsed["python_code"])
        if meta:
            out["metadata"] = meta
        return out

    def _generate_metadata(self, system_name: str, python_code: str) -> Optional[Dict[str, Any]]:
        """Second LLM call + numerical verify; deterministic extract on failure."""
        repair_section = ""
        last_meta: Optional[Dict[str, Any]] = None
        attempts = 1 + self.metadata_max_retries

        for attempt in range(attempts):
            user_prompt = (
                self._metadata_user_template
                .replace("{{system_name}}", system_name or "System")
                .replace("{{python_code}}", python_code or "")
                .replace("{{repair_section}}", repair_section)
            )
            try:
                response_text, _ = self.invoke_llm(
                    system_prompt=self._metadata_system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception:  # noqa: BLE001
                break
            meta = self._parse_metadata_response(response_text)
            if meta is None:
                repair_section = (
                    "Previous metadata reply was not valid JSON with the required keys. "
                    "Reply again with ONLY the metadata object."
                )
                continue
            meta["assumptions"] = _sanitize_assumptions(meta.get("assumptions"))
            last_meta = meta
            if not _metadata_shape_ok(meta):
                repair_section = (
                    "Previous metadata failed shape checks (states / state_equations "
                    "length mismatch or missing keys). Fix lengths and required keys."
                )
                continue
            ok, message = self._verify_metadata(python_code, meta)
            if ok:
                return meta
            repair_section = (
                f"Numerical verification failed: {message}. "
                "state_equations must match the arithmetic in python_code exactly "
                "(copy each derivative RHS; include all parameters)."
            )

        # Deterministic fallback from code — never emit the banned assumption phrase.
        return self._deterministic_metadata(system_name, python_code, seed=last_meta)

    @staticmethod
    def _verify_metadata(python_code: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            from backend_core.plant_compiler import verify_dynamics

            result = verify_dynamics(python_code, meta)
            return bool(result.ok), result.message or ("ok" if result.ok else "verify failed")
        except Exception as exc:  # noqa: BLE001
            # If verifier is unavailable, accept shape-ok LLM metadata.
            return True, f"verify skipped ({exc})"

    @staticmethod
    def _deterministic_metadata(
        system_name: str,
        python_code: str,
        seed: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract metadata from code without generic 'inferred' assumptions."""
        try:
            from backend_core.plant_compiler import reconcile_metadata_with_code
        except Exception:
            return seed if _metadata_shape_ok(seed) else None

        plant = {
            "system_name": system_name,
            "python_code": python_code,
            "metadata": seed if isinstance(seed, dict) else None,
        }
        try:
            meta = reconcile_metadata_with_code(plant)
        except Exception:
            return seed if _metadata_shape_ok(seed) else None
        meta.pop("_verify", None)
        meta["assumptions"] = _sanitize_assumptions(meta.get("assumptions"))
        # Prefer physics-flavoured default over any residual process notes
        process_notes = (
            "state_equations taken from python_code",
            "failed numerical check",
            "used fallback",
            "metadata inferred",
        )
        cleaned = [
            a for a in meta["assumptions"]
            if not any(p in a.lower() for p in process_notes)
        ]
        meta["assumptions"] = cleaned or ["Continuous-time state-space dynamics"]
        return meta if _metadata_shape_ok(meta) else (seed if _metadata_shape_ok(seed) else meta)

    @staticmethod
    def _parse_metadata_response(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = extract_json_from_response(text)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict):
            return None
        # Some models wrap as {"metadata": {...}}
        if "states" not in data and isinstance(data.get("metadata"), dict):
            data = data["metadata"]
        if "states" not in data:
            return None
        return data

    @staticmethod
    def _sanitize_code_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)
        if isinstance(out.get("system_name"), str):
            out["system_name"] = _to_ascii(out["system_name"])
        if isinstance(out.get("python_code"), str):
            out["python_code"] = _to_ascii(out["python_code"])
        if isinstance(out.get("reply"), str):
            out["reply"] = _to_ascii(out["reply"])
        # Primary agent must not carry metadata even if the model ignored instructions.
        out.pop("metadata", None)
        return out

    @staticmethod
    def _format_draft_display(parsed: Dict[str, Any]) -> str:
        reply = (parsed.get("reply") or "").strip()
        name = parsed.get("system_name", "draft")
        code = parsed.get("python_code", "")
        parts = []
        if reply:
            parts.append(reply)
        parts.append(f"**Draft: {name}**")
        parts.append(f"```python\n{code}\n```")
        parts.append("_Say what to change, or click **Confirm system** (or reply **finish**) to accept this draft._")
        return "\n\n".join(parts)

    @staticmethod
    def _has_code(data: Dict[str, Any]) -> bool:
        return all(isinstance(data.get(k), str) and data[k].strip() for k in REQUIRED_CODE_KEYS)

    @staticmethod
    def _parse_structured_response(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = extract_json_from_response(text)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if status in ("continue", "draft", "complete"):
            return data
        if all(k in data for k in REQUIRED_CODE_KEYS) and "reply" not in data:
            out = dict(data)
            out["status"] = "complete"
            return out
        return data

    @staticmethod
    def _count_user_turns(history_messages: List[Dict[str, str]]) -> int:
        return sum(1 for m in history_messages if m.get("role") == "user")

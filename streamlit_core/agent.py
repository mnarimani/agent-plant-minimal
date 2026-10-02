"""PlantModelAgent — the core agent contract.

Two-call architecture, ported from the AgentPlant / plant-model reference
design and reimplemented against the OpenAI Responses API:

1. **Primary agent** (``plant_model_agent.yaml``) — conversation only.
   Emits ``continue`` / ``draft`` / ``complete`` with ``system_name`` +
   ``python_code``. Never emits structured metadata.

2. **Metadata agent** (``plant_model_metadata.yaml``) — after every
   successful draft or complete, a second call reads the finished
   ``python_code`` and emits metadata (states, inputs, outputs, parameters,
   assumptions) shape-checked against the code, with one repair retry.

Deliberately kept separate from the RAG / web-search tool calls: this
module never sets ``tools=`` itself, so it never combines OpenAI's hosted
tools with the strict JSON-object contract in a single call. Retrieved
context is instead handed in as plain text (``retrieved_context`` /
``web_context``) by the caller (``app.py``), which keeps the model call
that has to produce reliable JSON free of tool-call side effects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from . import openai_client
from .json_extract import extract_json

PROMPTS_DIR = Path(__file__).parent / "prompts"

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
    "\n\nInternal note: your previous reply was not a single valid JSON object "
    "in one of the three allowed shapes. Reply again with ONLY one JSON object — "
    'status "draft" if the plant is already known from the history, otherwise '
    '"continue".'
)

_FORCE_DRAFT_NOTE = (
    '\n\nInternal note: do not emit status "complete" yet — the conversation is '
    'not finished. If the plant is already known, emit status "draft" with real '
    "python_code and ask what to change or whether to finish. Otherwise emit "
    'status "continue" with one concrete question. Output ONLY JSON.'
)

_FINISH_RE = re.compile(
    r"\b(finish|done|confirm(?:\s+system)?|accept(?:\s+draft)?|ship\s*it|"
    r"looks?\s+good|totally\s+good|finalize|finalise|good\s+to\s+go|"
    r"that'?s\s+(fine|good|ok|okay)|perfect|approved)\b",
    re.IGNORECASE,
)

_FORBIDDEN_ASSUMPTION_SNIPPETS = (
    "metadata inferred from",
    "review before control design",
)

_ASCII_REPLACEMENTS = (
    ("θ", "theta"), ("Θ", "Theta"), ("ω", "omega"), ("Ω", "Omega"),
    ("τ", "tau"), ("α", "alpha"), ("β", "beta"), ("γ", "gamma"),
    ("δ", "delta"), ("Δ", "Delta"), ("φ", "phi"), ("Φ", "Phi"),
    ("ψ", "psi"), ("Ψ", "Psi"), ("ρ", "rho"), ("σ", "sigma"),
    ("Σ", "Sigma"), ("μ", "mu"), ("λ", "lambda"), ("π", "pi"),
    ("·", "*"), ("×", "*"), ("²", "^2"), ("³", "^3"), ("¹", "^1"),
    ("⁰", "^0"), ("⁻", "-"), ("≈", "~="), ("≤", "<="), ("≥", ">="),
    ("≠", "!="), ("→", "->"), ("←", "<-"), ("°", " deg"), ("\u00a0", " "),
)


def _to_ascii(text: str) -> str:
    if not text:
        return text
    out = text
    for src, dst in _ASCII_REPLACEMENTS:
        out = out.replace(src, dst)
    return "".join(ch if (32 <= ord(ch) <= 126) or ch in "\n\r\t" else "?" for ch in out)


def _clean_assumptions(assumptions: Any) -> List[str]:
    if not isinstance(assumptions, list):
        return ["Continuous-time state-space dynamics"]
    cleaned: List[str] = []
    for item in assumptions:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        if any(bad in text.lower() for bad in _FORBIDDEN_ASSUMPTION_SNIPPETS):
            continue
        cleaned.append(text)
    return cleaned or ["Continuous-time state-space dynamics"]


def _load_prompt(name: str) -> Dict[str, str]:
    path = PROMPTS_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


class PlantModelAgent:
    """Conversational plant-model agent: continue / draft / complete.

    One instance is created per Streamlit session and kept alive in
    ``st.session_state`` for the life of the session, so ``draft_count`` /
    ``latest_draft`` naturally persist across reruns without a separate
    export/import step.
    """

    def __init__(
        self,
        client,
        *,
        model: Optional[str] = None,
        max_drafts: int = DEFAULT_MAX_DRAFTS,
        min_user_turns_before_completion: int = DEFAULT_MIN_USER_TURNS_BEFORE_COMPLETION,
        metadata_max_retries: int = DEFAULT_METADATA_MAX_RETRIES,
    ) -> None:
        self.client = client
        self.model = model or openai_client.DEFAULT_MODEL

        primary = _load_prompt("plant_model_agent")
        metadata = _load_prompt("plant_model_metadata")
        self._system_prompt: str = primary["system_prompt"]
        self._user_template: str = primary["user_prompt_template"]
        self._meta_system_prompt: str = metadata["system_prompt"]
        self._meta_user_template: str = metadata["user_prompt_template"]

        self.max_drafts = max(1, max_drafts)
        self.min_user_turns_before_completion = max(1, min_user_turns_before_completion)
        self.metadata_max_retries = max(0, metadata_max_retries)

        # Conversation state.
        self.draft_count = 0
        self.latest_draft: Optional[Dict[str, Any]] = None

        # Per-turn introspection consumed by the Streamlit UI to build the
        # staged step log (see backend_core.steps.Step).
        self.last_status: Optional[str] = None
        self.last_metadata_generated: bool = False
        self.latest_code: Optional[str] = None

    def reset(self) -> None:
        self.draft_count = 0
        self.latest_draft = None
        self.latest_code = None

    # -- prompt rendering ----------------------------------------------

    @staticmethod
    def format_history(messages: Iterable[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)

    def _render_user_prompt(
        self, history_text: str, user_message: str, retrieved_context: str, web_context: str
    ) -> str:
        return (
            self._user_template.replace("{{conversation_history}}", history_text or "(no previous turns)")
            .replace("{{user_message}}", user_message)
            .replace("{{retrieved_context}}", retrieved_context.strip() or "(none)")
            .replace("{{web_context}}", web_context.strip() or "(none)")
        )

    def _call_primary(self, user_prompt: str, system_suffix: str = "") -> str:
        resp = openai_client.responses_call(
            self.client,
            kind="agent_primary",
            model=self.model,
            instructions=self._system_prompt + system_suffix,
            input_text=user_prompt,
            temperature=0.2,
        )
        return resp.output_text or ""

    # -- main entry point -------------------------------------------------

    def step(
        self,
        history_messages: Iterable[Dict[str, str]],
        user_message: str,
        retrieved_context: str = "",
        web_context: str = "",
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Run one conversational turn.

        Returns ``(display_text, final_payload)``. ``final_payload`` is
        non-None only on the turn a model becomes final (status complete,
        or the draft cap is hit).
        """
        self.last_status = None
        self.last_metadata_generated = False
        self.latest_code = None

        history_messages = list(history_messages)
        history_text = self.format_history(history_messages)
        user_prompt = self._render_user_prompt(history_text, user_message, retrieved_context, web_context)
        user_turns = sum(1 for m in history_messages if m.get("role") == "user") + 1
        user_accepts = bool(_FINISH_RE.search(user_message))

        raw = self._call_primary(user_prompt)
        parsed = extract_json(raw)

        if parsed is None:
            raw = self._call_primary(user_prompt, system_suffix=_REPAIR_NOTE)
            parsed = extract_json(raw)
            if parsed is None:
                self.last_status = "unparsed"
                return raw.strip() or "(empty model response)", None

        # If the user clearly accepted and we already have a draft on file,
        # promote it even if this turn's status wasn't literally "complete".
        if user_accepts and self.latest_draft is not None and parsed.get("status") != "complete":
            if parsed.get("status") == "complete" and self._has_code(parsed):
                return self._accept_complete(parsed)
            return self._accept_complete(self.latest_draft)

        status = parsed.get("status")

        if status == "continue":
            reply = _to_ascii((parsed.get("reply") or "").strip())
            self.last_status = "continue"
            if reply:
                return reply, None
            # Empty reply on "continue" — nudge the model to actually say
            # something useful instead of showing a blank bubble.
            raw = self._call_primary(user_prompt, system_suffix=_FORCE_DRAFT_NOTE)
            parsed = extract_json(raw)
            if parsed is None:
                return raw.strip() or "(empty model response)", None
            status = parsed.get("status")
            if status == "continue":
                return (parsed.get("reply") or raw).strip(), None
            # else fall through to draft/complete handling below

        if status == "draft" and self._has_code(parsed):
            return self._accept_draft(parsed)

        if status == "complete" and self._has_code(parsed):
            allowed = user_accepts or self.latest_draft is not None or user_turns >= self.min_user_turns_before_completion
            if allowed:
                return self._accept_complete(parsed)
            # Too early — ask the model to draft/continue instead of finalising.
            raw = self._call_primary(
                user_prompt
                + f"\n\n(Your previous reply claimed status complete too early. "
                f"Rejected payload:\n{raw}\n)",
                system_suffix=_FORCE_DRAFT_NOTE,
            )
            parsed = extract_json(raw)
            if parsed is None:
                return raw.strip() or "(empty model response)", None
            if parsed.get("status") == "draft" and self._has_code(parsed):
                return self._accept_draft(parsed)
            if parsed.get("status") == "continue":
                self.last_status = "continue"
                return (parsed.get("reply") or raw).strip(), None
            if parsed.get("status") == "complete" and self._has_code(parsed):
                return self._accept_complete(parsed)
            return raw.strip(), None

        # Unknown/malformed shape — surface whatever text we have.
        reply = (parsed.get("reply") or "").strip()
        self.last_status = status or "unknown"
        return reply or raw.strip() or "(empty model response)", None

    # -- draft / complete handling ----------------------------------------

    def _accept_draft(self, parsed: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        parsed = self._sanitize(parsed)
        self.draft_count += 1
        self.last_status = "draft"
        self.latest_code = parsed["python_code"]
        self.latest_draft = self._attach_metadata(parsed)
        display = self._format_draft(parsed)
        final = dict(self.latest_draft) if self.draft_count >= self.max_drafts else None
        return display, final

    def _accept_complete(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        payload = self._sanitize(payload)
        final = self._attach_metadata(payload)
        self.latest_draft = final
        self.latest_code = final["python_code"]
        self.last_status = "complete"
        return f"Model ready — **{final['system_name']}**.", final

    def _attach_metadata(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "system_name": parsed["system_name"],
            "python_code": parsed["python_code"],
        }
        meta = self._generate_metadata(parsed["system_name"], parsed["python_code"])
        if meta:
            out["metadata"] = meta
            self.last_metadata_generated = True
        return out

    def _generate_metadata(self, system_name: str, python_code: str) -> Optional[Dict[str, Any]]:
        repair_section = ""
        attempts = 1 + self.metadata_max_retries
        for _ in range(attempts):
            user_prompt = (
                self._meta_user_template.replace("{{system_name}}", system_name or "System")
                .replace("{{python_code}}", python_code or "")
                .replace("{{repair_section}}", repair_section)
            )
            try:
                resp = openai_client.responses_call(
                    self.client,
                    kind="agent_metadata",
                    model=self.model,
                    instructions=self._meta_system_prompt,
                    input_text=user_prompt,
                    temperature=0.0,
                )
                raw = resp.output_text or ""
            except Exception:
                return None

            meta = extract_json(raw)
            if isinstance(meta, dict) and "states" not in meta and isinstance(meta.get("metadata"), dict):
                meta = meta["metadata"]
            if not isinstance(meta, dict) or not self._metadata_shape_ok(meta):
                repair_section = (
                    "Previous metadata reply was invalid JSON, or had mismatched "
                    "list lengths / missing required keys. Fix it and reply again "
                    "with ONLY the metadata object."
                )
                continue
            meta["assumptions"] = _clean_assumptions(meta.get("assumptions"))
            return meta
        return None

    @staticmethod
    def _metadata_shape_ok(meta: Dict[str, Any]) -> bool:
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
        return bool(meta.get("system_type"))

    # -- small helpers ------------------------------------------------------

    @staticmethod
    def _has_code(data: Dict[str, Any]) -> bool:
        return all(isinstance(data.get(k), str) and data[k].strip() for k in REQUIRED_CODE_KEYS)

    @staticmethod
    def _sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)
        for key in ("system_name", "python_code", "reply"):
            if isinstance(out.get(key), str):
                out[key] = _to_ascii(out[key])
        out.pop("metadata", None)  # the primary agent must never carry metadata
        return out

    @staticmethod
    def _format_draft(parsed: Dict[str, Any]) -> str:
        reply = (parsed.get("reply") or "").strip()
        name = parsed.get("system_name", "draft")
        code = parsed.get("python_code", "")
        parts = []
        if reply:
            parts.append(reply)
        parts.append(f"**Draft: {name}**")
        parts.append(f"```python\n{code}\n```")
        parts.append("_Say what to change, or reply **finish** to accept this draft._")
        return "\n\n".join(parts)

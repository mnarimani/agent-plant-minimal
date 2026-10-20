"""Thin, logged wrapper around the OpenAI Responses API.

Everything in this app — the primary agent, the metadata extractor, RAG
grounding, web search, and the Simulink conversion agent
(``backend_core.simulink``) — goes through :func:`responses_call` so every
call is uniformly logged (see ``logging_utils``) and uniformly resilient
to the handful of ways different models reject parameters (e.g. some
reasoning models reject ``temperature``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from . import logging_utils

# Load .env once, on first import, regardless of who imports us first.
load_dotenv()

DEFAULT_MODEL = os.environ.get("AGENTPLANT_MODEL", "gpt-4.1")


def get_client() -> OpenAI:
    """Build an OpenAI client from ``OPENAI_API_KEY``.

    Raises a plain ``RuntimeError`` with a friendly message if the key is
    missing, so the Streamlit layer can show it directly instead of a raw
    SDK traceback.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it, or put it in a .env file "
            "in the project root, then restart the app."
        )
    return OpenAI(api_key=api_key)


def _is_param_rejection(exc: Exception, param_name: str) -> bool:
    msg = str(exc).lower()
    return param_name in msg and (
        "unsupported" in msg or "not supported" in msg or "unknown parameter" in msg
        or "invalid" in msg
    )


def responses_call(
    client: OpenAI,
    *,
    kind: str,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    input_text: Union[str, List[Dict[str, str]]],
    tools: Optional[List[Dict[str, Any]]] = None,
    include: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    text_format: Optional[Dict[str, Any]] = None,
):
    """Call ``client.responses.create`` and log the interaction.

    Falls back to omitting ``temperature`` if the model rejects it (some
    reasoning models only accept the default). Returns the raw SDK
    response object; callers read ``.output_text`` / ``.output`` off it.

    ``input_text`` is either one user message or a list of
    ``{"role", "content"}`` messages (a multi-turn exchange, e.g. a draft
    sent back with the reason it was rejected). ``text_format`` is passed as
    ``text={"format": ...}``, e.g. ``{"type": "json_object"}`` for JSON mode.
    """
    model = model or DEFAULT_MODEL
    kwargs: Dict[str, Any] = {"model": model, "input": input_text}
    if instructions is not None:
        kwargs["instructions"] = instructions
    if tools is not None:
        kwargs["tools"] = tools
    if include is not None:
        kwargs["include"] = include
    if temperature is not None:
        kwargs["temperature"] = temperature
    if text_format is not None:
        kwargs["text"] = {"format": text_format}
    logged_input = input_text if isinstance(input_text, str) else json.dumps(input_text, ensure_ascii=False)

    try:
        resp = client.responses.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - defensive param-compat retry
        if temperature is not None and _is_param_rejection(exc, "temperature"):
            kwargs.pop("temperature", None)
            resp = client.responses.create(**kwargs)
        else:
            logging_utils.log_interaction(
                kind=f"{kind}_error",
                system_prompt=instructions or "",
                user_prompt=logged_input,
                response_text=f"ERROR: {exc}",
                model=model,
                extra={"tools": tools or []},
            )
            raise

    text = getattr(resp, "output_text", "") or ""
    logging_utils.log_interaction(
        kind=kind,
        system_prompt=instructions or "",
        user_prompt=logged_input,
        response_text=text,
        model=model,
        extra={"tools": tools or []},
    )
    return resp

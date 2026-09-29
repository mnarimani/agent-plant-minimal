"""Thin, logged wrapper around the OpenAI Responses API.

Everything in this app — the primary agent, the metadata extractor, RAG
grounding, and web search — goes through :func:`responses_call` so every
call is uniformly logged (see ``logging_utils``) and uniformly resilient
to the handful of ways different models reject parameters (e.g. some
reasoning models reject ``temperature``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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
    input_text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    include: Optional[List[str]] = None,
    temperature: Optional[float] = None,
):
    """Call ``client.responses.create`` and log the interaction.

    Falls back to omitting ``temperature`` if the model rejects it (some
    reasoning models only accept the default). Returns the raw SDK
    response object; callers read ``.output_text`` / ``.output`` off it.
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
                user_prompt=input_text,
                response_text=f"ERROR: {exc}",
                model=model,
                extra={"tools": tools or []},
            )
            raise

    text = getattr(resp, "output_text", "") or ""
    logging_utils.log_interaction(
        kind=kind,
        system_prompt=instructions or "",
        user_prompt=input_text,
        response_text=text,
        model=model,
        extra={"tools": tools or []},
    )
    return resp

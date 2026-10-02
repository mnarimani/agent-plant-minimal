"""OpenAI-native web search (``web_search``), default OFF, focus-gated.

Two layers keep search calls rare and worthwhile instead of noisy:

1. **Query planning** — a cheap, tool-free LLM call looks at the
   conversation and either proposes one focused search query (a real
   component's datasheet parameter, a named standard, a specific numeric
   constant, recent information — something the model wouldn't already
   know confidently) or replies ``NONE``.
2. **Focus filter** (:func:`looks_focused`) — a plain heuristic that
   rejects generic textbook queries even if the planner slips one
   through (e.g. "state variable definition", "plant model", "dynamics
   equation").

Only a query that survives both is actually sent to OpenAI's hosted
``web_search`` tool. No Tavily, no other search provider.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import yaml

from . import openai_client

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_PROMPTS_PATH = _PROMPTS_DIR / "websearch.yaml"


def _load_websearch_prompts() -> dict:
    """Load query-planner / search instructions from prompts/websearch.yaml."""
    with _PROMPTS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {_PROMPTS_PATH}")
    return data


_PROMPTS = _load_websearch_prompts()
_QUERY_PLANNER_INSTRUCTIONS = (_PROMPTS.get("query_planner_instructions") or "").strip()
_SEARCH_INSTRUCTIONS = (_PROMPTS.get("search_instructions") or "").strip()
if not _QUERY_PLANNER_INSTRUCTIONS or not _SEARCH_INSTRUCTIONS:
    raise RuntimeError(
        f"websearch prompts missing keys in {_PROMPTS_PATH} "
        "(need query_planner_instructions and search_instructions)"
    )

# Generic phrases that should never trigger a real web search on their own.
_GENERIC_BLOCKLIST = {
    "plant model", "state variable", "state variable definition", "dynamics equation",
    "control system", "differential equation", "ode", "state space", "transfer function",
    "system dynamics", "physics", "newtons law", "newton's law", "equations of motion",
    "dynamics function", "what is a dynamics function", "plant modeling", "plant modelling",
}


def _strip_punct(s: str) -> str:
    return re.sub(r"[?.!,:;]+$", "", s.strip())


# Phrases that mark the planner's reply as talking BACK to the user (asking
# them for something, apologizing, requesting an upload) rather than
# proposing a search query. Caught this empirically: the planner, given no
# context to work with, replied "Please upload the content you want me to
# refer to" -- a well-formed-looking, 9-word, capitalized sentence that the
# rest of the heuristic alone would have waved through.
_CONVERSATIONAL_REPLY_MARKERS = (
    "please upload", "please provide", "please share", "please attach",
    "please paste", "i don't have", "i do not have", "i don't see",
    "you haven't", "you have not", "could you", "can you provide",
    "no content", "no attachment", "no file",
)


def _looks_conversational(low: str) -> bool:
    return any(marker in low for marker in _CONVERSATIONAL_REPLY_MARKERS)


# Domain tokens that signal a real plant / product / vehicle class worth searching.
_DOMAIN_TOKENS = {
    "auv", "uuv", "rov", "quadrotor", "quadcopter", "helicopter", "quanser",
    "aero", "pendulum", "cartpole", "cart-pole", "dcmotor", "dc-motor", "servo",
    "robot", "manipulator", "drone", "uav", "ship", "submarine", "glider",
    "ballbeam", "ball-and-beam", "maglev", "levitation", "turbine", "boiler",
    "cstr", "heat-exchanger", "inverted", "rotary", "flexible", "arm",
    "pitch", "yaw", "heave", "surge", "sway", "depth", "buoyancy",
}

_RESEARCH_INTENT = re.compile(
    r"\b("
    r"search|look\s*up|look\s*for|find|fetch|google|"
    r"literature|paper|papers|arxiv|datasheet|manual|parameters?|"
    r"typical\s+model|state[- ]space|equations?\s+of\s+motion"
    r")\b",
    re.I,
)


def user_wants_research(user_message: str) -> bool:
    """True when the user explicitly asks to search / look up literature."""
    return bool(_RESEARCH_INTENT.search(user_message or ""))


def looks_focused(query: Optional[str], *, research_intent: bool = False) -> bool:
    """Reject vague/generic queries so we only search when it's worth it.

    When ``research_intent`` is True (user asked to search/look up), accept
    domain plant names (AUV, Quanser, …) even without digits.
    """
    if not query:
        return False
    q = _strip_punct(query)
    if not q or q.strip().upper() == "NONE":
        return False

    low = re.sub(r"[^a-z0-9\s\-]", "", q.lower()).strip()
    if not low:
        return False
    words = [w for w in low.replace("-", " ").split() if w]
    if len(words) < 2:
        return False

    if _looks_conversational(low):
        return False

    has_digit = any(ch.isdigit() for ch in q)
    has_acronym = any(re.search(r"\b[A-Z]{2,}\b", w) for w in q.split())
    non_leading_words = q.split()[1:]
    has_proper_noun = any(w[:1].isupper() for w in non_leading_words if w.strip(",."))
    has_domain = any(w in _DOMAIN_TOKENS for w in words)

    contains_generic_phrase = any(phrase in low for phrase in _GENERIC_BLOCKLIST)
    if low in _GENERIC_BLOCKLIST:
        return False
    if contains_generic_phrase and not (has_digit or has_proper_noun or has_acronym or has_domain):
        return False

    has_long_token = any(len(w) >= 10 for w in words)
    if has_digit or has_proper_noun or has_acronym or has_domain or has_long_token:
        return True
    if research_intent and len(words) >= 3:
        return True
    return False


def plan_query(
    client,
    history_text: str,
    user_message: str,
    *,
    retrieved_context: str = "",
    model: Optional[str] = None,
) -> Optional[str]:
    """Ask the model whether a focused web search is warranted right now.

    ``retrieved_context`` (whatever RAG already found this turn, if
    anything) is passed in so the planner isn't blind to it: without this,
    a message like "refer to the attached content" gives the planner
    nothing to work with, and it was empirically observed to reply with a
    request back to the user instead of a query or NONE.
    """
    context_block = (
        f"\n\nDocument context already retrieved this turn (from the user's "
        f"uploaded file, if any):\n{retrieved_context.strip()}"
        if retrieved_context.strip()
        else ""
    )
    resp = openai_client.responses_call(
        client,
        kind="web_query_plan",
        model=model or openai_client.DEFAULT_MODEL,
        instructions=_QUERY_PLANNER_INSTRUCTIONS,
        input_text=(
            f"Conversation so far:\n{history_text or '(no previous turns)'}\n\n"
            f"Latest user message:\n{user_message}"
            f"{context_block}"
        ),
        temperature=0.0,
    )
    text = (resp.output_text or "").strip()
    if not text or text.upper().startswith("NONE"):
        return None
    return text


def search(client, query: str, *, model: Optional[str] = None) -> Tuple[str, List[Dict[str, str]]]:
    """Run OpenAI's hosted ``web_search`` tool for ``query``.

    Returns ``(brief_text, links)``. Tries the current tool name first and
    falls back to the legacy name for older SDK/model combinations.
    """
    model = model or openai_client.DEFAULT_MODEL
    last_err: Optional[Exception] = None
    for tool_type in ("web_search", "web_search_preview"):
        try:
            resp = openai_client.responses_call(
                client,
                kind="web_search",
                model=model,
                instructions=_SEARCH_INSTRUCTIONS,
                input_text=query,
                tools=[{"type": tool_type}],
            )
            brief = resp.output_text or ""
            links = _extract_citations(resp)
            return brief, links
        except Exception as exc:  # noqa: BLE001 - try the next tool variant
            last_err = exc
            continue
    raise RuntimeError(f"Web search failed: {last_err}")


def _extract_citations(resp: Any) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    for item in getattr(resp, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                if getattr(ann, "type", None) == "url_citation":
                    url = getattr(ann, "url", "") or ""
                    title = getattr(ann, "title", "") or url
                    if url:
                        links.append({"title": title, "url": url})
    seen = set()
    deduped = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            deduped.append(link)
    return deduped

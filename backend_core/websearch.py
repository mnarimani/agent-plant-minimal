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

from . import openai_client

_QUERY_PLANNER_INSTRUCTIONS = """\
You decide whether a plant-modelling assistant needs to search the web right now.

Only propose a search when the user needs a SPECIFIC real-world fact the
assistant would not already know confidently: a named component's datasheet
parameter, a real standard or regulation, a specific numeric constant, a named
product or material, or recent/changed information.

Never propose a search for generic textbook concepts the assistant already
knows, e.g. "state variable", "plant model", "differential equation",
"Newton's second law", "transfer function", "what is a dynamics function".

If the user is referring to a file or document already uploaded (e.g. "refer
to the attached content", "use the PDF I gave you"), do NOT search the web for
that — check the "document context already retrieved this turn" section below
first. Only propose a web search if a specific fact is still missing after
considering that context.

Your reply is read by code, not a person: it is either the search query text
itself, or exactly NONE. Never reply with a question, a request for the user
to upload/provide/clarify something, or any other sentence addressed to the
user — if you don't have enough to search on, or nothing further is needed,
that is exactly what NONE means, not a reason to ask for more.

Reply with ONLY the search query text (a few specific words — include part/
model numbers or proper nouns when relevant), or reply with exactly NONE if
no search is needed right now. No punctuation-only replies, no explanation.
"""

_SEARCH_INSTRUCTIONS = (
    "Search the web and answer with a short, focused, factual brief (3-5 "
    "sentences). Prefer authoritative sources such as datasheets, standards "
    "bodies, and official documentation over forums or blogs."
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


def looks_focused(query: Optional[str]) -> bool:
    """Reject vague/generic queries so we only search when it's worth it."""
    if not query:
        return False
    q = _strip_punct(query)
    if not q or q.strip().upper() == "NONE":
        return False

    low = re.sub(r"[^a-z0-9\s]", "", q.lower()).strip()
    if not low:
        return False
    words = [w for w in low.split() if w]
    if len(words) < 3:
        return False

    if _looks_conversational(low):
        return False

    has_digit = any(ch.isdigit() for ch in q)
    # Sentence-initial capitalization is just English orthography and
    # carries no signal about properness -- "Please upload..." is
    # capitalized for the same reason "What is..." is. Only a capitalized
    # word AFTER the first counts as proper-noun evidence.
    non_leading_words = q.split()[1:]
    has_proper_noun = any(w[:1].isupper() for w in non_leading_words if w.strip(",."))
    # A generic phrase anywhere in the query (e.g. "what IS a transfer function")
    # disqualifies the query unless something concretely specific is also present.
    contains_generic_phrase = any(phrase in low for phrase in _GENERIC_BLOCKLIST)
    if contains_generic_phrase and not (has_digit or has_proper_noun):
        return False
    if low in _GENERIC_BLOCKLIST:
        return False

    # Only a genuinely long/technical single token counts on its own; the
    # threshold is high enough to skip common words like "transfer" or
    # "dynamics" that show up inside otherwise-generic questions.
    has_long_token = any(len(w) >= 10 for w in words)
    return has_digit or has_proper_noun or has_long_token


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

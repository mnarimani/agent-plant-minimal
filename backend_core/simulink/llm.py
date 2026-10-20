"""The C_to_Python agent's LLM, served by this project's OpenAI client.

``agent.run`` takes ``llm(messages) -> text`` with ``llm.model`` and
``llm.usage`` (one ``(input_tokens, output_tokens)`` per call, for the cost
report). This builds that callable on ``openai_client.responses_call``, so the
conversion uses the same Responses API and the same ``.logs/`` as the rest of the app.

Model: its own setting, ``AGENTPLANT_SIMULINK_MODEL`` (default ``gpt-5.4-mini``), not
``AGENTPLANT_MODEL``. The agent's prompts were tuned with gpt-5.4-mini; with gpt-4o-mini
the same prompt gave numerically correct code without the model's parameter names
(K/m written as 0.002), which the validation cannot see.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .. import openai_client

DEFAULT_MODEL = "gpt-5.4-mini"


def model_name() -> str:
    """the model of the Simulink conversion: AGENTPLANT_SIMULINK_MODEL, else gpt-5.4-mini"""
    return (os.environ.get("AGENTPLANT_SIMULINK_MODEL") or "").strip() or DEFAULT_MODEL


def make_llm(client, model: Optional[str] = None):
    """-> llm(messages) for ``agent.run``. messages: [{"role", "content"}, ...]."""
    model = model or model_name()

    def llm(messages: List[Dict[str, str]]) -> str:
        # The messages go in as they are - system prompt included - the way the agent's prompts
        # were written (chat messages). The system prompt must not move to `instructions`:
        # JSON mode of the Responses API requires the word "json" in the INPUT messages, and
        # in the agent's prompts it is the system prompt that says "Return ONE JSON object".
        resp = openai_client.responses_call(
            client,
            kind="simulink_agent",
            model=model,
            input_text=[{"role": m["role"], "content": m["content"]} for m in messages],
            text_format={"type": "json_object"},
        )
        usage = getattr(resp, "usage", None)
        llm.usage.append((int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)))
        return resp.output_text or ""

    llm.model, llm.usage = model, []
    return llm

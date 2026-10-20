"""Who decides that the user accepted a draft: the model, not keywords.

The agent finalises a model only when the model answers with status
"complete". It does not scan the user's message for words like "done".

Runs offline with a scripted fake OpenAI client — no API key, no network:

    python -m pytest tests/test_finish_detection.py

These tests pin the agent's side of the contract. Whether a real model picks
the right status for a given message is a prompt-quality question and needs a
run against the real API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_core import agent as agent_mod  # noqa: E402
from backend_core import logging_utils  # noqa: E402

CODE_V1 = (
    "import numpy as np\n\n"
    "def dynamics(t, x, u):\n"
    "    y = x[0]  # output (-)\n"
    "    m = 0.05  # mass (kg)\n"
    "    return np.array([(-y + u) / m])\n"
)
CODE_V2 = CODE_V1.replace("m = 0.05", "m = 2.0")

METADATA = {
    "states": ["y"],
    "state_meanings": ["output"],
    "inputs": ["u"],
    "outputs": ["y"],
    "state_equations": ["(-y + u) / m"],
    "parameters": {"m": 0.05},
    "system_type": "1st order",
    "assumptions": ["linear"],
}

HISTORY = [
    {"role": "user", "content": "a first-order lag"},
    {"role": "assistant", "content": "draft"},
]


class _Response:
    def __init__(self, text: str) -> None:
        self.output_text = text
        self.output = []


class FakeClient:
    """Answers metadata calls with METADATA and primary calls from a queue."""

    def __init__(self) -> None:
        self.responses = self
        self.primary_replies: list[dict] = []

    def create(self, **kwargs):
        if "extract structured metadata" in kwargs.get("instructions", ""):
            return _Response(json.dumps(METADATA))
        return _Response(json.dumps(self.primary_replies.pop(0)))


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def agent_with_draft(client):
    """An agent that has already shown the user a draft with m = 0.05."""
    agent = agent_mod.PlantModelAgent(client, model="fake-model")
    client.primary_replies.append(
        {"status": "draft", "reply": "draft", "system_name": "Lag", "python_code": CODE_V1}
    )
    _, final = agent.step([], "a first-order lag")
    assert final is None and agent.draft_count == 1
    return agent


@pytest.mark.parametrize(
    "message",
    [
        "I'm not done yet, change the mass to 2 kg",
        "that's not perfect, mass should be 2 kg",
        "looks good but make the mass 2 kg",
        "do not accept this, use 2 kg",
    ],
)
def test_acceptance_words_in_a_revision_request_do_not_finalise(agent_with_draft, client, message):
    client.primary_replies.append(
        {"status": "draft", "reply": "mass is now 2 kg", "system_name": "Lag", "python_code": CODE_V2}
    )

    _, final = agent_with_draft.step(HISTORY, message)

    assert final is None, "the model returned a revised draft; it must be kept, not finalised"
    assert agent_with_draft.last_status == "draft"
    assert "m = 2.0" in agent_with_draft.latest_draft["python_code"]
    assert agent_with_draft.draft_count == 2


@pytest.mark.parametrize("message", ["finish", "yes", "تمومه، همینو نهایی کن", "c'est bon"])
def test_model_saying_complete_finalises_in_any_language(agent_with_draft, client, message):
    client.primary_replies.append({"status": "complete", "system_name": "Lag", "python_code": CODE_V1})

    text, final = agent_with_draft.step(HISTORY, message)

    assert final is not None and final["python_code"] == CODE_V1
    assert agent_with_draft.last_status == "complete"
    assert "Model ready" in text


def test_complete_without_code_finalises_the_draft_the_user_saw(agent_with_draft, client):
    client.primary_replies.append({"status": "complete"})

    _, final = agent_with_draft.step(HISTORY, "finish")

    assert final is not None and final["python_code"] == CODE_V1


def test_agent_does_not_override_the_model_on_keywords(agent_with_draft, client):
    client.primary_replies.append({"status": "continue", "reply": "Done with what, exactly?"})

    text, final = agent_with_draft.step(HISTORY, "done")

    assert final is None
    assert text == "Done with what, exactly?"


def test_complete_on_the_first_turn_is_sent_back_for_a_draft(client):
    agent = agent_mod.PlantModelAgent(client, model="fake-model")
    client.primary_replies += [
        {"status": "complete", "system_name": "Lag", "python_code": CODE_V1},
        {"status": "draft", "reply": "draft", "system_name": "Lag", "python_code": CODE_V1},
    ]

    _, final = agent.step([], "a first-order lag")

    assert final is None
    assert agent.last_status == "draft"


def test_complete_before_any_draft_and_without_code_does_not_finalise(client):
    agent = agent_mod.PlantModelAgent(client, model="fake-model")
    client.primary_replies.append({"status": "complete", "reply": "No draft yet. Which system?"})

    text, final = agent.step([], "finish")

    assert final is None
    assert text == "No draft yet. Which system?"

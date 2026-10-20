"""Simulink upload package -> draft of the plant-model agent (backend_core.simulink).

Runs offline with a scripted fake OpenAI client — no MATLAB, no API key:

    python -m pytest tests/test_simulink_convert.py

fixtures/maglev_c2py.zip is a small upload package in the format of c2py_export.m: the
Embedded Coder code of the maglev model of the C_to_Python project and the first 2 s of its
Simulink step / sine / PRBS runs.
"""

from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_core import agent as plant_agent  # noqa: E402
from backend_core import logging_utils, sandbox  # noqa: E402
from backend_core.simulink import llm as simulink_llm  # noqa: E402
from backend_core.simulink.c2py import agent as c2py_agent  # noqa: E402
from backend_core.simulink.c2py import convert  # noqa: E402

MAGLEV = Path(__file__).resolve().parent / "fixtures" / "maglev_c2py.zip"


class FakeResponses:
    """Responses API stand-in. A rewrite prompt gets the exact translation it was given back
    (which passes the validation); the naming prompts get an empty, valid answer.
    Like the real API, JSON mode is refused unless an INPUT message mentions "json"
    (`instructions` do not count)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["input"]
        if (kwargs.get("text") or {}).get("format", {}).get("type") == "json_object" and \
                "json" not in " ".join(m["content"] for m in messages).lower():
            raise ValueError("Error code: 400 - Response input messages must contain the word 'json' in some form "
                             "to use 'text.format' of type 'json_object'.")
        system = " ".join(m["content"] for m in messages if m["role"] == "system") + (kwargs.get("instructions") or "")
        last = messages[-1]["content"]
        if '"python_code"' in system and "```python\n" in last:
            code = last.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
            answer = {"python_code": code, "system": "Magnetic levitation", "outputs": [], "notes": ""}
        else:
            answer = {"system": "Magnetic levitation", "rename": {}, "meaning": {}, "notes": ""}
        return types.SimpleNamespace(output_text=json.dumps(answer), output=[],
                                     usage=types.SimpleNamespace(input_tokens=1000, output_tokens=100))


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path / "logs")


@pytest.fixture(scope="module")
def maglev_result(tmp_path_factory):
    fake = types.SimpleNamespace(responses=FakeResponses())
    runs = tmp_path_factory.mktemp("runs")
    result = convert.convert_package(MAGLEV, runs, simulink_llm.make_llm(fake, model="gpt-5.4-mini"), log=lambda m: None)
    return result, fake


def test_maglev_package_becomes_a_validated_draft(maglev_result):
    result, _ = maglev_result
    assert result.ok, result.message
    assert [s.ok for s in result.steps] == [True] * len(result.steps)
    draft = result.draft
    assert set(draft) == {"system_name", "python_code", "metadata"}
    meta = draft["metadata"]
    for key in plant_agent.REQUIRED_METADATA_KEYS:          # the format of the plant-model agent
        assert key in meta, key
    assert len(meta["states"]) == len(meta["state_meanings"]) == len(meta["state_equations"]) == 3
    sim = meta["simulink"]
    assert sim["validated"] and sim["status"] == c2py_agent.STATUS_OK
    assert sim["python_vs_simulink"]["passed"]
    assert sim["python_vs_simulink"]["inputs"] == ["step", "sin", "prbs"]
    assert sim["state_equations_checked"] == [True, True, True]
    assert "maglev" in result.message and "PASS" in result.message


def test_converted_code_runs_in_the_app_sandbox(maglev_result):
    result, _ = maglev_result
    sim = sandbox.run_default_simulation(result.draft["python_code"])
    assert sim.success, sim.message
    assert sim.n_states == 3


def test_llm_calls_go_through_the_project_client_in_json_mode(maglev_result):
    result, fake = maglev_result
    calls = fake.responses.calls
    assert calls, "the agent made no LLM call"
    call = calls[0]
    assert call["model"] == "gpt-5.4-mini"
    assert call["text"] == {"format": {"type": "json_object"}}
    assert isinstance(call["input"], list)
    assert [m["role"] for m in call["input"]][:2] == ["system", "user"]       # the agent's chat messages, as written
    assert result.cost_usd is not None and result.cost_usd > 0


def test_an_upload_that_is_not_a_package_is_rejected_with_a_reason(tmp_path):
    bad = tmp_path / "model.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("model.c", "void model_initialize(void) {}")
    result = convert.convert_package(bad, tmp_path / "runs", llm=None, log=lambda m: None)
    assert not result.ok and result.draft is None
    assert "c2py_manifest.json" in result.message
    assert not result.steps[-1].ok


def test_the_model_file_itself_is_rejected_with_a_reason(tmp_path):
    slx = tmp_path / "model.slx"
    slx.write_bytes(b"PK")
    result = convert.convert_package(slx, tmp_path / "runs", llm=None, log=lambda m: None)
    assert not result.ok
    assert "c2py_export.m" in result.message


def test_reconcile_keeps_the_validated_metadata_while_the_code_is_unchanged(maglev_result):
    origin = maglev_result[0].draft
    draft = {"system_name": "x", "python_code": origin["python_code"] + "\n\n", "metadata": {"states": ["llm"]}}
    convert.reconcile(draft, origin)
    assert draft["metadata"] == origin["metadata"]
    assert draft["metadata"] is not origin["metadata"]


def test_reconcile_marks_a_changed_model_as_not_verified(maglev_result):
    origin = maglev_result[0].draft
    changed = origin["python_code"].replace("0.002", "0.004")
    assert changed != origin["python_code"]
    draft = {"system_name": "x", "python_code": changed, "metadata": {"states": ["a"]}}
    convert.reconcile(draft, origin)
    assert draft["metadata"]["states"] == ["a"]               # the chat agent's metadata stays
    assert draft["metadata"]["simulink"]["validated"] is False
    assert draft["metadata"]["simulink"]["status"] == convert.MODIFIED


def test_card_of_a_model_with_discrete_memory():
    card = {
        "model": "m", "system": "s", "status": c2py_agent.STATUS_PART, "method": "exact",
        "notes": "PID memory frozen.",
        "model_details": {
            "states": ["a"], "state_meanings": ["position (m)"], "inputs": ["u1", "u2"],
            "input_meanings": ["force (N)", c2py_agent.NA], "outputs": [c2py_agent.NA],
            "state_equations": ["u1 - a"], "parameters": c2py_agent.NA, "system_type": "MISO",
        },
        "source_names": {"states": ["X"], "inputs": ["In1", "In2"], "outputs": ["Out1"]},
    }
    meta = convert.card_to_metadata(card)
    assert meta["inputs"] == ["u1: force (N)", "u2"]
    assert meta["outputs"] == ["Out1"]
    assert meta["parameters"] == {}
    assert meta["assumptions"][0].startswith("Continuous part only")
    assert "PID memory frozen." in meta["assumptions"]
    assert meta["simulink"]["validated"] is True


def test_the_conversion_has_its_own_model_setting(monkeypatch):
    monkeypatch.delenv("AGENTPLANT_SIMULINK_MODEL", raising=False)
    monkeypatch.setenv("AGENTPLANT_MODEL", "gpt-4o-mini")         # the chat's model is not used
    assert simulink_llm.make_llm(object()).model == "gpt-5.4-mini"
    monkeypatch.setenv("AGENTPLANT_SIMULINK_MODEL", "gpt-5.5")
    assert simulink_llm.make_llm(object()).model == "gpt-5.5"


# ---- simulation the way Simulink runs the model --------------------------------------------
def test_the_draft_records_how_simulink_runs_the_model(maglev_result):
    result, _ = maglev_result
    sim = result.draft["metadata"]["simulink"]["simulation"]
    assert sim["solver"] == "ode8" and sim["fixed_step"] == pytest.approx(0.01)
    assert len(sim["initial_state"]) == 3 and sim["n_inputs"] == 1
    c, A, b = result.tableau
    assert len(c) == len(A) == len(b) == 13                          # ode8: 13 stages


def test_fixed_step_simulation_reproduces_the_python_model_of_the_c_code(maglev_result):
    from backend_core.simulink.c2py import validation
    result, _ = maglev_result
    sim_info = result.draft["metadata"]["simulink"]["simulation"]
    sim = convert.simulate(result.draft["python_code"], tableau=result.tableau, fixed_step=sim_info["fixed_step"],
                           x0=sim_info["initial_state"], n_inputs=1, t_final=1.0,
                           input_kind="step", input_params={"t0": 0.2, "amplitude": 2.0})
    assert sim.success and not sim.diverged, sim.message
    Model = validation.load_model_class(result.out_dir / "maglev_py.py")
    ref = Model(solver="ode8").simulate(1.0, sim.u.copy())["x"][: len(sim.t)]
    assert np.max(np.abs(sim.x - ref)) <= 1e-9 * max(1.0, np.max(np.abs(ref)))



def _ode8(maglev_result):
    return maglev_result[0].tableau


def test_a_model_undefined_at_zero_runs_from_its_own_initial_state(maglev_result):
    code = "import numpy as np\n\ndef dynamics(t, x, u):\n    v = x[0]\n    return np.array([-(v * v) / abs(v)])\n"
    kw = dict(tableau=_ode8(maglev_result), fixed_step=0.01, n_inputs=1, t_final=1.0)
    from_zero = convert.simulate(code, x0=[0.0], **kw)
    assert not from_zero.success and "non-finite" in from_zero.message
    from_own = convert.simulate(code, x0=[1.0], **kw)
    assert from_own.success and from_own.x[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_a_run_away_state_stops_early_with_the_trajectory(maglev_result):
    code = "import numpy as np\n\ndef dynamics(t, x, u):\n    return np.array([x[0]])\n"
    sim = convert.simulate(code, tableau=_ode8(maglev_result), fixed_step=0.01, x0=[1.0], n_inputs=1, t_final=30.0)
    assert sim.success and sim.diverged
    assert 13.0 < sim.t[-1] < 14.5                                      # e^t passes 1e6 at t = 13.8 s
    assert np.all(np.isfinite(sim.x))


def test_a_long_simulation_is_cut_to_the_step_limit(maglev_result):
    code = "import numpy as np\n\ndef dynamics(t, x, u):\n    return np.array([-x[0]])\n"
    sim = convert.simulate(code, tableau=_ode8(maglev_result), fixed_step=0.01, x0=[1.0], n_inputs=1,
                           t_final=100.0, max_steps=500)
    assert sim.success and len(sim.t) == 501 and "Shortened to 5 s" in sim.message


def test_simulated_code_still_goes_through_the_sandbox(maglev_result):
    code = "import os\n\ndef dynamics(t, x, u):\n    return x\n"
    sim = convert.simulate(code, tableau=_ode8(maglev_result), fixed_step=0.01, x0=[1.0], n_inputs=1, t_final=1.0)
    assert not sim.success and "not allowed" in sim.message

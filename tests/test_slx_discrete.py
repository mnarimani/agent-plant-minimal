"""Blocks with discrete memory in the direct .slx route (slx2py/model_converter/discrete_semantics.py).

The blocks themselves are checked against Simulink in test_slx_blocks.py (group C) and in whole
models (PID_Controller with Derivative blocks, Longitidunal_Vehicle_Dynamics with Memory blocks).
Here: the draft the app makes of such a model - dynamics(t, x, u) plus update(t, x, u) - its
checks, and its simulation in the app (convert.simulate, the sandbox).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from backend_core import sandbox
from backend_core.simulink.c2py import convert
from backend_core.simulink.direct import slx_direct

MODELS = Path(__file__).parent / "fixtures" / "slx_models_reference.zip"
FOLDER = "slx2py_models_reference/"


def _model(name: str) -> bytes:
    if not MODELS.is_file():
        pytest.skip("no Simulink reference models")
    with zipfile.ZipFile(MODELS) as z:
        return z.read(FOLDER + name + ".slx")


@pytest.fixture(scope="module")
def pid(tmp_path_factory):
    return slx_direct.convert_slx("PID_Controller.slx", _model("PID_Controller"), tmp_path_factory.mktemp("runs"),
                                  log=lambda m: None)


@pytest.fixture(scope="module")
def longitudinal(tmp_path_factory):
    return slx_direct.convert_slx("Longitidunal_Vehicle_Dynamics.slx", _model("Longitidunal_Vehicle_Dynamics"),
                                  tmp_path_factory.mktemp("runs"), log=lambda m: None)


@pytest.mark.parametrize("which", ["pid", "longitudinal"])
def test_models_with_memory_become_drafts_with_an_update(which, request):
    r = request.getfixturevalue(which)
    assert r.ok, r.reason
    assert all(s.ok for s in r.steps)
    code = r.draft["python_code"]
    assert "def dynamics(t, x, u):" in code and "def update(t, x, u):" in code
    assert sandbox.compile_update(code) is not None
    md = r.draft["metadata"]
    assert md["system_type"].startswith("Hybrid model")
    assert any("update(t, x, u)" in a for a in md["assumptions"])
    sim = md["simulink"]["simulation"]
    assert sim["discrete"] and sim["solver"] == "ode8" and sim["fixed_step"] == pytest.approx(0.01)
    assert "discrete memory" in r.message


def test_the_memory_is_named_in_the_states(longitudinal):
    meanings = longitudinal.draft["metadata"]["state_meanings"]
    assert any(m.startswith("Memory 'Frolling Front/Memory'") for m in meanings)


def test_the_draft_simulates_with_its_update_in_the_app(pid):
    """convert.simulate (the app, the Simulation Lab) applies update() at every step, like the engine's
    block model that matches Simulink: the result is the block model's, step for step"""
    sim = pid.draft["metadata"]["simulink"]["simulation"]
    code = pid.draft["python_code"]
    r = convert.simulate(code, tableau=pid.tableau, fixed_step=sim["fixed_step"], x0=sim["initial_state"],
                         n_inputs=sim["n_inputs"], t_final=3.0)
    assert r.success and not r.diverged
    # the Derivative memory: the time of the last major step is the time of the step
    assert r.x[-1, 0] == pytest.approx(3.0)
    assert r.x[-1, 4] == 2.0                                     # two major steps stored


def test_without_update_the_memory_would_stay_put(pid):
    """the reason update() matters: dropped, the memory keeps its initial value"""
    sim = pid.draft["metadata"]["simulink"]["simulation"]
    code = pid.draft["python_code"].split("\n\ndef update(t, x, u):")[0] + "\n"
    assert sandbox.compile_update(code) is None
    r = convert.simulate(code, tableau=pid.tableau, fixed_step=sim["fixed_step"], x0=sim["initial_state"],
                         n_inputs=sim["n_inputs"], t_final=1.0)
    assert r.x[-1, 0] == sim["initial_state"][0]


def test_the_sandbox_runs_a_changed_draft_with_its_update(pid):
    """a draft changed in the chat runs in the general sandbox: still with update(), at fixed step"""
    r = sandbox.simulate(pid.draft["python_code"], x0=pid.draft["metadata"]["simulink"]["simulation"]["initial_state"],
                         t_final=2.0, dt=0.01, input_channel=0, input_kind="step", input_params={"t0": 1.0})
    assert r.success, r.message
    assert "discrete update" in r.solver_used
    assert r.x[-1, 0] == pytest.approx(2.0)


def test_a_fixed_step_divides_the_sample_times():
    assert slx_direct._discrete_step(0.01, []) == 0.01
    assert slx_direct._discrete_step(0.01, [0.05]) == pytest.approx(0.01)
    assert slx_direct._discrete_step(0.01, [0.05, 0.02]) == pytest.approx(0.01)
    assert slx_direct._discrete_step(0.01, [0.015]) == pytest.approx(0.0075)
    assert slx_direct._discrete_step(0.1, [0.03, 0.05]) == pytest.approx(0.01)

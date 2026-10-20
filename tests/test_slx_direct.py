"""Route 0: an .slx converted directly from its XML (backend_core/simulink/direct/slx_direct.py).

Runs offline (no MATLAB, no API key) on example models of the SLX_TO_Python project, in
fixtures/:

    python -m pytest tests/test_slx_direct.py

    tst5   subsystems, MATLAB Function, 3rd-order Transfer Fcn, Saturation -> converted
    tst3   aircraft 6DOF built from Aerospace Blockset library blocks      -> library blocks
    tst4   Stateflow temperature controller                                  -> Stateflow
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_core import sandbox  # noqa: E402
from backend_core.simulink import intake  # noqa: E402
from backend_core.simulink.c2py import convert  # noqa: E402
from backend_core.simulink.direct import slx_direct  # noqa: E402

EXAMPLES = ROOT / "tests" / "fixtures"


def _slx(name: str) -> bytes:
    return (EXAMPLES / ("%s.slx" % name)).read_bytes()


@pytest.fixture(scope="module")
def tst5(tmp_path_factory):
    return slx_direct.convert_slx("tst5.slx", _slx("tst5"), tmp_path_factory.mktemp("runs"), log=lambda m: None)


def test_a_basic_model_becomes_a_draft_checked_against_its_block_model(tst5):
    assert tst5.ok, tst5.reason
    md = tst5.draft["metadata"]
    assert md["states"] == ["x1", "x2", "x3", "x4", "x5"]
    assert md["inputs"] == ["u1: root Inport In1", "u2: root Inport In2", "u3: root Inport In3"]
    info = md["simulink"]
    assert info["compared_with_simulink"] is False and info["verified_against"] == slx_direct.VERIFIED_AGAINST
    assert info["dynamics_vs_python_model"]["points"]["passed"]
    assert info["dynamics_vs_python_model"]["simulation"]["passed"]
    assert info["simulation"]["initial_state"] == [1.0, 0.0, 0.0, 0.0, 0.0]   # Integrator IC of the model
    assert all(s.ok for s in tst5.steps)
    assert "no MATLAB" in tst5.message


def test_the_draft_is_plain_dynamics_code_for_the_sandbox(tst5):
    code = tst5.draft["python_code"]
    fn = sandbox.compile_dynamics(code)
    assert sandbox.infer_dims(code)[:2] == (5, 3)
    # dx1 = 0.1*(0.2 x3 + 0.1 x4 + 0.5 x5); dx3 = -0.2 x3 - 0.5 x4 - 0.4 x5 + 0.1*(u1 + x1 + u3)
    x, u = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.5, 7.0, -1.0])
    dx = fn(0.0, x, u)
    assert dx[0] == pytest.approx(0.1 * (0.2 * 3 + 0.1 * 4 + 0.5 * 5))
    assert dx[1] == pytest.approx(1.0)
    assert dx[2] == pytest.approx(-0.2 * 3 - 0.5 * 4 - 0.4 * 5 + 0.1 * (0.5 + 1.0 - 1.0))


def test_the_draft_runs_in_the_simulation_of_converted_models(tst5):
    sim = tst5.draft["metadata"]["simulink"]["simulation"]
    r = convert.simulate(tst5.draft["python_code"], tableau=tst5.tableau, fixed_step=sim["fixed_step"],
                         x0=sim["initial_state"], n_inputs=sim["n_inputs"], t_final=10.0)
    assert r.success and not r.diverged and r.x.shape[1] == 5


def test_library_blocks_are_named_in_the_reason(tmp_path):
    r = slx_direct.convert_slx("tst3.slx", _slx("tst3"), tmp_path, log=lambda m: None)
    assert not r.ok and r.draft is None
    assert "not stored in the .slx file" in r.reason and "6DOF (Euler Angles)" in r.reason
    assert not r.steps[-1].ok


def test_a_stateflow_chart_is_the_reason(tmp_path):
    r = slx_direct.convert_slx("tst4.slx", _slx("tst4"), tmp_path, log=lambda m: None)
    assert not r.ok
    assert "Stateflow" in r.reason and "Temperature Control Chart" in r.reason


def test_mdl_and_broken_files_fail_with_a_reason_and_never_raise(tmp_path):
    r = slx_direct.convert_slx("plant.mdl", b"Model { BlockType Gain }", tmp_path, log=lambda m: None)
    assert not r.ok and ".slx" in r.reason
    r = slx_direct.convert_slx("broken.slx", b"not a zip", tmp_path, log=lambda m: None)
    assert not r.ok and r.reason


def test_intake_keeps_the_model_file_and_the_reason_reaches_the_agent():
    up = intake.classify("tst3.slx", _slx("tst3"))
    assert up.kind == intake.MODEL and up.model_data == _slx("tst3")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("models/tst3.slx", _slx("tst3"))
    zipped = intake.classify("tst3.zip", buf.getvalue())
    assert zipped.kind == intake.MODEL and zipped.model_data == _slx("tst3")
    up.model.direct_error = "The model uses library blocks: 6DOF (Euler Angles)."
    ctx = intake.agent_context(up.model, intake.CodeStage())
    assert "directly from the model file" in ctx and "6DOF (Euler Angles)" in ctx
    assert "did not work (method 1)" in intake.just_uploaded_note(up.model)


# ---- problems of the model itself: the user changes the model, the C-code routes do not help
def _tst5_changed(change) -> bytes:
    import io
    import zipfile
    source = zipfile.ZipFile(io.BytesIO(_slx("tst5")))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item)
            if item.filename == "simulink/systems/system_root.xml":
                data = change(data.decode("utf-8")).encode("utf-8")
            target.writestr(item, data)
    return out.getvalue()


def _without(block_type):
    import re
    return lambda text: re.sub(r'<Block BlockType="%s".*?</Block>' % block_type, "", text, flags=re.S)


def test_a_model_without_root_outport_converts_whole_and_says_it_has_no_output(tst5, tmp_path):
    r = slx_direct.convert_slx("tst5.slx", _tst5_changed(_without("Outport")), tmp_path, log=lambda m: None)
    assert r.ok and r.no_input == slx_direct.NO_OUTPUT and r.draft["metadata"]["outputs"] == []
    assert r.draft["metadata"]["states"] == tst5.draft["metadata"]["states"]        # every state still


def test_a_model_fix_is_never_sent_to_the_c_route():
    info = intake.inspect_model("tst5.slx", _slx("tst5"))
    info.direct_error = info.model_fix = slx_direct.NO_OUTPORT
    context = intake.agent_context(info, None)
    assert "Outport" in context and "Do NOT offer the C-code routes" in context and "c2py_export" not in context
    assert "Do not mention the C-code routes" in intake.just_uploaded_note(info)


def test_a_model_without_root_inport_converts_and_says_how_to_add_inputs(tmp_path):
    r = slx_direct.convert_slx("tst5.slx", _tst5_changed(_without("Inport")), tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert r.draft["metadata"]["simulink"]["simulation"]["n_inputs"] == 0
    assert slx_direct.NO_INPORT in r.message


def test_inputs_from_matlab_data_are_replaced_by_inports(tmp_path):
    change = lambda text: text.replace('<Block BlockType="Inport" Name="In1"',              # noqa: E731
                                       '<Block BlockType="FromWorkspace" Name="simin"', 1)
    r = slx_direct.convert_slx("tst5.slx", _tst5_changed(change), tmp_path, log=lambda m: None)
    assert not r.ok and r.model_fix and "From Workspace 'simin'" in r.model_fix and "Inport" in r.model_fix

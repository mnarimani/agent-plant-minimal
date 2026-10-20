"""One chat for every upload: what was uploaded, what is missing, and the C-code-only route.

Runs offline (no MATLAB, no API key):

    python -m pytest tests/test_simulink_intake.py

The C code is the one in fixtures/maglev_c2py.zip (see test_simulink_convert.py).
"""

from __future__ import annotations

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend_core import agent as agent_mod  # noqa: E402
from backend_core import logging_utils  # noqa: E402
from backend_core.simulink import intake, llm as simulink_llm  # noqa: E402
from backend_core.simulink.c2py import convert  # noqa: E402
from test_simulink_convert import MAGLEV, FakeResponses  # noqa: E402


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "LOG_DIR", tmp_path / "logs")


@pytest.fixture(scope="module")
def maglev_code():
    with zipfile.ZipFile(MAGLEV) as z:
        return {n.split("/")[-1]: z.read(n) for n in z.namelist() if n.endswith((".c", ".h"))}


def _zip(files) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


SLX_ROOT = """<?xml version="1.0" encoding="utf-8"?>
<System>
  <Block BlockType="Inport" Name="Voltage" SID="1"/>
  <Block BlockType="Integrator" Name="Integrator" SID="2"/>
  <Block BlockType="SubSystem" Name="Plant" SID="3"/>
  <Block BlockType="Outport" Name="Position" SID="4"/>
</System>
"""
SLX_SUB = """<?xml version="1.0" encoding="utf-8"?>
<System>
  <Block BlockType="Inport" Name="In1" SID="5"/>
  <Block BlockType="Reference" Name="PID" SID="6"><P Name="SourceBlock">simulink/Continuous/PID Controller</P></Block>
  <Block BlockType="Outport" Name="Out1" SID="7"/>
</System>
"""
SLX_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<ConfigSet><Object ClassName="Simulink.SolverCC">
  <P Name="SolverType">Variable-step</P><P Name="Solver">ode45</P><P Name="FixedStep">auto</P>
</Object></ConfigSet>
"""


def _slx(root=SLX_ROOT) -> bytes:
    return _zip({"simulink/systems/system_root.xml": root, "simulink/systems/system_3.xml": SLX_SUB,
                 "simulink/configSet0.xml": SLX_CONFIG, "metadata/coreProperties.xml": "<x/>"})


# ---- what was uploaded ------------------------------------------------------------------------
def test_uploads_are_sorted_by_kind(maglev_code):
    assert intake.classify("paper.PDF", b"%PDF").kind == intake.DOCUMENT
    assert intake.classify("notes.md", b"#").kind == intake.DOCUMENT
    assert intake.classify("maglev_c2py.zip", MAGLEV.read_bytes()).kind == intake.PACKAGE
    assert intake.classify("maglev.c", maglev_code["maglev.c"]).kind == intake.CODE
    assert intake.classify("model.slx", _slx()).kind == intake.MODEL
    other = intake.classify("model.docx", b"x")
    assert other.kind == intake.OTHER and ".docx" in other.note
    assert intake.classify("broken.zip", b"not a zip").kind == intake.OTHER


def test_a_zip_of_the_code_folder_is_code_and_ert_main_is_ignored(maglev_code):
    files = {"maglev_ert_rtw/" + n: d for n, d in maglev_code.items()}
    files["maglev_ert_rtw/ert_main.c"] = b"int main(void) { return 0; }"
    files["maglev_ert_rtw/html/report.html"] = b"<html/>"
    files["__MACOSX/maglev_ert_rtw/._maglev.c"] = b"junk"
    up = intake.classify("code.zip", _zip(files))
    assert up.kind == intake.CODE
    assert set(up.code_files) == set(maglev_code) | {"ert_main.c"}
    chk = intake.check_code(up.code_files)
    assert chk.ok and chk.model == "maglev" and chk.ignored == ["ert_main.c"]


def test_a_zip_of_the_model_folder_is_the_model():
    up = intake.classify("project.zip", _zip({"project/maglev.slx": _slx(), "project/setup.m": "a=1;"}))
    assert up.kind == intake.MODEL and up.model.name == "maglev"


# ---- the Simulink model file ------------------------------------------------------------------
def test_the_slx_file_is_read_for_advice():
    info = intake.inspect_model("maglev.slx", _slx())
    assert info.read and info.name == "maglev"
    assert (info.n_inports, info.n_outports) == (1, 1)            # root level only, not the subsystem's
    assert info.solver == "ode45" and info.solver_type == "Variable-step"
    assert "Integrator" in info.continuous and "PID Controller" in info.continuous
    assert any("variable-step" in n for n in info.notes)


def test_a_model_without_outport_or_continuous_blocks_is_warned_about():
    root = '<System><Block BlockType="Inport" Name="u" SID="1"/><Block BlockType="Gain" Name="g" SID="2"/></System>'
    zipped = _zip({"simulink/systems/system_root.xml": root, "simulink/configSet0.xml": SLX_CONFIG})
    info = intake.inspect_model("m.slx", zipped)
    assert info.n_outports == 0
    assert any("no Outport" in n for n in info.notes)
    assert any("no continuous-time block" in n for n in info.notes)


def test_an_unreadable_model_file_is_still_a_model():
    up = intake.classify("weird.slx", b"not a zip")
    assert up.kind == intake.MODEL and not up.model.read
    assert "could not read" in up.model.notes[0]


def test_the_mdl_text_format():
    text = 'Model {\n Name "m"\n Solver "ode4"\n SolverType "Fixed-step"\n System {\n' \
           '  Block {\n BlockType Integrator\n Name "I"\n }\n  Block {\n BlockType Outport\n Name "y"\n }\n }\n}\n'
    info = intake.inspect_model("m.mdl", text.encode())
    assert info.read and info.solver == "ode4" and info.continuous == ["Integrator"]


# ---- generated C code, collected over several messages ----------------------------------------
THREE = ("maglev.c", "maglev.h", "maglev_data.c")


def test_three_files_are_enough_and_nothing_else_is_asked_for(maglev_code):
    chk = intake.check_code({n: maglev_code[n] for n in THREE})
    assert chk.ok and chk.model == "maglev", (chk.missing, chk.problems)


def test_missing_files_are_named(maglev_code):
    chk = intake.check_code({n: maglev_code[n] for n in ("maglev.c", "maglev_data.c")})
    assert not chk.ok and chk.model == "maglev" and chk.missing == ["maglev.h"]


def test_three_files_translate_like_all_of_them(maglev_code, tmp_path):
    """the translator writes the typedefs of maglev_types.h and needs no rtwtypes.h: the same model"""
    import importlib.util
    import numpy as np
    from backend_core.simulink.c2py import c2py

    def simulate(files, tag):
        folder = tmp_path / tag
        folder.mkdir()
        for n, d in files.items():
            (folder / n).write_bytes(d)
        _, src, _ = c2py.transpile([str(folder)], "maglev")
        sources[tag] = [line for line in src.splitlines() if not line.startswith("# Generated by")]
        (folder / "m.py").write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("maglev_" + tag, folder / "m.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        r = module.Model().simulate(2.0, lambda t: [np.sin(t) + (t >= 1)])
        return np.asarray(r["y"], float), np.asarray(r["x"], float)

    sources = {}
    y_all, x_all = simulate(maglev_code, "all")
    y3, x3 = simulate({n: maglev_code[n] for n in THREE}, "three")
    assert sources["all"] == sources["three"]          # the same Python model, line for line
    assert np.array_equal(y_all, y3) and np.array_equal(x_all, x3)


def test_missing_parameter_file_is_caught(maglev_code):
    # without maglev_data.c the parameters would silently be zero
    chk = intake.check_code({n: d for n, d in maglev_code.items() if n != "maglev_data.c"})
    assert chk.missing == ["maglev_data.c"]


def test_headers_alone_name_the_model(maglev_code):
    chk = intake.check_code({"maglev.h": maglev_code["maglev.h"]})
    assert chk.model == "maglev" and "maglev.c" in chk.missing


def test_code_without_continuous_states_is_a_problem(maglev_code):
    files = dict(maglev_code)
    files["maglev.c"] = files["maglev.c"].replace(b"maglev_derivatives", b"maglev_deriv_gone")
    chk = intake.check_code(files)
    assert not chk.ok and any("no continuous states" in p for p in chk.problems)


def test_code_of_two_models_is_a_problem(maglev_code):
    files = dict(maglev_code, **{"other.c": b"void other_initialize(void) {}\nvoid other_step(void) {}\n"})
    chk = intake.check_code(files)
    assert not chk.ok and "more than one model" in chk.problems[0]


def test_the_stage_collects_files_over_messages(maglev_code):
    stage = intake.CodeStage()
    first = {n: d for n, d in maglev_code.items() if n.endswith(".c")}
    assert stage.add(first) is False
    assert not stage.check().ok
    assert stage.add({n: d for n, d in maglev_code.items() if n.endswith(".h")}) is False
    assert stage.check().ok


def test_the_stage_starts_over_for_another_model(maglev_code):
    stage = intake.CodeStage()
    stage.add(maglev_code)
    restarted = stage.add({"other.c": b"void other_initialize(void) {}\n"})
    assert restarted and list(stage.files) == ["other.c"]


# ---- what the chat agent is told --------------------------------------------------------------
def test_agent_context_names_what_is_missing(maglev_code):
    stage = intake.CodeStage()
    stage.add({n: d for n, d in maglev_code.items() if n != "maglev.h"})
    ctx = intake.agent_context(intake.inspect_model("maglev.slx", _slx()), stage)
    assert "maglev.slx" in ctx and "ask for them: maglev.h" in ctx
    assert "c2py_export.m" in ctx and "Embedded Coder" in ctx
    assert intake.agent_context(None, intake.CodeStage()) == ""


class _PromptClient:
    """Records the prompts of the chat agent; answers with a fixed reply."""

    def __init__(self, reply: dict) -> None:
        self.responses, self.reply, self.calls = self, reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(output_text=json.dumps(self.reply), output=[])


def test_the_agent_prompt_carries_the_upload_context_and_keeps_the_users_language():
    client = _PromptClient({"status": "continue", "reply": "مدل سیمولینک رسید. کدام روش را ترجیح می‌دهید؟"})
    agent = agent_mod.PlantModelAgent(client, model="fake-model")
    text, final = agent.step([], "[Attached: maglev.slx]", upload_context="The user uploaded the Simulink model X")
    assert final is None and text.startswith("مدل سیمولینک")              # not turned into "????"
    prompt = client.calls[0]["input"]
    assert "Simulink conversion status" in prompt and "The user uploaded the Simulink model X" in prompt
    assert "Simulink models and generated C code" in client.calls[0]["instructions"]
    agent.step([], "hello")
    assert "Simulink conversion status (set by the application; may be empty):\n(none)" in client.calls[1]["input"]


def test_the_agent_can_attach_the_export_script():
    client = _PromptClient({"status": "continue", "reply": "steps ...", "attach": ["c2py_export.m"]})
    agent = agent_mod.PlantModelAgent(client, model="fake-model")
    agent.step([], "I am not familiar with code generation", upload_context="pending")
    assert agent.last_attachments == ["c2py_export.m"]
    client.reply = {"status": "continue", "reply": "ok"}
    agent.step([], "thanks")
    assert agent.last_attachments == []


def test_agent_context_holds_the_steps_of_both_routes_with_the_model_name():
    ctx = intake.agent_context(intake.inspect_model("maglev.slx", _slx()), intake.CodeStage())
    assert "c2py_export('maglev')" in ctx and "maglev_c2py.zip" in ctx      # route A
    assert "maglev_ert_rtw" in ctx and "ert.tlc" in ctx                      # route B
    assert "variable-step" in ctx                                            # what was found in the file


# ---- the C-code-only route --------------------------------------------------------------------
def test_complete_c_code_becomes_a_draft_not_compared_with_simulink(maglev_code, tmp_path):
    fake = types.SimpleNamespace(responses=FakeResponses())
    result = convert.convert_c_code({n: maglev_code[n] for n in THREE}, tmp_path, simulink_llm.make_llm(fake, model="gpt-5.4-mini"),
                                    log=lambda m: None)
    assert result.ok, result.message
    sim = result.draft["metadata"]["simulink"]
    assert sim["compared_with_simulink"] is False and sim["source"] == convert.SOURCE_CODE
    assert sim["python_vs_simulink"] is None and sim["validated"]
    assert "not checked" in result.message and "PASS" not in result.message.split("\n")[2]
    assert any("not checked" in s.label for s in result.steps)
    assert sim["simulation"]["solver"] == "ode8"                    # the solver the code was generated with
    assert result.tableau is not None


def test_incomplete_c_code_is_not_converted(maglev_code, tmp_path):
    files = {n: d for n, d in maglev_code.items() if n != "maglev.h"}
    result = convert.convert_c_code(files, tmp_path, llm=None, log=lambda m: None)
    assert not result.ok and "maglev.h" in result.message
    assert not (tmp_path / "maglev").exists()


def test_the_package_route_still_records_its_source(tmp_path):
    fake = types.SimpleNamespace(responses=FakeResponses())
    result = convert.convert_package(MAGLEV, tmp_path, simulink_llm.make_llm(fake, model="gpt-5.4-mini"),
                                     log=lambda m: None)
    sim = result.draft["metadata"]["simulink"]
    assert result.ok and sim["compared_with_simulink"] is True and sim["source"] == convert.SOURCE_PACKAGE

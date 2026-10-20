"""MATLAB workspace variables in the direct .slx route (slx2py/model_converter/workspace.py).

Block parameters often name MATLAB variables. Their values come, as in Simulink, from the user's
scripts (an attached .m file or ``name = value`` lines in the chat) and from the model's own
callbacks (InitFcn ...). The interpreter evaluates MATLAB without MATLAB; what it cannot evaluate
is reported, never guessed.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from backend_core.simulink import intake
from backend_core.simulink.direct import slx_direct
from backend_core.simulink.direct.slx2py.model_converter import workspace

FIXTURES = Path(__file__).parent / "fixtures"


def run(script: str) -> workspace.Workspace:
    ws = workspace.Workspace()
    ws.run(script, "test")
    return ws


def value(ws: workspace.Workspace, name: str) -> np.ndarray:
    return ws.values[name]


# ---------------------------------------------------------------------------------------------
# the interpreter: MATLAB's syntax and semantics
# ---------------------------------------------------------------------------------------------
def test_matrices_follow_matlab_spacing_rules():
    ws = run("a = [1 -2]; b = [1 - 2]; c = [1 , 2 ; 3 4]; d = [1 2]'; e = [c; 5 6]; f = [];\n"
             "g = [1 2\n3 4]; h = [a' a']; k = [1 -2]*2; m = 2 -1; n = [a (1)]")
    assert value(ws, "a").tolist() == [[1, -2]]
    assert value(ws, "b").tolist() == [[-1]]
    assert value(ws, "c").tolist() == [[1, 2], [3, 4]]
    assert value(ws, "d").tolist() == [[1], [2]]
    assert value(ws, "e").shape == (3, 2)
    assert value(ws, "f").shape == (0, 0)
    assert value(ws, "g").tolist() == [[1, 2], [3, 4]]
    assert value(ws, "h").tolist() == [[1, 1], [-2, -2]]
    assert value(ws, "k").tolist() == [[2, -4]]
    assert value(ws, "m").tolist() == [[1]]
    assert value(ws, "n").tolist() == [[1, -2, 1]]


def test_operators_precedence_and_matrix_algebra():
    ws = run("a = -2^2; b = 2^-1; c = 2^3^2; d = [1 2; 3 4]*[1; 1]; e = [1 2].*[3 4]; f = [2 4]./2;\n"
             "g = [1 2; 3 4]\\[5; 6]; h = [1 2; 3 4]^2; k = 1:3 + 1; m = ~0 + (2 > 1) * 3; n = [1 2; 3 4]';\n"
             "p = (1:3)' + [10 20]; q = 5:-2:0; r = 1e3; s = 3i; t = sqrt(-4); u = 1 && 0 || 1")
    assert value(ws, "a")[0, 0] == -4 and value(ws, "b")[0, 0] == 0.5 and value(ws, "c")[0, 0] == 64
    assert value(ws, "d").tolist() == [[3], [7]]
    assert value(ws, "e").tolist() == [[3, 8]] and value(ws, "f").tolist() == [[1, 2]]
    assert np.allclose(value(ws, "g"), np.linalg.solve([[1, 2], [3, 4]], [[5], [6]]))
    assert value(ws, "h").tolist() == [[7, 10], [15, 22]]
    assert value(ws, "k").tolist() == [[1, 2, 3, 4]]
    assert value(ws, "m")[0, 0] == 4
    assert value(ws, "n").tolist() == [[1, 3], [2, 4]]
    assert value(ws, "p").shape == (3, 2) and value(ws, "p")[2, 1] == 23
    assert value(ws, "q").tolist() == [[5, 3, 1]]
    assert value(ws, "r")[0, 0] == 1000 and value(ws, "s")[0, 0] == 3j and value(ws, "t")[0, 0] == 2j
    assert value(ws, "u")[0, 0] == 1


def test_indexing_end_structs_and_indexed_assignment():
    ws = run("x = 10:10:50; a = x(2); b = x(end); c = x([1 end]); d = x(x > 25);\n"
             "M = magic_free = 0;\n"            # a statement it cannot read: a note, not a crash
             "M = [1 2 3; 4 5 6]; e = M(2, :); f = M(:, end); g = M(end); h = M(:)';\n"
             "p.m = 2; p.g.v = 9.81; w = p.m * p.g.v; x(7) = 1; Z = zeros(2); Z(2, 3) = 5;\n"
             "[nr, nc] = size(M); n = numel(M);")
    assert value(ws, "a")[0, 0] == 20 and value(ws, "b")[0, 0] == 50
    assert value(ws, "c").tolist() == [[10, 50]] and value(ws, "d").tolist() == [[30, 40, 50]]
    assert value(ws, "e").tolist() == [[4, 5, 6]] and value(ws, "f").tolist() == [[3], [6]]
    assert value(ws, "g")[0, 0] == 6 and value(ws, "h").tolist() == [[1, 4, 2, 5, 3, 6]]
    assert value(ws, "p.m")[0, 0] == 2 and value(ws, "w")[0, 0] == pytest.approx(19.62)
    assert value(ws, "x").tolist() == [[10, 20, 30, 40, 50, 0, 1]]
    assert value(ws, "Z").tolist() == [[0, 0, 0], [0, 0, 5]]
    assert value(ws, "nr")[0, 0] == 2 and value(ws, "nc")[0, 0] == 3 and value(ws, "n")[0, 0] == 6


def test_comments_continuations_commands_and_control_flow():
    ws = run("%{\nblock comment\nb = 99;\n%}\nclear all; clc; close all; format long\n"
             "a = [1, 2, ... comment\n 3]; % the end\n"
             "if a(1) > 0, s = 1; elseif a(1) < 0, s = -1; else, s = 0; end\n"
             "total = 0;\nfor k = 1:4\n  total = total + k;\nend\n"
             "n = 0; while n < 3, n = n + 1; end\n"
             "disp(a); fprintf('%d\\n', 3); name = 'plant'; ")
    assert "b" not in ws.values
    assert value(ws, "a").tolist() == [[1, 2, 3]]
    assert value(ws, "s")[0, 0] == 1 and value(ws, "total")[0, 0] == 10 and value(ws, "n")[0, 0] == 3
    assert ws.values["name"] == "plant"
    assert not ws.unresolved


def test_math_and_matrix_functions():
    ws = run("a = sqrt(16) + abs(-2) + round(2.5) + round(-2.5) + fix(-2.7) + mod(-1, 3) + rem(-1, 3);\n"
             "b = eye(2) + ones(2) + zeros(2, 2); c = diag([1 2]); d = inv([2 0; 0 4]); e = det([1 2; 3 4]);\n"
             "f = max([1 5 3]); [g, i] = min([4 2 8]); h = sum([1 2; 3 4]); k = linspace(0, 1, 3);\n"
             "m = repmat([1 2], 2, 1); n = norm([3 4]); p = deg2rad(180); q = atan2(1, 1); r = expm(zeros(2));\n"
             "s = pi; t = Inf; u = exp(1); v = kron([1 2], [1; 1]); w = trace([1 2; 3 4]); x = reshape(1:4, 2, 2)")
    assert value(ws, "a")[0, 0] == 4 + 2 + 3 - 3 - 2 + 2 - 1
    assert value(ws, "b").tolist() == [[2, 1], [1, 2]]
    assert value(ws, "c").tolist() == [[1, 0], [0, 2]] and value(ws, "d").tolist() == [[0.5, 0], [0, 0.25]]
    assert value(ws, "e")[0, 0] == pytest.approx(-2)
    assert value(ws, "f")[0, 0] == 5 and value(ws, "g")[0, 0] == 2 and value(ws, "i")[0, 0] == 2
    assert value(ws, "h").tolist() == [[4, 6]] and value(ws, "k").tolist() == [[0, 0.5, 1]]
    assert value(ws, "m").tolist() == [[1, 2], [1, 2]] and value(ws, "n")[0, 0] == 5
    assert value(ws, "p")[0, 0] == pytest.approx(np.pi) and value(ws, "q")[0, 0] == pytest.approx(np.pi / 4)
    assert value(ws, "r").tolist() == [[1, 0], [0, 1]] and np.isinf(value(ws, "t")[0, 0])
    assert value(ws, "v").tolist() == [[1, 2], [1, 2]] and value(ws, "w")[0, 0] == 5
    assert value(ws, "x").tolist() == [[1, 3], [2, 4]]


def test_control_design_functions_solve_their_equations():
    """lqr / care / are / lyap / acker: checked by the equations they solve (as MATLAB defines them)"""
    ws = run("A = [0 1; -2 3]; B = [0; 1]; C = [1 1]; Q = eye(2); R = 0.25;\n"
             "[K, S, P] = lqr(A, B, Q, R); X = are(A', C'*C, B*B'); L = lyap(A - B*K, Q);\n"
             "Ka = acker(A, B, [-1 -2]);")
    A, B, C = np.array([[0, 1], [-2, 3.0]]), np.array([[0], [1.0]]), np.array([[1, 1.0]])
    K, S, X, L, Ka = (value(ws, n) for n in ("K", "S", "X", "L", "Ka"))
    assert np.allclose(A.T @ S + S @ A - S @ B @ B.T @ S / 0.25 + np.eye(2), 0, atol=1e-9)
    assert np.allclose(K, B.T @ S / 0.25)
    assert np.all(np.real(value(ws, "P")) < 0)
    assert np.allclose(A @ X + X @ A.T - X @ C.T @ C @ X + B @ B.T, 0, atol=1e-9)    # are(A', ...) form
    Acl = A - B @ K
    assert np.allclose(Acl @ L + L @ Acl.T + np.eye(2), 0, atol=1e-9)
    assert np.allclose(sorted(np.linalg.eigvals(A - B @ Ka).real), [-2, -1])


def test_what_cannot_be_evaluated_is_reported_with_everything_computed_from_it():
    ws = run("a = 2;\nb = rand(3);\nc = b * a;\nd = undefined_thing + 1;\nload params.mat\n"
             "switch a\n case 2\n  e = 1;\nend\nf = a + 1;")
    assert set(ws.unresolved) == {"b", "c", "d", "e"}
    assert "rand" in ws.unresolved["b"] and "uses b" in ws.unresolved["c"]
    assert "undefined_thing" in ws.unresolved["d"]
    assert value(ws, "f")[0, 0] == 3
    assert any("params.mat" in n for n in ws.notes)


def test_a_script_that_cannot_be_read_is_a_note():
    ws = run("a = 'unterminated")
    assert not ws.values and ws.notes


def test_block_parameters_use_the_variables():
    values = {"k": 2.0, "A": np.array([[0, 1], [-2, -3.0]])}
    assert workspace.evaluate("[k 1]*k", values).tolist() == [[4, 2]]
    assert workspace.evaluate("A(2, :)", values).tolist() == [[-2, -3]]
    assert workspace.evaluate("eye(2)*k", values).tolist() == [[2, 0], [0, 2]]
    with pytest.raises(workspace.UnknownName) as error:
        workspace.evaluate("m*g", values)
    assert error.value.name == "m"


def _chat(text):
    ws = run(workspace.assignments_in_text(text))
    return {k: (v.ravel().tolist() if v.size > 1 else float(v.ravel()[0])) for k, v in ws.values.items()}, ws


@pytest.mark.parametrize("text, values", [
    ("m = 1200\nk = 50", {"m": 1200, "k": 50}),
    ("m = 1200, k = 50", {"m": 1200, "k": 50}),
    ("m=1200 k=50", {"m": 1200, "k": 50}),
    ("m = 1200 و k = 50", {"m": 1200, "k": 50}),                       # inside a Persian sentence
    ("مقدار m = 1200 و k = 50 هست", {"m": 1200, "k": 50}),
    ("جرم m = 1200 است\nسختی k = 50", {"m": 1200, "k": 50}),
    ("m = ۱۲۰۰\nk = ۵۰", {"m": 1200, "k": 50}),                       # Persian digits
    ("m = ۱۲٫۵", {"m": 12.5}),                                           # Persian decimal separator
    ("m=1200، k=50", {"m": 1200, "k": 50}),                              # Persian comma
    ("m = 1200 kg\nk = 50 N/m", {"m": 1200, "k": 50}),                  # units are not values
    ("g = 9.81 m/s^2", {"g": 9.81}),
    ("m: 1200\nk: 50", {"m": 1200, "k": 50}),
    ("Sure:\nm = 1200 % kg\n  A = [0 1;\n -2 -3];\n`k = 4;`\nThanks!", {"m": 1200, "A": [0, 1, -2, -3], "k": 4}),
    ("num = [2 1 5], den = [10 2 5 0]", {"num": [2, 1, 5], "den": [10, 2, 5, 0]}),
    ("p.x0 = 1", {"p.x0": 1}),
    ("k = 2*pi", {"k": 2 * np.pi}),
    ("x == 3 is a comparison", {}),
    ("سلام، مدل رو ببین", {}),
])
def test_values_are_found_in_a_chat_message_however_they_are_written(text, values):
    found, ws = _chat(text)
    assert found.keys() == values.keys(), (text, found)
    for name, value in values.items():
        assert np.allclose(found[name], value)
    assert not ws.unresolved


def test_text_that_is_not_matlab_is_a_note_never_a_crash():
    ws = run("m = 1200 و k = 50")                 # a Persian letter in a MATLAB script
    assert ws.notes and not ws.values
    ws = run("جرم = 1200")
    assert ws.notes or ws.unresolved


# ---------------------------------------------------------------------------------------------
# the direct route: a model whose parameters are workspace variables
# ---------------------------------------------------------------------------------------------
def _slx(name: str) -> bytes:
    return (FIXTURES / f"{name}.slx").read_bytes()


def _with_variables(init_fcn: str | None = None) -> bytes:
    """tst5 with its Gain, Transfer Fcn and Integrator parameters written as MATLAB variables"""
    replacements = {'<P Name="Gain">0.1</P>': '<P Name="Gain">k</P>',
                    '<P Name="Numerator">[2 1 5]</P>': '<P Name="Numerator">num</P>',
                    '<P Name="Denominator">[10 2 5 4]</P>': '<P Name="Denominator">[den(1:3) a]</P>',
                    '<P Name="InitialCondition">1</P>': '<P Name="InitialCondition">p.x0</P>'}
    source = zipfile.ZipFile(io.BytesIO(_slx("tst5")))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item)
            if item.filename == "simulink/systems/system_root.xml":
                text = data.decode("utf-8")
                for old, new in replacements.items():
                    assert old in text
                    text = text.replace(old, new)
                data = text.encode("utf-8")
            if item.filename == "simulink/blockdiagram.xml" and init_fcn:
                text = data.decode("utf-8")
                text = text.replace("<Model>", '<Model>\n    <P Name="InitFcn">%s</P>' % init_fcn, 1)
                data = text.encode("utf-8")
            target.writestr(item, data)
    return out.getvalue()


SCRIPT = "k = 0.1;\nnum = [2 1 5];\nden = [10 2 5 0]; a = 4;\np.x0 = 1;"


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return slx_direct.convert_slx("tst5.slx", _slx("tst5"), tmp_path_factory.mktemp("runs"), log=lambda m: None)


def test_without_values_the_variables_are_named_and_nothing_else(tmp_path):
    r = slx_direct.convert_slx("tst5.slx", _with_variables(), tmp_path, log=lambda m: None)
    assert not r.ok
    assert set(r.missing_variables) == {"k", "num", "den", "a", "p.x0"} and r.only_variables
    listed = r.reason.split("file): ", 1)[1]
    for name in ("k", "num", "den", "a", "p.x0"):
        assert re.search(r"(?<![\w.])%s(?![\w])" % re.escape(name), listed), name
    assert "InitFcn" in r.reason


def test_the_users_script_gives_the_same_model_as_the_numbers(tmp_path, plain):
    r = slx_direct.convert_slx("tst5.slx", _with_variables(), tmp_path, log=lambda m: None,
                               scripts=[("the chat", SCRIPT)])
    assert r.ok, r.reason
    assert r.draft["python_code"] == plain.draft["python_code"]
    assert r.draft["metadata"]["parameters"] == {"k": 0.1, "a": 4.0, "p.x0": 1.0}
    assert "MATLAB variables used for the parameters (from the chat)" in r.message
    assert any("MATLAB workspace variables in the parameters" in s.label for s in r.steps)


def test_the_models_own_init_fcn_is_enough(tmp_path, plain):
    r = slx_direct.convert_slx("tst5.slx", _with_variables(SCRIPT.replace("\n", " ")), tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert r.draft["python_code"] == plain.draft["python_code"]
    assert "the model's InitFcn" in r.message


def test_the_init_fcn_runs_after_the_users_script_as_in_simulink(tmp_path):
    r = slx_direct.convert_slx("tst5.slx", _with_variables("k = 2 * base;"), tmp_path, log=lambda m: None,
                               scripts=[("the chat", "base = 0.05; num = [2 1 5]; den = [10 2 5 0]; a = 4; p.x0 = 1;")])
    assert r.ok, r.reason
    assert r.draft["metadata"]["parameters"]["k"] == pytest.approx(0.1)


def test_values_sent_over_two_messages_complete_the_model(tmp_path, plain):
    """the user sends some values, the app asks for the rest, the user sends them: then it converts"""
    first = workspace.assignments_in_text("مقادیر: k = ۰٫۱ و num = [2 1 5] و a = 4 هست")
    r = slx_direct.convert_slx("tst5.slx", _with_variables(), tmp_path / "1", log=lambda m: None,
                               scripts=[("the chat", first)])
    assert not r.ok and set(r.missing_variables) == {"den", "p.x0"} and r.only_variables
    second = workspace.assignments_in_text("den = [10 2 5 0]\np.x0 = 1")
    r = slx_direct.convert_slx("tst5.slx", _with_variables(), tmp_path / "2", log=lambda m: None,
                               scripts=[("the chat", first), ("the chat", second)])
    assert r.ok, r.reason
    assert r.draft["python_code"] == plain.draft["python_code"]


def test_a_variable_the_script_cannot_compute_is_explained(tmp_path):
    r = slx_direct.convert_slx("tst5.slx", _with_variables("k = rand(1);"), tmp_path, log=lambda m: None,
                               scripts=[("the chat", "num = [2 1 5]; den = [10 2 5 0]; a = 4; p.x0 = 1;")])
    assert not r.ok and r.missing_variables == ["k"]
    assert re.search(r"k has no value: .*rand", r.reason)


def test_the_users_kalman_model_reads_its_init_fcn():
    """Kalman_FSFB_Sim (matlab_block_tests/my_models) computes its gains in InitFcn with are / lqr:
    the values must solve the equations MATLAB solves"""
    root = Path(__file__).resolve().parents[1] / "matlab_block_tests" / "my_models"
    if not (root / "Kalman_FSFB_Sim.slx").is_file():
        pytest.skip("the user's models are not here")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(root / "Kalman_FSFB_Sim.slx") as z:
            z.extractall(tmp)
        ws = workspace.model_workspace(tmp)
    assert not ws.unresolved and not ws.notes
    A, B, B1, C, D, P, L, K = (value(ws, n) for n in ("A", "B", "B1", "C", "D", "P", "L", "K"))
    # P = are(A', C'*C./D.^2, B*B'):  A P + P A' - P C'C P / D^2 + B B' = 0
    assert np.allclose(A @ P + P @ A.T - P @ C.T @ C @ P / D[0, 0] ** 2 + B @ B.T, 0, atol=1e-9)
    assert np.allclose(L, P @ C.T / (D @ D.T))
    # K = lqr(A, B1, eye(2), 0.25) is optimal: with S of the closed loop
    # (Acl' S + S Acl + Q + K' R K = 0) it is K = R^-1 B1' S, and the closed loop is stable
    from scipy.linalg import solve_continuous_lyapunov
    R, Acl = 0.25, A - B1 @ K
    S = solve_continuous_lyapunov(Acl.T, -(np.eye(2) + K.T @ K * R))
    assert np.allclose(K, B1.T @ S / R, rtol=1e-8)
    assert np.all(np.linalg.eigvals(Acl).real < 0)


# ---------------------------------------------------------------------------------------------
# intake: .m scripts are accepted, also inside a zip with the model
# ---------------------------------------------------------------------------------------------
def test_m_files_are_accepted_as_scripts():
    assert "m" in intake.ACCEPTED_TYPES
    u = intake.classify("init_params.m", b"m = 1200;\n")
    assert u.kind == intake.SCRIPT and u.scripts == [("the attached script init_params.m", "m = 1200;\n")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("model/tst5.slx", _slx("tst5"))
        z.writestr("model/params.m", "k = 0.1;")
    u = intake.classify("bundle.zip", buffer.getvalue())
    assert u.kind == intake.MODEL and u.scripts == [("the attached script params.m", "k = 0.1;")]

"""Upload package (or generated C code) -> draft of the plant-model agent.

:func:`convert_package` runs the C_to_Python pipeline on a ``<model>_c2py.zip``
made by ``c2py_export.m`` (the ``--package`` path of its main.py: no MATLAB
needed) and returns the result as a draft in the format of
``backend_core.agent.PlantModelAgent``:

    {"system_name": ..., "python_code": "<dynamics(t, x, u)>", "metadata": {...}}

:func:`convert_c_code` does the same for generated C code uploaded without a package
(``intake`` collects it): there are no Simulink results then, so the comparison with
Simulink is skipped and the metadata says so (``simulink.compared_with_simulink``).

``metadata`` has the keys of this project's metadata format (states,
state_meanings, inputs, outputs, state_equations, parameters, system_type,
assumptions). What that format has no place for - the validation against
Simulink, which equations were checked, the definitions - is kept, not
dropped, under one extra key ``"simulink"`` (:func:`card_to_metadata`).

:func:`reconcile` keeps that key honest once the user changes the model in the
chat: the same code keeps the validated metadata; changed code is marked as no
longer verified against Simulink.

:func:`simulate` runs a converted model the way Simulink ran it: from the model's
own initial state, with its fixed-step solver (ode8) and step size. The app's
general sandbox (``backend_core.sandbox``: zero initial state, adaptive RK45) is
not suited to these models - an initial state of zero can be exactly the 0/0 the
model avoids with its own initial conditions, and switching / saturating blocks
make the adaptive step size collapse (a vehicle model that ode8 runs in 0.4 s did
not finish 0.5 s of simulated time in 2 minutes with RK45).
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ... import agent as plant_agent
from ... import sandbox
from ...steps import Step
from .. import intake
from . import agent as c2py_agent
from . import c2py, package, validation

NA = c2py_agent.NA

# The settings of the C_to_Python pipeline (CONFIG in its main.py), without the
# ones that only matter when MATLAB runs on this machine.
CONFIG: Dict[str, Any] = dict(
    tf=10,                      # stop time of the reference runs of step 4 [s]
    dt=None,                    # None = step size of the C model
    inputs=["step"],            # test signals of step 4 (step 3 uses the runs of the package)
    amp=2,
    seed=0,
    tol=1e-3,                   # step 3: PASS if RMSE / range(Simulink output) <= tol
    solver="ode8",              # the solver c2py_export.m forces for the code and the Simulink runs
    sweep=False,
    no_interp=False,
    agent_mode="auto",          # rewrite (small models) / annotate (big models)
    agent_rewrite_max_lines=60,
    agent_max_attempts=3,
    agent_naming="model",       # only names of the model itself; the LLM adds meanings
    agent_json=True,
    json_max_terms=30,
    agent_tol=1e-6,             # step 4: PASS if RMSE / range <= agent_tol for every state
    agent_max_steps=2000,
    agent_samples=150,
    agent_window=25,
)

MODIFIED = "Modified after the Simulink validation - no longer verified against Simulink"

SOURCE_PACKAGE = "c2py package (generated C code + Simulink test runs)"
SOURCE_CODE = "generated C code only (no Simulink test runs)"

SIM_MAX_STEPS = 20000       # a simulation is cut to this many fixed steps (the step size is the model's)


@dataclass
class ConversionResult:
    ok: bool
    model: Optional[str]
    message: str                                  # what to tell the user (markdown)
    steps: List[Step] = field(default_factory=list)
    draft: Optional[Dict[str, Any]] = None        # PlantModelAgent draft, only when ok
    card: Optional[Dict[str, Any]] = None         # the JSON card of the C_to_Python agent
    rmse: Optional[Dict[str, Any]] = None         # Python model vs Simulink
    out_dir: Optional[Path] = None                # every file of the run
    cost_usd: Optional[float] = None
    tableau: Optional[List[Any]] = None           # (c, A, b) of the model's fixed-step solver, for simulate()


# ---------------------------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------------------------
def convert_package(
    zip_path,
    runs_dir,
    llm,
    log: Callable[[str], None] = print,
    cfg: Optional[Dict[str, Any]] = None,
) -> ConversionResult:
    """Steps 2-4 of C_to_Python on an upload package. Never raises for a bad upload or a
    failed step: the reason is in ``message`` and in the failed ``Step``.

    llm: ``llm(messages) -> text`` with ``.model`` / ``.usage`` (see ``llm.make_llm``).
    The files of the run are written to ``<runs_dir>/<model>/`` (replaced on every run)."""
    cfg = dict(CONFIG, **(cfg or {}))
    steps: List[Step] = []

    def fail(model, step, message, out=None, **kw):
        steps.append(step)
        return ConversionResult(False, model, message, steps, out_dir=out, **kw)

    # ---- the upload
    log("Checking the upload package ...")
    rep = package.check(zip_path)
    if not rep.ok:
        rep.cleanup()
        return fail(rep.model, Step("simulink", "Upload package rejected", rep.text(), False),
                    "This upload cannot be used:\n\n" + "\n".join("- " + e for e in rep.errors)
                    + "\n\nRun `c2py_export.m` next to your model again and upload the zip it creates.")
    name = rep.model
    out = Path(runs_dir) / name
    try:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        data = package.install(rep, out)
    finally:
        rep.cleanup()
    steps.append(Step("simulink", "Checked the upload package of %s" % name, rep.text(), True))
    return _convert(name, out, data, cfg, llm, log, steps, SOURCE_PACKAGE)


def convert_c_code(
    files: Dict[str, bytes],
    runs_dir,
    llm,
    log: Callable[[str], None] = print,
    cfg: Optional[Dict[str, Any]] = None,
) -> ConversionResult:
    """Generated C code without a c2py package (route B of ``intake``): the same pipeline as
    :func:`convert_package` without the comparison with Simulink - there are no Simulink results.
    The Python model runs with the solver the code was generated with. Never raises."""
    cfg = dict(CONFIG, solver="auto", **(cfg or {}))
    steps: List[Step] = []
    log("Checking the C code ...")
    chk = intake.check_code(files)
    if not chk.ok:
        reasons = chk.problems + ["missing: %s" % m for m in chk.missing]
        steps.append(Step("simulink", "The C code is not complete", "\n".join(reasons), False))
        return ConversionResult(False, chk.model, "This C code cannot be converted yet:\n\n"
                                + "\n".join("- " + r for r in reasons), steps)
    name = chk.model
    out = Path(runs_dir) / name
    if out.exists():
        shutil.rmtree(out)
    rtw = out / "codegen" / ("%s_ert_rtw" % name)
    rtw.mkdir(parents=True)
    for n, d in files.items():
        if n not in chk.ignored:
            (rtw / n).write_bytes(d)
    steps.append(Step("simulink", "Checked the generated C code of %s (%d files)" % (name, len(chk.files)),
                      "\n".join(chk.files + ["ignored: " + n for n in chk.ignored]), True))
    return _convert(name, out, None, cfg, llm, log, steps, SOURCE_CODE)


def _convert(name, out: Path, data, cfg, llm, log, steps: List[Step], source: str) -> ConversionResult:
    """C code in <out>/codegen/<name>_ert_rtw -> ConversionResult.
    data: the Simulink runs of a package ({signal: (t, U, t_sim, Y_sim, error)}), None = there are
    none (generated C code only): the comparison with Simulink is skipped, not failed."""

    def fail(model, step, message, out=None, **kw):
        steps.append(step)
        return ConversionResult(False, model, message, steps, out_dir=out, **kw)

    res = None
    try:
        # ---- C -> Python
        log("Translating the C code to Python ...")
        u, src, files = c2py.transpile([str(out / "codegen" / ("%s_ert_rtw" % name))], name)
        py_path = out / ("%s_py.py" % name)
        py_path.write_text(src, encoding="utf-8")
        # how Simulink runs the model: solver, step size, initial state (for simulate())
        Model = validation.load_model_class(py_path)
        model0 = Model(solver=cfg["solver"])
        c, A, b = sys.modules[Model.__module__]._TAB[model0._method()]
        tableau = [list(c), [list(row) for row in A], list(b)]
        simulation = {"solver": model0._method(), "fixed_step": float(model0.step_size),
                      "initial_state": [float(v) for v in model0.get_x()], "n_inputs": len(model0.input_names)}
        steps.append(Step("c2py", "Translated the generated C code to Python (%d C/H files)" % len(files),
                          "\n".join("warning: " + w for w in u.warnings) or None, True))

        # ---- Python vs Simulink
        if data is None:
            steps.append(Step("validation", "Python model vs Simulink: not checked - no Simulink results "
                              "were uploaded (only the C code)", None, True))
        elif not data:
            return fail(name, Step("validation", "No usable Simulink results in the package", None, False),
                        "The package has no usable Simulink results, so the Python model cannot be checked "
                        "against Simulink. Run `c2py_export.m` again and upload the complete zip.", out)
        else:
            log("Comparing the Python model with the Simulink results ...")
            res = validation.validate(name, py_path, out / "validation", cfg, None, data)
            res["c2py_warnings"] = u.warnings
            (out / "rmse.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            cmp_ = _comparison(res)
            worst = _max(cmp_["worst_nrmse"].values())
            steps.append(Step("validation", "Python model vs Simulink: %s (worst RMSE/range %s, tolerance %g)"
                              % ("PASS" if res["passed"] else "FAIL", _e(worst), res["tolerance"]),
                              _comparison_text(cmp_), bool(res["passed"])))
            if not res["passed"]:
                return ConversionResult(False, name, (
                    "The Python translation of the generated C code does not match the Simulink results "
                    "(worst RMSE/range %s, tolerance %g), so it is not a valid reference and the conversion "
                    "stops here. Details are in the step log; the files are in `%s`." % (_e(worst), res["tolerance"], out)),
                    steps, rmse=res, out_dir=out)

        # ---- readable dynamics(t, x, u) + JSON card
        log("Writing a readable dynamics(t, x, u) and validating it ...")
        card = c2py_agent.run(name, py_path, out, cfg, llm=llm, log=log)
    except SystemExit as ex:                      # the C_to_Python modules stop with SystemExit
        return fail(name, Step("simulink", "Conversion stopped", str(ex), False), "The conversion stopped: %s" % ex, out)
    except Exception as ex:                       # noqa: BLE001 - report, never crash the app
        missing = re.search(r"has no attribute '(\w+)' \(function/macro not found", str(ex))
        if missing:
            # code the model calls but that was not sent (shared utility code): ask for that file
            what = missing.group(1)
            if what.startswith(name + "_"):         # the model's own data (<model>_P ...)
                return fail(name, Step("simulink", "C code incomplete: %s is not defined" % what, str(ex), False),
                            "The C code needs `%s`, which is defined in `%s_data.c` (the parameter values). Send "
                            "that file too; the files already received are kept." % (what, name), out)
            return fail(name, Step("simulink", "C code incomplete: %s is not defined" % what, str(ex), False),
                        "The C code calls `%s`, which none of the files sent defines. It is shared utility code "
                        "of Simulink: send the file that defines it (usually `%s.c`, in the folder "
                        "`slprj/ert/_sharedutils` next to `%s_ert_rtw`). The files already received are kept."
                        % (what, what, name), out)
        return fail(name, Step("simulink", "Conversion failed", "%s: %s" % (type(ex).__name__, ex), False),
                    "The conversion failed: %s: %s" % (type(ex).__name__, ex), out)

    cost = _cost(out)
    ok = card["status"] != c2py_agent.STATUS_FAIL
    steps.append(Step("agent", "dynamics(t, x, u): %s" % card["status"], card.get("method"), ok))
    if not ok:
        return ConversionResult(False, name, (
            "No version of dynamics(t, x, u) passed the validation against the Python model, so nothing is "
            "delivered. The drafts and their validation results are in `%s`." % (out / "agent")),
            steps, card=card, rmse=res, out_dir=out, cost_usd=cost)

    metadata = card_to_metadata(card, res)
    metadata["simulink"]["simulation"] = simulation
    metadata["simulink"]["source"] = source
    metadata["simulink"]["compared_with_simulink"] = res is not None
    checked = metadata["simulink"]["state_equations_checked"]
    if checked:
        steps.append(Step("metadata", "State equations traced back by program: %d of %d checked numerically"
                          % (sum(checked), len(checked)), None, all(checked)))
    code = (out / ("%s_dynamics.py" % name)).read_text(encoding="utf-8")
    system = card.get("system")
    draft = {"system_name": system if _known(system) else name, "python_code": code, "metadata": metadata}
    result = ConversionResult(True, name, "", steps, draft=draft, card=card, rmse=res, out_dir=out, cost_usd=cost,
                              tableau=tableau)
    result.message = summary(result)
    return result


# ---------------------------------------------------------------------------------------------
# JSON card -> metadata format of this project
# ---------------------------------------------------------------------------------------------
def card_to_metadata(card: Dict[str, Any], rmse: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The card of the C_to_Python agent in the metadata format of backend_core.agent, plus
    everything that format has no field for under "simulink"."""
    md = card["model_details"]
    src = card.get("source_names") or {}
    src_out = list(src.get("outputs") or [])
    inputs = ["%s: %s" % (n, m) if _known(m) else n for n, m in zip(md["inputs"], md["input_meanings"])]
    outputs = [o if _known(o) else (src_out[i] if i < len(src_out) else NA)
               for i, o in enumerate(md.get("outputs") or [])]
    assumptions = []
    if card["status"] == c2py_agent.STATUS_PART:
        assumptions.append("Continuous part only: the discrete-time memory of the model (unit delays, discrete "
                           "integrators, filters ...) is held at its initial value")
    if _known(card.get("notes")):
        assumptions.append(card["notes"].strip())
    pm = md.get("parameter_meanings")
    return {
        "states": list(md["states"]),
        "state_meanings": list(md["state_meanings"]),
        "inputs": inputs,
        "outputs": outputs,
        "state_equations": list(md["state_equations"]),
        "parameters": dict(md["parameters"]) if isinstance(md.get("parameters"), dict) else {},
        "system_type": md.get("system_type") or NA,
        "assumptions": assumptions or ["Continuous-time state-space dynamics"],
        "simulink": {
            "model": card["model"],
            "status": card["status"],
            "validated": card["status"] != c2py_agent.STATUS_FAIL,
            "method": card.get("method"),
            "naming": card.get("naming"),
            "python_vs_simulink": _comparison(rmse) if rmse else None,
            "dynamics_vs_python_model": card.get("validation"),
            "state_equations_checked": list(md.get("state_equations_checked") or []),
            "definitions": list(md.get("definitions") or []),
            "computed_in_code": list(md.get("computed_in_code") or []),
            "parameter_meanings": dict(pm) if isinstance(pm, dict) else {},
            "source_names": src,
            "notes": card.get("notes"),
            "warnings": list(card.get("warnings") or []),
        },
    }


def same_code(a: str, b: str) -> bool:
    """True if two drafts are the same code (the chat agent writes code as ASCII)."""
    return plant_agent._to_ascii(a or "").strip() == plant_agent._to_ascii(b or "").strip()


def reconcile(draft: Optional[Dict[str, Any]], origin: Optional[Dict[str, Any]]) -> None:
    """After a chat turn on a converted model (in place).
    origin: the converted draft. Same code -> the validated metadata (the chat agent would
    otherwise replace it with metadata an LLM read from the code); changed code -> the
    metadata of the chat agent, marked as no longer verified against Simulink."""
    if not draft or not origin or not draft.get("python_code"):
        return
    if same_code(draft["python_code"], origin["python_code"]):
        draft["metadata"] = copy.deepcopy(origin["metadata"])
        return
    meta = draft.get("metadata")
    if isinstance(meta, dict):
        sim = origin["metadata"]["simulink"]
        meta["simulink"] = {"model": sim["model"], "status": MODIFIED, "validated": False,
                            "converted_status": sim["status"]}


# ---------------------------------------------------------------------------------------------
# simulation the way Simulink runs the model
# ---------------------------------------------------------------------------------------------
def simulate(
    python_code: str,
    *,
    tableau,
    fixed_step: float,
    x0,
    n_inputs: int,
    t_final: float,
    input_channel: int = 0,
    input_kind: str = "step",
    input_params: Optional[Dict[str, float]] = None,
    state_names: Optional[List[str]] = None,
    max_steps: int = SIM_MAX_STEPS,
    solver_name: str = "ode8",
    output_names: Optional[List[str]] = None,
) -> "sandbox.SimResult":
    """Fixed-step simulation of a converted dynamics(t, x, u) -> sandbox.SimResult (what the app
    plots). The code runs through the same restricted sandbox as every other draft
    (sandbox.compile_dynamics); the input is held over each step, like the Python model of
    step 2. A state that runs away (|x| > the sandbox's threshold) or a non-finite derivative
    after the first step stops the run early with the trajectory up to there (diverged=True)."""
    attempts = ["%s fixed-step, dt = %g s" % (solver_name, fixed_step)]
    try:
        fn = sandbox.compile_dynamics(python_code)
        update = sandbox.compile_update(python_code)
        output = sandbox.compile_output(python_code)
    except sandbox.DynamicsCodeError as exc:
        return sandbox.SimResult(False, str(exc), attempts=attempts)
    h = float(fixed_step)
    x = np.asarray(x0, dtype=float).reshape(-1)
    nx = x.size
    n = int(round(float(t_final) / h))
    note = ""
    if n > max_steps:
        n = max_steps
        note = " Shortened to %g s (%d steps of the model's fixed step %g s)." % (n * h, max_steps, h)
    if n < 1 or nx == 0:
        return sandbox.SimResult(False, "Nothing to simulate (duration %g s, step %g s, %d states)." % (t_final, h, nx),
                                 attempts=attempts)
    if input_kind not in sandbox._WAVEFORMS:
        return sandbox.SimResult(False, "Unknown input kind '%s'." % input_kind, attempts=attempts)
    nu = max(int(n_inputs), int(input_channel) + 1, 1)
    t = np.arange(n + 1) * h
    U = np.zeros((n + 1, nu))
    wave = sandbox._WAVEFORMS[input_kind]
    U[:, int(input_channel)] = [wave(float(tt), **(input_params or {})) for tt in t]
    c, A, b = tableau

    def f(tt, xx, uu):
        dx = np.atleast_1d(np.asarray(fn(float(tt), xx, uu), dtype=float)).reshape(-1)
        if dx.shape[0] != nx:
            raise sandbox.DynamicsCodeError("dynamics() returned %d value(s), the model has %d states."
                                            % (dx.shape[0], nx))
        if not np.all(np.isfinite(dx)):
            raise sandbox.DynamicsCodeError(
                "dynamics() produced a non-finite value (inf/nan) at t=%.4g, x=%s." % (tt, np.round(xx, 6).tolist()))
        return dx

    def major(k, xx):
        """Simulink's Update phase of the discrete memory at the major step k (drafts with update())"""
        if update is None:
            return xx
        new = np.atleast_1d(np.asarray(update(float(t[k]), xx.copy(), U[k]), dtype=float)).reshape(-1)
        if new.shape[0] != nx:
            raise sandbox.DynamicsCodeError("update() returned %d value(s), the model has %d states." % (new.shape[0], nx))
        return new

    X = np.zeros((n + 1, nx))
    end, diverged, message = n, False, "ok"
    with np.errstate(all="ignore"):
        try:
            x = major(0, x)
        except sandbox.DynamicsCodeError as exc:
            return sandbox.SimResult(False, "%s (first step)" % exc, attempts=attempts)
        X[0] = x
        for k in range(n):
            u = U[k]
            try:
                ks = [f(t[k], x, u)]
                for i in range(1, len(b)):
                    xs = x + h * sum(A[i][j] * ks[j] for j in range(i) if A[i][j] != 0)
                    ks.append(f(t[k + 1] if c[i] == 1.0 else t[k] + c[i] * h, xs, u))
            except sandbox.DynamicsCodeError as exc:
                if k == 0:
                    return sandbox.SimResult(False, "%s (first step, from the model's own initial state)" % exc,
                                             attempts=attempts)
                end, diverged = k, True
                message = "Stopped at t=%.4g s: %s" % (t[k], exc)
                break
            x = x + h * sum(b[i] * ks[i] for i in range(len(b)) if b[i] != 0)
            try:
                x = major(k + 1, x)
            except sandbox.DynamicsCodeError as exc:
                end, diverged = k, True
                message = "Stopped at t=%.4g s: %s" % (t[k + 1], exc)
                break
            X[k + 1] = x
            if not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) > sandbox._DIVERGENCE_THRESHOLD:
                end, diverged = k + 1, True
                message = ("State magnitude exceeded %.0e at t=%.3g s, so the simulation was stopped early. The "
                           "model is validated against Simulink, so this is how the model itself responds to "
                           "this input." % (sandbox._DIVERGENCE_THRESHOLD, t[k + 1]))
                break
    Y = None
    if output is not None:                    # the outputs y at every step, from the states there
        try:
            with np.errstate(all="ignore"):
                Y = np.array([np.atleast_1d(np.asarray(output(float(t[k]), X[k].copy(), U[k].copy()), dtype=float))
                              for k in range(end + 1)])
        except Exception as exc:              # noqa: BLE001 - the states are still shown
            note += " The outputs y could not be computed (%s: %s), so the states are shown." % (type(exc).__name__, exc)
            Y = None
    return sandbox.SimResult(
        True, message + note, t=t[:end + 1], x=X[:end + 1], u=U[:end + 1], y=Y, output_names=output_names,
        n_states=nx, n_inputs=int(n_inputs),
        input_channel=int(input_channel), input_kind=input_kind, state_names=state_names, diverged=diverged,
        solver_used=attempts[0], attempts=attempts)


# ---------------------------------------------------------------------------------------------
# text for the user
# ---------------------------------------------------------------------------------------------
def summary(r: ConversionResult) -> str:
    md, sim = r.card["model_details"], r.draft["metadata"]["simulink"]
    cmp_ = sim["python_vs_simulink"] or {}
    lines = ["Converted the Simulink model **%s**: %s, %s, %s."
             % (r.model, _n(len(md["states"]), "state"), _n(len(md["inputs"]), "input"),
                _n(len(md["outputs"]), "output")), ""]
    if cmp_:
        until = cmp_.get("compared_until_s") or {}
        lines.append("- Python translation of the generated C code vs Simulink: **PASS** for %s (worst RMSE/range %s, "
                     "tolerance %g)%s." % (", ".join(cmp_["inputs"]), _e(_max(cmp_["worst_nrmse"].values())),
                                          cmp_["tolerance"],
                                          "; compared only up to %s, where the model itself becomes unstable for that "
                                          "input" % ", ".join("%g s for %s" % (t, s) for s, t in until.items())
                                          if until else ""))
    else:
        lines.append("- Python translation of the generated C code vs Simulink: **not checked** - only the C code "
                     "was uploaded, without Simulink's own results. The `<model>_c2py.zip` of `c2py_export.m` "
                     "includes them, if you want this check too.")
    lines.append("- This `dynamics(t, x, u)` vs that Python model: **%s** (%s)." % (sim["status"], sim["method"]))
    if sim["state_equations_checked"]:
        c = sim["state_equations_checked"]
        lines.append("- State equations in the system sheet: %d of %d checked numerically." % (sum(c), len(c)))
    if r.cost_usd is not None:
        lines.append("- LLM cost of the conversion: $%.4f." % r.cost_usd)
    lines += ["", "If you change the model in the chat, it is no longer verified against %s."
              % ("Simulink" if cmp_ else "the generated code")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
def _n(k: int, word: str) -> str:
    return "%d %s%s" % (k, word, "" if k == 1 else "s")


def _known(v) -> bool:
    return isinstance(v, str) and bool(v.strip()) and v.strip() != NA


def _finite(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def _max(values):
    v = [x for x in values if x is not None]
    return max(v) if v else None


def _e(v):
    return "n/a" if v is None else "%.1e" % v


def _comparison(res: Dict[str, Any]) -> Dict[str, Any]:
    worst, until = {}, {}
    for sig, outs in res["scenarios"].items():
        worst[sig] = _max(_finite(o["nrmse"]) for o in outs.values())
        t = next((o["compared_until_s"] for o in outs.values() if o.get("compared_until_s") is not None), None)
        if t is not None:
            until[sig] = t
    return {"passed": bool(res["passed"]), "tolerance": res["tolerance"],
            "tolerance_meaning": res.get("tolerance_meaning"), "solver": res.get("python_solver"),
            "dt": res.get("dt"), "inputs": list(res["scenarios"]), "worst_nrmse": worst,
            "compared_until_s": until, "warnings": list(res.get("warnings") or [])}


def _comparison_text(c: Dict[str, Any]) -> str:
    rows = ["%s: worst RMSE/range %s%s" % (s, _e(v), " (compared up to %g s)" % c["compared_until_s"][s]
                                           if s in c["compared_until_s"] else "") for s, v in c["worst_nrmse"].items()]
    return "\n".join(rows + ["warning: " + w for w in c["warnings"]])


def _cost(out: Path) -> Optional[float]:
    try:
        return json.loads((out / "agent" / "cost.json").read_text(encoding="utf-8")).get("cost_usd")
    except Exception:                             # no LLM call, or no price for the model
        return None

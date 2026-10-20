"""Whole models: the slx2py engine against Simulink itself.

matlab_block_tests/simulate_models_reference.m simulates real models in MATLAB (a unit step on each
root input, the model's own solver, outputs every 10 ms) and writes slx2py_models_reference.zip.
This converts each model with the engine, simulates its block model the same way (the model's
solver; variable-step with the same tight tolerances) and compares the outputs:

    python tests/slx_models_check.py [path/to/slx2py_models_reference.zip]

A model the engine cannot convert is listed with its reason.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_core.simulink.direct.slx2py.model_converter import pipeline  # noqa: E402
from backend_core.simulink.direct.slx2py.model_simulation import backends, contracts, inputs, integrate  # noqa: E402

DEFAULT = ROOT / "tests" / "fixtures" / "slx_models_reference.zip"
FOLDER = "slx2py_models_reference/"
TOL = 1e-5               # max |y - y_simulink| <= TOL * max(1, max |y_simulink|)

# outputs not compared, with the reason (checked: the reason is shown with the result)
KNOWN_DEVIATIONS = {
    "Longitidunal_Vehicle_Dynamics": (
        ("Traction Slip", "Traction Slip1"),
        "the wheel slips divide by the vehicle speed, which is exactly 0 in the first step: the slips are +-inf "
        "and the wheel speeds chatter between the solver's stages by the sign of a floating-point zero (with -0 "
        "instead of +0 the engine's rear slip flips from -1157 to +1180, Simulink's is -1174). The rest of the run "
        "keeps that start; Out1 (the vehicle, through the Memory blocks) matches"),
}


def _names(name: str, width: int) -> list[str]:
    return [name] if width == 1 else [f"{name}({i + 1})" for i in range(width)]


def load(reference: Path):
    with zipfile.ZipFile(reference) as z:
        manifest = json.loads(z.read(FOLDER + "models_manifest.json"))
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    cases = manifest["cases"] if isinstance(manifest["cases"], list) else [manifest["cases"]]
    return cases, files


def run_case(case: dict, files: dict, work: Path) -> str:
    """-> "PASS ...", "FAIL ...", "not converted: ...", "Simulink run failed: ..." """
    name = case["name"]
    if case["status"] != "ok":
        return "Simulink run failed: %s" % case["error"][:120]
    source = work / (name + ".slx")
    source.write_bytes(files[FOLDER + name + ".slx"])
    try:
        pipeline.convert_slx(source, work / "x" / name, work / "m")
    except Exception as exc:                  # noqa: BLE001
        return "not converted: %s" % " ".join(str(exc).split())[:140]
    rows = list(csv.reader(io.StringIO(files[FOLDER + name + ".csv"].decode("utf-8"))))
    data = np.array(rows[1:], dtype=float)
    return _compare(case, work / "m", data[:, 0], data[:, 1:])


def check(reference: Path) -> None:
    cases, files = load(reference)
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            work = Path(tmp) / case["name"]
            work.mkdir()
            print("%-34s %s" % (case["name"], run_case(case, files, work)), flush=True)


def _compare(case: dict, model_dir: Path, t_ref: np.ndarray, y_ref: np.ndarray) -> str:
    contract = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    as_list = lambda v: v if isinstance(v, list) else [v]                       # noqa: E731
    specs = {}
    for item in as_list(case["inputs"]):
        for element in _names(item["name"], int(item["width"])):
            specs[element] = {"kind": "step", "step_time": float(item["step_time"]), "initial_value": 0.0, "final_value": 1.0}
    missing = set(contract["inputs"]) ^ set(specs)
    if missing:
        return "FAIL: the inputs differ from Simulink's: %s" % sorted(missing)
    sim = dict(contract["simulation"])
    variable = case["solver_type"] == "Variable-step"
    sim.update(start_time=0.0, stop_time=float(case["stop_time"]), max_step="auto", initial_step="auto", min_step="auto")
    if variable:
        sim.update(solver="ode45", step_size=0.01, relative_tolerance=1e-8, absolute_tolerance=1e-10,
                   auto_scale_absolute_tolerance=False)
    else:
        # Simulink's own fixed-step solver, step for step: its stages see the input at their own
        # times, so a step at t = 1 already acts in the last stage of the step that ends there
        return _fixed_step(case, contract, model_dir, specs, t_ref, y_ref)
    settings = contracts.SimulationSettings.from_mapping(sim)
    signals = inputs.InputSet(tuple(contract["inputs"]), specs)
    backend = backends.ModelBackend(model_dir, contract)
    trace = integrate.simulate(backend, settings, signals, transitions=backend.transition_times(0.0, settings.stop_time))
    expected = [n for item in as_list(case["outputs"]) for n in _names(item["name"], int(item["width"]))]
    if sorted(expected) != sorted(trace.output_names):
        return "FAIL: the outputs differ from Simulink's: %s vs %s" % (expected, list(trace.output_names))
    ours = np.column_stack([np.interp(t_ref, trace.time, trace.outputs[:, trace.output_names.index(n)]) for n in expected])
    scale = max(1.0, float(np.max(np.abs(y_ref)))) if y_ref.size else 1.0
    error = float(np.max(np.abs(ours - y_ref))) if y_ref.size else 0.0
    return "%s  max error %.1e (%s, %d outputs)" % ("PASS" if error <= TOL * scale else "FAIL", error, sim["solver"],
                                                   len(expected))


def _known(case, expected, y_ref):
    """(outputs compared, their reference columns, note) without the known deviations"""
    skip, why = KNOWN_DEVIATIONS.get(case["name"], ((), ""))
    keep = [i for i, n in enumerate(expected) if n not in skip]
    note = " - not compared: %s (%s)" % (", ".join(n for n in expected if n in skip), why) if len(keep) < len(expected) else ""
    return [expected[i] for i in keep], y_ref[:, keep] if y_ref.size else y_ref, note


def _fixed_step(case, contract, model_dir, specs, t_ref, y_ref) -> str:
    import runpy
    from backend_core.simulink.direct import slx_direct
    solver = contract["simulation"]["solver"]
    h = float(case.get("fixed_step") or contract["simulation"]["step_size"])
    try:
        c, a, b = slx_direct._tableau(solver)
    except KeyError:
        return "not checked: fixed-step solver %s" % solver
    model = runpy.run_path(str(model_dir / "model.py"))["build_model"]()
    names = list(contract["inputs"])

    def u(t):
        return {n: (specs[n]["final_value"] if t >= specs[n]["step_time"] else specs[n]["initial_value"]) for n in names}

    def f(t, x):
        return np.asarray(model.evaluate(t, x, u(t)).derivatives, dtype=float)

    expected = [n for item in (case["outputs"] if isinstance(case["outputs"], list) else [case["outputs"]])
                for n in _names(item["name"], int(item["width"]))]
    x = np.asarray(model.initial_state_at(0.0, u(0.0)), dtype=float)
    x = model.update(0.0, x, u(0.0))          # every major step: the Update phase of the discrete memory first
    steps = int(round(float(case["stop_time"]) / h))
    times, rows = [0.0], [model.evaluate(0.0, x, u(0.0)).outputs]
    for k in range(steps):
        t = k * h
        if x.size:
            ks = [f(t, x)]
            for i in range(1, len(b)):
                ks.append(f(t + c[i] * h, x + h * sum(a[i][j] * ks[j] for j in range(i) if a[i][j])))
            x = x + h * sum(b[i] * ks[i] for i in range(len(b)) if b[i])
        times.append((k + 1) * h)
        x = model.update((k + 1) * h, x, u((k + 1) * h))
        rows.append(model.evaluate((k + 1) * h, x, u((k + 1) * h)).outputs)
    expected, y_ref, note = _known(case, expected, y_ref)
    ours_all = np.array([[row[n] for n in expected] for row in rows])
    ours = np.column_stack([np.interp(t_ref, times, ours_all[:, j]) for j in range(len(expected))])
    scale = max(1.0, float(np.max(np.abs(y_ref)))) if y_ref.size else 1.0
    error = float(np.max(np.abs(ours - y_ref))) if y_ref.size else 0.0
    return "%s  max error %.1e of %.1e (%s, h = %g, %d outputs)%s" % ("PASS" if error <= TOL * scale else "FAIL", error,
                                                                    scale, solver, h, len(expected), note)


if __name__ == "__main__":
    check(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)

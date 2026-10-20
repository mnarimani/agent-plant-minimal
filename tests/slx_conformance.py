"""Block-by-block comparison of the slx2py engine with Simulink itself.

fixtures/slx_blocks_reference.zip is made by matlab_block_tests/make_slx_block_reference.m (run in
MATLAB R2025b): for every test case one small model (Inports -> the block -> Outports), its .slx and
what Simulink answered (.csv: t, u1..un, y1..ym; fixed-step ode4, h = 1 ms, outputs every 10 ms, the
inputs interpolated linearly between the samples of the csv). manifest.json lists the cases, the
parameters that were set and ALL dialog parameters of each block with their values.

For every case this converts the .slx with the engine and runs, the way Simulink did,

    model     the engine's block model (model.py)
    dynamics  the extracted equations (dynamics.py: f, g, initial_state), if any

and compares the outputs with Simulink's. Run it for a table of every case:

    python tests/slx_conformance.py [case-name-substring ...]
"""

from __future__ import annotations

import csv
import io
import json
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_core.simulink.direct.slx2py.model_converter import pipeline  # noqa: E402

REFERENCE = ROOT / "tests" / "fixtures" / "slx_blocks_reference.zip"
FOLDER = "slx2py_block_reference/"
H = 1e-3                 # Simulink's fixed step (ode4)
DECIMATION = 10          # outputs every 10 ms
TOL = 1e-6               # max |y - y_simulink| <= TOL * max(1, max |y_simulink|)
JUMP = 0.25              # an input step this large between two samples is a jump of the test signal

# known, explained differences: case -> (tolerance, why)
DEVIATIONS = {
    "b_integrator_limited": (1e-3, "Simulink resets the state to the limit at every major step; an ODE "
                                   "cannot, so in the step that reaches the limit the state passes it by up to "
                                   "one step of the input (the output itself is clipped)"),
}


def manifest() -> Dict[str, Any]:
    with zipfile.ZipFile(REFERENCE) as z:
        return json.loads(z.read(FOLDER + "manifest.json"))


def cases(group: Optional[str] = None) -> List[Dict[str, Any]]:
    return [c for c in manifest()["cases"] if c["status"] == "ok" and (group is None or c["group"] == group)]


def _reference(name: str):
    with zipfile.ZipFile(REFERENCE) as z:
        slx = z.read(FOLDER + name + ".slx")
        rows = list(csv.reader(io.StringIO(z.read(FOLDER + name + ".csv").decode("utf-8"))))
    head, data = rows[0], np.array(rows[1:], dtype=float)
    u_cols = [i for i, h in enumerate(head) if h.startswith("u")]
    y_cols = [i for i, h in enumerate(head) if h.startswith("y")]
    return slx, data[:, 0], data[:, u_cols], data[:, y_cols]


def run_case(name: str, workdir: Path) -> Dict[str, Any]:
    """-> {"status": pass | fail | unsupported, "model": err, "dynamics": err | None, "error": text}"""
    slx, t_ref, U, Y = _reference(name)
    source = workdir / (name + ".slx")
    source.write_bytes(slx)
    out = workdir / "model"
    try:
        pipeline.convert_slx(source, workdir / "extracted" / name, out)
    except Exception as exc:                  # noqa: BLE001 - the engine raises ValueError / OSError
        return {"status": "unsupported", "model": None, "dynamics": None, "error": _one_line(exc)}
    contract = json.loads((out / "model.json").read_text(encoding="utf-8"))
    inputs, outputs = contract["inputs"], contract["outputs"]
    scale = max(1.0, float(np.max(np.abs(Y)))) if Y.size else 1.0
    result: Dict[str, Any] = {"status": "pass", "model": None, "dynamics": None, "error": ""}
    # Samples next to a jump of a test input (switch controls, logic inputs) are not compared:
    # Simulink's time there is a rounding off the csv's time (7.6900000000000004 for 7.69) and the
    # interpolated input lands on the other side of the jump.
    keep = np.ones(len(t_ref), dtype=bool)
    keep &= _logged(name, t_ref)
    if U.size:
        jump = np.any(np.abs(np.diff(U, axis=0)) > JUMP, axis=1)
        keep[1:] &= ~jump
        keep[:-1] &= ~jump

    def uf(t: float) -> np.ndarray:
        return np.array([np.interp(t, t_ref, U[:, i]) for i in range(U.shape[1])])

    # ---- the block model
    try:
        model = runpy.run_path(str(out / "model.py"))["build_model"]()
        named = lambda t: dict(zip(inputs, uf(t).tolist()))      # noqa: E731
        x0 = np.asarray(model.initial_state_at(0.0, named(0.0)), dtype=float)

        def g(t, x):
            values = model.evaluate(t, x, named(t)).outputs
            return np.concatenate([np.atleast_1d(np.asarray(values[n], dtype=float)) for n in outputs])

        update = (lambda t, x: model.update(t, x, named(t))) if model.discrete else None
        Yhat = _simulate(lambda t, x: model.evaluate(t, x, named(t)).derivatives, g, x0, t_ref, update)
        result["model"] = _error(Yhat[keep], Y[keep], scale)
    except Exception as exc:                  # noqa: BLE001
        result.update(status="fail", error="model: " + _one_line(exc))
        return result

    # ---- the extracted equations
    if contract.get("dynamics", {}).get("status") == "success":
        try:
            ns = runpy.run_path(str(out / "dynamics.py"))
            xd0 = np.asarray(ns["initial_state"](uf(0.0), t=0.0), dtype=float)
            update = (lambda t, x: ns["update"](x, uf(t), t=t)) if "update" in ns else None
            Yd = _simulate(lambda t, x: ns["f"](x, uf(t), t=t), lambda t, x: ns["g"](x, uf(t), t=t), xd0, t_ref, update)
            result["dynamics"] = _error(Yd[keep], Y[keep], scale)
        except Exception as exc:              # noqa: BLE001
            result["dynamics"] = float("inf")
            result["error"] = "dynamics: " + _one_line(exc)
    errors = [e for e in (result["model"], result["dynamics"]) if e is not None]
    tolerance = DEVIATIONS.get(name, (TOL, ""))[0]
    if not errors or max(errors) > tolerance * scale:
        result["status"] = "fail"
    return result


def _logged(name: str, t_ref: np.ndarray) -> np.ndarray:
    """Samples Simulink really logged. Before the reference script logged every step, it logged
    every 10th sample of each Outport (Decimation 10): a discrete Outport of rate Ts only every
    10*Ts; the csv holds the value in between. Only those times are compared then."""
    case = next((c for c in manifest()["cases"] if c["name"] == name), {})
    if case.get("output_logging"):           # logged every step (the script since 2026-10-16)
        return np.ones(len(t_ref), dtype=bool)
    rates = []
    for key, value in (case.get("sample_times") or {}).items():
        if key.startswith("y") and isinstance(value, str):
            ts = float(value.strip("[]").split()[0])
            if 0 < ts < float("inf") and ts > H * (1 + 1e-9):
                rates.append(ts)
    keep = np.ones(len(t_ref), dtype=bool)
    for ts in rates:
        period = DECIMATION * ts
        keep &= np.abs(t_ref / period - np.round(t_ref / period)) < 1e-6
    return keep


def _simulate(f: Callable, g: Callable, x0: np.ndarray, t_ref: np.ndarray, update: Optional[Callable] = None) -> np.ndarray:
    """Simulink's ode4 with h = 1 ms; outputs at the times of the csv (every 10th step). Every
    major step first updates the discrete memory (``update``, Simulink's Update phase), then
    gives the outputs and the solver's stages."""
    x = x0.astype(float)
    if x.size == 0:
        return np.array([g(float(t), x) for t in t_ref])
    major = update or (lambda t, state: state)
    x = np.asarray(major(0.0, x), dtype=float)
    rows = [g(0.0, x)]
    steps = int(round(t_ref[-1] / H))
    for k in range(steps):
        t = k * H
        k1 = np.asarray(f(t, x), dtype=float)
        k2 = np.asarray(f(t + H / 2, x + H / 2 * k1), dtype=float)
        k3 = np.asarray(f(t + H / 2, x + H / 2 * k2), dtype=float)
        k4 = np.asarray(f(t + H, x + H * k3), dtype=float)
        x = x + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x = np.asarray(major((k + 1) * H, x), dtype=float)
        if (k + 1) % DECIMATION == 0:
            rows.append(g((k + 1) * H, x))
    return np.array(rows)


def _error(Yhat: np.ndarray, Y: np.ndarray, scale: float) -> float:
    Yhat = np.asarray(Yhat, dtype=float).reshape(Y.shape[0], -1)
    if Yhat.shape != Y.shape:
        raise ValueError("outputs: %s columns, Simulink: %s" % (Yhat.shape[1], Y.shape[1]))
    diff = np.abs(Yhat - Y)
    diff[np.isnan(Yhat) & np.isnan(Y)] = 0.0
    return float(np.max(diff)) if np.all(np.isfinite(diff)) else float("inf")


def _one_line(exc: Exception) -> str:
    return " ".join(str(exc).split())[:240]


def report(filters: List[str]) -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases():
            name = c["name"]
            if filters and not any(f in name for f in filters):
                continue
            work = Path(tmp) / name
            work.mkdir()
            r = run_case(name, work)
            rows.append((c["group"], name[2:], r))
            fmt = lambda v: "-" if v is None else "%.1e" % v      # noqa: E731
            print("%s  %-28s %-11s model %-8s dyn %-8s %s" % (c["group"], name[2:], r["status"], fmt(r["model"]),
                                                              fmt(r["dynamics"]), r["error"][:110]), flush=True)
    for group in "ABC":
        g = [r for gr, _, r in rows if gr == group]
        if g:
            print("group %s: %d pass of %d" % (group, sum(r["status"] == "pass" for r in g), len(g)))


if __name__ == "__main__":
    report(sys.argv[1:])

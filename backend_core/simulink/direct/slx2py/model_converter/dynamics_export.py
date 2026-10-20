"""Write equations, a machine-readable report, and standalone f/g functions."""
from __future__ import annotations

import json
import math
import inspect
from pathlib import Path

from .dynamics import DynamicsResult, extract_dynamics
from .dynamics_expressions import python_prelude
from .model import Model


def render_equations(report: dict) -> str:
    lines = [f"Dynamic model: {report['model_name']}", f"Status: {report['status']}", ""]
    if report["status"] != "success":
        for error in report["errors"]:
            block = f"{error.get('block_type')} '{error.get('block_name')}' (ID {error.get('block_id')})" if error.get("block_id") else "Model"
            lines.append(f"[{error['code']}] {block}: {error['message']}")
        return "\n".join(lines) + "\n"
    lines.extend([report["form"], "", "States (runtime vector order):"])
    for state in report["states"]:
        lines.append(f"  {state['symbol']} = {state['block_name']} / local state {state['local_index'] + 1}; initial = {state['initial_expression']}")
    if not report["states"]:
        lines.append("  Empty state vector: this model is algebraic.")
    lines.append("\nInputs:")
    lines.extend(f"  {item['symbol']} = {item['name']}" for item in report["inputs"])
    lines.append("\nState derivatives:")
    lines.extend(f"  d{item['state']}/dt = {item['expression']}" for item in report["f"])
    lines.append("\nOutputs:")
    lines.extend(f"  {item['output']} ({item['name']}) = {item['expression']}" for item in report["g"])
    if report.get("update"):
        lines.append("\nDiscrete memory, at every major solver step (before its stages):")
        lines.extend(f"  {item['state']}+ = {item['expression']}" for item in report["update"])
    lines.extend(["", "Transfer Function states use the runtime's tf2ss coordinates. A/B/C/D are in dynamics.json.",
                  "Conditions use lazy 'value if condition else other' notation; min/max are scalar.",
                  "Initialization samples all external IC expressions with the configured default x vector.",
                  "Executable API: dynamics.py f(x,u,t=...), g(x,u,t=...), initial_state(u,t=...).",
                  "Time is optional and defaults to 0; supply t for explicitly time-dependent models."])
    return "\n".join(lines) + "\n"


def _render_python(result: DynamicsResult) -> str:
    equations = result.equations
    if equations is None:
        raise ValueError("Failed extraction has no executable equations.")
    report = result.report
    names = [state["symbol"] for state in report["states"]] + [item["symbol"] for item in report["inputs"]] + ["t"]
    binding = ", ".join(names) + ", = _values(x, u, t)"
    lines = [
        '"""Extracted continuous equations. Requires only NumPy; no original model or converter."""',
        "from math import sin, cos, tan, exp, sqrt, log",
        "import math",
        "import numpy as np", "",
        f"STATE_NAMES = {tuple(state['symbol'] for state in report['states'])!r}",
        f"INPUT_NAMES = {tuple(item['name'] for item in report['inputs'])!r}",
        f"OUTPUT_NAMES = {tuple(item['name'] for item in report['outputs'])!r}",
        f"STATE_DEFAULTS = {equations.state_defaults!r}",
        f"ZERO_CROSSINGS = {equations.zero_crossing_metadata!r}", "",
        "def _values(x, u, t):",
        "    x = np.asarray(x, dtype=float)",
        "    u = np.asarray(u, dtype=float)",
        "    if x.shape != (len(STATE_NAMES),) or u.shape != (len(INPUT_NAMES),):",
        "        raise ValueError('State/input vector shape does not match STATE_NAMES/INPUT_NAMES.')",
        "    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)) or not np.isfinite(t):",
        "        raise ValueError('State, inputs and time must be finite.')",
        "    return (*x.tolist(), *u.tolist(), float(t))", "",
    ]
    used = " ".join(expr.python() for expr in (*equations.derivatives, *equations.outputs,
                                                 *equations.zero_crossings, *equations.initializers, *equations.updates))
    lines.extend(python_prelude(used + " _divide("))
    for name, expressions in (("f", equations.derivatives), ("g", equations.outputs), ("zero_crossings", equations.zero_crossings)):
        lines.extend([f"def {name}(x, u, *, t=0.0):", "    " + binding,
                      "    return np.array([" + ", ".join(expr.python() for expr in expressions) + "], dtype=float)", ""])
    if equations.updates:
        lines.extend(["def update(x, u, *, t=0.0):",
                      '    """x+ at the major step at t: the discrete memory (Simulink\'s Update phase)"""',
                      "    " + binding,
                      "    return np.array([" + ", ".join(expr.python() for expr in equations.updates) + "], dtype=float)", ""])
    lines.extend(["def initial_state(u=(), *, t=0.0):", "    x = STATE_DEFAULTS", "    " + binding,
                  "    return np.array([" + ", ".join(expr.python() for expr in equations.initializers) + "], dtype=float)", ""])
    return "\n".join(lines)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def export_dynamics(model: Model, directory: Path) -> dict:
    result = extract_dynamics(model)
    report = result.report
    python_path = directory / "dynamics.py"
    if result.equations is not None:
        try:
            source = _render_python(result)
            compile(source, str(python_path), "exec")
        except Exception as error:
            report = {**report, "status": "error", "form": None,
                      "errors": [{"code": "render_error", "message": str(error), "block_id": None}]}
            report.pop("f", None)
            report.pop("g", None)
        else:
            python_path.write_text(source, encoding="utf-8")
    if report["status"] != "success":
        # Do not leave executable f/g from an earlier successful conversion.
        python_path.unlink(missing_ok=True)
    report = _json_safe(report)
    (directory / "dynamics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "dynamics.txt").write_text(render_equations(report), encoding="utf-8")
    return report

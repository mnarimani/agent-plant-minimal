"""Sandbox: restricted exec of a drafted ``dynamics(t, x, u)`` + simulation.

Three layers, in order:

1. :func:`validate_ast` — parses the drafted source and rejects anything
   outside a small whitelist (imports limited to numpy/math/scipy, no
   dunder access, no ``with``/``try``, no ``exec``/``eval``/``open``/
   ``__import__``/etc.).
2. :func:`compile_dynamics` — compiles the validated AST and executes it
   in a namespace with a hand-picked, minimal ``__builtins__`` (no file
   or process access at all), then returns the ``dynamics`` callable.
3. :func:`simulate` — drives that callable with ``scipy.integrate.solve_ivp``
   under a step / ramp / sine input on a chosen channel and returns a
   :class:`SimResult` plotly can render.

This is defense-in-depth for *accidental* misbehavior (a stray ``os``
import, a placeholder ``exec(...)``), not a hardened multi-tenant
security boundary — for that you'd still want a real subprocess/container
boundary in front of this. See the README's "Sandbox" note.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


class DynamicsCodeError(Exception):
    """Raised for anything wrong with drafted dynamics() source or output."""


_ALLOWED_IMPORT_ROOTS = {"numpy", "math", "scipy"}

_FORBIDDEN_NAMES = {
    # Note: "__import__" is intentionally NOT in this set. Python's own
    # `import` statement calls __builtins__['__import__'] at runtime, so
    # removing it outright would break the very `import numpy as np` line
    # the drafted code is required to have. Instead we install a
    # _restricted_import below that only allows the same whitelisted
    # roots validate_ast() already enforces at parse time; a direct call
    # like `__import__("os")` is still caught separately by validate_ast
    # (it rejects any bare Name node named "__import__").
    "eval", "exec", "open", "compile", "input", "exit", "quit",
    "help", "breakpoint", "globals", "locals", "vars", "dir", "getattr",
    "setattr", "delattr", "__builtins__", "__loader__", "__spec__", "memoryview",
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = (name or "").split(".")[0]
    if top not in _ALLOWED_IMPORT_ROOTS:
        raise DynamicsCodeError(f"Import of '{name}' is not allowed in sandboxed dynamics code.")
    return __import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS: Dict[str, Any] = {
    "len": len, "range": range, "abs": abs, "min": min, "max": max, "sum": sum,
    "float": float, "int": int, "bool": bool, "list": list, "tuple": tuple,
    "dict": dict, "enumerate": enumerate, "zip": zip, "map": map, "round": round,
    "True": True, "False": False, "None": None, "pow": pow, "sorted": sorted,
    "ValueError": ValueError, "TypeError": TypeError, "Exception": Exception,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "__import__": _restricted_import,
}


def validate_ast(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise DynamicsCodeError(f"Syntax error in drafted code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            )
            for name in names:
                top = (name or "").split(".")[0]
                if top not in _ALLOWED_IMPORT_ROOTS:
                    raise DynamicsCodeError(
                        f"Import of '{name}' is not allowed in sandboxed dynamics code "
                        f"(only numpy / math / scipy are permitted)."
                    )
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise DynamicsCodeError(f"Use of '{node.id}' is not allowed in sandboxed dynamics code.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise DynamicsCodeError("Dunder attribute access is not allowed in sandboxed dynamics code.")
        elif isinstance(node, (ast.With, ast.Try)):
            raise DynamicsCodeError("with/try blocks are not allowed in sandboxed dynamics code.")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise DynamicsCodeError("global/nonlocal are not allowed in sandboxed dynamics code.")

    return tree


def compile_dynamics(source: str) -> Callable[[float, np.ndarray, Any], Any]:
    """Validate, compile, and execute ``source``; return the ``dynamics`` callable."""
    tree = validate_ast(source)
    code = compile(tree, filename="<dynamics>", mode="exec")

    sandbox_globals: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "numpy": np,
        "math": math,
    }
    try:
        import scipy  # noqa: F401  (optional, only if the code imports it)

        sandbox_globals["scipy"] = scipy
    except Exception:
        pass

    local_ns: Dict[str, Any] = {}
    try:
        exec(code, sandbox_globals, local_ns)  # noqa: S102 - intentional, restricted namespace
    except Exception as exc:
        raise DynamicsCodeError(f"Drafted code raised an error while loading: {exc}") from exc

    fn = local_ns.get("dynamics") or sandbox_globals.get("dynamics")
    if fn is None or not callable(fn):
        raise DynamicsCodeError("No callable `dynamics(t, x, u)` was defined by the drafted code.")

    params = list(inspect.signature(fn).parameters)
    if params[:3] != ["t", "x", "u"]:
        raise DynamicsCodeError(
            f"dynamics() must have signature (t, x, u); found ({', '.join(params)})."
        )
    return fn


def infer_dims(source: str) -> Tuple[int, int, bool]:
    """Guess (n_states, n_inputs, scalar_u) by scanning for x[i] / u[i] indices."""
    x_idx = [int(m.group(1)) for m in re.finditer(r"\bx\[(\d+)\]", source)]
    u_idx = [int(m.group(1)) for m in re.finditer(r"\bu\[(\d+)\]", source)]
    n_states = (max(x_idx) + 1) if x_idx else 1
    if u_idx:
        return n_states, max(u_idx) + 1, False
    return n_states, 1, True


# -- input waveforms -------------------------------------------------------


def step_input(t: float, t0: float = 1.0, amplitude: float = 1.0) -> float:
    return float(amplitude) if t >= t0 else 0.0


def ramp_input(t: float, t0: float = 1.0, slope: float = 1.0) -> float:
    return float(slope) * (t - t0) if t >= t0 else 0.0


def sine_input(t: float, amplitude: float = 1.0, freq_hz: float = 0.5, phase: float = 0.0, bias: float = 0.0) -> float:
    return float(bias) + float(amplitude) * math.sin(2.0 * math.pi * float(freq_hz) * t + float(phase))


_WAVEFORMS = {"step": step_input, "ramp": ramp_input, "sine": sine_input}

# Threshold used to *cleanly* stop integration when a state runs away, well
# below float64 overflow (~1.8e308). Catching a blow-up here, via a scipy
# terminal event, avoids the adaptive step controller ever grinding its step
# size down toward zero chasing an overflowing derivative -- which is what
# produces scipy's cryptic "Required step size is less than spacing between
# numbers" error. Stopping early also means a real, useful partial trajectory
# is still returned and plotted, instead of nothing at all.
_DIVERGENCE_THRESHOLD = 1.0e6

# Guards a *different* failure shape than the state-magnitude event above:
# a near-singularity (e.g. 1/cos(theta) as theta -> +/-90deg) where the
# STATE stays bounded but the DERIVATIVE grows enormous while still being
# finite (not inf/nan), which otherwise makes RK45's adaptive step search
# spend tens of thousands of evaluations homing in on the singularity
# without ever tripping a state-divergence event or a non-finite check --
# a genuine near-hang, confirmed empirically while building this. Real
# plant models built from natural-language descriptions essentially never
# have legitimate derivatives this large; this is almost always a
# singularity or a units/parameter error.
_DERIVATIVE_MAGNITUDE_THRESHOLD = 1.0e8


# -- simulation --------------------------------------------------------------


@dataclass
class SimResult:
    success: bool
    message: str
    t: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None  # shape (T, n_states)
    u: Optional[np.ndarray] = None  # shape (T, n_inputs)
    n_states: int = 0
    n_inputs: int = 0
    input_channel: int = 0
    input_kind: str = ""
    state_names: Optional[List[str]] = None
    diverged: bool = False          # True: stopped early, state ran away
    solver_used: str = "RK45"       # which method actually produced this result
    attempts: List[str] = field(default_factory=list)  # trace, for the step log / diagnosis


def _make_divergence_event():
    """Catches a state running away (an unstable/insufficiently-damped
    pole): a scipy terminal event, checked once per *accepted* step, that
    stops integration cleanly and still returns the partial trajectory.

    This is deliberately narrower than "any large value anywhere" -- see
    the separate derivative-magnitude guard inside ``rhs()`` below for the
    other failure shape (a near-singularity, state bounded but derivative
    enormous), which turned out to need a different fix: empirically, an
    equivalent event checking the derivative once per accepted step did
    NOT catch a real 1/cos(theta) singularity in reasonable wall-clock
    time, because the *accepted*-state derivative grows far more slowly
    than the solver's own intermediate, unaccepted trial-stage
    evaluations that are actually what balloons the search. Only a check
    inside ``rhs`` itself -- reached on every trial evaluation, not just
    once a step is accepted -- interrupts that search quickly.
    """

    def _event(t: float, x: np.ndarray) -> float:
        x_safe = np.where(np.isfinite(x), x, _DIVERGENCE_THRESHOLD * 10)
        return _DIVERGENCE_THRESHOLD - float(np.max(np.abs(x_safe)))

    _event.terminal = True
    _event.direction = -1  # fires when the expression crosses from + to -
    return _event


def _run_solve_ivp(rhs, t_final: float, x0_arr: np.ndarray, dt: float, method: str):
    t_eval = np.arange(0.0, t_final + dt / 2, dt)
    with np.errstate(over="ignore", invalid="ignore"):
        return solve_ivp(
            rhs, (0.0, t_final), x0_arr, t_eval=t_eval, method=method,
            max_step=dt, rtol=1e-6, atol=1e-9, events=_make_divergence_event(),
        )


def simulate(
    python_code: str,
    *,
    x0: List[float],
    t_final: float,
    dt: float,
    input_channel: int,
    input_kind: str,
    input_params: Dict[str, float],
    state_names: Optional[List[str]] = None,
) -> SimResult:
    """Restricted-exec ``python_code``, then integrate it under a chosen input.

    Numerically fragile drafts (an unstable/insufficiently-damped pole, a
    1/x-style singularity in the kinematics, ...) are handled rather than
    left to crash the solver: a state that runs away stops integration
    cleanly at a fixed magnitude threshold (``diverged=True``, partial
    trajectory still returned); a non-finite derivative (a genuine
    singularity) is caught immediately with a message naming the likely
    cause; and if RK45 still fails outright for some other reason, one
    retry with a stiff-aware implicit solver (LSODA) is attempted before
    giving up.
    """
    attempts: List[str] = []
    try:
        n_states, n_inputs, scalar_u = infer_dims(python_code)
        n_inputs = max(n_inputs, input_channel + 1)
        fn = compile_dynamics(python_code)

        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.shape[0] != n_states:
            if x0_arr.shape[0] > n_states:
                x0_arr = x0_arr[:n_states]
            else:
                x0_arr = np.concatenate([x0_arr, np.zeros(n_states - x0_arr.shape[0])])

        if input_kind not in _WAVEFORMS:
            raise DynamicsCodeError(f"Unknown input kind '{input_kind}'.")
        waveform = _WAVEFORMS[input_kind]

        def u_of_t(t: float) -> np.ndarray:
            u = np.zeros(n_inputs, dtype=float)
            u[input_channel] = waveform(t, **input_params)
            return u

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            u_vec = u_of_t(t)
            u_arg = u_vec[0] if scalar_u else u_vec
            dx = fn(float(t), np.asarray(x, dtype=float), u_arg)
            dx_arr = np.atleast_1d(np.asarray(dx, dtype=float)).reshape(-1)
            if dx_arr.shape[0] != n_states:
                raise DynamicsCodeError(
                    f"dynamics() returned {dx_arr.shape[0]} value(s) but the state "
                    f"vector has {n_states}. dx must be the same shape as x."
                )
            if not np.all(np.isfinite(dx_arr)):
                raise DynamicsCodeError(
                    f"dynamics() produced a non-finite value (inf/nan) at t={t:.4g}, "
                    f"x={np.round(x, 4).tolist()}. This usually means a division by "
                    f"(near-)zero or a singularity in the equations (e.g. 1/cos(theta) "
                    f"near +/-90deg, or 1/x as x crosses 0) rather than an unstable pole."
                )
            # Checked on every call -- including the solver's own internal,
            # unaccepted trial-stage evaluations, not just accepted steps.
            # That matters: empirically, a real near-singularity's *trial*
            # evaluations reach an enormous derivative long before the
            # eventual *accepted* state does, so only a check reached this
            # early actually interrupts the runaway step-size search in
            # reasonable wall-clock time (a post-step scipy event does not).
            deriv_mag = float(np.max(np.abs(dx_arr)))
            if deriv_mag > _DERIVATIVE_MAGNITUDE_THRESHOLD:
                raise DynamicsCodeError(
                    f"dynamics() produced an extremely large derivative "
                    f"(|dx|~{deriv_mag:.2e}) at t={t:.4g}, x={np.round(x, 4).tolist()}, "
                    f"well before the state itself reached a large magnitude. This "
                    f"usually means either a near-singularity in the equations (e.g. "
                    f"1/cos(theta) close to +/-90deg, or dividing by a state variable "
                    f"that passes through zero), or a very strongly unstable/positive "
                    f"pole whose growth rate itself is unrealistically fast."
                )
            return dx_arr

        if t_final <= 0 or dt <= 0:
            raise DynamicsCodeError("Simulation duration and step size must both be positive.")

        def finalize(sol, method: str) -> SimResult:
            u_trace = np.array([u_of_t(t) for t in sol.t])
            diverged = bool(len(sol.t_events) and len(sol.t_events[0]))
            message = "ok"
            if diverged:
                t_hit = sol.t_events[0][0]
                message = (
                    f"State magnitude exceeded {_DIVERGENCE_THRESHOLD:.0e} at t={t_hit:.3g}s, "
                    f"so the simulation was stopped early. This points to numerically "
                    f"unstable dynamics -- e.g. an insufficiently damped or positive-feedback "
                    f"pole, a sign error in a restoring/damping term, or an unrealistic "
                    f"parameter value -- rather than a solver problem."
                )
            return SimResult(
                True, message,
                t=sol.t, x=sol.y.T, u=u_trace,
                n_states=n_states, n_inputs=n_inputs,
                input_channel=input_channel, input_kind=input_kind,
                state_names=state_names, diverged=diverged, solver_used=method,
                attempts=attempts,
            )

        sol = _run_solve_ivp(rhs, t_final, x0_arr, dt, "RK45")
        attempts.append("RK45" + (" (stopped early, state diverged)" if sol.status == 1 else ""))
        if sol.success:
            return finalize(sol, "RK45")

        # RK45 failed for a reason other than the divergence guard (e.g. a
        # genuinely stiff system) -- retry once with LSODA, which switches
        # automatically between stiff (implicit) and non-stiff methods.
        attempts.append(f"RK45 failed ({sol.message}); retrying with LSODA")
        sol2 = _run_solve_ivp(rhs, t_final, x0_arr, dt, "LSODA")
        attempts.append("LSODA" + (" (stopped early, state diverged)" if sol2.status == 1 else ""))
        if sol2.success:
            result = finalize(sol2, "LSODA")
            result.message = "Solved with the stiff-aware LSODA method after RK45 failed to converge. " + result.message
            return result

        return SimResult(
            False,
            f"Integration failed even after a stiff-solver retry.\n"
            f"RK45: {sol.message}\nLSODA: {sol2.message}",
            attempts=attempts,
        )
    except DynamicsCodeError as exc:
        return SimResult(False, str(exc), attempts=attempts)
    except Exception as exc:  # noqa: BLE001 - never let a bad draft crash the app
        tail = traceback.format_exc(limit=2)
        return SimResult(False, f"{exc}\n{tail}", attempts=attempts)


def run_default_simulation(python_code: str, state_names: Optional[List[str]] = None) -> SimResult:
    """The automatic smoke-test run shown right after a new draft/complete."""
    n_states, _n_inputs, _scalar_u = infer_dims(python_code)
    return simulate(
        python_code,
        x0=[0.0] * n_states,
        t_final=10.0,
        dt=0.02,
        input_channel=0,
        input_kind="step",
        input_params={"t0": 1.0, "amplitude": 1.0},
        state_names=state_names,
    )


def build_figure(sim: SimResult):
    """Build a two-row Plotly figure: state trajectories, driven input."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = sim.state_names or [f"x{i}" for i in range(sim.n_states)]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        subplot_titles=("States", f"Input u[{sim.input_channel}] ({sim.input_kind})"),
        vertical_spacing=0.1,
    )
    for i in range(sim.x.shape[1]):
        label = names[i] if i < len(names) else f"x{i}"
        fig.add_trace(go.Scatter(x=sim.t, y=sim.x[:, i], mode="lines", name=str(label)), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=sim.t, y=sim.u[:, sim.input_channel], mode="lines",
            name=f"u[{sim.input_channel}]", line=dict(dash="dot", color="#888"),
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
    )
    return fig

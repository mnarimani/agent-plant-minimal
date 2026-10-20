"""Blocks with discrete memory as DiscreteBlock templates (see blocks.DiscreteBlock).

Simulink runs a model in major steps. In each, the Output phase computes every block's output,
then the Update phase stores the blocks' memory; the solver's stages in between see that memory
unchanged. Each block here is its Simulink semantics in that form: the outputs of the stages
(``outputs``), of the Output phase (``major``, when different) and the new memory (``updates``),
as expressions over t, the input in1 (e1..en for a vector block) and the memory m1..mK.

    continuous sample time (every major step):  Memory, Derivative, Rate Limiter, Relay, Backlash
    discrete sample time [Ts, offset]:          Unit Delay, Zero-Order Hold, Discrete-Time
                                                Integrator (Forward Euler), Discrete Transfer
                                                Fcn, Discrete State-Space

A discrete block acts at its sample hits t = offset + k*Ts (``hit``), and holds its output in
between, also in the stages of the steps between two hits. A sample time -1 (inherited) is
resolved after the model is wired (sample_times.resolve).
Every block here is checked against Simulink (tests/test_slx_blocks.py, group C).
"""
from __future__ import annotations

import numpy as np

from .blocks import DiscreteBlock
from .dynamics_expressions import Expr, constant, linear_combination, operation, symbol

T = symbol("t")
IN = symbol("in1")


class DiscreteSemanticsError(ValueError):
    pass


def _m(k: int) -> Expr:
    return symbol(f"m{k}")


def _select(condition: Expr, yes: Expr, no: Expr) -> Expr:
    return operation("select", condition, yes, no)


def _sampled(block: DiscreteBlock, sample: tuple[float, float]) -> DiscreteBlock:
    block.sample = sample
    return block


def _hit(sample: tuple[float, float]) -> Expr:
    return operation("ne", operation("hit", T, constant(sample[0]), constant(sample[1])), constant(0.0))


def sample_time(value, name: str) -> tuple[float, float] | None:
    """[Ts] or [Ts, offset] of a discrete block; None: inherited (-1), resolved once the model is
    wired (sample_times.resolve). Continuous (0) and constant (inf) sample times are not supported."""
    values = np.real(np.asarray(value, dtype=float)).ravel()
    if values.size not in (1, 2):
        raise DiscreteSemanticsError(f"Sample time of '{name}' must be Ts or [Ts, offset].")
    ts, offset = float(values[0]), float(values[1]) if values.size == 2 else 0.0
    if ts == -1:
        if offset != 0:
            raise DiscreteSemanticsError(f"Sample time {values.tolist()} of '{name}' is not supported.")
        return None
    if not np.isfinite(ts) or ts <= 0 or not (0 <= offset < ts):
        raise DiscreteSemanticsError(f"Sample time {values.tolist()} of '{name}' is not supported (Ts > 0, "
                                     "0 <= offset < Ts).")
    return ts, offset


def memory(block_id: str, name: str, initial) -> DiscreteBlock:
    """Memory (continuous sample time): the input of the previous major step, held over the whole
    step (also in its stages). m1: the output held, m2: the input stored at the last major step."""
    return DiscreteBlock(block_id, name, "Memory", 1, [_m(1)], [_m(2), IN], [initial, initial], major=[_m(2)],
                         feedthrough=False, labels=["output held (input of the major step before)",
                                                    "input at the last major step"])


def derivative(block_id: str, name: str, start_time: float) -> DiscreteBlock:
    """Derivative: (u - u_last) / (t - t_last) with the latest stored major step strictly before
    t (Simulink keeps two), 0 before there is one. m1, m2: time and input of the last major step,
    m3, m4: of the one before, m5: how many are stored (0, 1, 2)."""
    def slope(time: Expr, value: Expr) -> Expr:
        return operation("div", operation("sub", IN, value), operation("sub", T, time))

    first = operation("and", operation("ge", _m(5), constant(1.0)), operation("lt", _m(1), T))
    second = operation("and", operation("ge", _m(5), constant(2.0)), operation("lt", _m(3), T))
    output = _select(first, slope(_m(1), _m(2)), _select(second, slope(_m(3), _m(4)), constant(0.0)))
    return DiscreteBlock(block_id, name, "Derivative", 1, [output],
                         [T, IN, _m(1), _m(2), operation("min", operation("add", _m(5), constant(1.0)), constant(2.0))],
                         [start_time - 1.0, 0.0, start_time - 1.0, 0.0, 0.0], feedthrough=True,
                         labels=["time of the last major step", "input at the last major step",
                                 "time of the major step before", "input at the major step before",
                                 "major steps stored (0-2)"])


def rate_limiter(block_id: str, name: str, rising: float, falling: float, initial, start_time: float,
                 step: float) -> DiscreteBlock:
    """Rate Limiter: the change since the output of the last major step is limited to
    rising * dt / falling * dt (dt: the time since that step). Before the first step Simulink
    starts from the initial condition one fixed step before the start. m1: time of the last
    major step, m2: the output then."""
    if rising < falling:
        raise DiscreteSemanticsError(f"Rate Limiter '{name}': the rising slew rate is below the falling one.")
    dt = operation("sub", T, _m(1))
    change = operation("sub", IN, _m(2))
    up, down = operation("mul", constant(rising), dt), operation("mul", constant(falling), dt)
    output = _select(operation("gt", change, up), operation("add", _m(2), up),
                     _select(operation("lt", change, down), operation("add", _m(2), down), IN))
    return DiscreteBlock(block_id, name, "Rate Limiter", 1, [output], [T, output],
                         [start_time - step, initial], feedthrough=True,
                         labels=["time of the last major step", "output at the last major step"])


def relay(block_id: str, name: str, on: float, off: float, on_value: float, off_value: float) -> DiscreteBlock:
    """Relay: switches on when the input reaches the on point, off when it reaches the off
    point, at major steps (the stages keep the state of the step). m1: 1 = on, 0 = off."""
    if on < off:
        raise DiscreteSemanticsError(f"Relay '{name}': the switch-on point is below the switch-off point.")
    state = _select(operation("ge", IN, constant(on)), constant(1.0),
                    _select(operation("le", IN, constant(off)), constant(0.0), _m(1)))

    def value(of: Expr) -> Expr:
        return _select(operation("ne", of, constant(0.0)), constant(on_value), constant(off_value))

    return DiscreteBlock(block_id, name, "Relay", 1, [value(_m(1))], [state], [0.0], major=[value(state)],
                         feedthrough=True, labels=["on (1) / off (0)"])


def backlash(block_id: str, name: str, width: float, initial) -> DiscreteBlock:
    """Backlash (play of width w): the output follows the input once the input is more than w/2
    away from the output of the last major step, else it stays there. m1: that output."""
    half = constant(width / 2.0)
    output = _select(operation("lt", IN, operation("sub", _m(1), half)), operation("add", IN, half),
                     _select(operation("le", IN, operation("add", _m(1), half)), _m(1), operation("sub", IN, half)))
    return DiscreteBlock(block_id, name, "Backlash", 1, [output], [output], [initial], feedthrough=True,
                         labels=["output at the last major step"])


def unit_delay(block_id: str, name: str, initial, sample: tuple[float, float]) -> DiscreteBlock:
    """Unit Delay: at a hit the output is the input of the previous hit. m1: the output held,
    m2: the input stored at the last hit."""
    hit = _hit(sample)
    return _sampled(DiscreteBlock(block_id, name, "Unit Delay", 1, [_m(1)],
                         [_select(hit, _m(2), _m(1)), _select(hit, IN, _m(2))], [initial, initial],
                         major=[_select(hit, _m(2), _m(1))], feedthrough=False,
                         labels=["output held", "input at the last sample hit"]), sample)


def zero_order_hold(block_id: str, name: str, sample: tuple[float, float]) -> DiscreteBlock:
    """Zero-Order Hold: the input at the last hit. m1: the value held."""
    hit = _hit(sample)
    return _sampled(DiscreteBlock(block_id, name, "Zero-Order Hold", 1, [_m(1)], [_select(hit, IN, _m(1))], [0.0],
                         major=[_select(hit, IN, _m(1))], feedthrough=True, labels=["value held"]), sample)


def discrete_integrator(block_id: str, name: str, gain: float, initial, sample: tuple[float, float]) -> DiscreteBlock:
    """Discrete-Time Integrator, Forward Euler: y(n) = x(n), x(n+1) = x(n) + K*Ts*u(n).
    m1: the output held, m2: the state x."""
    hit = _hit(sample)
    step = operation("add", _m(2), operation("mul", constant(gain * sample[0]), IN))
    return _sampled(DiscreteBlock(block_id, name, "Discrete-Time Integrator", 1, [_m(1)],
                         [_select(hit, _m(2), _m(1)), _select(hit, step, _m(2))], [initial, initial],
                         major=[_select(hit, _m(2), _m(1))], feedthrough=False,
                         labels=["output held", "integrator state"]), sample)


def discrete_state_space(block_id: str, name: str, kind: str, a, b, c, d, initial,
                         sample: tuple[float, float]) -> DiscreteBlock:
    """x(n+1) = A x(n) + B u(n), y(n) = C x(n) + D u(n) at the hits (Discrete State-Space, and
    the Discrete Transfer Fcn in this form). Memory: the outputs held (p), then the states x (n)."""
    a, b, c, d = (np.atleast_2d(np.asarray(v, dtype=float)) for v in (a, b, c, d))
    n = 0 if a.size == 0 else a.shape[0]
    m = d.shape[1] if b.size == 0 else b.shape[1]
    p = d.shape[0] if c.size == 0 else c.shape[0]
    a, b, c = a.reshape(n, n), b.reshape(n, m), c.reshape(p, n)
    if d.shape != (p, m) or not all(np.all(np.isfinite(v)) for v in (a, b, c, d)):
        raise DiscreteSemanticsError(f"The matrices of '{name}' have inconsistent sizes or non-finite values.")
    x0 = np.zeros(n) if np.asarray(initial).size == 1 and float(np.asarray(initial).ravel()[0]) == 0 else \
        np.broadcast_to(np.asarray(initial, dtype=float).ravel(), (n,)).copy()
    hit = _hit(sample)
    held = [_m(i + 1) for i in range(p)]
    states = [_m(p + i + 1) for i in range(n)]
    inputs = [symbol(f"e{j + 1}") for j in range(m)]

    def row(matrix_x, matrix_u, i):
        return operation("add", linear_combination(matrix_x[i], states) if n else constant(0.0),
                         linear_combination(matrix_u[i], inputs) if m else constant(0.0))

    y = [row(c, d, i) for i in range(p)]
    x_next = [row(a, b, i) for i in range(n)]
    major = [_select(hit, y[i], held[i]) for i in range(p)]
    updates = major + [_select(hit, x_next[i], states[i]) for i in range(n)]
    initial_held = (c @ x0 if n else np.zeros(p)).tolist()
    return _sampled(DiscreteBlock(block_id, name, kind, 1, held, updates, [*initial_held, *x0.tolist()], major=major,
                         feedthrough=bool(np.any(d != 0)), vector=(m, p),
                         labels=[f"output {i + 1} held" for i in range(p)] + [f"state {i + 1}" for i in range(n)]), sample)


def discrete_transfer_fcn(block_id: str, name: str, numerator, denominator, sample: tuple[float, float]) -> DiscreteBlock:
    """Discrete Transfer Fcn num(z)/den(z) (zero initial states), as a discrete state space."""
    from scipy.signal import tf2ss
    num = np.atleast_1d(np.asarray(numerator, dtype=float)).ravel()
    den = np.atleast_1d(np.asarray(denominator, dtype=float)).ravel()
    if den.size == 0 or den[0] == 0:
        raise DiscreteSemanticsError(f"The leading denominator coefficient of '{name}' must not be zero.")
    if num.size > den.size:
        raise DiscreteSemanticsError(f"Discrete Transfer Fcn '{name}' is not proper.")
    a, b, c, d = tf2ss(num, den)
    return discrete_state_space(block_id, name, "Discrete Transfer Fcn", a, b, c, d, 0.0, sample)

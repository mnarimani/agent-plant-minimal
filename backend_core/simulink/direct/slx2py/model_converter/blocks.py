from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from collections.abc import Mapping, Sequence
import math
import re
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import tf2ss

from .dynamics_expressions import Expr, constant, fcn_python, parse_expression, symbol
from .stateflow import MatlabVector, StateflowEngine, compile_matlab_function
from .stateflow_machine import compile_state_machine
from .waveforms import pulse_value


FloatVector: TypeAlias = NDArray[np.float64]
SignalValue: TypeAlias = float | FloatVector


class Block(ABC):
    direct_feedthrough = True
    state_size = 0
    scalar_only = False          # True: every signal of the block must be a scalar
    elementwise = True           # inputs of width 1 or of the output's width (scalar expansion)

    def __init__(self, block_id: str, name: str, input_count: int, output_count: int) -> None:
        self.block_id = block_id
        self.name = name
        self.input_count = input_count
        self.output_count = output_count
        self._state_start: int | None = None

    def bind_state(self, state_start: int) -> None:
        if self.state_size:
            self._state_start = state_start

    def initial_state(self) -> FloatVector:
        return np.zeros(self.state_size, dtype=float)

    def state_view(self, state: FloatVector) -> FloatVector:
        if self.state_size == 0 or self._state_start is None:
            raise RuntimeError(f"Block '{self.block_id}' has no continuous state.")
        return state[self._state_start : self._state_start + self.state_size]

    @abstractmethod
    def output(
        self,
        time: float,
        state: FloatVector,
        inputs: Sequence[SignalValue],
        external_inputs: Mapping[str, float],
    ) -> tuple[SignalValue, ...]:
        pass

    def derivatives(
        self,
        time: float,
        state: FloatVector,
        inputs: Sequence[SignalValue],
        external_inputs: Mapping[str, float],
    ) -> FloatVector:
        return np.zeros(self.state_size, dtype=float)

    def initial_state_from_inputs(self, inputs: Sequence[SignalValue]) -> FloatVector | None:
        """Return an input-provided initial state, when the block supports one.

        Most continuous blocks use only their configured initial-state
        parameter, so the default deliberately has no effect.
        """
        return None

    # ---- signal widths (1 = scalar), resolved by Model.infer_widths before the states are bound
    def infer_output_widths(self, input_widths: Sequence[int | None]) -> list[int | None]:
        """The widths of the outputs for these input widths (None = not known yet). Default:
        element-wise with scalar expansion, like most Simulink blocks."""
        if self.input_count == 0:
            return [1] * self.output_count
        if any(width is None for width in input_widths):
            return [None] * self.output_count
        return [max(input_widths)] * self.output_count

    def apply_widths(self, input_widths: Sequence[int], output_widths: Sequence[int]) -> None:
        """Check the resolved widths and size the block for them (e.g. its states)."""
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)
        if self.scalar_only and any(width != 1 for width in (*input_widths, *output_widths)):
            raise ValueError(f"Block '{self.name}' ({type(self).__name__}) supports scalar signals only; "
                             f"its signals have widths {list(input_widths)} -> {list(output_widths)}.")
        width = max(output_widths, default=1)
        for port, value in enumerate(input_widths):
            if self.elementwise and value not in (1, width):
                raise ValueError(f"Block '{self.name}': input {port + 1} has width {value}, the output {width}.")


def element_names(name: str, width: int) -> list[str]:
    """The names of the elements of a model input / output: In1, or In1(1), In1(2) ... for a vector"""
    return [name] if width == 1 else [f"{name}({i + 1})" for i in range(width)]


class Inport(Block):
    direct_feedthrough = False

    def __init__(self, block_id: str, name: str, input_key: str | None = None, width: int = 1,
                 inherited: bool = False) -> None:
        super().__init__(block_id, name, input_count=0, output_count=1)
        self.input_key = input_key or name
        self.width = int(width)
        self.inherited = bool(inherited)      # no dimensions in the model: taken as a scalar

    def infer_output_widths(self, input_widths):
        return [self.width]

    @property
    def element_names(self) -> list[str]:
        return element_names(self.input_key, self.width)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        try:
            values = [float(external_inputs[key]) for key in self.element_names]
        except KeyError as error:
            raise KeyError(f"Missing external input '{error.args[0]}'.") from error
        return (values[0] if self.width == 1 else np.asarray(values, dtype=float),)


class Outport(Block):
    def __init__(self, block_id: str, name: str) -> None:
        super().__init__(block_id, name, input_count=1, output_count=1)

    @property
    def width(self) -> int:
        return getattr(self, "input_widths", [1])[0]

    @property
    def element_names(self) -> list[str]:
        return element_names(self.name, self.width)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        value = np.asarray(inputs[0], dtype=float)
        return (float(value) if value.ndim == 0 else value.ravel().copy(),)


class MatlabFunction(Block):
    """A MATLAB Function block executed through the independent Stateflow engine."""

    def __init__(self, block_id: str, name: str, script: str) -> None:
        self.script = script
        self.chart = compile_matlab_function(script)
        super().__init__(block_id, name, input_count=len(self.chart.input_names),
                         output_count=len(self.chart.output_names))
        self.input_names = self.chart.input_names
        self.output_names = self.chart.output_names
        self.output_name = self.chart.output_names[0]
        self._engine = StateflowEngine()

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        # a vector input (from a Mux) is indexed one-based in the code: u(1)
        values = {name: value if np.ndim(value) == 0 else MatlabVector(np.asarray(value, dtype=float))
                  for name, value in zip(self.input_names, inputs, strict=True)}
        result = self._engine.evaluate(self.chart, values, time)
        return tuple(_signal(result[name]) for name in self.output_names)

    def infer_output_widths(self, input_widths):
        """the sizes of the outputs: run the code once with inputs of these widths"""
        if any(width is None for width in input_widths):
            return [None] * self.output_count
        probe = [0.37 + 0.11 * i if width == 1 else np.linspace(0.31, 0.83, width)
                 for i, width in enumerate(input_widths)]
        try:
            outputs = self.output(0.0, np.zeros(0), probe, {})
        except Exception:                     # noqa: BLE001 - a probe value the code cannot take
            return [1] * self.output_count
        return [int(np.size(value)) for value in outputs]

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)


class StateflowChart(Block):
    scalar_only = True
    """A multi-output Stateflow Chart block backed by the Stateflow engine."""

    def __init__(self, block_id: str, name: str, chart_xml: str) -> None:
        self.chart_xml = chart_xml
        self.chart = compile_state_machine(chart_xml)
        super().__init__(block_id, name, input_count=len(self.chart.input_names), output_count=len(self.chart.output_names))
        self.input_names = self.chart.input_names
        self.output_names = self.chart.output_names
        self._engine = StateflowEngine()
        self._trigger_value: FloatVector | None = None
        self._pending_events: tuple[str, ...] = ()
        self._cached_outputs = self.chart.initial_outputs()
        self.allow_updates = True

    def reset_runtime(self):
        self.chart.reset()
        self._trigger_value = None
        self._pending_events = ()
        self._cached_outputs = self.chart.initial_outputs()

    def set_trigger_value(self, value: SignalValue) -> None:
        """Observe zero crossings using each event's declared edge policy."""
        if not self.allow_updates:
            return
        current = np.atleast_1d(np.asarray(value, dtype=float)).copy()
        if current.shape != (len(self.chart.event_names),):
            raise ValueError(f"Stateflow trigger width differs from event count on '{self.name}'.")
        if self._trigger_value is None:
            previous = np.zeros_like(current)
        else:
            if self._trigger_value.shape != current.shape:
                raise ValueError(f"Stateflow trigger shape changed on '{self.name}'.")
            previous = self._trigger_value
        rising = ((previous < 0) & (current >= 0)) | ((previous == 0) & (current > 0))
        falling = ((previous > 0) & (current <= 0)) | ((previous == 0) & (current < 0))
        changed = [(rising[i] if trigger == "RISING_EDGE_EVENT" else falling[i] if trigger == "FALLING_EDGE_EVENT" else rising[i] or falling[i])
                   for i, trigger in enumerate(self.chart.event_triggers)]
        self._trigger_value = current
        self._pending_events = tuple(
            self.chart.event_names[index]
            for index in np.flatnonzero(changed)
            if index < len(self.chart.event_names) and self.chart.event_names[index]
        )

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        if not self.allow_updates:
            return tuple(float(self._cached_outputs[name]) for name in self.output_names)
        values = {name: _scalar(value) for name, value in zip(self.input_names, inputs, strict=True)}
        if self.allow_updates:
            self._cached_outputs = self._engine.evaluate(self.chart, values, time, self._pending_events)
        self._pending_events = ()
        return tuple(float(self._cached_outputs[name]) for name in self.output_names)


class Step(Block):
    direct_feedthrough = False
    scalar_only = True

    def __init__(
        self,
        block_id: str,
        name: str,
        step_time: float = 1.0,
        initial_value: float = 0.0,
        final_value: float = 1.0,
    ) -> None:
        super().__init__(block_id, name, input_count=0, output_count=1)
        self.step_time = float(step_time)
        self.initial_value = float(initial_value)
        self.final_value = float(final_value)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        value = self.final_value if time >= self.step_time else self.initial_value
        return (float(value),)


class PulseGenerator(Block):
    scalar_only = True
    """Scalar, simulation-time pulse with explicit rising and falling edges."""
    direct_feedthrough = False

    def __init__(self, block_id, name, amplitude=1., period=10., width=5., delay=0.):
        if not all(math.isfinite(v) for v in (amplitude, period, width, delay)) or period <= 0 or not 0 < width < 100 or delay < 0:
            raise ValueError("Time-based Pulse Generator requires finite parameters, period > 0, 0 < width < 100 and delay >= 0.")
        super().__init__(block_id, name, input_count=0, output_count=1)
        self.amplitude, self.period, self.width, self.delay = map(float, (amplitude, period, width, delay))

    def output(self, time, state, inputs, external_inputs):
        return (pulse_value(time, self.amplitude, self.period, self.width, self.delay),)

    def transition_times(self, start, stop):
        if self.amplitude == 0:
            return ()
        first = max(0, math.floor((start - self.delay) / self.period) - 1)
        last = math.floor((stop - self.delay) / self.period)
        if 2 * (last - first + 1) > 200000:
            raise ValueError("Pulse Generator exceeds 200000 events; shorten the simulation range.")
        return tuple(edge for cycle in range(first, last + 1)
                     for edge in (self.delay + cycle * self.period, self.delay + cycle * self.period + self.period * self.width / 100)
                     if start < edge <= stop)


class Constant(Block):
    direct_feedthrough = False

    def __init__(self, block_id: str, name: str, value: float | Sequence[float]) -> None:
        super().__init__(block_id, name, input_count=0, output_count=1)
        values = np.asarray(value, dtype=float).ravel()
        self.value = float(values[0]) if values.size == 1 else values

    def infer_output_widths(self, input_widths):
        return [int(np.size(self.value))]

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (self.value if np.ndim(self.value) == 0 else self.value.copy(),)


class Sum(Block):
    def __init__(self, block_id: str, name: str, signs: Sequence[float]) -> None:
        if not signs:
            raise ValueError("Sum requires at least one input sign.")
        super().__init__(block_id, name, input_count=len(signs), output_count=1)
        self.signs = np.asarray(signs, dtype=float)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        if len(self.signs) == 1:
            # one input: the sum of its elements (Sum of Elements)
            return (float(self.signs[0] * np.sum(np.asarray(inputs[0], dtype=float))),)
        # element-wise; a scalar input is expanded to the width of the vectors
        total = sum(sign * np.asarray(value, dtype=float) for sign, value in zip(self.signs, inputs, strict=True))
        return (_signal(total),)

    def infer_output_widths(self, input_widths):
        return [1] if len(self.signs) == 1 else super().infer_output_widths(input_widths)

    @property
    def elementwise(self) -> bool:
        return len(self.signs) > 1


GAIN_MODES = ("Element-wise(K.*u)", "Matrix(K*u)", "Matrix(u*K)", "Matrix(K*u) (u vector)")


class Gain(Block):
    """Gain: a scalar or a vector (element-wise), or a matrix (K*u / u*K)."""

    def __init__(self, block_id: str, name: str, gain: float | Sequence, multiplication: str = "Element-wise(K.*u)") -> None:
        super().__init__(block_id, name, input_count=1, output_count=1)
        if multiplication not in GAIN_MODES:
            raise ValueError(f"Gain multiplication {multiplication!r} is not supported.")
        self.multiplication = "Matrix(K*u)" if multiplication == "Matrix(K*u) (u vector)" else multiplication
        k = np.asarray(gain, dtype=float)
        if self.multiplication == "Element-wise(K.*u)":
            k = k.ravel()
            self.gain = float(k[0]) if k.size == 1 else k
        else:
            self.gain = np.atleast_2d(k)

    def infer_output_widths(self, input_widths):
        if self.multiplication == "Matrix(K*u)":
            return [self.gain.shape[0]]
        if self.multiplication == "Matrix(u*K)":
            return [self.gain.shape[1]]
        if input_widths[0] is None:
            return [None]
        return [max(int(np.size(self.gain)), input_widths[0])]

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)
        size = {"Matrix(K*u)": lambda: self.gain.shape[1], "Matrix(u*K)": lambda: self.gain.shape[0]}
        if self.multiplication in size and input_widths[0] != size[self.multiplication]():
            raise ValueError(f"Gain '{self.name}': the matrix needs an input of width {size[self.multiplication]()}, "
                             f"got {input_widths[0]}.")
        if self.multiplication == "Element-wise(K.*u)" and np.size(self.gain) not in (1, output_widths[0]):
            raise ValueError(f"Gain '{self.name}': the gain has {np.size(self.gain)} elements, the input {input_widths[0]}.")

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        u = np.asarray(inputs[0], dtype=float)
        if self.multiplication == "Matrix(K*u)":
            return (_signal(self.gain @ np.atleast_1d(u)),)
        if self.multiplication == "Matrix(u*K)":
            return (_signal(np.atleast_1d(u) @ self.gain),)
        return (_signal(self.gain * u),)


class Integrator(Block):
    """Integrator, optionally limited ("Limit output"): the output stays in [lower, upper] and
    the state stops while it is at a limit and the input pushes it further out. Simulink also
    resets the state to the limit at each major step; an ODE cannot, so in the step that
    reaches a limit the state may pass it by up to one step of the input (the output does not)."""

    direct_feedthrough = False
    state_size = 1

    def __init__(
        self,
        block_id: str,
        name: str,
        initial_value: float | Sequence[float] = 0.0,
        external_initial_condition: bool = False,
        lower_limit: float = -math.inf,
        upper_limit: float = math.inf,
    ) -> None:
        self.external_initial_condition = bool(external_initial_condition)
        super().__init__(block_id, name, input_count=2 if self.external_initial_condition else 1, output_count=1)
        values = np.asarray(initial_value, dtype=float).ravel()
        self.initial_value = float(values[0]) if values.size == 1 else values
        self.state_size = int(values.size)     # the width of the signal, set again by apply_widths
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        if self.lower_limit > self.upper_limit:
            raise ValueError("Integrator lower saturation limit exceeds the upper limit.")

    @property
    def limited(self) -> bool:
        return math.isfinite(self.lower_limit) or math.isfinite(self.upper_limit)

    def infer_output_widths(self, input_widths):
        if np.size(self.initial_value) > 1:
            return [int(np.size(self.initial_value))]
        return [input_widths[0]]

    def apply_widths(self, input_widths, output_widths):
        super().apply_widths(input_widths, output_widths)
        if np.size(self.initial_value) not in (1, output_widths[0]):
            raise ValueError(f"Integrator '{self.name}': initial condition of {np.size(self.initial_value)} "
                             f"elements for a signal of width {output_widths[0]}.")
        self.state_size = output_widths[0]

    def _clip(self, values: FloatVector) -> FloatVector:
        return np.clip(values, self.lower_limit, self.upper_limit)

    def initial_state(self) -> FloatVector:
        return self._clip(np.broadcast_to(np.asarray(self.initial_value, dtype=float), (self.state_size,)).astype(float))

    def initial_state_from_inputs(self, inputs: Sequence[SignalValue]) -> FloatVector | None:
        if not self.external_initial_condition:
            return None
        return self._clip(np.broadcast_to(np.asarray(inputs[1], dtype=float), (self.state_size,)).astype(float))

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (_signal(self._clip(self.state_view(state))),)

    def derivatives(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> FloatVector:
        rate = np.broadcast_to(np.asarray(inputs[0], dtype=float), (self.state_size,)).astype(float)
        if self.limited:
            value = self.state_view(state)
            rate[((value >= self.upper_limit) & (rate > 0)) | ((value <= self.lower_limit) & (rate < 0))] = 0.0
        return rate


class SecondOrderIntegrator(Block):
    """Integrator, Second-Order: states (x, dx/dt), input d2x/dt2; outputs x and/or dx/dt."""

    scalar_only = True

    direct_feedthrough = False
    state_size = 2

    def __init__(self, block_id: str, name: str, initial_x: float = 0.0, initial_dxdt: float = 0.0,
                 show_output: str = "both") -> None:
        if show_output not in {"both", "x", "dxdt"}:
            raise ValueError(f"Unsupported Second-Order Integrator output {show_output!r}.")
        self.initial_x = float(initial_x)
        self.initial_dxdt = float(initial_dxdt)
        self.show_output = show_output
        super().__init__(block_id, name, input_count=1, output_count=2 if show_output == "both" else 1)

    def initial_state(self) -> FloatVector:
        return np.array([self.initial_x, self.initial_dxdt], dtype=float)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        x, dxdt = (float(v) for v in self.state_view(state))
        return {"both": (x, dxdt), "x": (x,), "dxdt": (dxdt,)}[self.show_output]

    def derivatives(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> FloatVector:
        return np.array([float(self.state_view(state)[1]), _scalar(inputs[0])], dtype=float)


class StateSpace(Block):
    """Continuous state space dx/dt = A x + B u, y = C x + D u (State-Space, and the PID
    Controller written in this form). u and y are vectors when B / C have several columns / rows."""

    def __init__(self, block_id: str, name: str, a, b, c, d, initial_condition: float | Sequence[float] = 0.0) -> None:
        a, b, c, d = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (a, b, c, d))
        n = 0 if a.size == 0 else a.shape[0]
        a = a.reshape(n, n)
        m = d.shape[1] if b.size == 0 else b.shape[1]
        p = d.shape[0] if c.size == 0 else c.shape[0]
        b, c = b.reshape(n, m), c.reshape(p, n)
        if a.shape != (n, n) or d.shape != (p, m) or not all(np.all(np.isfinite(x)) for x in (a, b, c, d)):
            raise ValueError("State-Space matrices have inconsistent sizes or non-finite values.")
        self.a, self.b, self.c, self.d = a, b, c, d
        self.input_width, self.output_width = m, p
        self.state_size = n
        self.direct_feedthrough = bool(np.any(d != 0.0))
        self.initial_conditions = _initial_conditions(initial_condition, n) if n else np.zeros(0)
        super().__init__(block_id, name, input_count=1, output_count=1)

    def initial_state(self) -> FloatVector:
        return self.initial_conditions.copy()

    def infer_output_widths(self, input_widths):
        return [self.output_width]

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)
        if input_widths[0] != self.input_width:
            raise ValueError(f"State-Space '{self.name}' expects an input of width {self.input_width}, got {input_widths[0]}.")

    def _input(self, value: SignalValue) -> FloatVector:
        u = np.atleast_1d(np.asarray(value, dtype=float))
        if u.size != self.input_width:
            raise ValueError(f"State-Space '{self.name}' expects an input of width {self.input_width}, got {u.size}.")
        return u

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        x = self.state_view(state) if self.state_size else np.zeros(0)
        y = self.c @ x
        if self.direct_feedthrough:
            y = y + self.d @ self._input(inputs[0])
        return (float(y[0]) if self.output_width == 1 else np.asarray(y, dtype=float),)

    def derivatives(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> FloatVector:
        return self.a @ self.state_view(state) + self.b @ self._input(inputs[0])


class MathBlock(Block):
    """A memoryless block given by one expression per output over its inputs in1..inN and the
    time t (block_semantics writes them from the block's type and parameters). The expressions
    are the block's whole semantics: the runtime evaluates them, and the equations substitute
    the expressions of the inputs into them (dynamics_rules)."""

    def __init__(self, block_id: str, name: str, kind: str, input_count: int, outputs: Sequence,
                 reduce: str | None = None) -> None:
        self.kind = kind
        self.templates = tuple(o if isinstance(o, Expr) else Expr.from_dict(o) for o in outputs)
        if not self.templates:
            raise ValueError(f"Block '{name}' has no output.")
        if reduce not in (None, "min", "max"):
            raise ValueError(f"Unsupported reduction {reduce!r}.")
        self.reduce = reduce                  # MinMax with one input: over the elements of a vector
        super().__init__(block_id, name, input_count=input_count, output_count=len(self.templates))
        self.direct_feedthrough = input_count > 0
        self.time_dependent = any("t" in e.symbols() for e in self.templates)

    def infer_output_widths(self, input_widths):
        return [1] * self.output_count if self.reduce else super().infer_output_widths(input_widths)

    @property
    def elementwise(self) -> bool:
        return self.reduce is None

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        if self.reduce:
            values = np.asarray(inputs[0], dtype=float)
            return (float(np.min(values) if self.reduce == "min" else np.max(values)),)
        arrays = [np.asarray(v, dtype=float) for v in inputs]
        width = max((a.size for a in arrays), default=1)
        rows = []
        with np.errstate(all="ignore"):
            for k in range(width):           # element-wise, scalars expanded
                values = {"t": float(time), **{f"in{i + 1}": float(a if a.ndim == 0 else a.ravel()[k if a.size > 1 else 0])
                                               for i, a in enumerate(arrays)}}
                rows.append([float(e.evaluate(values)) for e in self.templates])
        if width == 1 and all(a.ndim == 0 for a in arrays):
            return tuple(rows[0])
        return tuple(np.array([row[j] for row in rows], dtype=float) for j in range(self.output_count))


class DiscreteBlock(Block):
    """A block with discrete memory (Memory, Derivative, Unit Delay, Zero-Order Hold, discrete
    integrators and filters ...), given by expression templates like a MathBlock. Simulink runs
    such a block in two phases of every major solver step, and so does this one:

        major     y  = major(t, in, m)    the Output phase of a major step (default: ``outputs``)
        updates   m+ = update(t, in, m)   the Update phase right after it (Model.update)
        outputs   y  = outputs(t, in, m+) the solver's stages until the next major step

    Symbols: t, the inputs in1..inN and the memory m1..mK (both of one element: element-wise, one
    set of K memory values per element). Mode ``vector=(n, p)``: one input of width n (e1..en),
    one output of width p (one template per output element), m1..mK the whole memory.
    The memory is part of the state vector with derivative 0: between major steps it is constant.
    ``outputs`` evaluated right after the update must give what ``major`` gave (the stage at the
    time of the major step uses the outputs of the major step, as in Simulink)."""

    discrete = True

    def __init__(self, block_id: str, name: str, kind: str, input_count: int, outputs: Sequence,
                 updates: Sequence, initial: Sequence, *, major: Sequence | None = None,
                 feedthrough: bool = True, vector: Sequence[int] | None = None,
                 labels: Sequence[str] | None = None) -> None:
        def expressions(items):
            return tuple(e if isinstance(e, Expr) else Expr.from_dict(e) for e in items)

        self.kind = kind
        self.templates = expressions(outputs)
        self.major_templates = expressions(major) if major is not None else None
        self.update_templates = expressions(updates)
        self.initial = [np.asarray(v, dtype=float) for v in initial]
        self.memory_size = len(self.update_templates)
        if len(self.initial) != self.memory_size or not self.templates:
            raise ValueError(f"Block '{name}': memory and its initial values differ.")
        self.vector = tuple(int(v) for v in vector) if vector is not None else None
        self.labels = list(labels) if labels is not None else [f"memory {k + 1}" for k in range(self.memory_size)]
        output_count = 1 if self.vector else len(self.templates)
        super().__init__(block_id, name, input_count=input_count, output_count=output_count)
        self.direct_feedthrough = bool(feedthrough) and input_count > 0
        self.elementwise = self.vector is None
        self.width = 1
        self.state_size = self.memory_size

    def infer_output_widths(self, input_widths):
        if self.vector:
            return [self.vector[1]]
        return super().infer_output_widths(input_widths)

    def apply_widths(self, input_widths, output_widths):
        super().apply_widths(input_widths, output_widths)
        if self.vector:
            if input_widths[0] != self.vector[0]:
                raise ValueError(f"Block '{self.name}' expects an input of width {self.vector[0]}, got {input_widths[0]}.")
            self.width = 1
        else:
            self.width = max([*input_widths, *output_widths], default=1)
        self.state_size = self.memory_size * self.width

    def initial_state(self) -> FloatVector:
        values = np.zeros(self.state_size, dtype=float)
        for k, initial in enumerate(self.initial):
            if initial.size not in (1, self.width):
                raise ValueError(f"Block '{self.name}': initial value of width {initial.size}, the signal {self.width}.")
            values[k::self.memory_size] = initial.ravel() if initial.size > 1 else float(initial.ravel()[0])
        return values

    def _environments(self, time, state, inputs):
        memory = self.state_view(state)
        if self.vector:
            values = {"t": float(time), **{f"m{k + 1}": float(v) for k, v in enumerate(memory)}}
            if self.input_count:
                values.update({f"e{i + 1}": float(v) for i, v in enumerate(np.atleast_1d(np.asarray(inputs[0], dtype=float)))})
            return [values]
        arrays = [np.atleast_1d(np.asarray(v, dtype=float)) for v in inputs]
        rows = []
        for j in range(self.width):
            values = {"t": float(time), **{f"in{i + 1}": float(a[j] if a.size > 1 else a[0]) for i, a in enumerate(arrays)},
                      **{f"m{k + 1}": float(memory[j * self.memory_size + k]) for k in range(self.memory_size)}}
            rows.append(values)
        return rows

    def _outputs(self, templates, time, state, inputs):
        with np.errstate(all="ignore"):
            rows = [[float(e.evaluate(values)) for e in templates] for values in self._environments(time, state, inputs)]
        if self.vector:
            values = rows[0]
            return (values[0] if len(values) == 1 else np.asarray(values, dtype=float),)
        if self.width == 1 and all(np.ndim(v) == 0 for v in inputs):
            return tuple(rows[0])
        return tuple(np.array([row[j] for row in rows], dtype=float) for j in range(self.output_count))

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return self._outputs(self.templates, time, state, inputs)

    def major_output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue]) -> tuple[SignalValue, ...]:
        return self._outputs(self.major_templates or self.templates, time, state, inputs)

    def update(self, time: float, state: FloatVector, inputs: Sequence[SignalValue]) -> FloatVector:
        """the block's memory after the Update phase of the major step at ``time``"""
        with np.errstate(all="ignore"):
            return np.array([float(e.evaluate(values)) for values in self._environments(time, state, inputs)
                             for e in self.update_templates], dtype=float)


class TransferFcn(Block):
    """Continuous SISO transfer function represented in state-space form."""

    scalar_only = True

    def __init__(
        self,
        block_id: str,
        name: str,
        numerator: Sequence[float],
        denominator: Sequence[float],
        initial_condition: float | Sequence[float] = 0.0,
    ) -> None:
        self.numerator = _polynomial_coefficients("numerator", numerator)
        self.denominator = _polynomial_coefficients("denominator", denominator)
        if self.denominator[0] == 0.0:
            raise ValueError("The leading denominator coefficient must not be zero.")
        if self.numerator.size > self.denominator.size:
            raise ValueError("Only proper transfer functions are supported.")

        try:
            state_matrix, input_matrix, output_matrix, feedthrough = tf2ss(
                self.numerator,
                self.denominator,
            )
        except ValueError as error:
            raise ValueError(f"Invalid transfer function: {error}") from error

        self.state_matrix = np.asarray(state_matrix, dtype=float)
        self.input_matrix = np.asarray(input_matrix, dtype=float)
        self.output_matrix = np.asarray(output_matrix, dtype=float)
        self.feedthrough = np.asarray(feedthrough, dtype=float)
        self.state_size = int(self.state_matrix.shape[0])
        self.direct_feedthrough = bool(np.any(self.feedthrough != 0.0))
        self.initial_conditions = _initial_conditions(initial_condition, self.state_size)
        super().__init__(block_id, name, input_count=1, output_count=1)

    def initial_state(self) -> FloatVector:
        return self.initial_conditions.copy()

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        current_state = self.state_view(state) if self.state_size else np.zeros(0, dtype=float)
        value = (self.output_matrix @ current_state + self.feedthrough[:, 0] * _scalar(inputs[0])).item()
        return (float(value),)

    def derivatives(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> FloatVector:
        current_state = self.state_view(state) if self.state_size else np.zeros(0, dtype=float)
        return self.state_matrix @ current_state + self.input_matrix[:, 0] * _scalar(inputs[0])


class Product(Block):
    """Scalar Product block, including Simulink's multiply/divide notation."""

    def __init__(self, block_id: str, name: str, operations: str) -> None:
        if not operations or any(operation not in "*/" for operation in operations):
            raise ValueError("Product operations must contain only '*' and '/'.")
        super().__init__(block_id, name, input_count=len(operations), output_count=1)
        self.operations = operations

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        if len(self.operations) == 1:
            # one input: the product of its elements (Product of Elements), or its reciprocal
            product = float(np.prod(np.asarray(inputs[0], dtype=float)))
            with np.errstate(divide="ignore", invalid="ignore"):
                return (product if self.operations == "*" else float(np.divide(1.0, product)),)
        result = np.asarray(inputs[0], dtype=float)
        for operation, value in zip(self.operations[1:], inputs[1:], strict=True):
            operand = np.asarray(value, dtype=float)
            if operation == "*":
                result = result * operand
            else:
                # Match IEEE-754/Simulink double arithmetic: a zero divisor
                # produces Inf or NaN for downstream blocks to handle, rather
                # than a Python ZeroDivisionError that aborts the simulation.
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = np.divide(result, operand)
        return (_signal(result),)

    def infer_output_widths(self, input_widths):
        return [1] if len(self.operations) == 1 else super().infer_output_widths(input_widths)

    @property
    def elementwise(self) -> bool:
        return len(self.operations) > 1


class Saturate(Block):
    def __init__(self, block_id: str, name: str, lower_limit: float, upper_limit: float, *, zero_crossing: bool = True) -> None:
        if lower_limit > upper_limit:
            raise ValueError("Saturation lower limit cannot exceed the upper limit.")
        super().__init__(block_id, name, input_count=1, output_count=1)
        self.zero_crossing = bool(zero_crossing)
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (_signal(np.clip(np.asarray(inputs[0], dtype=float), self.lower_limit, self.upper_limit)),)


class Lookup1D(Block):
    scalar_only = True
    """One-dimensional Lookup_n-D block with clipped linear interpolation."""

    def __init__(self, block_id: str, name: str, breakpoints: Sequence[float], table: Sequence[float]) -> None:
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        self.table = np.asarray(table, dtype=float)
        if self.breakpoints.ndim != 1 or self.table.ndim != 1 or self.breakpoints.size < 2:
            raise ValueError("Lookup1D requires at least two one-dimensional breakpoints.")
        if self.breakpoints.size != self.table.size or np.any(np.diff(self.breakpoints) <= 0.0):
            raise ValueError("Lookup1D breakpoints must be strictly increasing and match the table size.")
        super().__init__(block_id, name, input_count=1, output_count=1)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (float(np.interp(_scalar(inputs[0]), self.breakpoints, self.table)),)


class Mux(Block):
    elementwise = False
    """Combine scalar signals into one vector signal for downstream blocks."""

    def __init__(self, block_id: str, name: str, input_count: int) -> None:
        if input_count < 1:
            raise ValueError("Mux requires at least one input.")
        super().__init__(block_id, name, input_count=input_count, output_count=1)

    def infer_output_widths(self, input_widths):
        return [None] if any(w is None for w in input_widths) else [sum(input_widths)]

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (np.concatenate([np.atleast_1d(np.asarray(value, dtype=float)) for value in inputs]),)


class ManualSwitch(Block):
    """Two-input Manual Switch with the selection saved in the SLX file."""

    def __init__(self, block_id: str, name: str, current_setting: int) -> None:
        if current_setting not in {0, 1}:
            raise ValueError("Manual Switch CurrentSetting must be 0 or 1.")
        super().__init__(block_id, name, input_count=2, output_count=1)
        self.current_setting = current_setting

    @property
    def selected_input(self) -> int:
        """Simulink saves sw=1 for the first port and sw=0 for the second."""
        return 1 - self.current_setting

    def infer_output_widths(self, input_widths):
        return [input_widths[self.selected_input]]

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (inputs[self.selected_input],)


class Fcn(Block):
    """Evaluate a scalar Fcn expression against a scalar or Mux vector input."""

    def __init__(self, block_id: str, name: str, expression: str) -> None:
        self.expression = expression
        self._templates: dict[int, Expr] = {}
        if not re.search(r"\bu\s*[(\[]", expression):
            self._template(1)                 # a scalar expression: check it now
        super().__init__(block_id, name, input_count=1, output_count=1)

    def _template(self, width: int) -> Expr:
        """the expression over u1..u<width> (built once per input width)"""
        if width not in self._templates:
            elements = tuple(symbol(f"u{i + 1}") for i in range(width))
            try:
                self._templates[width] = parse_expression(fcn_python(self.expression), {"u": elements, "pi": constant(math.pi)},
                                                          ieee_division=True)
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Unsupported Fcn expression {self.expression!r}: {error}") from error
        return self._templates[width]

    def infer_output_widths(self, input_widths):
        return [1]                            # a Fcn gives a scalar, of a scalar or a vector u

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        vector = np.atleast_1d(np.asarray(inputs[0], dtype=float))
        values = {f"u{i + 1}": float(v) for i, v in enumerate(vector)}
        with np.errstate(all="ignore"):
            return (float(self._template(vector.size).evaluate(values)),)


class Demux(Block):
    """Split a vector into outputs of the given widths (-1: the rest, shared evenly)."""

    def __init__(self, block_id: str, name: str, outputs: int | Sequence[int]) -> None:
        sizes = [int(v) for v in np.atleast_1d(np.asarray(outputs, dtype=float))]
        self.sizes = sizes if len(sizes) > 1 else None
        super().__init__(block_id, name, input_count=1, output_count=len(sizes) if len(sizes) > 1 else sizes[0])

    def _split(self, width: int) -> list[int]:
        if self.sizes is None:
            base, extra = divmod(width, self.output_count)
            return [base + (1 if i < extra else 0) for i in range(self.output_count)]
        known = sum(s for s in self.sizes if s > 0)
        free = [i for i, s in enumerate(self.sizes) if s <= 0]
        sizes = list(self.sizes)
        for k, i in enumerate(free):
            base, extra = divmod(width - known, len(free))
            sizes[i] = base + (1 if k < extra else 0)
        return sizes

    def infer_output_widths(self, input_widths):
        return [None] * self.output_count if input_widths[0] is None else self._split(input_widths[0])

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)
        if sum(output_widths) != input_widths[0] or min(output_widths) < 1:
            raise ValueError(f"Demux '{self.name}' cannot split a signal of width {input_widths[0]} into {output_widths}.")

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        values = np.atleast_1d(np.asarray(inputs[0], dtype=float))
        result, start = [], 0
        for width in self.output_widths:
            result.append(_signal(values[start:start + width]))
            start += width
        return tuple(result)


class Selector(Block):
    """Selector of a vector: the elements at the given zero-based indices."""

    def __init__(self, block_id: str, name: str, indices: Sequence[int]) -> None:
        self.indices = [int(i) for i in indices]
        if not self.indices or min(self.indices) < 0:
            raise ValueError("Selector indices must be non-negative (zero-based here).")
        super().__init__(block_id, name, input_count=1, output_count=1)

    def infer_output_widths(self, input_widths):
        return [len(self.indices)]

    def apply_widths(self, input_widths, output_widths):
        self.input_widths, self.output_widths = list(input_widths), list(output_widths)
        if max(self.indices) >= input_widths[0]:
            raise ValueError(f"Selector '{self.name}': index {max(self.indices) + 1} is beyond the input width {input_widths[0]}.")

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (_signal(np.atleast_1d(np.asarray(inputs[0], dtype=float))[self.indices]),)


class Concatenate(Block):
    """Vector Concatenate: the inputs one after the other (like a Mux)."""

    elementwise = False

    def __init__(self, block_id: str, name: str, input_count: int) -> None:
        super().__init__(block_id, name, input_count=int(input_count), output_count=1)

    def infer_output_widths(self, input_widths):
        return [None] if any(w is None for w in input_widths) else [sum(input_widths)]

    def output(self, time: float, state: FloatVector, inputs: Sequence[SignalValue], external_inputs: Mapping[str, float]) -> tuple[SignalValue, ...]:
        return (_signal(np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in inputs])),)


def _signal(value) -> SignalValue:
    """A computed signal: a float when it has one element, else a 1-D array."""
    array = np.asarray(value, dtype=float)
    return float(array.reshape(-1)[0]) if array.size == 1 else array.reshape(-1).copy()


def _scalar(value: SignalValue) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError("This block requires a scalar input signal.")
    return float(array)


def _polynomial_coefficients(parameter_name: str, coefficients: Sequence[float]) -> FloatVector:
    values = np.atleast_1d(np.asarray(coefficients, dtype=float))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Transfer-function {parameter_name} must contain finite coefficients.")
    return values


def _initial_conditions(initial_condition: float | Sequence[float], state_size: int) -> FloatVector:
    values = np.atleast_1d(np.asarray(initial_condition, dtype=float))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Transfer-function initial condition must contain finite values.")
    if state_size == 0:
        return np.zeros(0, dtype=float)
    if values.size == 1:
        return np.full(state_size, values.item(), dtype=float)
    if values.size != state_size:
        raise ValueError(f"Transfer-function initial condition must be scalar or have {state_size} values.")
    return values.copy()

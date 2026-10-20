"""Validated simulation settings and the common sampled-trace contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping

import numpy as np

SUPPORTED_SOLVERS = ("ode45", "ode4", "ode1")
MAX_SAMPLES = 200000


@dataclass(frozen=True)
class SimulationSettings:
    solver: str
    start_time: float
    stop_time: float
    step_size: float
    relative_tolerance: float
    absolute_tolerance: float
    max_step: float | None = None
    initial_step: float | None = None
    min_step: float | None = None
    auto_scale_absolute_tolerance: bool = False
    zero_crossing_control: str = "UseLocalSettings"

    @staticmethod
    def _optional_step(value):
        if value is None or str(value).strip().lower() == "auto":
            return None
        return float(value)

    @classmethod
    def from_mapping(cls, values: Mapping) -> SimulationSettings:
        result = cls(str(values["solver"]), *(float(values[name]) for name in
            ("start_time", "stop_time", "step_size", "relative_tolerance", "absolute_tolerance")),
                     *(cls._optional_step(values.get(key)) for key in ("max_step", "initial_step", "min_step")),
                     bool(values.get("auto_scale_absolute_tolerance", False)),
                     str(values.get("zero_crossing_control", "UseLocalSettings")))
        result.validate()
        return result

    def validate(self) -> None:
        if self.solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Supported solvers: {', '.join(SUPPORTED_SOLVERS)}. Got {self.solver!r}.")
        numeric = (self.start_time, self.stop_time, self.step_size, self.relative_tolerance, self.absolute_tolerance)
        optional = (self.max_step, self.initial_step, self.min_step)
        if not all(math.isfinite(value) for value in (*numeric, *(v for v in optional if v is not None))):
            raise ValueError("Simulation settings must be finite.")
        if self.stop_time < self.start_time or min(self.step_size, self.relative_tolerance, self.absolute_tolerance) <= 0:
            raise ValueError("Stop must be >= start; step size and tolerances must be positive.")
        if any(value <= 0 for value in optional if value is not None):
            raise ValueError("Solver step limits must be positive or auto.")
        if self.solver == "ode45" and self.min_step is not None:
            raise ValueError("Explicit MinStep is not supported by the shared RK45 solver; use auto.")
        if self.zero_crossing_control not in ("UseLocalSettings", "EnableAll", "DisableAll"):
            raise ValueError("Unsupported zero-crossing control setting.")
        intervals = (self.stop_time - self.start_time) / self.step_size
        if intervals > MAX_SAMPLES - 2:
            raise ValueError(f"Use a larger output interval: at most {MAX_SAMPLES} samples are supported.")
        if self.solver != "ode45" and not np.isclose(intervals, round(intervals), rtol=0, atol=1e-9):
            raise ValueError("Fixed-step time range must be a multiple of step size.")

    def time_grid(self, transitions=()) -> np.ndarray:
        self.validate()
        count = int(np.floor((self.stop_time - self.start_time) / self.step_size + 1e-10))
        times = self.start_time + self.step_size * np.arange(count + 1)
        if np.isclose(times[-1], self.stop_time, rtol=0, atol=1e-10):
            times[-1] = self.stop_time
        else:
            times = np.append(times, self.stop_time)
        # A nominal grid point can round just below a Step onset (e.g. 3*.3).
        # Represent coincident points by the exact event time in every backend.
        for event in transitions:
            if not self.start_time <= event <= self.stop_time:
                continue
            position = int(np.searchsorted(times, event))
            candidates = [index for index in (position - 1, position) if 0 <= index < times.size]
            index = min(candidates, key=lambda i: abs(times[i] - event))
            tolerance = 8 * np.finfo(float).eps * max(1, abs(event))
            if abs(times[index] - event) <= tolerance:
                times[index] = event
        return times


@dataclass(frozen=True)
class Trace:
    time: np.ndarray
    output_names: tuple[str, ...]
    outputs: np.ndarray

    def validate(self) -> None:
        if (self.time.ndim != 1 or not self.time.size or self.outputs.shape != (self.time.size, len(self.output_names))
                or len(set(self.output_names)) != len(self.output_names)):
            raise ValueError("Trace shape or output identities are invalid.")
        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.outputs)):
            raise ValueError("Simulation returned non-finite outputs or times.")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("Trace time must be strictly increasing.")

    def save(self, path) -> None:
        self.validate()
        np.savez_compressed(path, time=self.time, output_names=np.array(self.output_names, dtype=str), outputs=self.outputs)

    @classmethod
    def load(cls, path) -> Trace:
        with np.load(path, allow_pickle=False) as data:
            result = cls(data["time"], tuple(data["output_names"].tolist()), data["outputs"])
        result.validate()
        return result

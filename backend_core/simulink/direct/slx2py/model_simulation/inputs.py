"""Named input specifications. New signal families register one provider factory."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StepSignal:
    step_time: float = 1.0
    initial_value: float = 0.0
    final_value: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.step_time, self.initial_value, self.final_value)):
            raise ValueError("Step time and values must be finite.")

    def value(self, time: float) -> float:
        return self.final_value if time >= self.step_time else self.initial_value

    def transitions(self) -> tuple[float, ...]:
        return (self.step_time,) if self.initial_value != self.final_value else ()


INPUT_FACTORIES = {"step": lambda spec: StepSignal(*(float(spec.get(key, default)) for key, default in
                  (("step_time", 1), ("initial_value", 0), ("final_value", 1))))}


class InputSet:
    def __init__(self, names: tuple[str, ...], specs: dict):
        if set(names) != set(specs) or len(set(names)) != len(names):
            raise ValueError("Supply exactly one input specification per declared model input.")
        self.names = names
        self.signals = {}
        self.specs = {}
        for name in names:
            spec = specs[name]
            kind = spec.get("kind", "step")
            if kind not in INPUT_FACTORIES:
                raise ValueError(f"Unsupported input kind {kind!r} for {name!r}.")
            signal = INPUT_FACTORIES[kind](spec)
            self.signals[name] = signal
            self.specs[name] = {"kind": kind, **signal.__dict__}

    def values(self, time: float) -> dict[str, float]:
        return {name: signal.value(time) for name, signal in self.signals.items()}

    def transitions(self) -> tuple[float, ...]:
        return tuple(sorted({t for signal in self.signals.values() for t in signal.transitions()}))


def default_steps(names) -> dict:
    return {name: {"kind": "step", "step_time": 1.0, "initial_value": 0.0, "final_value": 1.0} for name in names}

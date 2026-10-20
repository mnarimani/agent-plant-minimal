"""Root-event adapters with a small zero band to avoid repeated roots at restart."""
import numpy as np

class ContinuousEvents:
    def __init__(self, backend, settings, inputs):
        self.backend = backend
        self.inputs = inputs
        self.names = backend.zero_crossing_names(settings) if hasattr(backend, "zero_crossing_names") else ()
        self.sides = np.ones(len(self.names))
        self.last_roots = np.full(len(self.names), -np.inf)
        self.count = 0
        self.hits = []

    def functions(self, start, state, right, is_source_event):
        if not self.names:
            return None
        initial = self.backend.zero_crossing_values(start, state, self.inputs.values(start))
        restarted = np.abs(self.last_roots - start) <= 8 * np.finfo(float).eps * max(1, abs(start))
        self.sides = np.where((np.abs(initial) > 1e-12) & ~restarted, np.sign(initial), self.sides)
        functions = []
        for index in range(len(self.names)):
            def event(time, current_state, index=index):
                effective = float(np.nextafter(right, -np.inf)) if is_source_event and time >= right else float(time)
                values = self.backend.zero_crossing_values(effective, current_state, self.inputs.values(effective))
                value = float(values[index])
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite zero-crossing signal: {self.names[index]}")
                # Plateau/touching roots do not change branches. A true crossing
                # is localized to this 1e-12 signal band and then disarmed until
                # the signal crosses back; chart memory is never updated here.
                return value if abs(value) > 1e-12 else float(self.sides[index]) * 1e-12
            event.terminal = True
            event.direction = -float(self.sides[index])
            functions.append(event)
        return functions

    def record(self, result):
        indices = [index for index, roots in enumerate(result.t_events) if len(roots)]
        for index in indices:
            self.sides[index] *= -1
            self.last_roots[index] = float(result.t[-1])
            self.count += 1
            if len(self.hits) < 100:
                self.hits.append({"signal": self.names[index], "time": float(result.t[-1])})
        if self.count > 200000:
            raise RuntimeError("Too many continuous zero crossings; possible chattering.")

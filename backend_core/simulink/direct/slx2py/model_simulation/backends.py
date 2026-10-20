"""Independent adapters for the two exported computational representations."""
from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


class ModelBackend:
    def __init__(self, directory: Path, contract: dict):
        self.namespace = runpy.run_path(str(directory / "model.py"))
        self.model = self.namespace["build_model"]()
        self.input_names = tuple(contract["inputs"])
        self.output_names = tuple(contract["outputs"])
        self.state_size = self.model.state_size
        self.transitions = tuple(float(block.step_time) for block in self.model.blocks.values()
                                 if type(block).__name__ == "Step")
        self.charts = [block for block in self.model.blocks.values() if type(block).__name__ == "StateflowChart"]
        incoming = {}
        for connection in self.model.connections:
            incoming.setdefault(connection.destination.block_id, []).append(connection.source.block_id)
        def check_trigger(block_id, visited):
            if block_id in visited:
                raise ValueError("Cyclic Stateflow trigger graph is unsupported.")
            block = self.model.blocks[block_id]
            if block.state_size or type(block).__name__ == "StateflowChart" or block_id in self.model._enable_sources:
                raise ValueError("Stateflow triggers depending on continuous/discrete states require zero-crossing event scheduling, which is not implemented.")
            for source in incoming.get(block_id, ()):
                check_trigger(source, {*visited, block_id})
        for chart in self.charts:
            if not chart.chart.event_names or chart.block_id not in self.model._trigger_sources:
                raise ValueError("Periodic/untriggered Stateflow execution requires a sample-time scheduler, which is not implemented.")
            check_trigger(self.model._trigger_sources[chart.block_id].block_id, set())

    def zero_crossing_names(self, settings):
        self.model._compile_if_needed()
        self._zero_specs = []
        if settings.zero_crossing_control == "DisableAll":
            return ()
        incoming = {c.destination: c.source for c in self.model.connections}
        by_block = {}
        for c in self.model.connections:
            by_block.setdefault(c.destination.block_id, []).append(c.source.block_id)
        def continuous(block_id, visited):
            if block_id in visited:
                return False
            block = self.model.blocks[block_id]
            if block.state_size:
                return True
            if type(block).__name__ == "StateflowChart":
                return False  # Latched outputs change only at already scheduled events.
            return any(continuous(source, {*visited, block_id}) for source in by_block.get(block_id, ()))
        names = []
        for block in self.model.blocks.values():
            if type(block).__name__ != "Saturate":
                continue
            if settings.zero_crossing_control == "UseLocalSettings" and not getattr(block, "zero_crossing", True):
                continue
            source = next((src for dst, src in incoming.items() if dst.block_id == block.block_id and dst.port == 0), None)
            if source is None or not continuous(source.block_id, set()):
                continue
            width = self.model.signal_width(source)
            for element in range(width):          # a vector: each element crosses on its own
                suffix = f"({element + 1})" if width > 1 else ""
                for name, limit in (("lower", block.lower_limit), ("upper", block.upper_limit)):
                    if np.isfinite(limit):
                        self._zero_specs.append((source, float(limit), element))
                        names.append(f"{block.name}{suffix}:{name}")
        return tuple(names)

    def zero_crossing_values(self, time, state, inputs):
        if hasattr(self.model, "_evaluate_signals"):
            signals = self.model._evaluate_signals(time, state, inputs)
            return np.array([float(np.atleast_1d(signals[self.model._signal_indices[source]])[element]) - limit
                             for source, limit, element in self._zero_specs])
        signals, _ = self.model._evaluate_blocks(time, state, inputs)
        return np.array([float(np.atleast_1d(signals[source])[element]) - limit for source, limit, element in self._zero_specs])

    def initial_state(self, time, inputs):
        for chart in self.charts:
            chart.allow_updates = True
        state = np.asarray(self.model.initial_state_at(time, inputs), dtype=float)
        for chart in self.charts:
            chart.allow_updates = False
        return state

    def transition_times(self, start, stop):
        result = list(self.transitions)
        for block in self.model.blocks.values():
            if hasattr(block, "transition_times"):
                result.extend(block.transition_times(start, stop))
        if len(result) > 200000:
            raise ValueError("Simulation exceeds 200000 source events; shorten its time range.")
        return tuple(sorted(set(result)))

    def commit_event(self, time, state, inputs):
        """Update discrete chart memory once at a known signal discontinuity."""
        if not self.charts:
            return
        for chart in self.charts:
            chart.allow_updates = True
        try:
            self.model.evaluate(time, state, inputs)
        finally:
            for chart in self.charts:
                chart.allow_updates = False

    def derivative(self, time, state, inputs):
        return self.model.evaluate(time, state, inputs).derivatives

    def output(self, time, state, inputs):
        values = self.model.evaluate(time, state, inputs).outputs
        return np.array([values[name] for name in self.output_names], dtype=float)


class DynamicsBackend:
    def __init__(self, directory: Path, contract: dict):
        report = contract.get("dynamics", {})
        if report.get("status") != "success":
            details = "; ".join(f"{item.get('block_name', 'Model')}: {item['message']}" for item in report.get("errors", []))
            raise ValueError("Dynamic equations were not successfully extracted. " + details)
        self.namespace = runpy.run_path(str(directory / "dynamics.py"))
        self.input_names = tuple(self.namespace["INPUT_NAMES"])
        self.output_names = tuple(self.namespace["OUTPUT_NAMES"])
        self.state_size = len(self.namespace["STATE_NAMES"])
        if self.input_names != tuple(contract["inputs"]) or self.output_names != tuple(contract["outputs"]):
            raise ValueError("dynamics.py input/output order differs from model.json.")
        self.transitions = ()

    def zero_crossing_names(self, settings):
        specs = self.namespace.get("ZERO_CROSSINGS", ())
        self._zero_indices = tuple(index for index, spec in enumerate(specs)
                                   if settings.zero_crossing_control == "EnableAll"
                                   or settings.zero_crossing_control == "UseLocalSettings" and spec["enabled"])
        return tuple(specs[index]["name"] for index in self._zero_indices)

    def zero_crossing_values(self, time, state, inputs):
        values = self.namespace["zero_crossings"](state, self._inputs(inputs), t=time)
        return values[list(self._zero_indices)]

    def _inputs(self, inputs):
        return [inputs[name] for name in self.input_names]

    def initial_state(self, time, inputs):
        return self.namespace["initial_state"](self._inputs(inputs), t=time)

    def derivative(self, time, state, inputs):
        return self.namespace["f"](state, self._inputs(inputs), t=time)

    def output(self, time, state, inputs):
        return self.namespace["g"](state, self._inputs(inputs), t=time)


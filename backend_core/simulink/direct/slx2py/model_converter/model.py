from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .blocks import Block, FloatVector, Outport, SignalValue, StateflowChart, _scalar
from .graph import Connection, Endpoint


class ModelDefinitionError(ValueError):
    pass


class AlgebraicLoopError(ModelDefinitionError):
    pass


@dataclass(frozen=True, slots=True)
class Evaluation:
    outputs: dict[str, float]
    derivatives: FloatVector


class Model:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blocks: dict[str, Block] = {}
        self.connections: list[Connection] = []
        self._execution_order: tuple[str, ...] | None = None
        self._incoming: dict[Endpoint, Endpoint] = {}
        self._enable_sources: dict[str, Endpoint] = {}
        self._trigger_sources: dict[str, Endpoint] = {}
        self._state_size = 0
        self._widths: dict[Endpoint, int] | None = None

    @property
    def state_size(self) -> int:
        self.infer_widths()
        return self._state_size

    def add_block(self, block: Block) -> None:
        if block.block_id in self.blocks:
            raise ModelDefinitionError(f"Duplicate block id '{block.block_id}'.")
        self.blocks[block.block_id] = block
        self._execution_order = None
        self._widths = None

    def connect(self, source: Endpoint, destination: Endpoint) -> None:
        self.connections.append(Connection(source, destination))
        self._execution_order = None
        self._widths = None

    def signal_width(self, endpoint: Endpoint) -> int:
        """the width of an output port's signal (1 = scalar)"""
        self.infer_widths()
        return self._widths[endpoint]

    def infer_widths(self) -> None:
        """Resolve the width of every signal, like Simulink's propagation: each block gives the
        widths of its outputs from those of its inputs, repeated until nothing changes; a width
        nothing determines (a loop of element-wise blocks) is taken as 1. Then the blocks are
        sized for their widths and the states are bound in the order of the blocks."""
        if self._widths is not None:
            return
        incoming = {connection.destination: connection.source for connection in self.connections}
        widths: dict[Endpoint, int | None] = {Endpoint(b.block_id, p): None
                                              for b in self.blocks.values() for p in range(b.output_count)}

        def inputs_of(block):
            return [widths.get(incoming.get(Endpoint(block.block_id, port))) if Endpoint(block.block_id, port) in incoming
                    else 1 for port in range(block.input_count)]

        for _ in range(4 * len(widths) + 8):
            changed = False
            for block in self.blocks.values():
                try:
                    outputs = block.infer_output_widths(inputs_of(block))
                except (ValueError, IndexError) as error:
                    raise ModelDefinitionError(f"Block '{block.name}': {error}") from error
                for port, width in enumerate(outputs):
                    key = Endpoint(block.block_id, port)
                    if width is not None and widths[key] != width:
                        widths[key] = int(width)
                        changed = True
            if changed:
                continue
            unknown = [key for key, width in widths.items() if width is None]
            if not unknown:
                break
            widths[unknown[0]] = 1
        for block in self.blocks.values():
            outputs = [widths[Endpoint(block.block_id, port)] for port in range(block.output_count)]
            try:
                block.apply_widths(inputs_of(block), outputs)
            except ValueError as error:
                raise ModelDefinitionError(str(error)) from error
        offset = 0
        for block in self.blocks.values():
            block._state_start = None
            block.bind_state(offset)
            offset += block.state_size
        self._state_size = offset
        self._widths = widths

    def set_enable(self, block_id: str, source: Endpoint) -> None:
        if block_id not in self.blocks:
            raise ModelDefinitionError(f"Cannot enable unknown block '{block_id}'.")
        self._enable_sources[block_id] = source
        self._execution_order = None

    def set_trigger(self, block_id: str, source: Endpoint) -> None:
        block = self._get_block(block_id)
        if not isinstance(block, StateflowChart):
            raise ModelDefinitionError(f"Only Stateflow charts can have a trigger source; got '{block_id}'.")
        self._trigger_sources[block_id] = source
        self._execution_order = None

    def initial_state(self) -> FloatVector:
        state = np.zeros(self.state_size, dtype=float)
        for block in self.blocks.values():
            if block.state_size:
                block.state_view(state)[:] = block.initial_state()
        return state

    def initial_state_at(self, time: float, external_inputs: Mapping[str, float]) -> FloatVector:
        """Build continuous state, including explicitly wired initial ports.

        Simulink's external initial-condition port is sampled once at model
        initialization; it is not a second derivative input during the run.
        Evaluate the normal signal graph using configured defaults first, then
        replace only the states whose blocks explicitly request that port.
        """
        state = self.initial_state()
        for block in self.blocks.values():
            if isinstance(block, StateflowChart):
                block.reset_runtime()
        self._compile_if_needed()
        signal_values, _ = self._evaluate_blocks(time, state, external_inputs)
        for block in self.blocks.values():
            if block.state_size == 0:
                continue
            inputs = self._inputs_for(block, signal_values, require_values=True)
            supplied_state = block.initial_state_from_inputs(inputs)
            if supplied_state is not None:
                if supplied_state.shape != (block.state_size,):
                    raise ModelDefinitionError(
                        f"Initial state supplied to '{block.block_id}' has shape {supplied_state.shape}; "
                        f"expected ({block.state_size},)."
                    )
                block.state_view(state)[:] = supplied_state
        # Chart inputs must observe the final external initial conditions,
        # rather than the Integrator's fallback value used to resolve IC wiring.
        charts = [block for block in self.blocks.values() if isinstance(block, StateflowChart)]
        if charts:
            for block in charts:
                block.reset_runtime()
            self._evaluate_blocks(time, state, external_inputs)
        return state

    def evaluate(self, time: float, state: FloatVector, external_inputs: Mapping[str, float]) -> Evaluation:
        self._compile_if_needed()
        if state.shape != (self.state_size,):
            raise ValueError(f"Expected state shape ({self.state_size},), got {state.shape}.")

        signals = self._evaluate_signals(time, state, external_inputs)
        derivatives = np.zeros(self.state_size, dtype=float)
        for block, inputs, outputs, enable, trigger in self._state_nodes:
            if self._indexed_enabled(block, enable, signals):
                values = self._indexed_inputs(block, inputs, signals, True)
                block.state_view(derivatives)[:] = block.derivatives(time, state, values, external_inputs)
        # a vector output is given element by element: Out1(1), Out1(2) ...
        outputs = {}
        for block, index in self._output_nodes:
            values = np.atleast_1d(np.asarray(signals[index], dtype=float))
            outputs.update(zip(block.element_names, (float(v) for v in values), strict=True))
        return Evaluation(outputs=outputs, derivatives=derivatives)

    @property
    def discrete(self) -> bool:
        """True if some block keeps discrete memory (see blocks.DiscreteBlock and update)"""
        return any(getattr(block, "discrete", False) for block in self.blocks.values())

    def update(self, time: float, state: FloatVector, external_inputs: Mapping[str, float]) -> FloatVector:
        """Simulink's Update phase of the major step at ``time``: the signals of the Output phase
        (discrete blocks give their major-step outputs), then the new memory of every discrete
        block. -> the new state vector (continuous states unchanged). A model without discrete
        memory returns ``state`` itself.

        A major step of a fixed-step solver: x = update(t, x, u); outputs = evaluate(t, x, u).outputs;
        then the solver's stages from this x (evaluate(...).derivatives)."""
        self._compile_if_needed()
        nodes = [node for node in self._state_nodes if getattr(node[0], "discrete", False)]
        if not nodes:
            return state
        signals = self._evaluate_signals(time, state, external_inputs, major=True)
        new = np.array(state, dtype=float, copy=True)
        for block, inputs, outputs, enable, trigger in nodes:
            if self._indexed_enabled(block, enable, signals):
                values = self._indexed_inputs(block, inputs, signals, True)
                block.state_view(new)[:] = block.update(time, state, values)
        return new

    def _indexed_inputs(self, block, indices, signals, require_values):
        values = []
        for port, index in enumerate(indices):
            if index < 0:
                raise ModelDefinitionError(f"Input {port} of block '{block.block_id}' is not connected.")
            value = signals[index]
            if value is None:
                if require_values:
                    source = self._incoming[Endpoint(block.block_id, port)]
                    raise ModelDefinitionError(
                        f"Signal from {source.block_id}#{source.port} is unavailable for block '{block.block_id}'."
                    )
                value = 0.0
            values.append(value)
        return tuple(values)

    def _indexed_enabled(self, block, index, signals):
        if index < 0:
            return True
        value = signals[index]
        if value is None:
            source = self._enable_sources[block.block_id]
            raise ModelDefinitionError(
                f"Enable signal from {source.block_id}#{source.port} is unavailable for block '{block.block_id}'."
            )
        return _scalar(value) != 0.0

    def _evaluate_signals(self, time, state, external_inputs, major=False):
        # Wiring is compiled once; buffers remain local for independent evaluations.
        signals = [None] * len(self._signal_indices)
        for block, inputs, outputs, enable, trigger in self._runtime_nodes:
            if not self._indexed_enabled(block, enable, signals):
                output = (0.0,) * block.output_count
            else:
                if trigger >= 0:
                    value = signals[trigger]
                    if value is None:
                        source = self._trigger_sources[block.block_id]
                        raise ModelDefinitionError(
                            f"Trigger signal from {source.block_id}#{source.port} is unavailable for block '{block.block_id}'."
                        )
                    block.set_trigger_value(value)
                values = self._indexed_inputs(block, inputs, signals, block.direct_feedthrough)
                if major and getattr(block, "discrete", False):
                    output = tuple(block.major_output(time, state, values))
                else:
                    output = tuple(block.output(time, state, values, external_inputs))
            if len(output) != block.output_count:
                raise ModelDefinitionError(
                    f"Block '{block.block_id}' returned {len(output)} outputs; expected {block.output_count}."
                )
            for index, value in zip(outputs, output):
                signals[index] = value
        return signals

    def _evaluate_blocks(self, time, state, external_inputs):
        # Retain the graph representation for initialization and diagnostic callers.
        signals = self._evaluate_signals(time, state, external_inputs)
        signal_values = {endpoint: signals[index] for endpoint, index in self._signal_indices.items()}
        block_outputs = {
            block.block_id: tuple(signals[index] for index in outputs)
            for block, inputs, outputs, enable, trigger in self._runtime_nodes
        }
        return signal_values, block_outputs

    def _compile_if_needed(self) -> None:
        self.infer_widths()
        if self._execution_order is not None:
            return

        incoming: dict[Endpoint, Endpoint] = {}
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree = {block_id: 0 for block_id in self.blocks}

        for connection in self.connections:
            source_block = self._get_block(connection.source.block_id)
            destination_block = self._get_block(connection.destination.block_id)
            self._validate_port(source_block, connection.source.port, source_block.output_count, "source")
            self._validate_port(destination_block, connection.destination.port, destination_block.input_count, "destination")
            if connection.destination in incoming:
                raise ModelDefinitionError(f"Input {connection.destination} has more than one source.")
            incoming[connection.destination] = connection.source

            if destination_block.direct_feedthrough and destination_block.block_id not in adjacency[source_block.block_id]:
                adjacency[source_block.block_id].add(destination_block.block_id)
                indegree[destination_block.block_id] += 1

        for block_id, source in self._enable_sources.items():
            source_block = self._get_block(source.block_id)
            destination_block = self._get_block(block_id)
            self._validate_port(source_block, source.port, source_block.output_count, "enable source")
            if destination_block.block_id not in adjacency[source_block.block_id]:
                adjacency[source_block.block_id].add(destination_block.block_id)
                indegree[destination_block.block_id] += 1

        for block_id, source in self._trigger_sources.items():
            source_block = self._get_block(source.block_id)
            destination_block = self._get_block(block_id)
            self._validate_port(source_block, source.port, source_block.output_count, "trigger source")
            if destination_block.block_id not in adjacency[source_block.block_id]:
                adjacency[source_block.block_id].add(destination_block.block_id)
                indegree[destination_block.block_id] += 1

        ready = deque(block_id for block_id in self.blocks if indegree[block_id] == 0)
        execution_order: list[str] = []
        while ready:
            block_id = ready.popleft()
            execution_order.append(block_id)
            for dependent_id in adjacency[block_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(dependent_id)

        if len(execution_order) != len(self.blocks):
            loop_blocks = sorted(block_id for block_id, degree in indegree.items() if degree > 0)
            raise AlgebraicLoopError(
                "Direct-feedthrough algebraic loop detected among blocks: " + ", ".join(loop_blocks)
            )

        self._incoming = incoming
        self._execution_order = tuple(execution_order)
        self._signal_indices = {
            endpoint: index for index, endpoint in enumerate(
                Endpoint(block.block_id, port)
                for block in self.blocks.values() for port in range(block.output_count)
            )
        }
        def signal_index(endpoint):
            return self._signal_indices[endpoint] if endpoint is not None else -1
        self._runtime_nodes = tuple(
            (
                block,
                tuple(signal_index(incoming.get(Endpoint(block.block_id, port))) for port in range(block.input_count)),
                tuple(self._signal_indices[Endpoint(block.block_id, port)] for port in range(block.output_count)),
                signal_index(self._enable_sources.get(block.block_id)),
                signal_index(self._trigger_sources.get(block.block_id)),
            )
            for block in (self.blocks[block_id] for block_id in execution_order)
        )
        self._state_nodes = tuple(node for node in self._runtime_nodes if node[0].state_size)
        self._output_nodes = tuple(
            (block, self._signal_indices[Endpoint(block.block_id, 0)])
            for block in self.blocks.values() if isinstance(block, Outport)
        )

    def _inputs_for(self, block: Block, signal_values: Mapping[Endpoint, SignalValue], require_values: bool) -> tuple[SignalValue, ...]:
        values: list[SignalValue] = []
        for port in range(block.input_count):
            source = self._incoming.get(Endpoint(block.block_id, port))
            if source is None:
                raise ModelDefinitionError(f"Input {port} of block '{block.block_id}' is not connected.")
            if source not in signal_values:
                if require_values:
                    raise ModelDefinitionError(
                        f"Signal from {source.block_id}#{source.port} is unavailable for block '{block.block_id}'."
                    )
                values.append(0.0)
            else:
                values.append(signal_values[source])
        return tuple(values)

    def _is_enabled(self, block_id: str, signal_values: Mapping[Endpoint, SignalValue]) -> bool:
        source = self._enable_sources.get(block_id)
        if source is None:
            return True
        try:
            return _scalar(signal_values[source]) != 0.0
        except KeyError as error:
            raise ModelDefinitionError(f"Enable signal from {source.block_id}#{source.port} is unavailable for block '{block_id}'.") from error

    def _apply_trigger(self, block_id: str, signal_values: Mapping[Endpoint, SignalValue]) -> None:
        source = self._trigger_sources.get(block_id)
        if source is None:
            return
        block = self._get_block(block_id)
        assert isinstance(block, StateflowChart)
        try:
            block.set_trigger_value(signal_values[source])
        except KeyError as error:
            raise ModelDefinitionError(
                f"Trigger signal from {source.block_id}#{source.port} is unavailable for block '{block_id}'."
            ) from error

    def _get_block(self, block_id: str) -> Block:
        try:
            return self.blocks[block_id]
        except KeyError as error:
            raise ModelDefinitionError(f"Unknown block id '{block_id}'.") from error

    @staticmethod
    def _validate_port(block: Block, port: int, port_count: int, role: str) -> None:
        if port < 0 or port >= port_count:
            raise ModelDefinitionError(f"Invalid {role} port {port} on block '{block.block_id}'.")

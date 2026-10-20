"""Discover continuous states and compose xdot=f(x,u), y=g(x,u).

State coordinates follow Model's bound runtime ordering. Signal resolution
walks wiring and stops at state outputs, so ordinary continuous feedback is
supported while direct-feedthrough cycles are rejected. Block interpretation
is delegated to a RuleRegistry, independent of graph traversal and rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .blocks import Block, DiscreteBlock, Inport, Integrator, Outport, TransferFcn, Saturate
from .dynamics_expressions import Expr, constant, operation, symbol
from .dynamics_rules import DEFAULT_RULES, RuleRegistry, Signal
from .graph import Endpoint
from .model import Model


class DynamicsError(ValueError):
    def __init__(self, message: str, *, code: str = "extraction_error", block: Block | None = None):
        super().__init__(message)
        self.code = code
        self.block = block

    def diagnostic(self) -> dict:
        return {
            "code": self.code, "message": str(self),
            "block_id": self.block.block_id if self.block else None,
            "block_name": self.block.name if self.block else None,
            "block_type": type(self.block).__name__ if self.block else None,
        }


@dataclass(frozen=True)
class DynamicsModel:
    derivatives: tuple[Expr, ...]
    outputs: tuple[Expr, ...]
    initializers: tuple[Expr, ...]
    state_defaults: tuple[float, ...]
    input_count: int
    zero_crossings: tuple[Expr, ...] = ()
    zero_crossing_metadata: tuple[dict, ...] = ()
    updates: tuple[Expr, ...] = ()            # x+ at a major step (discrete memory); () = none

    def environment(self, x: Sequence[float], u: Sequence[float], t: float) -> dict[str, float]:
        if np.asarray(x).shape != (len(self.derivatives),) or np.asarray(u).shape != (self.input_count,):
            raise ValueError("State/input vector shape does not match the extracted contract.")
        return {"t": float(t), **{f"x{i + 1}": float(v) for i, v in enumerate(x)},
                **{f"u{i + 1}": float(v) for i, v in enumerate(u)}}

    def evaluate(self, x: Sequence[float], u: Sequence[float], t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        values = self.environment(x, u, t)
        return (np.array([expr.evaluate(values) for expr in self.derivatives], dtype=float),
                np.array([expr.evaluate(values) for expr in self.outputs], dtype=float))

    def initial_state(self, u: Sequence[float], t: float = 0.0) -> np.ndarray:
        values = self.environment(self.state_defaults, u, t)
        return np.array([expr.evaluate(values) for expr in self.initializers], dtype=float)


@dataclass(frozen=True)
class DynamicsResult:
    report: dict
    equations: DynamicsModel | None = None


class _Context:
    def __init__(self, extractor: _Extractor, block: Block):
        self.extractor = extractor
        self.block = block
        self.states = extractor.state_symbols[block.block_id]

    def input(self, port: int) -> Signal:
        source = self.extractor.incoming.get(Endpoint(self.block.block_id, port))
        if source is None:
            raise DynamicsError(f"Input port {port + 1} is not connected.", code="missing_input", block=self.block)
        return self.extractor.resolve(source)

    def scalar_input(self, port: int) -> Expr:
        value = self.input(port)
        if not isinstance(value, Expr):
            raise DynamicsError(f"Input port {port + 1} requires a scalar.", code="vector_signal", block=self.block)
        return value

    def external_input(self, name: str) -> Expr:
        return self.extractor.input_symbols[name]


class _Extractor:
    def __init__(self, model: Model, registry: RuleRegistry, major: bool = False):
        self.model = model
        self.registry = registry
        self.major = major                    # the signals of the Output phase of a major step
        self.incoming = {connection.destination: connection.source for connection in model.connections}
        self.cache: dict[Endpoint, Signal] = {}
        self.visiting: set[Endpoint] = set()
        self.state_symbols: dict[str, tuple[Expr, ...]] = {}
        self.input_symbols: dict[str, Expr] = {}
        self.states: list[dict] = []
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.realizations: list[dict] = []
        self.defaults: list[float] = []

    def discover(self) -> None:
        self.model.infer_widths()
        for block in self.model.blocks.values():
            if isinstance(block, Inport):
                # one symbol per element of a vector input: In1(1), In1(2) ...
                for element in block.element_names:
                    name = f"u{len(self.inputs) + 1}"
                    if element in self.input_symbols:
                        raise DynamicsError("Duplicate external input name.", block=block)
                    self.input_symbols[element] = symbol(name)
                    self.inputs.append({"symbol": name, "index": len(self.inputs), "name": element, "block_id": block.block_id})
            if isinstance(block, Outport):
                for element in block.element_names:
                    self.outputs.append({"symbol": f"y{len(self.outputs) + 1}", "index": len(self.outputs), "name": element,
                                         "block_id": block.block_id})
            if block.state_size and block._state_start != len(self.states):
                raise DynamicsError("Continuous state ordering differs from the runtime.", block=block)
            names = tuple(symbol(f"x{len(self.states) + i + 1}") for i in range(block.state_size))
            self.state_symbols[block.block_id] = names
            if block.state_size:
                defaults = block.initial_state()
                if defaults.shape != (block.state_size,):
                    raise DynamicsError("Invalid initial state vector.", block=block)
                for local_index, (name, value) in enumerate(zip(names, defaults, strict=True)):
                    self.defaults.append(float(value))
                    self.states.append({
                        "symbol": name.value, "index": len(self.states), "block_id": block.block_id,
                        "block_name": block.name, "block_type": type(block).__name__, "local_index": local_index,
                        "initial_value": constant(value).value,
                        "coordinate": "transfer_function_internal" if isinstance(block, TransferFcn) else
                                      "integrator_output" if isinstance(block, Integrator) else
                                      "discrete_memory" if isinstance(block, DiscreteBlock) else "continuous_state",
                    })
                    if isinstance(block, DiscreteBlock):
                        element, slot = divmod(local_index, block.memory_size)
                        self.states[-1].update(kind=block.kind, memory=block.labels[slot] + (
                            f" (element {element + 1})" if block.width > 1 else ""))
            if isinstance(block, TransferFcn):
                self.realizations.append({
                    "block_id": block.block_id, "block_name": block.name,
                    "coordinates": "Runtime SciPy tf2ss controllable canonical coordinates; not physical state names.",
                    "states": [name.value for name in names],
                    "numerator": block.numerator.tolist(), "denominator": block.denominator.tolist(),
                    "A": block.state_matrix.tolist(), "B": block.input_matrix.tolist(),
                    "C": block.output_matrix.tolist(), "D": block.feedthrough.tolist(),
                })
        if len(self.states) != self.model.state_size:
            raise DynamicsError("Discovered state count differs from the runtime.")

    def enabled(self, block: Block) -> Expr | None:
        source = self.model._enable_sources.get(block.block_id)
        if source is None:
            return None
        value = self.resolve(source)
        if not isinstance(value, Expr):
            raise DynamicsError("Enable signal must be scalar.", block=block)
        return operation("ne", value, constant(0))

    def resolve(self, endpoint: Endpoint) -> Signal:
        if endpoint in self.cache:
            return self.cache[endpoint]
        block = self.model.blocks[endpoint.block_id]
        if endpoint in self.visiting:
            raise DynamicsError("Algebraic signal/enable cycle cannot be expressed as an explicit ODE.", code="algebraic_loop", block=block)
        self.visiting.add(endpoint)
        try:
            condition = self.enabled(block)
            rule = self.registry.rule_for(block)
            produce = rule.major_outputs if self.major and rule.major_outputs is not None else rule.outputs
            values = produce(_Context(self, block))
            if len(values) != block.output_count:
                raise ValueError("Rule returned an incorrect number of output ports.")
            for port, value in enumerate(values):
                if condition is not None:
                    if isinstance(value, tuple):
                        value = tuple(operation("select", condition, v, constant(0)) for v in value)
                    else:
                        value = operation("select", condition, value, constant(0))
                self.cache[Endpoint(block.block_id, port)] = value
            return self.cache[endpoint]
        except DynamicsError:
            raise
        except Exception as error:
            raise DynamicsError(str(error), code="block_rule_error", block=block) from error
        finally:
            self.visiting.remove(endpoint)

    def equations(self) -> DynamicsModel:
        derivatives: list[Expr] = []
        initializers: list[Expr] = []
        # Resolve every port, including internal branches, so errors cannot be hidden.
        for block in self.model.blocks.values():
            for port in range(block.output_count):
                self.resolve(Endpoint(block.block_id, port))
            if not block.state_size:
                continue
            try:
                context = _Context(self, block)
                rule = self.registry.rule_for(block)
                if rule.derivatives is None:
                    raise ValueError("Stateful rule does not provide continuous derivatives.")
                values = rule.derivatives(context)
                if len(values) != block.state_size or any(not isinstance(value, Expr) for value in values):
                    raise ValueError("Rule derivative count/type differs from the runtime state slice.")
                condition = self.enabled(block)
                derivatives.extend(operation("select", condition, value, constant(0)) if condition is not None else value for value in values)
                initial = rule.initializers(context) if rule.initializers is not None else tuple(constant(value) for value in block.initial_state())
                if len(initial) != block.state_size or any(not isinstance(value, Expr) for value in initial):
                    raise ValueError("Rule initializer count/type differs from the runtime state slice.")
                initializers.extend(initial)
            except DynamicsError:
                raise
            except Exception as error:
                raise DynamicsError(str(error), code="block_rule_error", block=block) from error
        resolved: dict[str, list] = {}
        for item in self.outputs:
            if item["block_id"] not in resolved:
                value = self.resolve(Endpoint(item["block_id"], 0))
                resolved[item["block_id"]] = list(value if isinstance(value, tuple) else (value,))
        outputs = tuple(resolved[item["block_id"]].pop(0) for item in self.outputs)
        if any(not isinstance(value, Expr) for value in outputs) or any(resolved.values()):
            raise DynamicsError("The model outputs do not have the widths of their Outports.")
        crossings, crossing_metadata = [], []
        state_names = {item["symbol"] for item in self.states}
        for block in self.model.blocks.values():
            if not isinstance(block, Saturate):
                continue
            signal = _Context(self, block).input(0)
            elements = signal if isinstance(signal, tuple) else (signal,)
            for index, value in enumerate(elements):
                if not value.symbols().intersection(state_names):
                    continue
                suffix = f"({index + 1})" if len(elements) > 1 else ""
                for name, limit in (("lower", block.lower_limit), ("upper", block.upper_limit)):
                    if np.isfinite(limit):
                        crossings.append(operation("sub", value, constant(limit)))
                        crossing_metadata.append({"name": f"{block.name}{suffix}:{name}", "enabled": block.zero_crossing})
        return DynamicsModel(tuple(derivatives), outputs, tuple(initializers), tuple(self.defaults), len(self.inputs),
                             tuple(crossings), tuple(crossing_metadata), self.updates())

    def updates(self) -> tuple[Expr, ...]:
        """x+ of the Update phase of a major step: the new memory of the discrete blocks, from the
        signals of the Output phase (a second pass with the blocks' major-step outputs); every
        other state unchanged. () when the model has no discrete memory."""
        blocks = [b for b in self.model.blocks.values() if b.state_size and self.registry.rule_for(b).updates]
        if not blocks:
            return ()
        major = _Extractor(self.model, self.registry, major=True)
        major.state_symbols, major.input_symbols = self.state_symbols, self.input_symbols
        result = [symbol(state["symbol"]) for state in self.states]
        for block in blocks:
            try:
                values = self.registry.rule_for(block).updates(_Context(major, block))
                if len(values) != block.state_size or any(not isinstance(v, Expr) for v in values):
                    raise ValueError("Rule update count/type differs from the runtime state slice.")
            except DynamicsError:
                raise
            except Exception as error:
                raise DynamicsError(str(error), code="block_rule_error", block=block) from error
            condition = major.enabled(block)
            start = block._state_start
            for k, value in enumerate(values):
                result[start + k] = operation("select", condition, value, result[start + k]) if condition is not None else value
        return tuple(result)


def extract_dynamics(model: Model, registry: RuleRegistry | None = None) -> DynamicsResult:
    """Return a structured error report rather than breaking normal conversion."""
    extractor = _Extractor(model, registry or DEFAULT_RULES)
    report = {
        "schema_version": 1, "model_name": model.name, "status": "error", "form": None,
        "states": extractor.states, "inputs": extractor.inputs, "outputs": extractor.outputs,
        "transfer_realizations": extractor.realizations, "errors": [],
        "scope": "Continuous scalar ODE and memoryless algebraic outputs of the converted active graph.",
    }
    report["discrete_components"] = [{
        "block_id": block.block_id, "block_name": block.name,
        "events": list(block.chart.event_names), "latched_outputs": list(block.output_names),
        "states": [state["name"] for state in block.chart.states.values()],
        "required_representation": "xdot=f(x,u,q), y=g(x,u,q), q+=h(x,u,q,event)",
    } for block in model.blocks.values() if type(block).__name__ == "StateflowChart"]
    try:
        extractor.discover()
        for block in model.blocks.values():
            if type(block) not in extractor.registry.rules:
                chart = type(block).__name__ == "StateflowChart"
                report["errors"].append(DynamicsError(
                    "Stateflow keeps discrete state and latched outputs. A complete representation needs xdot=f(x,u,q), y=g(x,u,q) and an event update q+=h(x,u,q). Independent hybrid dynamics extraction is not yet implemented."
                    if chart else "No continuous dynamics rule is registered. Event-driven/cached or discrete state requires a separate update law.",
                    code="hybrid_dynamics_required" if chart else "unsupported_block", block=block).diagnostic())
        if report["errors"]:
            return DynamicsResult(report)
        model._compile_if_needed()
        equations = extractor.equations()
        allowed_symbols = {"t", *(state["symbol"] for state in extractor.states), *(item["symbol"] for item in extractor.inputs)}
        for expr in (*equations.derivatives, *equations.outputs, *equations.initializers, *equations.updates):
            expr.validate(allowed_symbols)
        time_dependent = any("t" in expr.symbols() for expr in (*equations.derivatives, *equations.outputs))
        for state, initial in zip(extractor.states, equations.initializers, strict=True):
            state["initial_expression"] = initial.text()
            state["initial_expression_ir"] = initial.to_dict()
        report["zero_crossings"] = [{**metadata, "python": expr.python(), "expression": expr.text()}
                                    for metadata, expr in zip(equations.zero_crossing_metadata, equations.zero_crossings)]
        report.update({
            "status": "success", "time_dependent": time_dependent,
            "form": ("xdot = f(t,x,u); y = g(t,x,u); x+ = update(t,x,u) at every major step" if equations.updates else
                     "xdot = f(t,x,u); y = g(t,x,u)" if time_dependent else "xdot = f(x,u); y = g(x,u)"),
            "f": [{"state": state["symbol"], "expression": expr.text(), "python": expr.python(), "ir": expr.to_dict()}
                  for state, expr in zip(extractor.states, equations.derivatives, strict=True)],
            "g": [{"output": output["symbol"], "name": output["name"], "expression": expr.text(), "python": expr.python(), "ir": expr.to_dict()}
                  for output, expr in zip(extractor.outputs, equations.outputs, strict=True)],
            # discrete memory: x+ = update(t, x, u) at every major step, before the solver's stages
            "update": [{"state": state["symbol"], "expression": expr.text(), "python": expr.python(), "ir": expr.to_dict()}
                       for state, expr in zip(extractor.states, equations.updates)
                       if state.get("coordinate") == "discrete_memory" or expr != symbol(state["symbol"])],
            "initialization": "All external initial-condition ports are sampled once using the configured default state vector, matching Model.initial_state_at.",
            "numeric_domain": "Real scalar arithmetic. Product/Fcn division preserves IEEE Inf/NaN. Finite state/input checks apply at the exported API boundary.",
        })
        return DynamicsResult(report, equations)
    except Exception as error:
        diagnostic = error.diagnostic() if isinstance(error, DynamicsError) else DynamicsError(str(error)).diagnostic()
        report["errors"].append(diagnostic)
        return DynamicsResult(report)

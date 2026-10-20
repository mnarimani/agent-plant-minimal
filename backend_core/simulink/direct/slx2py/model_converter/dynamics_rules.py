"""Extensible block rules; each rule describes signals and state derivatives.

Register an exact runtime block type with @registry.register(BlockType).
Rules consume symbolic inputs from a RuleContext and return symbolic outputs.
Stateful rules also return one derivative per bound continuous state. Rules
must not evaluate a block, invent state, or read model/user code with eval().
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

import math

import numpy as np

from .blocks import Block, Concatenate, Constant, DiscreteBlock, Demux, Fcn, Gain, Inport, Integrator, Lookup1D, ManualSwitch, MathBlock, MatlabFunction, Mux, Outport, Product, PulseGenerator, Saturate, SecondOrderIntegrator, Selector, StateSpace, Step, Sum, TransferFcn
from .dynamics_expressions import Expr, constant, fcn_python, linear_combination, operation, parse_expression, symbol
from .dynamics_matlab import extract_matlab_function

Signal = Expr | tuple[Expr, ...]


class RuleContext(Protocol):
    block: Block
    states: tuple[Expr, ...]

    def input(self, port: int) -> Signal: ...
    def scalar_input(self, port: int) -> Expr: ...
    def external_input(self, name: str) -> Expr: ...


@dataclass(frozen=True)
class BlockRule:
    outputs: Callable[[RuleContext], tuple[Signal, ...]]
    derivatives: Callable[[RuleContext], tuple[Expr, ...]] | None = None
    initializers: Callable[[RuleContext], tuple[Expr, ...]] | None = None
    # discrete memory: the outputs of the Output phase of a major step, and the new memory of its
    # Update phase (one per state of the block; see blocks.DiscreteBlock)
    major_outputs: Callable[[RuleContext], tuple[Signal, ...]] | None = None
    updates: Callable[[RuleContext], tuple[Expr, ...]] | None = None


class RuleRegistry:
    def __init__(self) -> None:
        self.rules: dict[type[Block], BlockRule] = {}

    def register(self, block_type: type[Block], *, derivatives=None, initializers=None, major_outputs=None,
                 updates=None):
        def decorate(function):
            self.rules[block_type] = BlockRule(function, derivatives, initializers, major_outputs, updates)
            return function
        return decorate

    def rule_for(self, block: Block) -> BlockRule:
        try:
            return self.rules[type(block)]
        except KeyError as error:
            raise ValueError(f"No dynamics rule is registered for {type(block).__name__}.") from error


DEFAULT_RULES = RuleRegistry()


# ---- vector signals: a Signal is an Expr (scalar) or a tuple of Exprs (vector)
def _elements(value: Signal) -> tuple[Expr, ...]:
    return value if isinstance(value, tuple) else (value,)


def _pack(elements) -> Signal:
    elements = tuple(elements)
    return elements[0] if len(elements) == 1 else elements


def _expand(value: Signal, width: int) -> tuple[Expr, ...]:
    """a scalar repeated to the width (scalar expansion), a vector as it is"""
    elements = _elements(value)
    if len(elements) == width:
        return elements
    if len(elements) == 1:
        return elements * width
    raise ValueError(f"A signal of width {len(elements)} where width {width} is needed.")


def _width(*signals: Signal) -> int:
    return max(len(_elements(signal)) for signal in signals)


@DEFAULT_RULES.register(Inport)
def inport(ctx):
    return (_pack(ctx.external_input(name) for name in ctx.block.element_names),)


@DEFAULT_RULES.register(Outport)
def outport(ctx):
    return (ctx.input(0),)


@DEFAULT_RULES.register(Constant)
def constant_block(ctx):
    return (_pack(constant(v) for v in np.atleast_1d(ctx.block.value)),)


@DEFAULT_RULES.register(Step)
def step(ctx):
    block = ctx.block
    return (operation("select", operation("ge", symbol("t"), constant(block.step_time)),
                      constant(block.final_value), constant(block.initial_value)),)


@DEFAULT_RULES.register(PulseGenerator)
def pulse(ctx):
    block = ctx.block
    return (operation("pulse", symbol("t"), *(constant(v) for v in (block.amplitude, block.period, block.width, block.delay))),)


@DEFAULT_RULES.register(Gain)
def gain(ctx):
    block, u = ctx.block, _elements(ctx.input(0))
    if block.multiplication == "Matrix(K*u)":
        return (_pack(linear_combination(row, list(u)) for row in block.gain),)
    if block.multiplication == "Matrix(u*K)":
        return (_pack(linear_combination(column, list(u)) for column in block.gain.T),)
    k = np.atleast_1d(block.gain)
    width = max(k.size, len(u))
    k_all = k if k.size == width else np.repeat(k, width)
    return (_pack(operation("mul", constant(k_all[i]), x) for i, x in enumerate(_expand(ctx.input(0), width))),)


@DEFAULT_RULES.register(Sum)
def sum_block(ctx):
    if ctx.block.input_count == 1:
        # one input: the sum of its elements (Sum of Elements)
        elements = _elements(ctx.input(0))
        return (linear_combination([ctx.block.signs[0]] * len(elements), list(elements)),)
    inputs = [ctx.input(i) for i in range(ctx.block.input_count)]
    width = _width(*inputs)
    columns = [_expand(value, width) for value in inputs]
    return (_pack(linear_combination(ctx.block.signs, [column[k] for column in columns]) for k in range(width)),)


@DEFAULT_RULES.register(Product)
def product(ctx):
    if ctx.block.input_count == 1:
        # one input: the product of its elements (Product of Elements), or its reciprocal
        result = constant(1.0)
        for element in _elements(ctx.input(0)):
            result = operation("mul", result, element)
        return (result if ctx.block.operations == "*" else operation("ieee_div", constant(1.0), result),)
    # Match the runtime's existing first-port convention exactly; element-wise for vectors.
    inputs = [ctx.input(i) for i in range(ctx.block.input_count)]
    width = _width(*inputs)
    columns = [_expand(value, width) for value in inputs]
    results = []
    for k in range(width):
        result = columns[0][k]
        for port, op in enumerate(ctx.block.operations[1:], start=1):
            result = operation("mul" if op == "*" else "ieee_div", result, columns[port][k])
        results.append(result)
    return (_pack(results),)


@DEFAULT_RULES.register(Saturate)
def saturate(ctx):
    return (_pack(operation("min", operation("max", x, constant(ctx.block.lower_limit)), constant(ctx.block.upper_limit))
                  for x in _elements(ctx.input(0))),)


def _clip(value: Expr, lower: float, upper: float) -> Expr:
    if math.isfinite(lower):
        value = operation("max", value, constant(lower))
    if math.isfinite(upper):
        value = operation("min", value, constant(upper))
    return value


def integrator_derivatives(ctx):
    block = ctx.block
    rates = _expand(ctx.input(0), block.state_size)
    if not block.limited:
        return rates
    result = []
    for x, rate in zip(ctx.states, rates):
        stops = []
        if math.isfinite(block.upper_limit):
            stops.append(operation("and", operation("ge", x, constant(block.upper_limit)), operation("gt", rate, constant(0.0))))
        if math.isfinite(block.lower_limit):
            stops.append(operation("and", operation("le", x, constant(block.lower_limit)), operation("lt", rate, constant(0.0))))
        stop = stops[0] if len(stops) == 1 else operation("or", *stops)
        result.append(operation("select", stop, constant(0.0), rate))
    return tuple(result)


def integrator_initializers(ctx):
    block = ctx.block
    if block.external_initial_condition:
        values = _expand(ctx.input(1), block.state_size)
    else:
        values = tuple(constant(v) for v in np.broadcast_to(np.asarray(block.initial_value, dtype=float), (block.state_size,)))
    return tuple(_clip(value, block.lower_limit, block.upper_limit) for value in values)


@DEFAULT_RULES.register(Integrator, derivatives=integrator_derivatives, initializers=integrator_initializers)
def integrator(ctx):
    return (_pack(_clip(x, ctx.block.lower_limit, ctx.block.upper_limit) for x in ctx.states),)


@DEFAULT_RULES.register(SecondOrderIntegrator, derivatives=lambda ctx: (ctx.states[1], ctx.scalar_input(0)),
                        initializers=lambda ctx: (constant(ctx.block.initial_x), constant(ctx.block.initial_dxdt)))
def second_order_integrator(ctx):
    x, dxdt = ctx.states
    return {"both": (x, dxdt), "x": (x,), "dxdt": (dxdt,)}[ctx.block.show_output]


def _state_space_input(ctx) -> tuple[Expr, ...]:
    u = _elements(ctx.input(0))
    if len(u) != ctx.block.input_width:
        raise ValueError(f"State-Space expects an input of width {ctx.block.input_width}, got {len(u)}.")
    return u


def state_space_derivatives(ctx):
    block, u = ctx.block, _state_space_input(ctx)
    return tuple(operation("add", linear_combination(block.a[i], ctx.states), linear_combination(block.b[i], u))
                 for i in range(block.state_size))


@DEFAULT_RULES.register(StateSpace, derivatives=state_space_derivatives,
                        initializers=lambda ctx: tuple(constant(v) for v in ctx.block.initial_conditions))
def state_space(ctx):
    block = ctx.block
    # Do not resolve the input without feedthrough: feedback through the state is no loop.
    u = _state_space_input(ctx) if block.direct_feedthrough else ()
    rows = tuple(operation("add", linear_combination(block.c[i], ctx.states),
                           linear_combination(block.d[i], u) if u else constant(0.0))
                 for i in range(block.output_width))
    return (rows[0] if len(rows) == 1 else rows,)


@DEFAULT_RULES.register(MathBlock)
def math_block(ctx):
    block = ctx.block
    inputs = [ctx.input(i) for i in range(block.input_count)]
    if block.reduce:
        elements = _elements(inputs[0])
        return (elements[0] if len(elements) == 1 else operation(block.reduce, *elements),)
    width = _width(*inputs) if inputs else 1
    columns = [_expand(value, width) for value in inputs]
    outputs = []
    for template in block.templates:
        outputs.append(_pack(template.substitute({f"in{i + 1}": column[k] for i, column in enumerate(columns)})
                             for k in range(width)))
    return tuple(outputs)


@DEFAULT_RULES.register(Demux)
def demux(ctx):
    elements, start, outputs = _elements(ctx.input(0)), 0, []
    for width in ctx.block.output_widths:
        outputs.append(_pack(elements[start:start + width]))
        start += width
    return tuple(outputs)


@DEFAULT_RULES.register(Selector)
def selector(ctx):
    elements = _elements(ctx.input(0))
    return (_pack(elements[i] for i in ctx.block.indices),)


@DEFAULT_RULES.register(Concatenate)
def concatenate(ctx):
    return (_pack(x for i in range(ctx.block.input_count) for x in _elements(ctx.input(i))),)


def transfer_derivatives(ctx):
    block = ctx.block
    return tuple(operation("add", linear_combination(row, ctx.states),
                           linear_combination([coefficient], [ctx.scalar_input(0)]))
                 for row, coefficient in zip(block.state_matrix, block.input_matrix[:, 0], strict=True))


@DEFAULT_RULES.register(TransferFcn, derivatives=transfer_derivatives)
def transfer(ctx):
    block = ctx.block
    result = linear_combination(block.output_matrix[0], ctx.states)
    # Do not resolve a zero D input: strictly proper feedback passes through state.
    if block.feedthrough[0, 0] != 0:
        result = operation("add", result, operation("mul", constant(block.feedthrough[0, 0]), ctx.scalar_input(0)))
    return (result,)


@DEFAULT_RULES.register(MatlabFunction)
def matlab_function(ctx):
    return extract_matlab_function(ctx.block.chart, tuple(ctx.input(i) for i in range(ctx.block.input_count)))


@DEFAULT_RULES.register(Mux)
def mux(ctx):
    return (tuple(x for i in range(ctx.block.input_count) for x in _elements(ctx.input(i))),)


@DEFAULT_RULES.register(ManualSwitch)
def manual_switch(ctx):
    return (ctx.input(ctx.block.selected_input),)


@DEFAULT_RULES.register(Fcn)
def fcn(ctx):
    value = ctx.input(0)
    vector = value if isinstance(value, tuple) else (value,)
    return (parse_expression(fcn_python(ctx.block.expression),
                             {"u": vector, "pi": constant(3.141592653589793)}, ieee_division=True),)


@DEFAULT_RULES.register(Lookup1D)
def lookup(ctx):
    block = ctx.block
    value = ctx.scalar_input(0)
    result = constant(block.table[-1])
    for i in reversed(range(len(block.breakpoints) - 1)):
        slope = (block.table[i + 1] - block.table[i]) / (block.breakpoints[i + 1] - block.breakpoints[i])
        interpolated = operation("add", constant(block.table[i]), operation("mul", constant(slope),
                                 operation("sub", value, constant(block.breakpoints[i]))))
        result = operation("select", operation("lt", value, constant(block.breakpoints[i + 1])), interpolated, result)
    return (operation("select", operation("le", value, constant(block.breakpoints[0])), constant(block.table[0]), result),)


# ---- discrete memory (blocks.DiscreteBlock): the templates over t, in1..inN / e1..en, m1..mK
def _discrete_rows(ctx, templates) -> list[tuple[Expr, ...]]:
    """the templates with the block's inputs and memory substituted: one row per element"""
    block = ctx.block
    used = set().union(*(t.symbols() for t in templates))
    if block.vector:
        values = {f"m{k + 1}": ctx.states[k] for k in range(block.memory_size)}
        if block.input_count and any(name.startswith("e") for name in used):
            values.update({f"e{i + 1}": e for i, e in enumerate(_expand(ctx.input(0), block.vector[0]))})
        return [tuple(t.substitute(values) for t in templates)]
    # only the inputs the templates use: a Memory's output does not depend on its input (no loop)
    columns = [_expand(ctx.input(i), block.width) if f"in{i + 1}" in used else None for i in range(block.input_count)]
    rows = []
    for j in range(block.width):
        values = {f"in{i + 1}": column[j] for i, column in enumerate(columns) if column is not None}
        values.update({f"m{k + 1}": ctx.states[j * block.memory_size + k] for k in range(block.memory_size)})
        rows.append(tuple(t.substitute(values) for t in templates))
    return rows


def _discrete_signals(ctx, templates) -> tuple[Signal, ...]:
    rows = _discrete_rows(ctx, templates)
    if ctx.block.vector:
        return (_pack(rows[0]),)
    return tuple(_pack(row[port] for row in rows) for port in range(ctx.block.output_count))


@DEFAULT_RULES.register(
    DiscreteBlock,
    derivatives=lambda ctx: tuple(constant(0.0) for _ in range(ctx.block.state_size)),
    initializers=lambda ctx: tuple(constant(float(v)) for v in ctx.block.initial_state()),
    major_outputs=lambda ctx: _discrete_signals(ctx, ctx.block.major_templates or ctx.block.templates),
    updates=lambda ctx: tuple(e for row in _discrete_rows(ctx, ctx.block.update_templates) for e in row))
def discrete_block(ctx):
    return _discrete_signals(ctx, ctx.block.templates)

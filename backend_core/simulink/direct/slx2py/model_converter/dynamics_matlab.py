"""Symbolically interpret the same scalar MATLAB statements as the runtime.

Assignments substitute expressions. if/elseif/else merge environments using
lazy piecewise expressions. Locals lacking an assignment on any possible path
cannot be used later. This frontend adds no continuous states: the supported
MATLAB Function language is memoryless. Persistent/discrete memory needs a
separate state/update contract and must not be mislabeled as a continuous ODE.
"""
from __future__ import annotations

import math

from .dynamics_expressions import Expr, constant, operation, parse_expression, symbol
from .stateflow import MatlabFunctionChart, _Assignment, _Conditional, matlab_expression_ast


def extract_matlab_function(chart: MatlabFunctionChart, inputs: tuple[Expr, ...]) -> tuple[Expr, ...]:
    environment = {**dict(zip(chart.input_names, inputs, strict=True)), "time": symbol("t"), "pi": constant(math.pi)}
    _execute(chart.statements, environment)
    missing = set(chart.output_names) - environment.keys()
    if missing:
        raise ValueError("MATLAB Function output is not assigned on every path: " + ", ".join(sorted(missing)))
    return tuple(environment[name] for name in chart.output_names)


def _select(condition, yes, no):
    """if / else of scalars, or element by element of vectors"""
    if isinstance(yes, tuple) or isinstance(no, tuple):
        yes = yes if isinstance(yes, tuple) else (yes,)
        no = no if isinstance(no, tuple) else (no,)
        if len(yes) != len(no):
            raise ValueError("A variable has different sizes in the branches of an if.")
        return tuple(operation("select", condition, a, b) for a, b in zip(yes, no))
    return operation("select", condition, yes, no)


def _execute(statements: tuple, environment: dict[str, Expr]) -> None:
    for statement in statements:
        if isinstance(statement, _Assignment):
            environment[statement.name] = parse_expression(matlab_expression_ast(statement.source).body, environment)
            continue
        if not isinstance(statement, _Conditional):
            raise ValueError(f"Unsupported MATLAB statement {type(statement).__name__}.")
        alternatives: list[tuple[Expr | None, dict[str, Expr]]] = []
        has_else = False
        for branch in statement.branches:
            condition = parse_expression(matlab_expression_ast(branch.source).body, environment) if branch.source is not None else None
            if condition is not None and condition.op == "constant" and not condition.value:
                continue
            branch_environment = dict(environment)
            _execute(branch.statements, branch_environment)
            alternatives.append((condition, branch_environment))
            if condition is None or (condition.op == "constant" and condition.value):
                has_else = True
                break
        if not has_else:
            alternatives.append((None, dict(environment)))
        # Only variables definitely assigned in all possible outcomes survive.
        names = set.intersection(*(set(values) for _, values in alternatives))
        merged = {}
        for name in names:
            expression = alternatives[-1][1][name]
            for condition, values in reversed(alternatives[:-1]):
                expression = _select(condition, values[name], expression)
            merged[name] = expression
        environment.clear()
        environment.update(merged)

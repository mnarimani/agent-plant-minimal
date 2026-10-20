"""Extensible execution primitives for Stateflow charts.

The first chart type is an EML/MATLAB Function chart. Additional chart types
(state machines, events, temporal logic) can implement ``StateflowChart`` and
be evaluated by ``StateflowEngine`` without changing Simulink block code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Protocol
from xml.etree import ElementTree

from .dynamics_expressions import CALLABLE_NAMES, FUNCTION_ALIASES, OPERATIONS

MAX_UNROLLED = 10000        # statements a for loop may unroll to


class StateflowError(ValueError):
    pass


class MatlabVector:
    """A vector input of a MATLAB Function / Fcn block: u(1), u[1] are one-based."""

    def __init__(self, values) -> None:
        self.values = values

    def __getitem__(self, index) -> float:
        if float(index) != int(index) or not 1 <= int(index) <= len(self.values):
            raise IndexError(f"Index {index} is outside the vector of {len(self.values)} elements (one-based).")
        return float(self.values[int(index) - 1])

    __call__ = __getitem__


def _matlab_value(value):
    """a computed MATLAB value: a float, or a vector ([a; b], or a vector input)"""
    if isinstance(value, MatlabVector):
        return value
    if isinstance(value, (list, tuple)):
        flat = [float(v) for item in value
                for v in (item.values if isinstance(item, MatlabVector) else [item])]
        return flat[0] if len(flat) == 1 else MatlabVector(flat)
    return float(value)


# the functions a MATLAB Function may call, with MATLAB's semantics
MATLAB_FUNCTIONS = {name: OPERATIONS[FUNCTION_ALIASES.get(name, name)] for name in CALLABLE_NAMES}


class StateflowChart(Protocol):
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def execute(self, values: dict[str, float], time: float, events: tuple[str, ...] = ()) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class _Assignment:
    name: str
    expression: object
    source: str = ""


@dataclass(frozen=True, slots=True)
class _Branch:
    condition: object | None
    statements: tuple[_Assignment | _Conditional, ...]
    source: str | None = None


@dataclass(frozen=True, slots=True)
class _Conditional:
    branches: tuple[_Branch, ...]


@dataclass(frozen=True, slots=True)
class MatlabFunctionChart:
    """A scalar EML chart with assignments and if/elseif/else branches."""

    source: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    statements: tuple[_Assignment | _Conditional, ...]

    def execute(self, values: dict[str, float], time: float, events: tuple[str, ...] = ()) -> dict[str, float]:
        context = {name: values[name] if isinstance(values[name], MatlabVector) else float(values[name])
                   for name in self.input_names}
        context["time"] = float(time)
        _execute_statements(self.statements, context)
        try:
            return {name: context[name].values.copy() if isinstance(context[name], MatlabVector) else float(context[name])
                    for name in self.output_names}
        except KeyError as error:
            raise StateflowError(f"MATLAB Function did not assign output '{error.args[0]}'.") from error


class StateflowEngine:
    """Executes a chart. Future stateful chart types belong behind this API."""

    def evaluate(
        self,
        chart: StateflowChart,
        values: dict[str, float],
        time: float,
        events: tuple[str, ...] = (),
    ) -> dict[str, float]:
        missing = set(chart.input_names) - set(values)
        if missing:
            raise StateflowError("Missing chart input(s): " + ", ".join(sorted(missing)))
        return chart.execute(values, time, events)


def compile_stateflow_chart(source: str):
    """Compile the documented scalar state-machine subset without model names."""
    factory = globals().get("compile_state_machine")
    if factory is None:
        from .stateflow_machine import compile_state_machine as factory
    return factory(source)


def _xml_parameter(element: ElementTree.Element | None, name: str) -> str | None:
    if element is None:
        return None
    for parameter in element.findall("P"):
        if parameter.get("Name") == name:
            return parameter.text or ""
    return None


def compile_matlab_function(source: str) -> MatlabFunctionChart:
    lines = _normalized_lines(source)
    if not lines:
        raise StateflowError("MATLAB Function source is empty.")
    signature = re.fullmatch(
        r"function\s+(?P<outputs>(?:\[[^]]+\]|[A-Za-z_]\w*))\s*=\s*\w+\s*\((?P<inputs>[^)]*)\)",
        lines[0],
    )
    if signature is None:
        raise StateflowError("Unsupported MATLAB Function signature.")

    output_text = signature.group("outputs").strip("[]")
    output_names = _identifier_list(output_text, "output")
    input_names = _identifier_list(signature.group("inputs"), "input")
    if not input_names:
        raise StateflowError("MATLAB Function requires at least one input.")
    # "function ... end": the last end closes the function, not an if / for
    openers = sum(1 for line in lines[1:] if line.split(maxsplit=1)[0] in {"if", "for", "while", "switch"})
    if lines[-1] == "end" and sum(1 for line in lines[1:] if line == "end") == openers + 1:
        lines = lines[:-1]
    statements, next_index = _parse_statements(lines, 1, stop_words=frozenset())
    if next_index != len(lines):
        raise StateflowError(f"Unexpected MATLAB Function statement: {lines[next_index]!r}.")
    known = set(CALLABLE_NAMES) | set(input_names)
    for tree in _expression_trees(statements):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.func.id not in known:
                raise StateflowError(f"Unsupported MATLAB Function call: {node.func.id}(...).")
    return MatlabFunctionChart(source, input_names, output_names, statements)


def _expression_trees(statements):
    for statement in statements:
        if isinstance(statement, _Assignment):
            yield matlab_expression_ast(statement.source)
        else:
            for branch in statement.branches:
                if branch.source is not None:
                    yield matlab_expression_ast(branch.source)
                yield from _expression_trees(branch.statements)


def _parse_for(lines: list[str], index: int) -> tuple[list, int]:
    """for k = a:b / a:step:b with constant bounds, unrolled: k = value; body; ..."""
    header = re.fullmatch(r"for\s+([A-Za-z_]\w*)\s*=\s*(.+)", lines[index])
    if header is None:
        raise StateflowError(f"Unsupported for loop: {lines[index]!r}.")
    parts = [part.strip() for part in header.group(2).split(":")]
    try:
        numbers = [float(eval(compile(matlab_expression_ast(part), "<for>", "eval"), {"__builtins__": {}}, {}))
                   for part in parts]
    except Exception as error:
        raise StateflowError(f"A for loop needs constant bounds: {lines[index]!r}.") from error
    if len(numbers) == 2:
        start, step, stop = numbers[0], 1.0, numbers[1]
    elif len(numbers) == 3:
        start, step, stop = numbers
    else:
        raise StateflowError(f"Unsupported for loop range: {lines[index]!r}.")
    if step == 0:
        raise StateflowError("A for loop step must not be zero.")
    count = int(math.floor((stop - start) / step + 1e-10)) + 1
    body, end_index = _parse_statements(lines, index + 1, frozenset({"end"}))
    if end_index >= len(lines) or lines[end_index] != "end":
        raise StateflowError("MATLAB Function for loop is missing 'end'.")
    if max(count, 0) * (len(body) + 1) > MAX_UNROLLED:
        raise StateflowError("The for loop is too long to unroll.")
    unrolled: list = []
    for k in range(max(count, 0)):
        value = repr(start + k * step)
        unrolled.append(_Assignment(header.group(1), _compile_expression(value), value))
        unrolled.extend(body)
    return unrolled, end_index + 1


def _normalized_lines(source: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw_line in source.splitlines():
        line = raw_line.split("%", maxsplit=1)[0].strip()
        if line.endswith("..."):              # continued on the next line
            pending += line[:-3] + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if line:
            lines.append(line.rstrip(";").strip())
    if pending.strip():
        lines.append(pending.strip().rstrip(";"))
    return lines


def _matlab_brackets(expression: str) -> str:
    """MATLAB vector literals in Python syntax: [a b; c] -> [a, b, c] (elements separated by
    blanks, commas or semicolons; [1 -2] is two elements, [1 - 2] one), and a transpose ' after
    the bracket dropped (a vector's orientation does not matter here)."""
    out, i = [], 0
    while i < len(expression):
        if expression[i] != "[":
            out.append(expression[i])
            i += 1
            continue
        depth, j = 0, i
        while j < len(expression):
            depth += {"[": 1, "]": -1}.get(expression[j], 0)
            if depth == 0:
                break
            j += 1
        inner = expression[i + 1:j]
        elements: list[str] = []
        for row in re.split(r"[;,]", inner):
            for token in row.split():
                if elements and (token in {"+", "-", "*", "/", "^", ".*", "./", ".^"} or elements[-1][-1] in "+-*/^(."
                                 or token[0] in "*/^)" or elements[-1].count("(") > elements[-1].count(")")):
                    elements[-1] += " " + token
                else:
                    elements.append(token)
        out.append("[" + ", ".join(elements) + "]")
        i = j + 1
        while i < len(expression) and expression[i] in ".'":
            i += 1                            # ' or .' after the bracket
    return "".join(out)


def _identifier_list(text: str, role: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names or any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in names):
        raise StateflowError(f"Invalid MATLAB Function {role} list.")
    if len(set(names)) != len(names):
        raise StateflowError(f"Duplicate MATLAB Function {role} name.")
    return names


def _parse_statements(
    lines: list[str],
    index: int,
    stop_words: frozenset[str],
) -> tuple[tuple[_Assignment | _Conditional, ...], int]:
    statements: list[_Assignment | _Conditional] = []
    while index < len(lines):
        line = lines[index]
        keyword = line.split(maxsplit=1)[0]
        if keyword in stop_words:
            return tuple(statements), index
        if line.startswith("if "):
            conditional, index = _parse_conditional(lines, index)
            statements.append(conditional)
            continue
        if keyword == "for":
            unrolled, index = _parse_for(lines, index)
            statements.extend(unrolled)
            continue
        assignment = re.fullmatch(r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<expression>.+)", line)
        if assignment is None:
            raise StateflowError(f"Unsupported MATLAB Function statement: {line!r}.")
        expression = assignment.group("expression")
        statements.append(_Assignment(assignment.group("name"), _compile_expression(expression), expression))
        index += 1
    if stop_words:
        raise StateflowError("MATLAB Function is missing 'end'.")
    return tuple(statements), index


def _parse_conditional(lines: list[str], index: int) -> tuple[_Conditional, int]:
    branches: list[_Branch] = []
    while True:
        header = lines[index]
        if header.startswith("if "):
            condition_text = header[3:].strip()
        elif header.startswith("elseif "):
            condition_text = header[7:].strip()
        elif header == "else":
            condition_text = None
        else:
            raise StateflowError(f"Invalid conditional header: {header!r}.")
        condition = _compile_expression(condition_text) if condition_text is not None else None
        statements, index = _parse_statements(lines, index + 1, frozenset({"elseif", "else", "end"}))
        branches.append(_Branch(condition, statements, condition_text))
        if index >= len(lines):
            raise StateflowError("MATLAB Function conditional is missing 'end'.")
        if lines[index] == "end":
            return _Conditional(tuple(branches)), index + 1
        if lines[index] == "else":
            continue
        if lines[index].startswith("elseif "):
            continue
        raise StateflowError(f"Invalid conditional terminator: {lines[index]!r}.")


def matlab_expression_ast(expression: str) -> ast.Expression:
    """One checked syntax tree for numerical execution and symbolic extraction."""
    python_expression = (
        _matlab_brackets(expression).replace(".*", "*")
        .replace("./", "/")
        .replace(".^", "**")
        .replace("~=", "!=")
        .replace("&&", " and ")
        .replace("||", " or ")
        .replace("^", "**")
        .replace("~", " not ")
    )
    try:
        tree = ast.parse(python_expression, mode="eval")
    except SyntaxError as error:
        raise StateflowError(f"Unsupported MATLAB Function expression: {expression!r}.") from error
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Name, ast.Load, ast.Constant, ast.Call, ast.List,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Not,
        ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        raise StateflowError(f"Unsupported MATLAB Function expression: {expression!r}.")
    if any(isinstance(node, ast.Call) and not isinstance(node.func, ast.Name) for node in ast.walk(tree)):
        raise StateflowError(f"Unsupported MATLAB Function expression: {expression!r}.")
    # calls are functions (CALLABLE_NAMES) or the indexing of a vector input, u(1);
    # compile_matlab_function checks which, as it knows the inputs
    return tree


def _compile_expression(expression: str):
    return compile(matlab_expression_ast(expression), "<stateflow-eml>", "eval")


def _execute_statements(statements: tuple[_Assignment | _Conditional, ...], context: dict[str, float]) -> None:
    for statement in statements:
        if isinstance(statement, _Assignment):
            statement_value = _evaluate_expression(statement.expression, context)
            context[statement.name] = _matlab_value(statement_value)
            continue
        for branch in statement.branches:
            if branch.condition is None or bool(_evaluate_expression(branch.condition, context)):
                _execute_statements(branch.statements, context)
                break


def _evaluate_expression(expression: object, context: dict[str, float]) -> float | bool:
    try:
        return eval(expression, {"__builtins__": {}}, {**MATLAB_FUNCTIONS, "pi": math.pi, **context})
    except NameError as error:
        raise StateflowError(f"Unknown MATLAB Function variable: {error.name!r}.") from error

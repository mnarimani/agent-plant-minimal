"""Small, serializable scalar expression IR shared by extraction and rendering.

No model code is executed during extraction. Symbols are generated x1/u1/t,
never identifiers copied from an uploaded model. New operations must define
both their numerical interpretation and Python rendering here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np
from .waveforms import pulse_value


@dataclass(frozen=True, slots=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    value: float | bool | str | None = None

    def to_dict(self) -> dict:
        if self.op in {"constant", "symbol"}:
            return {"op": self.op, "value": self.value}
        return {"op": self.op, "args": [arg.to_dict() for arg in self.args]}

    def symbols(self) -> set[str]:
        if self.op == "symbol":
            return {str(self.value)}
        return set().union(*(arg.symbols() for arg in self.args))

    def validate(self, allowed_symbols: set[str]) -> None:
        if self.op == "symbol":
            if self.value not in allowed_symbols or self.args:
                raise ValueError(f"Invalid/unbound dynamics symbol {self.value!r}.")
            return
        if self.op == "constant":
            if self.args or not isinstance(self.value, (float, int, bool)) or not math.isfinite(self.value):
                raise ValueError("Invalid dynamics constant.")
            return
        arity = (5 if self.op == "pulse" else 2 if self.op in BINARY_TEXT or self.op == "ieee_div" else
                 3 if self.op == "select" else 1 if self.op in {"neg", "not"} else
                 FUNCTIONS[self.op][0] if self.op in FUNCTIONS else None)
        if self.op in TABLE_FUNCTIONS:
            if len(self.args) < 3:
                raise ValueError(f"Table lookup {self.op!r} needs its table.")
        elif self.op in {"and", "or", "min", "max"}:
            if len(self.args) < 2:
                raise ValueError(f"Dynamics operation {self.op!r} requires at least two operands.")
        elif arity is None or len(self.args) != arity:
            raise ValueError(f"Unsupported dynamics operation or invalid arity: {self.op!r}.")
        for arg in self.args:
            arg.validate(allowed_symbols)

    def evaluate(self, values: Mapping[str, float]) -> float | bool:
        if self.op == "constant":
            return self.value
        if self.op == "symbol":
            return values[str(self.value)]
        # Branches and boolean expressions must be lazy (e.g. guarded division).
        if self.op == "select":
            condition, yes, no = self.args
            return (yes if condition.evaluate(values) else no).evaluate(values)
        if self.op == "and":
            return all(arg.evaluate(values) for arg in self.args)
        if self.op == "or":
            return any(arg.evaluate(values) for arg in self.args)
        operands = [arg.evaluate(values) for arg in self.args]
        if self.op == "ieee_div":
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(np.divide(*operands))
        return OPERATIONS[self.op](*operands)

    def python(self) -> str:
        if self.op == "constant":
            return repr(self.value)
        if self.op == "symbol":
            return str(self.value)
        parts = [arg.python() for arg in self.args]
        if self.op in BINARY_TEXT:
            return f"({parts[0]} {BINARY_TEXT[self.op]} {parts[1]})"
        if self.op in {"and", "or"}:
            return "(" + f" {self.op} ".join(parts) + ")"
        if self.op == "neg":
            return f"(-{parts[0]})"
        if self.op == "not":
            return f"(not {parts[0]})"
        if self.op == "select":
            return f"({parts[1]} if {parts[0]} else {parts[2]})"
        name = ("_divide" if self.op == "ieee_div" else "pulse_value" if self.op == "pulse" else
                FUNCTIONS[self.op][2] if self.op in FUNCTIONS else self.op)
        return f"{name}({', '.join(parts)})"

    @classmethod
    def from_dict(cls, data: Mapping) -> Expr:
        """Inverse of to_dict (the templates of MathBlock are exported this way)."""
        if data["op"] in {"constant", "symbol"}:
            return cls(data["op"], value=data["value"])
        return cls(data["op"], tuple(cls.from_dict(arg) for arg in data["args"]))

    def substitute(self, mapping: Mapping[str, Expr]) -> Expr:
        """Replace symbols (e.g. a block's in1..inN by the expressions of its inputs)."""
        if self.op == "symbol":
            return mapping.get(str(self.value), self)
        if self.op == "constant":
            return self
        return operation(self.op, *(arg.substitute(mapping) for arg in self.args))

    def text(self) -> str:
        # The documented equation notation differs only for IEEE division.
        if self.op == "ieee_div":
            return f"({self.args[0].text()} / {self.args[1].text()})"
        if self.op in {"constant", "symbol"}:
            return self.python()
        parts = [arg.text() for arg in self.args]
        if self.op in BINARY_TEXT:
            return f"({parts[0]} {BINARY_TEXT[self.op]} {parts[1]})"
        if self.op == "select":
            return f"({parts[1]} if {parts[0]} else {parts[2]})"
        if self.op in {"and", "or"}:
            return "(" + f" {self.op} ".join(parts) + ")"
        if self.op == "neg":
            return f"(-{parts[0]})"
        if self.op == "not":
            return f"(not {parts[0]})"
        return f"{self.op}({', '.join(parts)})"


BINARY_TEXT = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
               "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
OPERATIONS = {
    "pulse": pulse_value,
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "pow": lambda a, b: a ** b, "neg": lambda a: -a, "not": lambda a: not a,
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "min": min, "max": max,
}


def _np(function):
    """IEEE semantics like Simulink's double arithmetic: log(-1) is NaN, not an exception."""
    def apply(*args):
        with np.errstate(all="ignore"):
            return float(function(*args))
    return apply


def _round_half_away(value):
    """MATLAB / Simulink round: halves away from zero (numpy rounds them to even)."""
    return np.sign(value) * np.floor(np.abs(value) + 0.5)


def _matlab_mod(a, b):
    """MATLAB mod: the sign of the divisor, and mod(x, 0) = x."""
    return a if b == 0 else a - np.floor(a / b) * b


def _hit(t, period, offset):
    """1 at the sample times offset + k * period (k = 0, 1, ...) of a discrete block, else 0"""
    k = round((t - offset) / period)
    return float(k >= 0 and abs(t - offset - k * period) <= 1e-9 * max(1.0, abs(t)))


# name -> (arity, numerical function, Python rendering); the rendering runs in plain Python with
# numpy as np (helpers named _... are defined by python_prelude)
FUNCTIONS = {
    "abs": (1, abs, "abs"),
    "exp": (1, _np(np.exp), "np.exp"), "log": (1, _np(np.log), "np.log"), "log10": (1, _np(np.log10), "np.log10"),
    "sqrt": (1, _np(np.sqrt), "np.sqrt"),
    "sin": (1, _np(np.sin), "np.sin"), "cos": (1, _np(np.cos), "np.cos"), "tan": (1, _np(np.tan), "np.tan"),
    "asin": (1, _np(np.arcsin), "np.arcsin"), "acos": (1, _np(np.arccos), "np.arccos"),
    "atan": (1, _np(np.arctan), "np.arctan"),
    "sinh": (1, _np(np.sinh), "np.sinh"), "cosh": (1, _np(np.cosh), "np.cosh"), "tanh": (1, _np(np.tanh), "np.tanh"),
    "asinh": (1, _np(np.arcsinh), "np.arcsinh"), "acosh": (1, _np(np.arccosh), "np.arccosh"),
    "atanh": (1, _np(np.arctanh), "np.arctanh"),
    "floor": (1, _np(np.floor), "np.floor"), "ceil": (1, _np(np.ceil), "np.ceil"), "fix": (1, _np(np.trunc), "np.trunc"),
    "round": (1, _np(_round_half_away), "_round"), "sign": (1, _np(np.sign), "np.sign"),
    "atan2": (2, _np(np.arctan2), "np.arctan2"), "hypot": (2, _np(np.hypot), "np.hypot"),
    "rem": (2, _np(np.fmod), "np.fmod"), "mod": (2, _np(_matlab_mod), "_mod"), "power": (2, _np(np.power), "np.power"),
    "hit": (3, _hit, "_hit"),
}
OPERATIONS = {**OPERATIONS, **{name: spec[1] for name, spec in FUNCTIONS.items()}}   # an assignment: model.py keeps only definitions

# definitions a rendered expression may call besides numpy (see python_prelude)
PYTHON_HELPERS = {
    "_divide": "def _divide(a, b):\n    # IEEE division (x/0 -> inf, 0/0 -> nan), like Simulink's Product / Fcn blocks\n"
               "    return float(np.float64(a) / np.float64(b))\n",
    "_round": "def _round(a):\n    # MATLAB round: halves away from zero\n"
              "    return float(np.sign(a) * np.floor(np.abs(a) + 0.5))\n",
    "_hit": "def _hit(t, period, offset):\n    # 1 at the sample times offset + k * period of a discrete block, else 0\n"
            "    k = round((t - offset) / period)\n"
            "    return float(k >= 0 and abs(t - offset - k * period) <= 1e-9 * max(1.0, abs(t)))\n",
    "_mod": "def _mod(a, b):\n    # MATLAB mod: the sign of the divisor, mod(a, 0) = a\n"
            "    return float(a if b == 0 else a - np.floor(a / b) * b)\n",
}


# ---- table lookups (Lookup_n-D, Direct Lookup, Repeating Sequence). The tables are constant
# arguments: f(u, n, breakpoints..., values...). Plain Python, no numpy needed, so the same
# source runs in the block model and in a rendered dynamics(t, x, u) (python_prelude).
def _segment(u, n, bp):
    """index i of the segment [bp[i], bp[i+1]] used for u (the end segments extrapolate)"""
    i = 0
    while i < n - 2 and u >= bp[i + 1]:
        i += 1
    return i


def _interp1(u, n, *data):
    """linear interpolation, linear extrapolation (Simulink's default)"""
    n = int(n)
    bp, tb = data[:n], data[n:2 * n]
    i = _segment(u, n, bp)
    return tb[i] + (tb[i + 1] - tb[i]) * (u - bp[i]) / (bp[i + 1] - bp[i])


def _interp1_clip(u, n, *data):
    n = int(n)
    return _interp1(min(max(u, data[0]), data[n - 1]), n, *data)


def _interp1_flat(u, n, *data):
    """the value of the breakpoint at or below u; the end values beyond the table"""
    n = int(n)
    bp, tb = data[:n], data[n:2 * n]
    i = 0
    while i < n - 1 and u >= bp[i + 1]:
        i += 1
    return tb[i]


def _interp1_nearest(u, n, *data):
    n = int(n)
    bp, tb = data[:n], data[n:2 * n]
    u = min(max(u, bp[0]), bp[n - 1])
    i = _segment(u, n, bp)
    return tb[i] if u - bp[i] < bp[i + 1] - u else tb[i + 1]


def _spline1(u, n, clip, *data):
    """natural cubic spline (second derivatives m given); beyond the table: the end tangents,
    or the end values (clip)"""
    n = int(n)
    bp, tb, m = data[:n], data[n:2 * n], data[2 * n:3 * n]
    if u < bp[0] or u > bp[n - 1]:
        i, x = (0, bp[0]) if u < bp[0] else (n - 2, bp[n - 1])
        h = bp[i + 1] - bp[i]
        a, b = (x - bp[i]) / h, (bp[i + 1] - x) / h
        slope = (tb[i + 1] - tb[i]) / h + h * ((3 * a * a - 1) * m[i + 1] - (3 * b * b - 1) * m[i]) / 6
        value = tb[0] if u < bp[0] else tb[n - 1]
        return value if clip else value + slope * (u - x)
    i = _segment(u, n, bp)
    h = bp[i + 1] - bp[i]
    a, b = (u - bp[i]) / h, (bp[i + 1] - u) / h
    return b * tb[i] + a * tb[i + 1] + h * h * ((b ** 3 - b) * m[i] + (a ** 3 - a) * m[i + 1]) / 6


def _interp2(u1, u2, n1, n2, clip, *data):
    """bilinear interpolation over table[i][j] (row i: breakpoint i of u1), linear extrapolation
    or the edges (clip)"""
    n1, n2 = int(n1), int(n2)
    bp1, bp2, tb = data[:n1], data[n1:n1 + n2], data[n1 + n2:]
    if clip:
        u1, u2 = min(max(u1, bp1[0]), bp1[n1 - 1]), min(max(u2, bp2[0]), bp2[n2 - 1])
    i, j = _segment(u1, n1, bp1), _segment(u2, n2, bp2)
    a = (u1 - bp1[i]) / (bp1[i + 1] - bp1[i])
    b = (u2 - bp2[j]) / (bp2[j + 1] - bp2[j])
    v00, v01 = tb[i * n2 + j], tb[i * n2 + j + 1]
    v10, v11 = tb[(i + 1) * n2 + j], tb[(i + 1) * n2 + j + 1]
    return (1 - a) * ((1 - b) * v00 + b * v01) + a * ((1 - b) * v10 + b * v11)


def _direct1(u, n, *table):
    """Direct Lookup Table: zero-based index, truncated and kept inside the table"""
    n = int(n)
    k = int(min(max(u, 0), n - 1))
    return table[k]


def _repeat(t, n, *data):
    """Repeating Sequence: the times / values repeat with the period of the last time"""
    n = int(n)
    times, values = data[:n], data[n:2 * n]
    period = times[n - 1] - times[0]
    tau = times[0] + (t - times[0]) % period if period > 0 else times[0]
    return _interp1_clip(tau, n, *times, *values)


# variable number of arguments; all are part of the plain-Python prelude
TABLE_FUNCTIONS = {"_interp1": _interp1, "_interp1_clip": _interp1_clip, "_interp1_flat": _interp1_flat,
                   "_interp1_nearest": _interp1_nearest, "_spline1": _spline1, "_interp2": _interp2,
                   "_direct1": _direct1, "_repeat": _repeat}
OPERATIONS = {**OPERATIONS, **TABLE_FUNCTIONS}


def fcn_python(expression: str) -> str:
    """A Simulink Fcn expression (C-like operators, ^ for power) in Python syntax."""
    import re
    text = expression.replace("^", "**").replace("&&", " and ").replace("||", " or ").replace("~=", "!=")
    return re.sub(r"!(?!=)", " not ", text)


# other names of the same functions in Simulink Fcn blocks (C-like) and MATLAB code
FUNCTION_ALIASES = {"fabs": "abs", "ln": "log", "sgn": "sign", "pow": "power", "asin": "asin",
                    "arcsin": "asin", "arccos": "acos", "arctan": "atan"}
# the names a Fcn / MATLAB Function expression may call
CALLABLE_NAMES = frozenset({"min", "max", *FUNCTIONS, *FUNCTION_ALIASES})


def python_prelude(source: str) -> list[str]:
    """The helper definitions the rendered expressions in ``source`` call (numpy is np)."""
    import inspect
    lines = [code for name, code in PYTHON_HELPERS.items() if name + "(" in source]
    if "pulse_value(" in source:
        lines.append(inspect.getsource(pulse_value))
    used = [name for name in TABLE_FUNCTIONS if name + "(" in source]
    if used:
        # the table functions call each other: give all of them
        lines += [inspect.getsource(f) for f in (_segment, *TABLE_FUNCTIONS.values())]
    return lines


def truth(value: Expr) -> Expr:
    """A logical value as Simulink's double signal: 1 or 0."""
    return operation("select", value, constant(1.0), constant(0.0))


def nonzero(value: Expr) -> Expr:
    return operation("ne", value, constant(0.0))


def constant(value: float | bool) -> Expr:
    value = value if isinstance(value, bool) else float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite constants cannot be exported as real scalar equations.")
    return Expr("constant", value=value)


def symbol(name: str) -> Expr:
    return Expr("symbol", value=name)


def operation(op: str, *args: Expr) -> Expr:
    if op == "select":
        condition, yes, no = args
        if condition.op == "constant":
            return yes if condition.value else no
        if yes == no:
            return yes
    if op in {"add", "sub"} and args[1] == constant(0):
        return args[0]
    if op == "add" and args[0] == constant(0):
        return args[1]
    if op == "mul" and args[0] == constant(1):
        return args[1]
    if op == "mul" and args[1] == constant(1):
        return args[0]
    if op in {"div", "pow"} and args[1] == constant(1):
        return args[0]
    result = Expr(op, tuple(args))
    # Fold only defined finite scalar constants; preserve singular operations.
    if op in OPERATIONS and all(arg.op == "constant" for arg in args):
        try:
            return constant(result.evaluate({}))
        except (ArithmeticError, ValueError, TypeError):
            pass
    return result


def linear_combination(coefficients: Sequence[float], expressions: Sequence[Expr]) -> Expr:
    result = constant(0)
    for coefficient, expression in zip(coefficients, expressions, strict=True):
        if coefficient != 0:
            result = operation("add", result, operation("mul", constant(coefficient), expression))
    return result


def parse_expression(source: str | ast.AST, context: Mapping[str, Expr | tuple[Expr, ...]], *, ieee_division: bool = False) -> Expr:
    """Translate a whitelisted AST, substituting local variables structurally."""
    node = ast.parse(source, mode="eval").body if isinstance(source, str) else source

    def scalar(value: Expr | tuple[Expr, ...]) -> Expr:
        if isinstance(value, tuple) and len(value) == 1:
            return value[0]
        if not isinstance(value, Expr):
            raise ValueError("A vector was used where a scalar expression is required.")
        return value

    def element(vector: Expr | tuple[Expr, ...], index: ast.AST) -> Expr:
        """u(1) / u[1]: MATLAB one-based indexing with a constant index"""
        vector = vector if isinstance(vector, tuple) else (vector,)
        if isinstance(index, ast.Constant) and type(index.value) in (int, float) and float(index.value).is_integer() \
                and 1 <= int(index.value) <= len(vector):
            return vector[int(index.value) - 1]
        raise ValueError("Vector indices must be constant, valid, one-based indices.")

    def visit(part: ast.AST) -> Expr:
        if isinstance(part, ast.Constant) and isinstance(part.value, (int, float, bool)):
            return constant(part.value)
        if isinstance(part, ast.Name):
            if part.id not in context:
                raise ValueError(f"Variable {part.id!r} is undefined on at least one execution path.")
            return scalar(context[part.id])
        if isinstance(part, ast.BinOp):
            names = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "ieee_div" if ieee_division else "div", ast.Pow: "pow"}
            if type(part.op) in names:
                return operation(names[type(part.op)], visit(part.left), visit(part.right))
        if isinstance(part, ast.UnaryOp):
            names = {ast.USub: "neg", ast.Not: "not"}
            if isinstance(part.op, ast.UAdd):
                return visit(part.operand)
            if type(part.op) in names:
                return operation(names[type(part.op)], visit(part.operand))
        if isinstance(part, ast.BoolOp):
            return operation("and" if isinstance(part.op, ast.And) else "or", *(visit(v) for v in part.values))
        if isinstance(part, ast.Compare):
            names = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
            terms = []
            left = part.left
            for operator, right in zip(part.ops, part.comparators, strict=True):
                terms.append(operation(names[type(operator)], visit(left), visit(right)))
                left = right
            return terms[0] if len(terms) == 1 else operation("and", *terms)
        if isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name) and part.value.id in context:
            return element(context[part.value.id], part.slice)
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and not part.keywords:
            name = part.func.id
            if name in context:
                if len(part.args) == 1:
                    return element(context[name], part.args[0])      # u(1): indexing, not a call
                raise ValueError(f"Function {name!r} is shadowed by a local variable.")
            if name in {"min", "max"} and len(part.args) >= 2:
                return operation(name, *(visit(arg) for arg in part.args))
            name = FUNCTION_ALIASES.get(name, name)
            if name in FUNCTIONS and len(part.args) == FUNCTIONS[name][0]:
                return operation(name, *(visit(arg) for arg in part.args))
        raise ValueError(f"Unsupported scalar expression node: {ast.dump(part, include_attributes=False)}")

    if isinstance(node, ast.List):
        # [a; b; c]: a vector (its elements may be vectors themselves)
        elements = []
        for item in node.elts:
            if isinstance(item, ast.Name) and isinstance(context.get(item.id), tuple):
                elements.extend(context[item.id])
            else:
                elements.append(visit(item))
        return tuple(elements)
    if isinstance(node, ast.Name) and isinstance(context.get(node.id), tuple):
        return context[node.id]               # a whole vector variable
    return visit(node)

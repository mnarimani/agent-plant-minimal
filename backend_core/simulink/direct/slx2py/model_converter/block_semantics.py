"""Memoryless Simulink blocks as expressions: block type + dialog parameters -> one expression
per output, over the block's inputs in1..inN and the time t (a MathBlock).

The defaults are Simulink's (R2025b): an .slx omits every parameter left at its default, so a
missing parameter means the default below. They, and the semantics, are checked block by
block against Simulink itself (tests/slx_conformance.py, with the models and results of
matlab_block_tests/make_slx_block_reference.m).

``Params`` reads a block's parameters; library blocks (BlockType "Reference") are keyed
"Reference:<SourceType>" and their mask parameters come from the block's InstanceData.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dynamics_expressions import Expr, constant, nonzero, operation, symbol, truth


class SemanticsError(ValueError):
    pass


@dataclass
class Params:
    text: Callable[[str], str | None]            # the raw parameter, None if not stored
    number: Callable[[str, float], float]        # a scalar parameter (MATLAB expression)
    matrix: Callable[[str, object], np.ndarray]  # a vector / matrix parameter
    start_time: float = 0.0

    def get(self, name: str, default: str) -> str:
        value = self.text(name)
        return default if value is None or value == "" else value


def build(block_type: str, p: Params) -> tuple | None:
    """-> (kind, input count, output expressions[, reduction]), or None: not a memoryless block of
    this module. The expressions are per element: on vectors the block works element-wise (scalars
    expanded); the reduction "min" / "max" works over the elements of its one input instead."""
    rule = RULES.get(block_type)
    if rule is None:
        return None
    return rule(p)


def _in(i: int) -> Expr:
    return symbol(f"in{i}")


T = symbol("t")
RULES: dict[str, Callable[[Params], tuple[str, int, list[Expr]]]] = {}


def rule(*names: str):
    def register(function):
        for name in names:
            RULES[name] = function
        return function
    return register


def op(name: str, *args: Expr) -> Expr:
    return operation(name, *args)


def c(value: float) -> Expr:
    return constant(float(value))


def _scalar(p: Params, name: str, default: float) -> float:
    value = np.asarray(p.matrix(name, default), dtype=float).ravel()
    if value.size != 1:
        raise SemanticsError(f"Parameter {name} must be a scalar here (vector signals are not supported yet).")
    return float(value[0])


# ---------------------------------------------------------------------------------------------
# sources (functions of t)
# ---------------------------------------------------------------------------------------------
@rule("Clock")
def clock(p):
    return "Clock", 0, [T]


@rule("Ground")
def ground(p):
    return "Ground", 0, [c(0)]


@rule("Sin")
def sine(p):
    if p.get("SineType", "Time based") != "Time based" or p.get("TimeSource", "Use simulation time") != "Use simulation time":
        raise SemanticsError("Only the time-based Sine Wave with simulation time is supported.")
    a, b, f, ph = (_scalar(p, n, d) for n, d in (("Amplitude", 1), ("Bias", 0), ("Frequency", 1), ("Phase", 0)))
    return "Sine Wave", 0, [op("add", op("mul", c(a), op("sin", op("add", op("mul", c(f), T), c(ph)))), c(b))]


@rule("SignalGenerator")
def signal_generator(p):
    wave = p.get("WaveForm", "sine")
    a, f = _scalar(p, "Amplitude", 1), _scalar(p, "Frequency", 1)
    hz = f if p.get("Units", "Hertz") == "Hertz" else f / (2 * math.pi)
    if wave == "sine":
        y = op("mul", c(a), op("sin", op("mul", c(2 * math.pi * hz), T)))
    else:
        phase = op("mod", op("mul", c(hz), T), c(1))           # 0 .. 1 over each period
        if wave == "square":                                   # -A for the first half period
            y = op("select", op("ge", phase, c(0.5)), c(a), c(-a))
        elif wave == "sawtooth":                               # from A down to -A
            y = op("mul", c(a), op("sub", c(1), op("mul", c(2), phase)))
        else:
            raise SemanticsError(f"Signal Generator wave form {wave!r} is not supported.")
    return "Signal Generator (%s)" % wave, 0, [y]


@rule("Reference:Ramp")
def ramp(p):
    slope, start = _scalar(p, "slope", 1), _scalar(p, "start", 0)
    x0 = _scalar(p, "InitialOutput", _scalar(p, "X0", 0))
    return "Ramp", 0, [op("select", op("ge", T, c(start)), op("add", c(x0), op("mul", c(slope), op("sub", T, c(start)))), c(x0))]


@rule("Reference:chirp")
def chirp(p):
    f1, period, f2 = _scalar(p, "f1", 0.1), _scalar(p, "T", 100), _scalar(p, "f2", 1)
    frequency = op("add", c(f1), op("mul", c((f2 - f1) / (2 * period)), T))
    return "Chirp Signal", 0, [op("sin", op("mul", c(2 * math.pi), op("mul", frequency, T)))]


@rule("Reference:Repeating table")
def repeating_sequence(p):
    times = np.asarray(p.matrix("rep_seq_t", [0, 2]), dtype=float).ravel()
    values = np.asarray(p.matrix("rep_seq_y", [0, 2]), dtype=float).ravel()
    if times.size != values.size or times.size < 2 or np.any(np.diff(times) < 0):
        raise SemanticsError("Repeating Sequence needs increasing times and one value per time.")
    return "Repeating Sequence", 0, [op("_repeat", T, c(times.size), *map(c, times), *map(c, values))]


# ---------------------------------------------------------------------------------------------
# math operations
# ---------------------------------------------------------------------------------------------
@rule("Abs")
def abs_block(p):
    return "Abs", 1, [op("abs", _in(1))]


@rule("Signum")
def signum(p):
    return "Sign", 1, [op("sign", _in(1))]


@rule("UnaryMinus")
def unary_minus(p):
    return "Unary Minus", 1, [op("neg", _in(1))]


@rule("Bias")
def bias(p):
    return "Bias", 1, [op("add", _in(1), c(_scalar(p, "Bias", 0)))]


@rule("Polyval")
def polynomial(p):
    coefficients = np.asarray(p.matrix("Coefs", [2.081618890e-19, -1.441693666e-14, 4.719686976e-10,
                                                 -8.536869453e-06, 0.1144057540, -2.035270840e-06]), dtype=float).ravel()
    y = c(coefficients[0])
    for k in coefficients[1:]:                                 # Horner
        y = op("add", op("mul", y, _in(1)), c(k))
    return "Polynomial", 1, [y]


@rule("Sqrt")
def square_root(p):
    operator = p.get("Operator", "sqrt")
    u = _in(1)
    if operator == "sqrt":
        y = op("sqrt", u)
    elif operator == "signedSqrt":
        y = op("mul", op("sign", u), op("sqrt", op("abs", u)))
    elif operator == "rSqrt":
        y = op("ieee_div", c(1), op("sqrt", u))
    else:
        raise SemanticsError(f"Sqrt function {operator!r} is not supported.")
    return "Sqrt (%s)" % operator, 1, [y]


_MATH_UNARY = {"exp": lambda u: op("exp", u), "log": lambda u: op("log", u), "log10": lambda u: op("log10", u),
               "10^u": lambda u: op("power", c(10), u), "magnitude^2": lambda u: op("mul", u, u),
               "square": lambda u: op("mul", u, u), "reciprocal": lambda u: op("ieee_div", c(1), u),
               "sqrt": lambda u: op("sqrt", u), "conj": lambda u: u, "transpose": lambda u: u,
               "hermitian": lambda u: u}
_MATH_BINARY = {"pow": "power", "hypot": "hypot", "rem": "rem", "mod": "mod"}


@rule("Math")
def math_function(p):
    operator = p.get("Operator", "exp")
    if operator in _MATH_UNARY:
        return "Math Function (%s)" % operator, 1, [_MATH_UNARY[operator](_in(1))]
    if operator in _MATH_BINARY:
        return "Math Function (%s)" % operator, 2, [op(_MATH_BINARY[operator], _in(1), _in(2))]
    raise SemanticsError(f"Math Function {operator!r} is not supported.")


_TRIG = {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}


@rule("Trigonometry")
def trigonometry(p):
    operator = p.get("Operator", "sin")
    if p.get("ApproximationMethod", "None") != "None":
        raise SemanticsError("Trigonometric Function approximations (CORDIC) are not supported.")
    if operator in _TRIG:
        return "Trigonometric Function (%s)" % operator, 1, [op(operator, _in(1))]
    if operator == "atan2":
        return "Trigonometric Function (atan2)", 2, [op("atan2", _in(1), _in(2))]
    if operator == "sincos":
        return "Trigonometric Function (sincos)", 1, [op("sin", _in(1)), op("cos", _in(1))]
    raise SemanticsError(f"Trigonometric Function {operator!r} is not supported.")


@rule("Rounding")
def rounding(p):
    operator = p.get("Operator", "floor")
    if operator not in {"floor", "ceil", "round", "fix"}:
        raise SemanticsError(f"Rounding function {operator!r} is not supported.")
    return "Rounding Function (%s)" % operator, 1, [op(operator, _in(1))]


@rule("MinMax")
def min_max(p):
    function = p.get("Function", "min")
    n = int(_scalar(p, "Inputs", 1))
    if n < 2:
        # one input: the min / max of its elements; of a scalar, the scalar
        return "MinMax (%s)" % function, 1, [_in(1)], function
    return "MinMax (%s)" % function, n, [op(function, *(_in(i) for i in range(1, n + 1)))]


# ---------------------------------------------------------------------------------------------
# discontinuities
# ---------------------------------------------------------------------------------------------
def _dead_zone(u: Expr, upper: Expr, lower: Expr) -> Expr:
    return op("select", op("ge", u, upper), op("sub", u, upper),
              op("select", op("gt", u, lower), c(0), op("sub", u, lower)))


@rule("DeadZone")
def dead_zone(p):
    return "Dead Zone", 1, [_dead_zone(_in(1), c(_scalar(p, "UpperValue", 0.5)), c(_scalar(p, "LowerValue", -0.5)))]


@rule("Quantizer")
def quantizer(p):
    q = _scalar(p, "QuantizationInterval", 0.5)
    return "Quantizer", 1, [op("mul", c(q), op("round", op("ieee_div", _in(1), c(q))))]


@rule("Reference:Coulombic and Viscous Friction")
def coulomb_viscous(p):
    offset, gain = _scalar(p, "offset", 1), _scalar(p, "gain", 1)
    u = _in(1)
    return "Coulomb & Viscous Friction", 1, [op("mul", op("sign", u), op("add", op("mul", c(gain), op("abs", u)), c(offset)))]


@rule("Reference:Saturation Dynamic")
def saturation_dynamic(p):
    return "Saturation Dynamic", 3, [op("min", op("max", _in(2), _in(3)), _in(1))]


@rule("Reference:Dead Zone Dynamic")
def dead_zone_dynamic(p):
    return "Dead Zone Dynamic", 3, [_dead_zone(_in(2), _in(1), _in(3))]


@rule("Reference:Wrap To Zero")
def wrap_to_zero(p):
    return "Wrap To Zero", 1, [op("select", op("gt", _in(1), c(_scalar(p, "Threshold", 255))), c(0), _in(1))]


# ---------------------------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------------------------
_RELATIONS = {"==": "eq", "~=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def _relation(relop: str, a: Expr, b: Expr) -> Expr:
    if relop not in _RELATIONS:
        raise SemanticsError(f"Relational operator {relop!r} is not supported.")
    return truth(op(_RELATIONS[relop], a, b))


@rule("RelationalOperator")
def relational(p):
    relop = p.get("Operator", ">=")
    return "Relational Operator (%s)" % relop, 2, [_relation(relop, _in(1), _in(2))]


@rule("Reference:Compare To Constant")
def compare_to_constant(p):
    relop = p.get("relop", "<=")
    return "Compare To Constant (%s)" % relop, 1, [_relation(relop, _in(1), c(_scalar(p, "const", 3)))]


@rule("Reference:Compare To Zero")
def compare_to_zero(p):
    relop = p.get("relop", "<=")
    return "Compare To Zero (%s)" % relop, 1, [_relation(relop, _in(1), c(0))]


@rule("Logic")
def logical(p):
    operator = p.get("Operator", "AND")
    n = 1 if operator == "NOT" else int(_scalar(p, "Inputs", 2))
    values = [nonzero(_in(i)) for i in range(1, n + 1)]
    if operator == "NOT":
        y = op("not", values[0])
    elif operator in {"AND", "NAND"}:
        y = values[0] if n == 1 else op("and", *values)
    elif operator in {"OR", "NOR"}:
        y = values[0] if n == 1 else op("or", *values)
    elif operator in {"XOR", "NXOR"}:
        y = values[0]
        for value in values[1:]:                             # odd number of true inputs
            y = op("ne", y, value)
    else:
        raise SemanticsError(f"Logical operator {operator!r} is not supported.")
    if operator in {"NAND", "NOR", "NXOR"}:
        y = op("not", y)
    return "Logical Operator (%s)" % operator, n, [truth(y)]


# ---------------------------------------------------------------------------------------------
# signal routing and attributes
# ---------------------------------------------------------------------------------------------
@rule("Switch")
def switch(p):
    criteria = p.get("Criteria", "u2 >= Threshold")
    threshold = c(_scalar(p, "Threshold", 0))
    condition = {"u2 >= Threshold": lambda: op("ge", _in(2), threshold),
                 "u2 > Threshold": lambda: op("gt", _in(2), threshold),
                 "u2 ~= 0": lambda: op("ne", _in(2), c(0))}.get(criteria)
    if condition is None:
        raise SemanticsError(f"Switch criteria {criteria!r} is not supported.")
    return "Switch (%s)" % criteria, 3, [op("select", condition(), _in(1), _in(3))]


@rule("MultiPortSwitch")
def multiport_switch(p):
    order = p.get("DataPortOrder", "One-based contiguous")
    n = int(_scalar(p, "Inputs", 3))
    if order == "One-based contiguous":
        indices = list(range(1, n + 1))
    elif order == "Zero-based contiguous":
        indices = list(range(0, n))
    elif order == "Specify indices":
        indices = [int(v) for v in np.asarray(p.matrix("DataPortIndices", list(range(1, n + 1))), dtype=float).ravel()]
        if len(indices) != n:
            raise SemanticsError("Multiport Switch: one index per data port is needed.")
    else:
        raise SemanticsError(f"Multiport Switch data port order {order!r} is not supported.")
    default = p.get("DataPortForDefault", "Last data port")
    extra = default == "Additional data port"
    if default not in {"Last data port", "Additional data port"}:
        raise SemanticsError(f"Multiport Switch default {default!r} is not supported.")
    control = op("fix", _in(1))                                # the control is truncated
    y = _in(n + 2) if extra else _in(n + 1)
    for k in range(n - (0 if extra else 1) - 1, -1, -1):
        y = op("select", op("eq", control, c(indices[k])), _in(k + 2), y)
    return "Multiport Switch", n + 1 + int(extra), [y]


_INTEGER_TYPES = {"int8": (-2 ** 7, 2 ** 7 - 1), "uint8": (0, 2 ** 8 - 1), "int16": (-2 ** 15, 2 ** 15 - 1),
                  "uint16": (0, 2 ** 16 - 1), "int32": (-2 ** 31, 2 ** 31 - 1), "uint32": (0, 2 ** 32 - 1)}


def _round_like(u: Expr, method: str) -> Expr:
    if method == "Zero":
        return op("fix", u)
    if method == "Floor":
        return op("floor", u)
    if method == "Ceiling":
        return op("ceil", u)
    if method in {"Round", "Simplest"}:
        return op("round", u)
    if method == "Nearest":                                    # halves towards +inf
        return op("floor", op("add", u, c(0.5)))
    raise SemanticsError(f"Integer rounding mode {method!r} is not supported.")


@rule("DataTypeConversion")
def data_type_conversion(p):
    target = p.get("OutDataTypeStr", "Inherit: Inherit via back propagation")
    u = _in(1)
    if target in {"double", "single"} or target.startswith("Inherit"):
        return "Data Type Conversion (%s)" % target, 1, [u]
    if target == "boolean":
        return "Data Type Conversion (boolean)", 1, [truth(nonzero(u))]
    if target in _INTEGER_TYPES:
        low, high = _INTEGER_TYPES[target]
        r = _round_like(u, p.get("RndMeth", "Zero"))
        if p.get("SaturateOnIntegerOverflow", "off") == "on":
            y = op("min", op("max", r, c(low)), c(high))
        else:                                                  # wraps around
            y = op("add", op("mod", op("sub", r, c(low)), c(high - low + 1)), c(low))
        return "Data Type Conversion (%s)" % target, 1, [y]
    raise SemanticsError(f"Data Type Conversion to {target!r} is not supported.")


@rule("InitialCondition")
def initial_condition(p):
    return "IC", 1, [op("select", op("le", T, c(p.start_time)), c(_scalar(p, "Value", 1)), _in(1))]


@rule("SignalSpecification")
def signal_specification(p):
    return "Signal Specification", 1, [_in(1)]


# ---------------------------------------------------------------------------------------------
# lookup tables
# ---------------------------------------------------------------------------------------------
def _breakpoints(p: Params, name: str, default) -> np.ndarray:
    values = np.asarray(p.matrix(name, default), dtype=float).ravel()
    if values.size < 2 or np.any(np.diff(values) <= 0):
        raise SemanticsError(f"{name} must be strictly increasing, with at least two values.")
    return values


def _natural_spline_second_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.interpolate import CubicSpline
    return CubicSpline(x, y, bc_type="natural")(x, 2)


@rule("Lookup_n-D")
def lookup_nd(p):
    dims = int(_scalar(p, "NumberOfTableDimensions", 2))
    if p.get("DataSpecification", "Table and breakpoints") != "Table and breakpoints" or \
            p.get("BreakpointsSpecification", "Explicit values") != "Explicit values" or \
            p.get("TableSource", "Dialog") != "Dialog":
        raise SemanticsError("Only lookup tables with the table and breakpoints in the dialog are supported.")
    interp = p.get("InterpMethod", "Linear point-slope")
    extrap = p.get("ExtrapMethod", "Linear")
    if extrap not in {"Linear", "Clip"}:
        raise SemanticsError(f"Lookup extrapolation {extrap!r} is not supported.")
    clip = extrap == "Clip"
    if dims == 1:
        bp = _breakpoints(p, "BreakpointsForDimension1", [1, 2, 3])
        table = np.asarray(p.matrix("Table", [4, 5, 6]), dtype=float).ravel()
        if table.size != bp.size:
            raise SemanticsError("Lookup table: one table value per breakpoint is needed.")
        data = [c(bp.size), *map(c, bp), *map(c, table)]
        if interp in {"Linear point-slope", "Linear Lagrange"}:
            y = op("_interp1_clip" if clip else "_interp1", _in(1), *data)
        elif interp == "Flat":
            y = op("_interp1_flat", _in(1), *data)
        elif interp == "Nearest":
            y = op("_interp1_nearest", _in(1), *data)
        elif interp == "Cubic spline":
            m = _natural_spline_second_derivatives(bp, table)
            y = op("_spline1", _in(1), c(bp.size), c(1.0 if clip else 0.0), *map(c, bp), *map(c, table), *map(c, m))
        else:
            raise SemanticsError(f"Lookup interpolation {interp!r} is not supported.")
        return "1-D Lookup Table (%s)" % interp, 1, [y]
    if dims == 2:
        if interp not in {"Linear point-slope", "Linear Lagrange"}:
            raise SemanticsError(f"2-D lookup interpolation {interp!r} is not supported.")
        bp1 = _breakpoints(p, "BreakpointsForDimension1", [1, 2, 3])
        bp2 = _breakpoints(p, "BreakpointsForDimension2", [1, 2, 3])
        table = np.asarray(p.matrix("Table", [[4, 5, 6], [16, 19, 20], [10, 18, 23]]), dtype=float)
        if table.shape != (bp1.size, bp2.size):
            raise SemanticsError("2-D lookup table: the table must have one row per breakpoint of input 1 and one "
                                 "column per breakpoint of input 2.")
        y = op("_interp2", _in(1), _in(2), c(bp1.size), c(bp2.size), c(1.0 if clip else 0.0),
               *map(c, bp1), *map(c, bp2), *map(c, table.ravel()))
        return "2-D Lookup Table", 2, [y]
    raise SemanticsError(f"{dims}-D lookup tables are not supported (1-D and 2-D are).")


@rule("LookupNDDirect")
def direct_lookup(p):
    if int(_scalar(p, "NumberOfTableDimensions", 2)) != 1 or p.get("TableIsInput", "off") != "off" or \
            p.get("InputsSelectThisObjectFromTable", "Element") != "Element":
        raise SemanticsError("Only the 1-D Direct Lookup Table with the table in the dialog is supported.")
    table = np.asarray(p.matrix("Table", [1, 2, 3]), dtype=float).ravel()
    return "Direct Lookup Table", 1, [op("_direct1", _in(1), c(table.size), *map(c, table))]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulink_c2py.py  -  deterministic (rule based, NO LLM) transpiler
====================================================================
Simulink Coder / Embedded Coder (ERT) generated C code  -->  executable Python model.

Usage
-----
    python simulink_c2py.py  <folder | file.c file.h ...>  [-o model_py.py] [--model NAME]

The tool
  1. preprocesses the C sources (comments, #define/#if, function-like macros),
  2. parses typedefs / structs / enums / global variables / functions,
  3. translates every function of the generated code (<model>_initialize, <model>_step,
     <model>_derivatives, <model>_output/_update, sub-system functions, helper functions ...)
     statement by statement into Python (if/else, for, while, switch, ternary, casts,
     pointers to scalars, math functions ...),
  4. replaces the C solver (rt_ertODEUpdateContinuousStates) by a small fixed-step
     Python solver (ode1..ode5, ode8 - the Simulink fixed-step solvers) that calls the translated step() and derivatives(),
  5. writes ONE self-contained python module (no dependency on this script, only numpy).

The generated module exposes a class  <model>  (alias  Model)  with
    initialize(), step(), terminate()      - exactly the C entry points
    simulate(t_final, inputs)              - convenience loop (like ert_main.c)
    f(t, x, u), g(t, x, u), linearize()    - dynamics  xdot = f(t,x,u),  y = g(t,x,u)
"""
import argparse
import glob
import keyword
import math
import os
import re
import sys
from collections import OrderedDict, namedtuple

VERSION = '1.0'

# --------------------------------------------------------------------------------------
# C scalar types   name -> (kind, bits, signed)      kind: f=float i=int b=bool v=void
# --------------------------------------------------------------------------------------
SCALARS = {
    'real_T': ('f', 64, True), 'real64_T': ('f', 64, True), 'time_T': ('f', 64, True),
    'real32_T': ('f', 32, True), 'double': ('f', 64, True), 'float': ('f', 32, True),
    'boolean_T': ('b', 8, False), '_Bool': ('b', 8, False),
    'int8_T': ('i', 8, True), 'uint8_T': ('i', 8, False),
    'int16_T': ('i', 16, True), 'uint16_T': ('i', 16, False),
    'int32_T': ('i', 32, True), 'uint32_T': ('i', 32, False),
    'int64_T': ('i', 64, True), 'uint64_T': ('i', 64, False),
    'int_T': ('i', 32, True), 'uint_T': ('i', 32, False), 'ulong_T': ('i', 64, False),
    'char_T': ('i', 8, True), 'uchar_T': ('i', 8, False), 'byte_T': ('i', 8, True),
    'char': ('i', 8, True), 'signed char': ('i', 8, True), 'unsigned char': ('i', 8, False),
    'short': ('i', 16, True), 'unsigned short': ('i', 16, False),
    'int': ('i', 32, True), 'unsigned int': ('i', 32, False),
    'long': ('i', 64, True), 'unsigned long': ('i', 64, False),
    'long long': ('i', 64, True), 'unsigned long long': ('i', 64, False),
    'size_t': ('i', 64, False), 'void': ('v', 0, False),
}
# unknown RTW header types that are really integers
BUILTIN_ALIASES = {'SimTimeStep': 'int_T', 'int_T': 'int_T'}

QUALS = {'static', 'const', 'extern', 'volatile', 'register', 'inline', '__inline',
         '__inline__', 'auto', 'restrict', '__restrict'}
BUILTIN_WORDS = {'unsigned', 'signed', 'short', 'long', 'int', 'char', 'float', 'double',
                 'void', '_Bool'}
STMT_KW = {'return', 'else', 'case', 'goto', 'break', 'continue', 'do', 'sizeof',
           'default', 'if', 'for', 'while', 'switch'}

MATH1 = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh',
         'acosh', 'atanh', 'exp', 'exp2', 'expm1', 'log', 'log10', 'log2', 'log1p', 'sqrt',
         'cbrt', 'fabs', 'floor', 'ceil', 'round', 'trunc', 'rint']
MATH2 = ['atan2', 'pow', 'fmod', 'fmin', 'fmax', 'hypot', 'copysign']
MATHF = {}
for _n in MATH1 + MATH2:
    MATHF[_n] = _n
    MATHF[_n + 'f'] = _n

# helper functions that the runtime provides itself (used when the .c file is not given)
_SIMD_OPS = ['set', 'set1', 'setzero', 'load', 'loadu', 'store', 'storeu', 'add', 'sub', 'mul',
             'div', 'min', 'max', 'sqrt', 'fmadd', 'fmsub']
SIMD_FUNCS = {p + o + '_' + x for p in ('_mm_', '_mm256_') for x in ('pd', 'ps') for o in _SIMD_OPS}
SIMD_FUNCS |= {'_mm_set_sd', '_mm_load_sd', '_mm_store_sd', '_mm_cvtsd_f64', '_mm_add_sd',
               '_mm_sub_sd', '_mm_mul_sd', '_mm_div_sd'}
# calls that only initialize the rtInf/rtNaN globals - our runtime already has them as
# constants, so the call site becomes a harmless no-op instead of an unresolved method
NOOP_FUNCS = {'rt_InitInfAndNaN', 'rt_InitInfAndNaNF'}
RUNTIME_FUNCS = SIMD_FUNCS | {'rtIsNaN', 'rtIsInf', 'rtIsNaNF', 'rtIsInfF', 'rt_roundd_snf',
                 'rt_powd_snf', 'rt_atan2d_snf', 'rt_hypotd_snf'}
# functions / globals that are NOT translated (infrastructure replaced by the runtime)
SKIP_FUNC_RE = re.compile(r'^(rt_ertODE\w*|rt_ODE\w*|rt_InitInfAndNaN|rtGet(Minus)?(Inf|NaN)F?|'
                          r'rtIsNaNF?|rtIsInfF?|rt_roundd_snf|rt_powd_snf|rt_atan2d_snf|'
                          r'rt_hypotd_snf|main|rt_OneStep)$')
SKIP_GLOBALS = {'rtInf', 'rtMinusInf', 'rtNaN', 'rtInfF', 'rtMinusInfF', 'rtNaNF'}
SKIP_FILE_RE = re.compile(r'(^|[\\/])(rt_nonfinite|rtGetInf|rtGetNaN)\.[ch]$|'
                          r'(^|[\\/])\w*main\w*\.c(pp)?$', re.I)

BUILTIN_CONSTS = {
    'rtInf': 'rtInf', 'rtMinusInf': 'rtMinusInf', 'rtNaN': 'rtNaN', 'rtInfF': 'rtInf',
    'rtMinusInfF': 'rtMinusInf', 'rtNaNF': 'rtNaN', 'NAN': 'nan', 'INFINITY': 'inf',
    'M_PI': 'math.pi', 'DBL_MAX': '1.7976931348623157e308', 'DBL_EPSILON': '2.220446049250313e-16',
    'MAJOR_TIME_STEP': '0', 'MINOR_TIME_STEP': '1', 'NULL': 'None', 'true': '1', 'false': '0',
}

PYRESERVED = set(keyword.kwlist) | {'self', 'np', 'math', 'float', 'int', 'bool', 'abs',
                                    'range', 'len', 'min', 'max', 'sum', 'isinstance'}


def pyname(n):
    return n + '_' if n in PYRESERVED else n


# ======================================================================================
# RUNTIME  (this text is pasted verbatim into every generated python module)
# ======================================================================================
RUNTIME = r'''
import math
import numpy as np

np.seterr(all='ignore')

inf = math.inf
nan = math.nan
rtInf = math.inf
rtMinusInf = -math.inf
rtNaN = math.nan
MAJOR_TIME_STEP = 0
MINOR_TIME_STEP = 1

_SCALARS = @@SCALARS@@
_NPDT = {}
_ZERO = {}
for _n, (_k, _b, _s) in _SCALARS.items():
    if _k == 'f':
        _NPDT[_n] = np.float32 if _b == 32 else np.float64
        _ZERO[_n] = 0.0
    elif _k in ('i', 'b'):
        _NPDT[_n] = np.int64
        _ZERO[_n] = 0
_INT = (int, np.integer)


# ---- explicit Runge-Kutta tableaus of the Simulink fixed-step solvers: (c, A, b) ---------
_DP5 = ([0.0, 1/5, 3/10, 4/5, 8/9, 1.0],
        [[0]*6, [1/5, 0, 0, 0, 0, 0], [3/40, 9/40, 0, 0, 0, 0], [44/45, -56/15, 32/9, 0, 0, 0],
         [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
         [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]],
        [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])
_TAB = {
    'ode1': ([0.0], [[0.0]], [1.0]),
    'ode2': ([0.0, 1.0], [[0, 0], [1, 0]], [0.5, 0.5]),
    'ode3': ([0.0, 0.5, 0.75], [[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]], [2/9, 1/3, 4/9]),
    'ode4': ([0.0, 0.5, 0.5, 1.0], [[0]*4, [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]],
             [1/6, 1/3, 1/3, 1/6]),
    'ode5': _DP5,
    'ode8': ([0.0, 0.05555555555555556, 0.08333333333333333, 0.125, 0.3125, 0.375, 0.1475, 0.465, 0.5648654513822596, 0.65, 0.9246562776405044, 1.0, 1.0],
             [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05555555555555556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.02083333333333333, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.03125, 0.0, 0.09375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.3125, 0.0, -1.171875, 1.171875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0375, 0.0, 0.0, 0.1875, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.04791013711111111, 0.0, 0.0, 0.1122487127777778, -0.02550567377777778, 0.01284682388888889, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.01691798978729228, 0.0, 0.0, 0.3878482784860432, 0.03597736985150033, 0.1969702142156661, -0.1727138523405018, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0690957533591923, 0.0, 0.0, -0.6342479767288542, -0.1611975752246041, 0.1386503094588253, 0.9409286140357563, 0.211636326481944, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1835569968390454, 0.0, 0.0, -2.468768084315592, -0.2912868878163005, -0.02647302023311738, 2.8478387641928, 0.2813873314698498, 0.1237448998633147, 0.0, 0.0, 0.0, 0.0], [-1.215424817395888, 0.0, 0.0, 16.67260866594577, 0.915741828416818, -6.056605804357471, -16.00357359415618, 14.84930308629766, -13.37157573528985, 5.134182648179638, 0.0, 0.0, 0.0], [0.2588609164382643, 0.0, 0.0, -4.774485785489205, -0.4350930137770325, -3.049483332072241, 5.577920039936099, 6.155831589861039, -5.062104586736938, 2.193926173180679, 0.1346279986593349, 0.0, 0.0], [0.8224275996265075, 0.0, 0.0, -11.65867325727766, -0.7576221166909362, 0.7139735881595818, 12.07577498689006, -2.127659113920403, 1.990166207048956, -0.2342864715440405, 0.1758985777079423, 0.0, 0.0]],
             [0.04174749114153025, 0.0, 0.0, 0.0, 0.0, -0.0554523286112393, 0.2393128072011801, 0.703510669403443, -0.7597596138144609, 0.6605630309222863, 0.1581874825101233, -0.2381095387528628, 0.25]),
}
_ALIAS = {'euler': 'ode1', 'heun': 'ode2', 'rk4': 'ode4'}


class _Struct(object):
    """C struct instance (fields are attributes, arrays are numpy arrays / lists)."""
    def __repr__(self):
        return '_Struct(%s)' % ', '.join('%s=%r' % kv for kv in self.__dict__.items())


class _Opaque(object):
    """Object of a type defined in a header that was not supplied (e.g. RTWSolverInfo)."""


class _Box(object):
    """Local scalar variable whose address is taken (&x)."""
    __slots__ = ('v',)

    def __init__(self, v):
        self.v = v

    def __getitem__(self, i):
        return self.v

    def __setitem__(self, i, val):
        self.v = val


class _Ref(object):
    """Pointer to a scalar struct member (&s.a) or array element (&a[i])."""
    def __init__(self, obj, key):
        self.o = obj
        self.k = key

    @property
    def v(self):
        return getattr(self.o, self.k) if isinstance(self.k, str) else self.o[self.k]

    @v.setter
    def v(self, val):
        if isinstance(self.k, str):
            setattr(self.o, self.k, val)
        else:
            self.o[self.k] = val

    def _name(self, i):
        names = list(self.o.__dict__)              # consecutive struct fields (SIMD stores)
        return names[names.index(self.k) + i]

    def __getitem__(self, i):
        if isinstance(self.k, str):
            return getattr(self.o, self.k if i == 0 else self._name(i))
        return self.o[self.k + i]

    def __setitem__(self, i, val):
        if isinstance(self.k, str):
            setattr(self.o, self.k if i == 0 else self._name(i), val)
        else:
            self.o[self.k + i] = val


def _new(base, dims=()):
    if dims:
        n = dims[0]
        if n == 0:
            return None                       # pointer
        rest = dims[1:]
        if base in _NPDT and all(d > 0 for d in rest):
            return np.zeros((n,) + tuple(rest), dtype=_NPDT[base])
        return [_new(base, rest) for _ in range(n)]
    if base in _NPDT:
        return _ZERO[base]
    if base in STRUCTS:
        o = _Struct()
        for (f, b, d) in STRUCTS[base]:
            setattr(o, f, _new(b, d))
        return o
    return _Opaque()


def _init_val(base, dims, init):
    if dims:
        n = dims[0]
        if n == 0:
            return None if isinstance(init, list) else init
        arr = _new(base, dims)
        items = init if isinstance(init, list) else [init]
        for i, it in enumerate(items[:n]):
            arr[i] = _init_val(base, dims[1:], it)
        return arr
    if base in STRUCTS:
        o = _new(base)
        items = init if isinstance(init, list) else [init]
        for (f, b, d), it in zip(STRUCTS[base], items):
            setattr(o, f, _init_val(b, d, it))
        return o
    if base in _NPDT:
        return float(init) if _SCALARS[base][0] == 'f' else int(init)
    return init


# ---- C arithmetic semantics --------------------------------------------------------
def _div(a, b):
    if isinstance(a, _INT) and isinstance(b, _INT):
        if b == 0:
            return 0
        q = abs(a) // abs(b)
        return q if (a < 0) == (b < 0) else -q
    try:
        return a / b
    except ZeroDivisionError:
        a = float(a)
        if a != a or a == 0.0:
            return nan
        return math.copysign(inf, a) * math.copysign(1.0, b)


def _mod(a, b):
    if isinstance(a, _INT) and isinstance(b, _INT):
        if b == 0:
            return 0
        r = abs(a) % abs(b)
        return r if a >= 0 else -r
    try:
        return math.fmod(a, b)
    except ValueError:
        return nan


def _ci(x, bits, signed):
    """C cast to an integer type (wrap-around)."""
    if isinstance(x, (float, np.floating)):
        if x != x:
            return 0
        if x in (inf, -inf):
            return (1 << (bits - 1)) - 1 if x > 0 else (-(1 << (bits - 1)) if signed else 0)
    x = int(x)
    m = 1 << bits
    x &= m - 1
    if signed and x >= (m >> 1):
        x -= m
    return x


def _f32(x):
    return float(np.float32(x))


def _deref(p):
    if isinstance(p, (_Ref, _Box)):
        return p.v
    if isinstance(p, (_Struct, _Opaque)):
        return p
    return p[0]


def _store(p, v):
    if isinstance(p, (_Ref, _Box)):
        p.v = v
    else:
        p[0] = v


def _zero(o):
    if isinstance(o, np.ndarray):
        o[...] = 0
    elif isinstance(o, _Struct):
        for k, v in list(o.__dict__.items()):
            if isinstance(v, (_Struct, np.ndarray, list, _Ref, _Box)):
                _zero(v)
            elif isinstance(v, float):
                setattr(o, k, 0.0)
            elif isinstance(v, _INT):
                setattr(o, k, 0)
    elif isinstance(o, list):
        for v in o:
            _zero(v)
    elif isinstance(o, _Ref) and isinstance(o.o, np.ndarray) and isinstance(o.k, int):
        o.o[o.k:] = 0                  # memset(&arr[i], 0, ...): zero i..end, not just arr[i]
    elif isinstance(o, (_Ref, _Box)):
        o.v = 0


def _memset(o, val):
    if val == 0:
        _zero(o)
    elif isinstance(o, np.ndarray):
        o[...] = val


def _memcpy(dst, src):
    if isinstance(dst, np.ndarray):
        dst[...] = src
    elif isinstance(dst, _Struct):
        for k, v in src.__dict__.items():
            cur = getattr(dst, k)
            if isinstance(cur, (np.ndarray, _Struct)):
                _memcpy(cur, v)
            else:
                setattr(dst, k, v)
    elif isinstance(dst, list):
        for i, v in enumerate(src):
            dst[i] = v


# ---- math.h ------------------------------------------------------------------------
def _guard(fn):
    def w(*a):
        try:
            return fn(*a)
        except ValueError:
            return nan
        except OverflowError:
            return inf
    return w


_c_sin, _c_cos, _c_tan = _guard(math.sin), _guard(math.cos), _guard(math.tan)
_c_asin, _c_acos, _c_atan = _guard(math.asin), _guard(math.acos), _guard(math.atan)
_c_sinh, _c_cosh, _c_tanh = _guard(math.sinh), _guard(math.cosh), _guard(math.tanh)
_c_asinh, _c_acosh, _c_atanh = _guard(math.asinh), _guard(math.acosh), _guard(math.atanh)
_c_exp, _c_expm1, _c_sqrt = _guard(math.exp), _guard(math.expm1), _guard(math.sqrt)
_c_atan2, _c_hypot, _c_copysign = math.atan2, math.hypot, math.copysign
_c_exp2 = _guard(lambda x: 2.0 ** x)


def _c_cbrt(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _mklog(fn):
    def w(x):
        if x > 0:
            return fn(x)
        if x == 0:
            return -inf
        return nan
    return w


_c_log, _c_log10, _c_log2 = _mklog(math.log), _mklog(math.log10), _mklog(math.log2)
_c_log1p = _guard(math.log1p)


def _c_pow(x, y):
    try:
        return math.pow(x, y)
    except OverflowError:
        return inf
    except ValueError:
        if x == 0 and y < 0:
            return inf
        return nan


def _c_fabs(x):
    return abs(float(x))


def _mkround(fn):
    def w(x):
        return float(fn(x)) if math.isfinite(x) else x
    return w


_c_floor, _c_ceil, _c_trunc = _mkround(math.floor), _mkround(math.ceil), _mkround(math.trunc)
_c_rint = _mkround(round)


def _c_round(x):
    if not math.isfinite(x):
        return x
    return float(math.floor(abs(x) + 0.5)) * (1.0 if x >= 0 else -1.0)


def _c_fmod(a, b):
    try:
        return math.fmod(a, b)
    except ValueError:
        return nan


def _c_fmin(a, b):
    return b if a != a else (a if b != b else min(a, b))


def _c_fmax(a, b):
    return b if a != a else (a if b != b else max(a, b))


# ---- rt_nonfinite / rtwlib helpers -------------------------------------------------
def rtIsNaN(x):
    return x != x


def rtIsInf(x):
    return x in (inf, -inf)


rtIsNaNF, rtIsInfF = rtIsNaN, rtIsInf


def rt_roundd_snf(u):
    if abs(u) < 4503599627370496.0:
        return _c_round(u)
    return u


def rt_atan2d_snf(a, b):
    return math.atan2(a, b)


def rt_hypotd_snf(a, b):
    return math.hypot(a, b)


def rt_powd_snf(u0, u1):
    if u0 != u0 or u1 != u1:
        return nan
    if math.isinf(u1):
        a = abs(u0)
        if a == 1.0:
            return nan
        if a > 1.0:
            return inf if u1 > 0 else 0.0
        return 0.0 if u1 > 0 else inf
    if u1 == 0.0:
        return 1.0
    return _c_pow(u0, u1)


# ---- SIMD intrinsics (SSE/SSE2/AVX) emulated with numpy vectors ---------------------
def _simd_store(n):
    def store(p, v):
        for i in range(n):
            p[i] = float(v[i])
    return store


def _simd_family():
    g = globals()
    for (pre, suf), n in {('_mm_', 'pd'): 2, ('_mm_', 'ps'): 4,
                          ('_mm256_', 'pd'): 4, ('_mm256_', 'ps'): 8}.items():
        dt = np.float64 if suf == 'pd' else np.float32

        def add(name, fn, pre=pre, suf=suf):
            g[pre + name + '_' + suf] = fn
        add('set', lambda *a, dt=dt: np.array(a[::-1], dtype=dt))
        add('set1', lambda x, n=n, dt=dt: np.full(n, x, dtype=dt))
        add('setzero', lambda n=n, dt=dt: np.zeros(n, dtype=dt))
        for nm in ('load', 'loadu'):
            add(nm, lambda p, n=n, dt=dt: np.array([p[i] for i in range(n)], dtype=dt))
        for nm in ('store', 'storeu'):
            add(nm, _simd_store(n))
        for nm, op in (('add', np.add), ('sub', np.subtract), ('mul', np.multiply),
                       ('div', np.divide), ('min', np.minimum), ('max', np.maximum)):
            add(nm, lambda a, b, op=op: op(a, b))
        add('sqrt', np.sqrt)
        add('fmadd', lambda a, b, c: a * b + c)
        add('fmsub', lambda a, b, c: a * b - c)


_simd_family()
_mm_set_sd = lambda x: np.array([x, 0.0])
_mm_load_sd = lambda p: np.array([p[0], 0.0])
_mm_store_sd = lambda p, v: p.__setitem__(0, float(v[0]))
_mm_cvtsd_f64 = lambda v: float(v[0])
_mm_add_sd = lambda a, b: np.array([a[0] + b[0], a[1]])
_mm_sub_sd = lambda a, b: np.array([a[0] - b[0], a[1]])
_mm_mul_sd = lambda a, b: np.array([a[0] * b[0], a[1]])
_mm_div_sd = lambda a, b: np.array([a[0] / b[0], a[1]])


# ---- flatten a struct into a vector of scalars -------------------------------------
def _get(p, k):
    return getattr(p, k) if isinstance(p, _Struct) else p[k]


def _set(p, k, v):
    if isinstance(p, _Struct):
        setattr(p, k, v)
    else:
        p[k] = v


def _leaves(val, name='', parent=None, key=None):
    if isinstance(val, _Struct):
        pre = name + '.' if name else ''
        res = []
        for k, v in val.__dict__.items():
            res += _leaves(v, pre + k, val, k)
        return res
    if isinstance(val, np.ndarray):
        return [('%s[%s]' % (name, ','.join(map(str, i))), val, i) for i in np.ndindex(*val.shape)]
    if isinstance(val, list):
        res = []
        for i, v in enumerate(val):
            res += _leaves(v, '%s[%d]' % (name, i), val, i)
        return res
    if isinstance(val, (int, float, np.number)) and parent is not None:
        return [(name, parent, key)]
    return []


class CModel(object):
    """Base class of the generated model: state handling + fixed step solver + helpers."""
    MODEL_NAME = ''
    STEP_FN = 'step'
    SOLVER = 'ode1'
    X_ATTR = 'X'
    U_ATTR = 'U'
    Y_ATTR = 'Y'
    XDOT_TYPE = None
    UPDATE_FN = None

    def __init__(self, solver='auto', h=None):
        self.solver = solver
        self._h_override = h
        self.reset()

    # ---- construction / reset ----
    def reset(self):
        self._tstop = None
        self._t_solver = 0.0
        self._ysnap = None
        self._ufun = None
        self._cache = None
        self._build_globals()
        self.Xdot = _new(self.XDOT_TYPE) if self.XDOT_TYPE else _Struct()
        M = getattr(self, 'M', None)
        if M is not None:
            if hasattr(M, 'derivs'):
                M.derivs = self.Xdot
            X = getattr(self, self.X_ATTR, None)
            if hasattr(M, 'contStates') and X is not None:
                M.contStates = X
        self.initialize()
        M = getattr(self, 'M', None)
        Tm = getattr(M, 'Timing', None)
        if Tm is not None and getattr(Tm, 't', 1) is None and hasattr(Tm, 'tArray'):
            Tm.t = Tm.tArray
        if self._h_override is not None and Tm is not None:
            Tm.stepSize0 = float(self._h_override)

    def __getattr__(self, name):
        if name.startswith(('rtsiSet', 'rtmSet')):
            return lambda *a, **k: None
        raise AttributeError("'%s' has no attribute '%s' (function/macro not found in the "
                             "supplied C files?)" % (type(self).__name__, name))

    # ---- rtw_solver.h / rtmodel.h replacements ----
    def rtsiGetT(self, *a):
        return self._t_solver

    def rtsiGetSolverStopTime(self, *a):
        if self._tstop is not None:
            return self._tstop
        return self._t_solver + self.step_size

    def rtsiSetSolverStopTime(self, si, v):
        self._tstop = v

    def rtsiGetStepSize(self, *a):
        return self.step_size

    def rtsiSetSimTimeStep(self, si, v):
        self.M.Timing.simTimeStep = v

    def rtmIsMajorTimeStep(self, *a):
        return self.M.Timing.simTimeStep == MAJOR_TIME_STEP

    def rtmIsMinorTimeStep(self, *a):
        return self.M.Timing.simTimeStep == MINOR_TIME_STEP

    def rtmGetT(self, *a):
        return self.M.Timing.t[0]

    # ---- state / io vectors ----
    @property
    def step_size(self):
        Tm = getattr(getattr(self, 'M', None), 'Timing', None)
        h = getattr(Tm, 'stepSize0', None)
        if h is None:
            raise RuntimeError('step size not found in the C code: pass h=... to the constructor')
        return float(h)

    def _lv(self):
        if self._cache is None:
            def L(name):
                o = getattr(self, name, None)
                return _leaves(o) if o is not None else []
            self._cache = dict(x=L(self.X_ATTR), xd=L('Xdot'), u=L(self.U_ATTR), y=L(self.Y_ATTR))
        return self._cache

    state_names = property(lambda s: [n for n, _, _ in s._lv()['x']])
    input_names = property(lambda s: [n for n, _, _ in s._lv()['u']])
    output_names = property(lambda s: [n for n, _, _ in s._lv()['y']])

    def _vec(self, key):
        return np.array([float(_get(p, k)) for _, p, k in self._lv()[key]])

    def get_x(self):
        return self._vec('x')

    def get_xdot(self):
        return self._vec('xd')

    def get_u(self):
        return self._vec('u')

    def get_y(self):
        return self._vec('y')

    def set_x(self, x):
        for (_, p, k), v in zip(self._lv()['x'], x):
            _set(p, k, float(v))

    def set_u(self, u):
        for (_, p, k), v in zip(self._lv()['u'], np.atleast_1d(u)):
            _set(p, k, float(v))

    # ---- fixed step solver (replaces rt_ertODEUpdateContinuousStates) ----
    def _method(self):
        m = _ALIAS.get(self.solver, self.solver)
        if m == 'auto':
            m = _ALIAS.get(self.SOLVER, self.SOLVER)
            if m not in _TAB:
                m = 'ode4'
        if m not in _TAB:
            raise ValueError('unknown solver %r (ode1..ode5, ode8, euler, heun, rk4)' % self.solver)
        return m

    def rt_ertODEUpdateContinuousStates(self, *a):
        Tm = self.M.Timing
        h = float(Tm.stepSize0)
        t = float(Tm.t[0])
        self._ysnap = self.get_y()              # outputs at the start of the major step
        x0 = self.get_x()
        if x0.size == 0:
            return
        step = getattr(self, self.STEP_FN)

        def F(tt, xx):
            self.set_x(xx)
            if self._ufun is not None:          # Simulink-like linear interpolation of the inputs
                self.set_u(self._ufun(tt))
            self._t_solver = tt
            Tm.simTimeStep = MINOR_TIME_STEP
            step()
            self.derivatives()
            return self.get_xdot()

        Tm.simTimeStep = MINOR_TIME_STEP
        self.derivatives()                      # outputs of this major step are already computed
        k1 = self.get_xdot()
        c, A, bw = _TAB[self._method()]
        tnew = self._tstop if self._tstop is not None else t + h
        ks = [k1]
        for i in range(1, len(bw)):
            xs = x0 + h * sum(A[i][j] * ks[j] for j in range(i) if A[i][j] != 0)
            ks.append(F(tnew if c[i] == 1.0 else t + c[i] * h, xs))
        xn = x0 + h * sum(bw[i] * ks[i] for i in range(len(bw)) if bw[i] != 0)
        self.set_x(xn)
        Tm.simTimeStep = MAJOR_TIME_STEP
        self._t_solver = t + h

    # ---- dynamics  xdot = f(t,x,u),  y = g(t,x,u) ----
    def _eval(self, t, x, u):
        if x is not None:
            self.set_x(x)
        if u is not None:
            self.set_u(u)
        Tm = self.M.Timing
        old = Tm.simTimeStep
        Tm.simTimeStep = MINOR_TIME_STEP
        self._t_solver = t
        getattr(self, self.STEP_FN)()
        self.derivatives()
        Tm.simTimeStep = old

    def f(self, t, x, u=None):
        """state derivative  dx/dt = f(t, x, u)   (continuous part)."""
        self._eval(t, x, u)
        return self.get_xdot()

    def g(self, t, x, u=None):
        """output  y = g(t, x, u)."""
        self._eval(t, x, u)
        return self.get_y()

    def linearize(self, x0=None, u0=None, t=0.0, eps=1e-6):
        """numerical A,B,C,D around (x0,u0)."""
        x0 = self.get_x() if x0 is None else np.asarray(x0, float)
        u0 = self.get_u() if u0 is None else np.asarray(u0, float)
        nx, nu = x0.size, u0.size
        y0 = self.g(t, x0, u0)
        A = np.zeros((nx, nx)); B = np.zeros((nx, nu))
        C = np.zeros((y0.size, nx)); D = np.zeros((y0.size, nu))
        for i in range(nx):
            d = np.zeros(nx); d[i] = eps
            A[:, i] = (self.f(t, x0 + d, u0) - self.f(t, x0 - d, u0)) / (2 * eps)
            C[:, i] = (self.g(t, x0 + d, u0) - self.g(t, x0 - d, u0)) / (2 * eps)
        for j in range(nu):
            d = np.zeros(nu); d[j] = eps
            B[:, j] = (self.f(t, x0, u0 + d) - self.f(t, x0, u0 - d)) / (2 * eps)
            D[:, j] = (self.g(t, x0, u0 + d) - self.g(t, x0, u0 - d)) / (2 * eps)
        return A, B, C, D

    # ---- simulation loop (what ert_main.c does) ----
    def _input_fn(self, inputs):
        names = self.input_names
        n = len(names)
        if inputs is None:
            return lambda t, k: [0.0] * n
        if callable(inputs):
            def fn(t, k):
                r = inputs(t)
                if isinstance(r, dict):
                    return [r.get(nm, 0.0) for nm in names]
                return list(np.atleast_1d(r))
            return fn
        if isinstance(inputs, dict):
            cols = [inputs.get(nm, 0.0) for nm in names]

            def fn(t, k):
                row = []
                for v in cols:
                    if callable(v):
                        row.append(v(t))
                    elif np.ndim(v) > 0:
                        row.append(v[min(k, len(v) - 1)])
                    else:
                        row.append(v)
                return row
            return fn
        arr = np.asarray(inputs, float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return lambda t, k: arr[min(k, len(arr) - 1)]

    def simulate(self, t_final, inputs=None, reset=True, interp_inputs=False):
        """Run from t=0 to t_final.  inputs: None | f(t) | {name: const|f(t)|array} | array(N,nu).
        interp_inputs=True: inputs are linearly interpolated between steps inside the solver stages
        (what a Simulink root Inport with 'Interpolate data' does).
        Returns dict(t, x, u, y, state_names, input_names, output_names)."""
        if reset:
            self.reset()
        h = self.step_size
        n = int(round(t_final / h))
        fn = self._input_fn(inputs)
        self._ufun = None
        if interp_inputs:      # sample inputs on the step grid, interpolate linearly inside a step
            tg = np.arange(n + 1) * h
            Ug = np.array([np.atleast_1d(np.asarray(fn(tg[k], k), float)) for k in range(n + 1)])
            fn = lambda t, k: Ug[k]
            self._ufun = lambda tt: [np.interp(tt, tg, Ug[:, j]) for j in range(Ug.shape[1])]
        T, X, U, Y = [], [], [], []
        step = getattr(self, self.STEP_FN)
        for k in range(n + 1):
            t = k * h
            self.set_u(fn(t, k))
            T.append(t)
            X.append(self.get_x())
            U.append(self.get_u())
            self._ysnap = None
            step()
            if self.UPDATE_FN:
                getattr(self, self.UPDATE_FN)()
            Y.append(self._ysnap if self._ysnap is not None else self.get_y())
        self._ufun = None
        return dict(t=np.array(T), x=np.array(X), u=np.array(U), y=np.array(Y),
                    state_names=self.state_names, input_names=self.input_names,
                    output_names=self.output_names)


def _main(cls):
    import argparse
    ap = argparse.ArgumentParser(description='run the generated %s model' % cls.__name__)
    ap.add_argument('--tf', type=float, default=10.0)
    ap.add_argument('--input', choices=['zero', 'step'], default='step')
    ap.add_argument('--solver', default='auto', help='auto | ode1..ode5 | ode8 | euler | heun | rk4')
    ap.add_argument('--h', type=float, default=None)
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    m = cls(solver=a.solver, h=a.h)
    print('states :', m.state_names)
    print('inputs :', m.input_names)
    print('outputs:', m.output_names)
    print('step size = %g s' % m.step_size)
    one = (lambda t: 1.0) if a.input == 'step' else 0.0
    r = m.simulate(a.tf, {n: one for n in m.input_names})
    print('y(t=%g) =' % r['t'][-1], dict(zip(r['output_names'], r['y'][-1])))
    if a.csv:
        hdr = ['t'] + r['input_names'] + r['output_names'] + r['state_names']
        data = np.column_stack([r['t'], r['u'], r['y'], r['x']])
        np.savetxt(a.csv, data, delimiter=',', header=','.join(hdr), comments='')
        print('written', a.csv)
'''


# ======================================================================================
# 1. preprocessing / tokenizing
# ======================================================================================
Tok = namedtuple('Tok', 'k v pre')
CT = namedtuple('CT', 'base dims')          # C type: base name + array/pointer dims (0 = pointer)


class CError(Exception):
    pass


def hide_comments(src, store):
    """Replace every comment by a placeholder ' @@C<n>@@ ' (text kept in `store`)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and src.startswith('/*', i):
            j = src.find('*/', i + 2)
            j = n if j < 0 else j
            store.append(src[i + 2:j])
            out.append(' @@C%d@@ ' % (len(store) - 1))
            i = j + 2
        elif c == '/' and src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            store.append(src[i + 2:j])
            out.append(' @@C%d@@ ' % (len(store) - 1))
            i = j
        elif c in '"\'':
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == '\\' else 1
            out.append(src[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


TOK_RE = re.compile(r'''
   (?P<ws>\s+)
 | (?P<cmt>@@C\d+@@)
 | (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*)
 | (?P<id>[A-Za-z_]\w*)
 | (?P<str>"(?:\\.|[^"\\])*")
 | (?P<chr>'(?:\\.|[^'\\])+')
 | (?P<op><<=|>>=|\.\.\.|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|[-+*/%&|^~!<>=?:;,.(){}\[\]#])
''', re.X)


def tokenize(text, comments):
    toks, pend = [], []
    for m in TOK_RE.finditer(text):
        k = m.lastgroup
        if k == 'ws':
            continue
        v = m.group()
        if k == 'cmt':
            raw = comments[int(v[3:-2])]
            for ln in raw.strip().splitlines():
                ln = ln.strip(' *\t')
                if ln:
                    pend.append(ln)
            continue
        toks.append(Tok(k, v, tuple(pend)))
        pend = []
    return toks


def num_value(txt):
    if txt[:2].lower() == '0x':
        return int(re.sub('[uUlL]+$', '', txt), 16)
    core = re.sub('[uUlL]+$', '', txt)
    isf = core.endswith(('f', 'F'))
    core = core.rstrip('fF')
    if isf or re.search(r'[.eE]', core):
        return float(core)
    return int(core, 10)


class PP(object):
    """Tiny C preprocessor: #define (object/function like), #if/#ifdef/#else/#endif."""
    def __init__(self):
        self.obj, self.fn, self.defined = {}, {}, set()
        self.comments = []
        self._cache = {}

    def _eval_if(self, cmd, rest):
        rest = re.sub(r'@@C\d+@@', ' ', rest).strip()
        if cmd == 'ifdef':
            return rest.split()[0] in self.defined
        if cmd == 'ifndef':
            return rest.split()[0] not in self.defined
        e = re.sub(r'defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)',
                   lambda m: '1' if (m.group(1) or m.group(2)) in self.defined else '0', rest)
        e = e.replace('&&', ' and ').replace('||', ' or ')
        e = re.sub(r'!(?!=)', ' not ', e)

        def repl(m):
            w = m.group(0)
            if w in ('and', 'or', 'not'):
                return w
            v = self.obj.get(w, '').strip()
            return v if re.fullmatch(r'-?\d+', v) else '0'
        e = re.sub(r'[A-Za-z_]\w*', repl, e)
        e = re.sub(r'(\d+)[uUlL]+', r'\1', e)
        try:
            return bool(eval(e, {}, {}))
        except Exception:
            return True

    def run(self, raw):
        text = hide_comments(raw, self.comments).replace('\\\n', ' ')
        out, stack, active = [], [], True
        for line in text.split('\n'):
            s = line.strip()
            if not s.startswith('#'):
                out.append(line if active else '')
                continue
            d = re.sub(r'@@C\d+@@', ' ', s[1:]).strip()
            m = re.match(r'(\w+)\s*(.*)', d)
            if not m:
                continue
            cmd, rest = m.group(1), m.group(2).strip()
            if cmd in ('if', 'ifdef', 'ifndef'):
                cond = self._eval_if(cmd, s[1:].split(None, 1)[1] if len(s[1:].split(None, 1)) > 1 else '') \
                    if active else False
                stack.append([active, cond])
                active = active and cond
            elif cmd == 'elif':
                e = stack[-1]
                if e[1] or not e[0]:
                    active = False
                else:
                    c = self._eval_if('if', rest)
                    e[1] = c
                    active = c
            elif cmd == 'else':
                e = stack[-1]
                active = e[0] and not e[1]
                e[1] = True
            elif cmd == 'endif':
                if stack:
                    active = stack.pop()[0]
            elif active and cmd == 'define':
                mm = re.match(r'(\w+)(\()?', rest)
                if not mm:
                    continue
                name = mm.group(1)
                self.defined.add(name)
                if mm.group(2):
                    close = rest.index(')')
                    params = [p.strip() for p in rest[mm.end():close].split(',') if p.strip()]
                    self.fn[name] = (params, rest[close + 1:])
                    self.obj.pop(name, None)
                else:
                    self.obj[name] = rest[mm.end():]
                    self.fn.pop(name, None)
            elif active and cmd == 'undef':
                self.defined.discard(rest)
                self.obj.pop(rest, None)
                self.fn.pop(rest, None)
            out.append('')
        return '\n'.join(out)

    def body(self, name, text):
        if name not in self._cache:
            self._cache[name] = tokenize(text, self.comments)
        return self._cache[name]

    def expand(self, toks, hide=frozenset()):
        out, i = [], 0
        while i < len(toks):
            t = toks[i]
            if t.k == 'id' and t.v not in hide:
                if t.v in self.fn and i + 1 < len(toks) and toks[i + 1].v == '(' and toks[i + 1].k == 'op':
                    params, btext = self.fn[t.v]
                    args, j = _collect_args(toks, i + 1)
                    if args is not None:
                        if not params:
                            args = []
                        m = dict(zip(params, args))
                        sub = []
                        for b in self.body(t.v, btext):
                            if b.k == 'id' and b.v in m:
                                sub.extend(m[b.v])
                            else:
                                sub.append(b)
                        out.extend(self.expand(sub, hide | {t.v}))
                        i = j
                        continue
                elif t.v in self.obj:
                    out.extend(self.expand(self.body(t.v, self.obj[t.v]), hide | {t.v}))
                    i += 1
                    continue
            out.append(t)
            i += 1
        return out


def _collect_args(toks, i):
    depth, args, j = 0, [[]], i
    while j < len(toks):
        t = toks[j]
        if t.k == 'op' and t.v == '(':
            depth += 1
            if depth > 1:
                args[-1].append(t)
        elif t.k == 'op' and t.v == ')':
            depth -= 1
            if depth == 0:
                return args, j + 1
            args[-1].append(t)
        elif t.k == 'op' and t.v == ',' and depth == 1:
            args.append([])
        else:
            args[-1].append(t)
        j += 1
    return None, i


def split_top(toks, sep):
    parts, cur, d = [], [], 0
    for t in toks:
        if t.k == 'op':
            if t.v in '([{' and len(t.v) == 1:
                d += 1
            elif t.v in ')]}' and len(t.v) == 1:
                d -= 1
            elif t.v == sep and d == 0:
                parts.append(cur)
                cur = []
                continue
        cur.append(t)
    parts.append(cur)
    return parts


def strip_wrap(toks):
    """remove redundant outer parentheses:  ((a = b))  ->  a = b"""
    while len(toks) >= 2 and toks[0].v == '(' and toks[0].k == 'op' and toks[-1].v == ')':
        d = 0
        ok = True
        for i, t in enumerate(toks):
            if t.k == 'op' and t.v == '(':
                d += 1
            elif t.k == 'op' and t.v == ')':
                d -= 1
                if d == 0 and i != len(toks) - 1:
                    ok = False
                    break
        if not ok:
            break
        toks = toks[1:-1]
    return toks


def norm_words(words):
    s = set(words)
    uns = 'unsigned' in s
    if '_Bool' in s:
        return '_Bool'
    if 'double' in s:
        return 'double'
    if 'float' in s:
        return 'float'
    if 'void' in s:
        return 'void'
    if 'char' in s:
        base = 'char'
    elif 'short' in s:
        base = 'short'
    else:
        n = list(words).count('long')
        base = 'long long' if n >= 2 else ('long' if n == 1 else 'int')
    return ('unsigned ' + base) if uns else base


class Func(object):
    def __init__(self, name, ret, params, body, static):
        self.name, self.ret, self.params, self.body, self.static = name, ret, params, body, static


class Glob(object):
    def __init__(self, name, ct, init, quals):
        self.name, self.ct, self.init, self.quals = name, ct, init, quals


# ======================================================================================
# 2. declaration parser + symbol tables
# ======================================================================================
class Unit(object):
    """Everything found in the C sources: types, globals, functions."""
    def __init__(self, pp):
        self.pp = pp
        self.structs = OrderedDict()      # name -> [(field, CT)]
        self.alias = {}                   # typedef name -> CT
        self.consts = {}                  # enum constants
        self.globals = OrderedDict()
        self.funcs = OrderedDict()
        self.warnings = []
        self.model = None
        self.name_override = {}   # raw C name -> forced python attr
        self._anon = 0
        self._ns = None

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def anon_name(self):
        self._anon += 1
        return '__anon%d' % self._anon

    # ---- names ----
    def strip_model(self, name):
        m = self.model
        if m and name.startswith(m + '_'):
            return name[len(m) + 1:]
        return name

    def gattr(self, name):
        if name in self.name_override:
            return self.name_override[name]
        return pyname(self.strip_model(name))

    pyfunc = gattr

    # ---- types ----
    def resolve(self, ct):
        base, dims, seen = ct.base, ct.dims, 0
        while base in self.alias and seen < 32 and self.alias[base].base != base:
            a = self.alias[base]
            base, dims = a.base, a.dims + dims
            seen += 1
        if base in BUILTIN_ALIASES and base not in self.alias:
            base = BUILTIN_ALIASES[base]
        return CT(base, dims)

    def kind(self, ct):
        if ct is None:
            return None
        r = self.resolve(ct)
        if r.base in SCALARS:
            return SCALARS[r.base][0]
        if r.base in self.structs:
            return 's'
        return 'o'

    def is_type_name(self, v):
        return v in SCALARS or v in self.alias or v in self.structs or v in BUILTIN_ALIASES

    def const_int(self, toks):
        try:
            tr = Tr(self, list(toks), const=True)
            e = tr.expr()
            if self._ns is None:
                self._ns = {'STRUCTS': {}}
                exec(build_runtime(), self._ns)
            return int(eval(e.code, self._ns))
        except Exception as ex:
            self.warn('cannot evaluate constant expression "%s": %s'
                      % (' '.join(t.v for t in toks), ex))
            return 0

    def parse(self, toks):
        Parser(self, toks).parse_top()


class Parser(object):
    EOF = Tok('eof', '', ())

    def __init__(self, unit, toks):
        self.u, self.t, self.p = unit, toks, 0

    # ---- token helpers ----
    def peek(self, k=0):
        i = self.p + k
        return self.t[i] if i < len(self.t) else self.EOF

    def next(self):
        t = self.peek()
        self.p += 1
        return t

    def at(self, v, k=0):
        t = self.peek(k)
        return t.v == v and t.k in ('op', 'id')

    def accept(self, v):
        if self.at(v):
            self.p += 1
            return True
        return False

    def expect(self, v):
        if not self.accept(v):
            raise CError("expected '%s' but found '%s'" % (v, self.peek().v))

    def need_id(self):
        t = self.next()
        if t.k != 'id':
            raise CError("identifier expected, found '%s'" % t.v)
        return t.v

    def find_end(self, i, stops):
        d, t = 0, self.t
        while i < len(t):
            if t[i].k == 'op':
                v = t[i].v
                if v in ('(', '[', '{'):
                    d += 1
                elif v in (')', ']', '}'):
                    if d == 0:
                        return i
                    d -= 1
                elif d == 0 and v in stops:
                    return i
            i += 1
        return i

    def skip_balanced(self, o, c):
        d = 0
        while self.p < len(self.t):
            v = self.t[self.p]
            if v.k == 'op' and v.v == o:
                d += 1
            elif v.k == 'op' and v.v == c:
                d -= 1
                if d == 0:
                    self.p += 1
                    return
            self.p += 1

    def take_braced(self):
        self.expect('{')
        j = self.p
        d = 1
        while j < len(self.t):
            v = self.t[j]
            if v.k == 'op' and v.v == '{':
                d += 1
            elif v.k == 'op' and v.v == '}':
                d -= 1
                if d == 0:
                    break
            j += 1
        body = self.t[self.p:j]
        self.p = j + 1
        return body

    # ---- declarations ----
    def unknown_type_ahead(self):
        n1, n2 = self.peek(1), self.peek(2)
        return n1.k == 'id' or (n1.v == '*' and n1.k == 'op' and (n2.k == 'id' or n2.v == '*'))

    def parse_specs(self):
        quals, words, base = set(), [], None
        while True:
            t = self.peek()
            if t.k != 'id':
                break
            v = t.v
            if v in QUALS:
                quals.add(v)
                self.p += 1
            elif v in BUILTIN_WORDS:
                words.append(v)
                self.p += 1
            elif v in ('struct', 'union'):
                self.p += 1
                tag = self.next().v if self.peek().k == 'id' else None
                if self.at('{'):
                    name = tag or self.u.anon_name()
                    self.u.structs[name] = self.parse_struct_body()
                else:
                    name = tag
                base = name
                break
            elif v == 'enum':
                self.p += 1
                if self.peek().k == 'id':
                    self.p += 1
                if self.at('{'):
                    self.parse_enum_body()
                base = 'int'
                break
            elif base is None and not words and (self.u.is_type_name(v) or self.unknown_type_ahead()):
                self.p += 1
                base = v
                break
            else:
                break
        if words:
            base = norm_words(words)
        return base, quals

    def parse_struct_body(self):
        self.expect('{')
        fields = []
        while not self.at('}'):
            if self.peek().k == 'eof':
                raise CError('unterminated struct')
            base, _ = self.parse_specs()
            if base is None:
                raise CError('bad struct member near "%s"' % self.peek().v)
            while not self.at(';'):
                name, ct, _ = self.parse_declarator(base)
                if self.accept(':'):
                    self.p += 1
                if name:
                    fields.append((name, ct))
                if not self.accept(','):
                    break
            self.expect(';')
        self.expect('}')
        return fields

    def parse_enum_body(self):
        self.expect('{')
        val = 0
        while not self.at('}'):
            name = self.need_id()
            if self.accept('='):
                j = self.find_end(self.p, {',', '}'})
                val = self.u.const_int(self.t[self.p:j])
                self.p = j
            self.u.consts[name] = val
            val += 1
            self.accept(',')
        self.expect('}')

    def parse_declarator(self, base):
        ptr = 0
        while True:
            if self.accept('*'):
                ptr += 1
            elif self.peek().k == 'id' and self.peek().v in QUALS:
                self.p += 1
            else:
                break
        if self.at('('):                       # function pointer / grouped declarator: skip
            self.skip_balanced('(', ')')
            if self.at('('):
                self.skip_balanced('(', ')')
            return None, CT(base, ()), False
        if self.peek().k != 'id':
            return None, CT(base, (0,) * ptr), False
        name = self.next().v
        dims = []
        while self.at('['):
            self.p += 1
            if self.at(']'):
                dims.append(0)
            else:
                j = self.find_end(self.p, {']'})
                dims.append(self.u.const_int(self.t[self.p:j]))
                self.p = j
            self.expect(']')
        return name, CT(base, tuple(dims) + (0,) * ptr), self.at('(')

    def parse_params(self):
        self.expect('(')
        params = []
        if self.at('void') and self.at(')', 1):
            self.p += 2
            return params
        while not self.at(')'):
            if self.accept('...'):
                break
            base, _ = self.parse_specs()
            if base is None:
                raise CError('bad parameter near "%s"' % self.peek().v)
            name, ct, _ = self.parse_declarator(base)
            params.append((name or '_p%d' % len(params), ct))
            if not self.accept(','):
                break
        self.expect(')')
        return params

    # ---- top level ----
    def parse_top(self):
        while self.p < len(self.t):
            tok = self.peek()
            start = self.p
            try:
                if tok.v in (';', '}') and tok.k == 'op':
                    self.p += 1
                elif tok.v == 'extern' and self.peek(1).k == 'str':
                    self.p += 2
                    self.accept('{')
                elif tok.v == 'typedef':
                    self.top_typedef()
                else:
                    self.top_decl()
            except CError as e:
                self.u.warn('parse problem near "%s": %s' % (tok.v, e))
                j = start
                while j < len(self.t) and self.t[j].v not in (';', '{'):
                    j += 1
                self.p = j
                if j < len(self.t) and self.t[j].v == '{':
                    self.skip_balanced('{', '}')
                else:
                    self.p = j + 1

    def top_typedef(self):
        self.p += 1
        base, _ = self.parse_specs()
        if base is None:
            raise CError('bad typedef')
        while True:
            name, ct, _ = self.parse_declarator(base)
            if name:
                if ct.base.startswith('__anon') and ct.base in self.u.structs and not ct.dims:
                    self.u.structs[name] = self.u.structs.pop(ct.base)      # give the struct its typedef name
                    base = name
                elif ct.base != name:
                    self.u.alias[name] = ct
            if not self.accept(','):
                break
        self.expect(';')

    def top_decl(self):
        base, quals = self.parse_specs()
        if base is None:
            raise CError('unknown declaration')
        if self.accept(';'):
            return
        while True:
            name, ct, isfn = self.parse_declarator(base)
            if name is None:
                raise CError('cannot parse declarator')
            if isfn:
                params = self.parse_params()
                if self.at('{'):
                    body = self.take_braced()
                    self.u.funcs[name] = Func(name, ct, params, body, 'static' in quals)
                    return
            else:
                init = None
                if self.accept('='):
                    j = self.find_end(self.p, {',', ';'})
                    init = self.t[self.p:j]
                    self.p = j
                if 'extern' not in quals:
                    self.u.globals[name] = Glob(name, ct, init, quals)
            if not self.accept(','):
                break
        self.expect(';')


# ======================================================================================
# 3. translator:  C expressions / statements  ->  python source
# ======================================================================================
class E(object):
    """translated expression: python code, C type, lvalue info, deref info"""
    __slots__ = ('code', 'ct', 'lv', 'deref')

    def __init__(self, code, ct=None, lv=None, deref=None):
        self.code, self.ct, self.lv, self.deref = code, ct, lv, deref


BINPREC = {'||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '<=': 7,
           '>': 7, '>=': 7, '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10}
ASSIGN_OPS = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='}
FLOAT_LIT = re.compile(r'^-?\d+\.\d*(e[+-]?\d+)?$|^-?\d+e[+-]?\d+$', re.I)
INT_CT = CT('int_T', ())
FLT_CT = CT('real_T', ())
BOOL_CT = CT('boolean_T', ())


def elem_ct(ct):
    if ct is None:
        return None
    return CT(ct.base, ct.dims[1:]) if ct.dims else ct


class Tr(Parser):
    def __init__(self, unit, toks, locals_=None, boxed=None, const=False, ret_ct=None):
        Parser.__init__(self, unit, toks)
        self.locals = locals_ if locals_ is not None else {}
        self.boxed = boxed if boxed is not None else set()
        self.const, self.ret_ct = const, ret_ct

    def child(self, toks):
        return Tr(self.u, toks, self.locals, self.boxed, self.const, self.ret_ct)

    # ------------------------------------------------------------------ expressions
    def expr(self):
        return self.ternary()

    def ternary(self):
        c = self.binary(1)
        if self.accept('?'):
            a = self.ternary()
            self.expect(':')
            b = self.ternary()
            return E('(%s if %s else %s)' % (a.code, c.code, b.code), a.ct or b.ct)
        return c

    def bin_ct(self, op, a, b):
        if op in ('==', '!=', '<', '<=', '>', '>=', '&&', '||'):
            return BOOL_CT
        ka, kb = self.u.kind(a), self.u.kind(b)
        if ka is None or kb is None or ka not in 'fib' or kb not in 'fib':
            return None
        if op in ('<<', '>>', '&', '|', '^', '%'):
            return INT_CT
        return FLT_CT if 'f' in (ka, kb) else INT_CT

    def mk_bin(self, op, a, b):
        ct = self.bin_ct(op, a.ct, b.ct)
        if op == '/':
            if FLOAT_LIT.match(b.code) and float(b.code) != 0.0:
                code = '(%s / %s)' % (a.code, b.code)
            else:
                code = '_div(%s, %s)' % (a.code, b.code)
        elif op == '%':
            code = '_mod(%s, %s)' % (a.code, b.code)
        elif op == '&&':
            code = '(%s and %s)' % (a.code, b.code)
        elif op == '||':
            code = '(%s or %s)' % (a.code, b.code)
        else:
            code = '(%s %s %s)' % (a.code, op, b.code)
        return E(code, ct)

    def binary(self, minp):
        left = self.unary()
        while True:
            t = self.peek()
            if t.k != 'op':
                break
            p = BINPREC.get(t.v)
            if p is None or p < minp:
                break
            self.p += 1
            right = self.binary(p + 1)
            left = self.mk_bin(t.v, left, right)
        return left

    def is_cast(self):
        if not self.at('('):
            return False
        j, t, seen, n = self.p + 1, self.t, False, len(self.t)
        while j < n and t[j].k == 'id':
            v = t[j].v
            if v in ('struct', 'union', 'enum'):
                j += 2
                seen = True
            elif v in QUALS or v in BUILTIN_WORDS:
                seen = seen or v not in QUALS
                j += 1
            elif self.u.is_type_name(v) and v not in self.locals:
                seen = True
                j += 1
            else:
                return False
        while j < n and t[j].k == 'op' and t[j].v == '*':
            j += 1
        return seen and j < n and t[j].k == 'op' and t[j].v == ')'

    def unary(self):
        t = self.peek()
        if t.k == 'op':
            v = t.v
            if v == '-':
                self.p += 1
                e = self.unary()
                return E('(-%s)' % e.code, e.ct)
            if v == '+':
                self.p += 1
                return self.unary()
            if v == '!':
                self.p += 1
                e = self.unary()
                return E('(not %s)' % e.code, BOOL_CT)
            if v == '~':
                self.p += 1
                e = self.unary()
                return E('(~%s)' % e.code, INT_CT)
            if v == '*':
                self.p += 1
                e = self.unary()
                return E('_deref(%s)' % e.code, elem_ct(e.ct), deref=e.code)
            if v == '&':
                self.p += 1
                return self.addr_of(self.unary())
            if v in ('++', '--'):
                raise CError('++/-- inside an expression is not supported')
            if v == '(' and self.is_cast():
                self.p += 1
                base, _ = self.parse_specs()
                ptr = 0
                while self.accept('*'):
                    ptr += 1
                self.expect(')')
                e = self.unary()
                return self.do_cast(base, ptr, e)
        if t.k == 'id' and t.v == 'sizeof':
            self.p += 1
            if self.at('(') and self.child_is_type():
                self.p += 1
                base, _ = self.parse_specs()
                while self.accept('*'):
                    base = 'ptr'
                self.expect(')')
                r = self.u.resolve(CT(base, ()))
                size = SCALARS[r.base][1] // 8 if r.base in SCALARS else 8
            else:
                self.unary()
                size = 8
            return E(str(max(size, 1)), INT_CT)
        return self.postfix()

    def child_is_type(self):
        save = self.p
        self.p += 0
        n = self.peek(1)
        ok = n.k == 'id' and (n.v in QUALS or n.v in BUILTIN_WORDS or n.v in ('struct', 'union', 'enum')
                              or (self.u.is_type_name(n.v) and n.v not in self.locals))
        self.p = save
        return ok

    def do_cast(self, base, ptr, e):
        if ptr:
            return E(e.code, CT(base, (0,) * ptr), e.lv)
        r = self.u.resolve(CT(base, ()))
        if r.base in SCALARS:
            kind, bits, signed = SCALARS[r.base]
            if kind == 'f':
                code = '_f32(%s)' % e.code if bits == 32 else 'float(%s)' % e.code
                return E(code, CT(base, ()))
            if kind == 'i':
                return E('_ci(%s, %d, %s)' % (e.code, bits, signed), CT(base, ()))
            if kind == 'b':
                return E('(1 if %s else 0)' % e.code, CT(base, ()))
        return E(e.code, CT(base, ()), e.lv)

    def addr_of(self, e):
        ct = e.ct
        if e.deref is not None:
            return E(e.deref, CT(ct.base, (0,) + ct.dims) if ct else None)
        if ct is not None and not ct.dims and self.u.kind(ct) in ('f', 'i', 'b') and e.lv:
            k = e.lv[0]
            if k == 'box':
                return E(e.lv[1], CT(ct.base, (0,)))
            if k == 'attr':
                return E('_Ref(%s, %r)' % (e.lv[1], e.lv[2]), CT(ct.base, (0,)))
            if k == 'idx':
                return E('_Ref(%s, %s)' % (e.lv[1], e.lv[2]), CT(ct.base, (0,)))
            self.u.warn('address of plain local scalar not supported: %s' % e.code)
        return E(e.code, CT(ct.base, (0,) + ct.dims) if ct else None)

    def member_ct(self, ct, name):
        if ct is None:
            return None
        r = self.u.resolve(ct)
        for f, fct in self.u.structs.get(r.base, []):
            if f == name:
                return fct
        return None

    def postfix(self):
        t = self.peek()
        if t.k == 'id' and self.at('(', 1) and t.v not in self.locals:
            self.p += 1
            e = self.call(t.v)
        else:
            e = self.primary()
        while True:
            t = self.peek()
            if t.k != 'op':
                break
            if t.v == '[':
                self.p += 1
                idx = self.expr()
                self.expect(']')
                e = E('%s[%s]' % (e.code, idx.code), elem_ct(e.ct), lv=('idx', e.code, idx.code))
            elif t.v in ('.', '->'):
                self.p += 1
                name = self.need_id()
                e = E('%s.%s' % (e.code, pyname(name)), self.member_ct(e.ct, name),
                      lv=('attr', e.code, pyname(name)))
            else:
                break
        return e

    def primary(self):
        t = self.next()
        if t.k == 'num':
            v = num_value(t.v)
            return E(repr(v), FLT_CT if isinstance(v, float) else INT_CT)
        if t.k == 'str':
            return E(t.v, CT('char', (0,)))
        if t.k == 'chr':
            s = t.v[1:-1].encode().decode('unicode_escape')
            return E(str(ord(s[0])), INT_CT)
        if t.k == 'op' and t.v == '(':
            e = self.expr()
            self.expect(')')
            return E('(%s)' % e.code, e.ct, e.lv, e.deref)
        if t.k == 'id':
            return self.sym(t.v)
        raise CError("unexpected token '%s'" % t.v)

    def sym(self, name):
        u = self.u
        if name in self.locals:
            ct = self.locals[name]
            # only scalars are boxed (see declare()); `&a[0]` of a local ARRAY also puts `a` in
            # self.boxed, but an array is passed by reference anyway and has no .v
            if name in self.boxed and not ct.dims and self.u.kind(ct) in ('f', 'i', 'b'):
                return E(pyname(name) + '.v', ct, lv=('box', pyname(name)))
            return E(pyname(name), ct, lv=('var', name))
        if name in BUILTIN_CONSTS:
            return E(BUILTIN_CONSTS[name], FLT_CT)
        if name in u.globals and not self.const:
            g = u.globals[name]
            a = u.gattr(name)
            return E('self.' + a, g.ct, lv=('attr', 'self', a))
        if name in u.consts:
            return E(repr(u.consts[name]), INT_CT)
        if not self.const:
            u.warn('unknown identifier "%s" (treated as self.%s)' % (name, name))
            return E('self.' + pyname(name))
        return E(pyname(name))

    def call(self, name):
        self.expect('(')
        args = []
        if not self.at(')'):
            while True:
                args.append(self.ternary())
                if not self.accept(','):
                    break
        self.expect(')')
        u = self.u
        a = ', '.join(x.code for x in args)
        if name == 'memset':
            return E('_memset(%s, %s)' % (args[0].code, args[1].code))
        if name in ('memcpy', 'memmove'):
            return E('_memcpy(%s, %s)' % (args[0].code, args[1].code))
        if name in ('abs', 'labs'):
            return E('abs(%s)' % a, args[0].ct)
        if name in MATHF and name not in u.funcs:
            return E('_c_%s(%s)' % (MATHF[name], a), FLT_CT)
        if name in u.funcs and not SKIP_FUNC_RE.match(name):
            return E('self.%s(%s)' % (u.pyfunc(name), a), u.funcs[name].ret)
        if name in NOOP_FUNCS:
            return E('None', None)
        if name in RUNTIME_FUNCS:
            return E('%s(%s)' % (name, a), None if name in SIMD_FUNCS else FLT_CT)
        if not name.startswith(('rtsi', 'rtm', 'rt_ertODE')):
            u.warn('call to function "%s" that is not defined in the supplied files' % name)
        return E('self.%s(%s)' % (pyname(name), a))

    # ------------------------------------------------------------------ coercion / init
    def coerce(self, ct, e):
        if ct is None or ct.dims:
            return e.code
        u = self.u
        k = u.kind(ct)
        ek = u.kind(e.ct) if (e.ct is not None and not e.ct.dims) else None
        if k == 'f':
            if u.resolve(ct).base in ('real32_T', 'float'):
                return '_f32(%s)' % e.code
            return e.code if (ek == 'f' or FLOAT_LIT.match(e.code)) else 'float(%s)' % e.code
        if k == 'i':
            return e.code if ek in ('i', 'b') else 'int(%s)' % e.code
        if k == 'b':
            return e.code if ek == 'b' else '(1 if %s else 0)' % e.code
        return e.code

    def init_code(self, toks):
        if toks and toks[0].k == 'op' and toks[0].v == '{':
            inner = toks[1:-1] if toks[-1].v == '}' else toks[1:]
            parts = [p for p in split_top(inner, ',') if p]
            return '[' + ', '.join(self.init_code(p) for p in parts) + ']'
        return self.child(toks).expr().code

    # ------------------------------------------------------------------ statements
    def emit_comments(self, tok, ind, out):
        for c in tok.pre:
            out.append('%s# %s' % ('    ' * ind, c))

    def stmt(self, ind, out):
        pad = '    ' * ind
        tok = self.peek()
        self.emit_comments(tok, ind, out)
        v, k = tok.v, tok.k
        if k == 'op':
            if v == '{':
                self.p += 1
                return self.block(ind, out)
            if v == ';':
                self.p += 1
                return
        if k == 'id':
            if v == 'if':
                return self.s_if(ind, out, 'if')
            if v == 'for':
                return self.s_for(ind, out)
            if v == 'while':
                self.p += 1
                self.expect('(')
                c = self.expr()
                self.expect(')')
                out.append('%swhile %s:' % (pad, c.code))
                return self.sub(ind + 1, out)
            if v == 'do':
                self.p += 1
                out.append(pad + 'while True:')
                self.sub(ind + 1, out)
                self.expect('while')
                self.expect('(')
                c = self.expr()
                self.expect(')')
                self.accept(';')
                out.append('%s    if not (%s): break' % (pad, c.code))
                return
            if v == 'switch':
                return self.s_switch(ind, out)
            if v == 'return':
                self.p += 1
                if self.accept(';'):
                    out.append(pad + 'return')
                    return
                e = self.expr()
                self.expect(';')
                out.append('%sreturn %s' % (pad, self.coerce(self.ret_ct, e)))
                return
            if v in ('break', 'continue'):
                self.p += 1
                self.expect(';')
                out.append(pad + v)
                return
            if v == 'goto':
                raise CError('goto is not supported')
        if self.decl_ahead():
            return self.s_decl(ind, out)
        j = self.find_end(self.p, {';'})
        seg = self.t[self.p:j]
        self.p = j
        self.expect(';')
        self.simple_seg(seg, ind, out)

    def block(self, ind, out):
        while not self.at('}'):
            if self.peek().k == 'eof':
                raise CError('missing }')
            self.stmt(ind, out)
        self.emit_comments(self.next(), ind, out)

    def sub(self, ind, out):
        start = len(out)
        self.stmt(ind, out)
        if all(l.strip().startswith('#') for l in out[start:]):
            out.append('    ' * ind + 'pass')

    def s_if(self, ind, out, kw):
        pad = '    ' * ind
        self.p += 1
        self.expect('(')
        c = self.expr()
        self.expect(')')
        out.append('%s%s %s:' % (pad, kw, c.code))
        self.sub(ind + 1, out)
        if self.at('else'):
            self.p += 1
            if self.at('if'):
                self.s_if(ind, out, 'elif')
            else:
                out.append(pad + 'else:')
                self.sub(ind + 1, out)

    # ---- declarations inside functions
    def decl_ahead(self):
        t = self.peek()
        if t.k != 'id' or t.v in STMT_KW:
            return False
        v = t.v
        if v in QUALS or v in BUILTIN_WORDS or v in ('struct', 'union', 'enum'):
            return True
        if v in self.locals:
            return False
        n1, n2, n3 = self.peek(1), self.peek(2), self.peek(3)
        if self.u.is_type_name(v):
            return n1.k == 'id' or n1.v == '*'
        if n1.k == 'id' and n2.v in (';', '=', '[', ','):
            return True
        return n1.v == '*' and n2.k == 'id' and n3.v in (';', '=', '[', ',')

    def s_decl(self, ind, out, end=True):
        base, quals = self.parse_specs()
        while True:
            name, ct, _ = self.parse_declarator(base)
            init = None
            if self.accept('='):
                j = self.find_end(self.p, {',', ';'})
                init = self.t[self.p:j]
                self.p = j
            if name:
                self.declare(name, ct, init, ind, out)
            if not self.accept(','):
                break
        if end:
            self.expect(';')

    def declare(self, name, ct, init, ind, out):
        pad = '    ' * ind
        u = self.u
        py = pyname(name)
        brace = bool(init) and init[0].v == '{' and init[0].k == 'op'
        if ct.dims == (0,) and brace:                       # int a[] = {...}
            ct = CT(ct.base, (len([p for p in split_top(init[1:-1], ',') if p]),))
        self.locals[name] = ct
        r = u.resolve(ct)
        kd = u.kind(ct)
        if not ct.dims:
            if kd in ('f', 'i', 'b'):
                if init is not None:
                    val = self.coerce(ct, self.child(init).expr())
                else:
                    val = '0.0' if kd == 'f' else '0'
                if name in self.boxed:
                    out.append('%s%s = _Box(%s)' % (pad, py, val))
                else:
                    out.append('%s%s = %s' % (pad, py, val))
            elif kd == 's':
                if brace:
                    out.append('%s%s = _init_val(%r, (), %s)' % (pad, py, r.base, self.init_code(init)))
                else:
                    out.append('%s%s = _new(%r)' % (pad, py, r.base))
            else:
                out.append('%s%s = _Opaque()' % (pad, py))
        elif all(d > 0 for d in ct.dims):
            if brace:
                out.append('%s%s = _init_val(%r, %r, %s)' % (pad, py, r.base, ct.dims, self.init_code(init)))
            else:
                out.append('%s%s = _new(%r, %r)' % (pad, py, r.base, ct.dims))
        else:                                                # pointer
            if init is not None and not brace:
                out.append('%s%s = %s' % (pad, py, self.child(init).expr().code))
            else:
                out.append('%s%s = None' % (pad, py))

    # ---- simple statements
    def simple_seg(self, seg, ind, out):
        for part in split_top(seg, ','):
            part = strip_wrap(part)
            if part:
                self.one_stmt(part, ind, out)

    def emit_assign(self, lhs, rhs, ind, out, op='='):
        pad = '    ' * ind
        if op != '=':
            cur = E('_deref(%s)' % lhs.deref if lhs.deref is not None else lhs.code, lhs.ct)
            rhs = self.mk_bin(op[:-1], cur, rhs)
        val = self.coerce(lhs.ct, rhs)
        if lhs.deref is not None:
            out.append('%s_store(%s, %s)' % (pad, lhs.deref, val))
        else:
            out.append('%s%s = %s' % (pad, lhs.code, val))

    def one_stmt(self, toks, ind, out):
        sub = self.child(toks)
        t0 = toks[0]
        if t0.k == 'op' and t0.v in ('++', '--'):
            sub.p += 1
            lv = sub.unary()
            return self.emit_assign(lv, E('1', INT_CT), ind, out, '+=' if t0.v == '++' else '-=')
        first = sub.ternary()
        t = sub.peek()
        if t.k == 'op' and t.v in ASSIGN_OPS:
            if t.v == '=':
                exprs = [first]
                while sub.accept('='):
                    exprs.append(sub.ternary())
                final = exprs[-1]
                for tgt in reversed(exprs[:-1]):
                    self.emit_assign(tgt, final, ind, out)
                    final = tgt
            else:
                sub.p += 1
                rhs = sub.ternary()
                self.emit_assign(first, rhs, ind, out, t.v)
            return
        if t.k == 'op' and t.v in ('++', '--'):
            return self.emit_assign(first, E('1', INT_CT), ind, out, '+=' if t.v == '++' else '-=')
        if sub.peek().k != 'eof':
            raise CError("unexpected token '%s' in statement" % sub.peek().v)
        if '(' in first.code and ('self.' in first.code or '_' in first.code):
            out.append('    ' * ind + first.code)

    # ---- loops
    def s_for(self, ind, out):
        pad = '    ' * ind
        self.p += 1
        self.expect('(')
        j1 = self.find_end(self.p, {';'})
        init = self.t[self.p:j1]
        self.p = j1 + 1
        j2 = self.find_end(self.p, {';'})
        cond = self.t[self.p:j2]
        self.p = j2 + 1
        j3 = self.find_end(self.p, {')'})
        step = self.t[self.p:j3]
        self.p = j3 + 1
        rng = self.try_range(init, cond, step)
        if rng:
            out.append('%sfor %s in range(%s):' % (pad, pyname(rng[0]), rng[1]))
            return self.sub(ind + 1, out)
        ch = self.child(init)
        if init:
            if ch.decl_ahead():
                ch.s_decl(ind, out, end=False)
            else:
                ch.simple_seg(init, ind, out)
        c = self.child(cond).expr().code if cond else 'True'
        out.append('%swhile %s:' % (pad, c))
        self.sub(ind + 1, out)
        if any(l.strip() == 'continue' for l in out[-50:]):
            self.u.warn('"continue" inside a generic for-loop: loop step may be skipped')
        self.simple_seg(step, ind + 1, out)

    def try_range(self, init, cond, step):
        it = list(init)
        k = 0
        while k + 1 < len(it) and it[k].k == 'id' and it[k + 1].k == 'id':
            k += 1
        if not (len(it) >= k + 3 and it[k].k == 'id' and it[k + 1].v == '='):
            return None
        var, lo = it[k].v, it[k + 2:]
        if len(cond) < 3 or cond[0].v != var or cond[1].v not in ('<', '<=', '>', '>='):
            return None
        op, hi = cond[1].v, cond[2:]
        s = step
        if len(s) == 2 and s[0].v == var and s[1].v in ('++', '--'):
            stp = 1 if s[1].v == '++' else -1
        elif len(s) == 2 and s[0].v in ('++', '--') and s[1].v == var:
            stp = 1 if s[0].v == '++' else -1
        elif len(s) >= 3 and s[0].v == var and s[1].v in ('+=', '-='):
            stp = self.u.const_int(s[2:]) * (1 if s[1].v == '+=' else -1)
        else:
            return None
        if stp == 0 or (op in ('<', '<=')) != (stp > 0):
            return None
        if k > 0:
            self.locals[var] = INT_CT
        lo_c = self.child(lo).expr().code
        hi_c = self.child(hi).expr().code
        if op == '<=':
            hi_c = '(%s) + 1' % hi_c
        elif op == '>=':
            hi_c = '(%s) - 1' % hi_c
        args = '%s, %s' % (lo_c, hi_c) + (', %d' % stp if stp != 1 else '')
        return var, args

    def s_switch(self, ind, out):
        pad = '    ' * ind
        self.p += 1
        self.expect('(')
        e = self.expr()
        self.expect(')')
        self.expect('{')
        groups, cur = [], None
        while not self.at('}'):
            if self.peek().k == 'eof':
                raise CError('missing } in switch')
            if self.at('case') or self.at('default'):
                is_case = self.at('case')
                self.p += 1
                lab = self.ternary().code if is_case else 'default'
                self.expect(':')
                if cur is None or cur['lines']:
                    cur = {'labels': [], 'lines': []}
                    groups.append(cur)
                cur['labels'].append(lab)
            else:
                if cur is None:
                    cur = {'labels': [], 'lines': []}
                    groups.append(cur)
                self.stmt(ind + 1, cur['lines'])
        self.expect('}')
        for i in range(len(groups) - 1, -1, -1):
            g = groups[i]
            code = [l for l in g['lines'] if not l.strip().startswith('#')]
            term = False
            if code and code[-1].strip() == 'break':
                idx = max(j for j, l in enumerate(g['lines']) if l.strip() == 'break')
                del g['lines'][idx]
                term = True
            elif code and code[-1].strip().startswith(('return', 'continue')):
                term = True
            g['full'] = g['lines'] + (groups[i + 1]['full'] if not term and i + 1 < len(groups) else [])
        var = '_sw%d' % ind
        out.append('%s%s = %s' % (pad, var, e.code))
        first, default = True, None
        for g in groups:
            if 'default' in g['labels']:
                default = g
                continue
            out.append('%s%s %s in (%s,):' % (pad, 'if' if first else 'elif', var, ', '.join(g['labels'])))
            out.extend(g['full'] or [pad + '    pass'])
            first = False
        if default:
            out.append(pad + ('if True:' if first else 'else:'))
            out.extend(default['full'] or [pad + '    pass'])


# ======================================================================================
# 4. code generation
# ======================================================================================
def build_runtime():
    return RUNTIME.replace('@@SCALARS@@', repr(SCALARS))


def gen_function(u, f):
    tr = Tr(u, f.body, ret_ct=f.ret)
    toks = f.body
    for i in range(len(toks) - 1):                    # locals whose address is taken -> boxed
        t = toks[i]
        if t.k == 'op' and t.v == '&' and toks[i + 1].k == 'id' and (i == 0 or toks[i - 1].v in ('(', ',', '=')):
            tr.boxed.add(toks[i + 1].v)
    for name, ct in f.params:
        tr.locals[name] = ct
    pnames = ''.join(', ' + pyname(n) for n, _ in f.params)
    out = ['    def %s(self%s):' % (u.pyfunc(f.name), pnames), '        # C function: %s' % f.name]
    for name, ct in f.params:                          # C converts int args to double params
        if not ct.dims and u.kind(ct) == 'f':
            out.append('        %s = float(%s)' % (pyname(name), pyname(name)))
    while tr.peek().k != 'eof':
        start, mark = tr.p, len(out)
        try:
            tr.stmt(2, out)
        except Exception as ex:                        # keep going, mark the place
            del out[mark:]
            tr.p = tr.find_end(start, {';'}) + 1
            msg = 'untranslated statement in %s: %s' % (f.name, ex)
            u.warn(msg)
            out.append('        raise NotImplementedError(%r)' % msg)
    if len(out) == 2 or all(l.strip().startswith('#') for l in out[2:]):
        out.append('        pass')
    return out


def mark_special_globals(u):
    """The <model>_M real-time-model pointer's C name is supposed to be <model>+'_M', but for
    long model names Simulink Coder's ANSI-C identifier-length limit can shorten just this one
    symbol (e.g. 'Longitidunal_Vehicle_Dynamics_M' -> 'Longitidunal_Vehicle_Dynamic_M'), which
    breaks the usual model-name-prefix stripping. Find it by TYPE instead (always a pointer to a
    'tag_RTM_..._T' struct - that tag is stable even when the variable name gets truncated) and
    force its python attribute to 'M', which is what the CModel runtime expects."""
    for g in u.globals.values():
        r = u.resolve(g.ct)
        if r.dims == (0,) and re.match(r'^tag_RTM_.*_T$', r.base):
            u.name_override[g.name] = 'M'
            return


def detect_model(u, forced=None):
    if forced:
        return forced
    cands = [n[:-len('_initialize')] for n in u.funcs if n.endswith('_initialize')]
    cands = [c for c in cands if any(('%s_%s' % (c, s)) in u.funcs for s in ('step', 'step0', 'output'))]
    if not cands:
        raise SystemExit('cannot find <model>_initialize / <model>_step in the supplied C files')
    if len(cands) > 1:
        u.warn('several models found %s - using %s (see --model)' % (cands, cands[0]))
    return cands[0]


def generate(u, model, solver, srcnames):
    L = ['# -*- coding: utf-8 -*-',
         '# Generated by simulink_c2py %s from: %s' % (VERSION, ', '.join(srcnames)),
         '# Python translation of the Simulink Coder C code of model "%s".' % model]
    L.append(build_runtime())
    L.append('')
    L.append('STRUCTS = {')
    for name, fields in u.structs.items():
        fl = []
        for fname, ct in fields:
            r = u.resolve(ct)
            fl.append((pyname(fname), r.base, r.dims))
        L.append('    %r: %r,' % (name, fl))
    L.append('}')
    L.append('')

    def find(prefix, attr):
        for g in u.globals.values():
            if g.ct.base.startswith(prefix) and not g.ct.dims:
                return u.gattr(g.name)
        return attr
    xdot = None
    for name in list(u.alias) + list(u.structs):
        if name.startswith('XDot_'):
            xdot = u.resolve(CT(name, ())).base
    if 'step' in [u.strip_model(n) for n in u.funcs]:
        step_fn, upd = 'step', None
    elif 'step0' in [u.strip_model(n) for n in u.funcs]:
        step_fn, upd = 'step0', None
        u.warn('multi-rate model: only step0 (base rate) is called by simulate()')
    else:
        step_fn, upd = 'output', ('update' if model + '_update' in u.funcs else None)

    L.append('class %s(CModel):' % model)
    L.append('    MODEL_NAME = %r' % model)
    L.append('    SOLVER = %r' % solver)
    L.append('    STEP_FN = %r' % step_fn)
    L.append('    UPDATE_FN = %r' % upd)
    L.append('    X_ATTR = %r' % find('X_', 'X'))
    L.append('    U_ATTR = %r' % find('ExtU_', 'U'))
    L.append('    Y_ATTR = %r' % find('ExtY_', 'Y'))
    L.append('    XDOT_TYPE = %r' % xdot)
    L.append('')
    L.append('    def _build_globals(self):')
    tr = Tr(u, [])
    n0 = len(L)
    for g in u.globals.values():
        if g.name in SKIP_GLOBALS:
            continue
        r = u.resolve(g.ct)
        a = u.gattr(g.name)
        if g.init is None:
            code = '_new(%r, %r)' % (r.base, r.dims)
        elif g.init[0].v == '{':
            code = '_init_val(%r, %r, %s)' % (r.base, r.dims, tr.init_code(g.init))
        else:
            code = tr.coerce(g.ct, tr.child(g.init).expr())
        L.append('        self.%s = %s' % (a, code))
    if len(L) == n0:
        L.append('        pass')
    for f in u.funcs.values():
        L.append('')
        L.extend(gen_function(u, f))
    L.append('')
    L.append('')
    L.append('Model = %s' % model)
    L.append('')
    L.append("if __name__ == '__main__':")
    L.append('    _main(Model)')
    L.append('')
    return '\n'.join(L)


# ======================================================================================
# 5. driver
# ======================================================================================
def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names if n.lower().endswith(('.c', '.h'))]
        else:
            files += glob.glob(p) or [p]
    files = [f for f in files if not SKIP_FILE_RE.search(f)]

    def key(f):
        b = os.path.basename(f).lower()
        return (0 if b.endswith('.h') else 1,
                0 if b == 'rtwtypes.h' else (1 if b.endswith('_types.h') else 2),
                0 if b.endswith('_data.c') else 1, b)
    return sorted(set(files), key=key)


# The parts of the generated headers that are the same for every model, so that <model>.c,
# <model>.h and <model>_data.c are enough: the type names of rtwtypes.h (used when it is not
# given) and the three solver macros of <model>_private.h (each only where nothing defines it).
STANDARD_TYPES = """#ifndef RTWTYPES_H
#define RTWTYPES_H
#ifndef false
#define false (0U)
#endif
#ifndef true
#define true (1U)
#endif
typedef signed char int8_T;
typedef unsigned char uint8_T;
typedef short int16_T;
typedef unsigned short uint16_T;
typedef int int32_T;
typedef unsigned int uint32_T;
typedef float real32_T;
typedef double real64_T;
typedef double real_T;
typedef double time_T;
typedef unsigned char boolean_T;
typedef int int_T;
typedef unsigned int uint_T;
typedef unsigned long ulong_T;
typedef char char_T;
typedef unsigned char uchar_T;
typedef char_T byte_T;
typedef struct { real32_T re; real32_T im; } creal32_T;
typedef struct { real64_T re; real64_T im; } creal64_T;
typedef struct { real_T re; real_T im; } creal_T;
typedef struct { int8_T re; int8_T im; } cint8_T;
typedef struct { uint8_T re; uint8_T im; } cuint8_T;
typedef struct { int16_T re; int16_T im; } cint16_T;
typedef struct { uint16_T re; uint16_T im; } cuint16_T;
typedef struct { int32_T re; int32_T im; } cint32_T;
typedef struct { uint32_T re; uint32_T im; } cuint32_T;
#endif
"""
STANDARD_MACROS = """#ifndef rtmIsMajorTimeStep
#define rtmIsMajorTimeStep(rtm) (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
#endif
#ifndef rtmIsMinorTimeStep
#define rtmIsMinorTimeStep(rtm) (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
#endif
#ifndef rtmSetTPtr
#define rtmSetTPtr(rtm, val) ((rtm)->Timing.t = (val))
#endif
"""


def missing_typedefs(raws):
    """The typedefs of <model>_types.h, written here when that file is not given.

    Embedded Coder puts only `typedef struct <tag> <name>;` lines there, for structs that
    <model>.h defines, named by fixed rules: the model  tag_RTM_<m>_T -> RT_MODEL_<m>_T  and the
    parameter structs  P_..._T_ -> P_..._T. So <model>.c, <model>.h and <model>_data.c are enough.
    -> the C text of the missing typedefs ('' when nothing is missing)."""
    text = '\n'.join(raws)
    lines = []
    for tag in re.findall(r'\bstruct\s+(\w+)\s*\{', text):
        if tag.startswith('tag_RTM_'):
            name = 'RT_MODEL_' + tag[len('tag_RTM_'):]
        elif tag.endswith('_') and len(tag) > 1:
            name = tag[:-1]
        else:
            continue
        declared = re.search(r'\btypedef\s+struct\s+%s\s+%s\s*;' % (tag, name), text) or \
            re.search(r'}\s*%s\s*;' % name, text)
        if not declared and re.search(r'\b%s\b' % name, text):
            lines.append('typedef struct %s %s;' % (tag, name))
    return '\n'.join(dict.fromkeys(lines))


def transpile(paths, model=None):
    files = gather(paths)
    if not files:
        raise SystemExit('no .c/.h files found')
    pp = PP()
    raws = []
    for f in files:
        with open(f, 'r', errors='replace') as fh:
            raws.append(fh.read())
    # what the headers that are the same for every model would give (see STANDARD_TYPES)
    given = {os.path.basename(f).lower() for f in files}
    extra = missing_typedefs(raws)                    # no <model>_types.h given: its typedefs
    if extra:
        raws.insert(0, extra)
    if 'rtwtypes.h' not in given:
        raws.insert(0, STANDARD_TYPES)
    raws.append(STANDARD_MACROS)                      # last: a macro of the model's headers wins
    texts = [pp.run(r) for r in raws]                 # all #defines are known before expansion
    u = Unit(pp)
    for t in texts:
        u.parse(pp.expand(tokenize(t, pp.comments)))
    u.model = detect_model(u, model)
    mark_special_globals(u)
    for n in [n for n in u.funcs if SKIP_FUNC_RE.match(n)]:
        del u.funcs[n]
    m = re.search(r'rtsiSetSolverName\s*\([^,]*,\s*"(\w+)"', '\n'.join(raws))
    solver = m.group(1) if m else 'ode1'
    return u, generate(u, u.model, solver, [os.path.basename(f) for f in files]), files


def main(argv=None):
    ap = argparse.ArgumentParser(description='Simulink Coder C code -> executable Python model')
    ap.add_argument('paths', nargs='+', help='folder(s) and/or .c/.h files of the generated code')
    ap.add_argument('-o', '--out', help='output .py (default <model>_py.py)')
    ap.add_argument('--model', help='model name (auto-detected from <model>_initialize)')
    ap.add_argument('--no-check', action='store_true', help='do not import/instantiate the result')
    a = ap.parse_args(argv)
    u, src, files = transpile(a.paths, a.model)
    out = a.out or '%s_py.py' % u.model
    with open(out, 'w') as fh:
        fh.write(src)
    print('model      :', u.model)
    print('C files    :', ', '.join(os.path.basename(f) for f in files))
    print('functions  :', ', '.join(u.pyfunc(n) for n in u.funcs))
    print('written    :', out)
    if not a.no_check:
        ns = {'__name__': 'generated_check'}
        exec(compile(src, out, 'exec'), ns)
        m = ns[u.model]()
        print('states     :', m.state_names)
        print('inputs     :', m.input_names)
        print('outputs    :', m.output_names)
        try:
            print('step size  : %g s' % m.step_size)
        except Exception as ex:
            print('step size  : ?', ex)
    for w in u.warnings:
        print('WARNING    :', w)


if __name__ == '__main__':
    main()

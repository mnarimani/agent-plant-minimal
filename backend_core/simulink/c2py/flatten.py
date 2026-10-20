# -*- coding: utf-8 -*-
"""
flatten.py  -  deterministic (NO LLM)   generated python model  ->  dynamics(t, x, u)

The generated <model>_py.py is itself python, so its step()/output() + derivatives() methods can
be turned into ONE plain function by program transformation instead of asking an LLM to re-type
several hundred lines:

    self.X.<state>            ->  x[i]           self.U.<input>   ->  u[i]
    _rtXdot.<state> = ...     ->  dx[i] = ...    self.B.<signal>  ->  local variable
    self.P.<param>            ->  named constant with its value
    self.DW.<memory>          ->  constant frozen at its initial value (writes are dropped)
    self.M.Timing.t[0]        ->  t              solver bookkeeping, outport writes  ->  removed
    _div/_c_sin/rt_powd_snf.. ->  numpy

followed by clean-up passes (unrolling of the small loops over states, splitting of re-used scratch
variables, state/input unpacking, dead code removal, constant hoisting). The result is exact by
construction and is still checked against the model by agent.Reference.

Layout of the emitted function (what agent.extract_card reads back):
    name = x[i]  # ...        every state
    name = value  # ...       every parameter that is used
    name = u[i]  # ...        every input
    ...                       block signals, in execution order
    dx = np.zeros(n)
    dx[i] = ...
    return dx

Constructs that are not handled (pointers into structs, calls of other model functions that
compute signals, 2-D state fields ...) raise Unsupported; the caller then falls back to the LLM.
"""
import ast
import copy
import io
import keyword
import re
import tokenize

import numpy as np


class Unsupported(Exception):
    pass


class _Drop(Exception):
    """raised while translating an expression that only concerns the solver bookkeeping"""


NP1 = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
       'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asinh': 'arcsinh', 'acosh': 'arccosh',
       'atanh': 'arctanh', 'exp': 'exp', 'exp2': 'exp2', 'expm1': 'expm1', 'log': 'log', 'log10': 'log10',
       'log2': 'log2', 'log1p': 'log1p', 'sqrt': 'sqrt', 'cbrt': 'cbrt', 'fabs': 'abs', 'floor': 'floor',
       'ceil': 'ceil', 'trunc': 'trunc', 'rint': 'rint', 'atan2': 'arctan2', 'pow': 'power', 'fmod': 'fmod',
       'fmin': 'fmin', 'fmax': 'fmax', 'hypot': 'hypot', 'copysign': 'copysign'}
CALLS = {'_c_' + k: v for k, v in NP1.items()}
CALLS.update({'rt_powd_snf': 'power', 'rt_atan2d_snf': 'arctan2', 'rt_hypotd_snf': 'hypot', '_mod': 'fmod',
              'rtIsNaN': 'isnan', 'rtIsNaNF': 'isnan', 'rtIsInf': 'isinf', 'rtIsInfF': 'isinf'})
CONST_NAMES = {'rtInf': 'inf', 'rtInfF': 'inf', 'inf': 'inf', 'rtNaN': 'nan', 'rtNaNF': 'nan', 'nan': 'nan'}
INT_TYPES = {'char', 'signed char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned int', 'long',
             'unsigned long', 'long long', 'unsigned long long', 'boolean_T', '_Bool', 'int8_T', 'uint8_T',
             'int16_T', 'uint16_T', 'int32_T', 'uint32_T', 'int64_T', 'uint64_T', 'int_T', 'uint_T', 'size_t'}
RESERVED = set(keyword.kwlist) | {'np', 't', 'x', 'u', 'dx', 'self', 'abs', 'int', 'float', 'range', 'min', 'max',
                                  'len', 'True', 'False', 'None', 'dynamics', '_y_'}
NOISE_CM = re.compile(r'^(End of |end MajorTimeStep|set solver stop time|Update absolute time|C function:|'
                      r'The "clockTick|been executed|and "Timing|overflow during|of the task|application lifespan)')


def _np(fn, *args):
    return ast.Call(ast.Attribute(ast.Name('np', ast.Load()), fn, ast.Load()), list(args), [])


def _const(v):
    if isinstance(v, (bool, np.bool_)):
        return ast.Constant(int(v))
    if isinstance(v, (int, np.integer)):
        v = int(v)
    else:
        v = float(v)
        if v != v:
            return ast.Attribute(ast.Name('np', ast.Load()), 'nan', ast.Load())
        if v in (float('inf'), float('-inf')):
            n = ast.Attribute(ast.Name('np', ast.Load()), 'inf', ast.Load())
            return n if v > 0 else ast.UnaryOp(ast.USub(), n)
    return ast.UnaryOp(ast.USub(), ast.Constant(-v)) if v < 0 else ast.Constant(v)


def _value_node(v):
    if isinstance(v, np.ndarray):
        def lst(a):
            return ast.List([lst(e) if isinstance(e, np.ndarray) else _const(e) for e in a], ast.Load())
        return _np('array', lst(v))
    return _const(v)


def _chain(node):
    """self.A.b.c -> ['A', 'b', 'c']   (None if the attribute chain does not start at self)"""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == 'self':
        return parts[::-1]
    return None


def _root(node):
    """(root Name id, [attrs]) of  name.a.b"""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return (node.id, parts[::-1]) if isinstance(node, ast.Name) else (None, None)


def _names(node, ctx=None):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and (ctx is None or isinstance(n.ctx, ctx))}


def _is_num(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _fold(node):
    """constant-fold an expression made of literals only (None if it is not constant)"""
    if any(isinstance(n, (ast.Name, ast.Call, ast.Attribute, ast.Subscript)) for n in ast.walk(node)):
        return None
    try:
        e = ast.fix_missing_locations(ast.Expression(copy.deepcopy(node)))
        return eval(compile(e, '<fold>', 'eval'), {'__builtins__': {}}, {})
    except Exception:
        return None


class _FoldIdx(ast.NodeTransformer):
    """3 + 2 -> 5 inside subscripts"""
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if not isinstance(node.slice, (ast.Constant, ast.Slice)):
            v = _fold(node.slice)
            if isinstance(v, int) and not isinstance(v, bool):
                node.slice = ast.Constant(v)
        return node


class _Subst(ast.NodeTransformer):
    def __init__(self, mapping):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            new = self.m[node.id]
            if isinstance(new, str):
                return ast.Name(new, node.ctx)
            if isinstance(node.ctx, ast.Load):
                return copy.deepcopy(new)
        return node


_SRC_CM = re.compile(r"^(:\s|'[^']*:\d+')")           # a source line of a MATLAB Function block


def _src_cm(stmt):
    """the MATLAB source lines among the comments of a statement (and of the statements inside it)"""
    return [c for q in _stmts([stmt]) for c in getattr(q, '_cm', ()) if _SRC_CM.match(c)]


def _give_cm(stmt, pend):
    """put the carried comments `pend` in front of the comments of stmt (and empty pend)"""
    if pend:
        stmt._cm = pend + [c for c in getattr(stmt, '_cm', ()) if c not in pend]
        del pend[:]


def _as_store(node):
    if isinstance(node, (ast.Name, ast.Subscript)):
        node.ctx = ast.Store()
    return node


def _stmts(body):
    """every statement in a statement list, nested ones included"""
    for s in body:
        yield s
        for attr in ('body', 'orelse'):
            if isinstance(getattr(s, attr, None), list):
                for y in _stmts(getattr(s, attr)):
                    yield y


def _store_base(tgt):
    """a[i][j] = ... -> the Name node a   (None if the target is not a subscript of a plain name)"""
    while isinstance(tgt, ast.Subscript):
        tgt = tgt.value
    return tgt if isinstance(tgt, ast.Name) else None


def _loads(body):
    """names that are READ somewhere (the base of a subscript store  a[i] = ...  does not count)"""
    bases = set()
    for s in _stmts(body):
        tg = s.targets if isinstance(s, ast.Assign) else []
        for tt in tg:
            if isinstance(tt, ast.Subscript):
                b = _store_base(tt)
                if b is not None:
                    bases.add(id(b))
    out = set()
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in bases:
                out.add(n.id)
    return out


# ======================================================================================
class Flat(object):
    """result of flatten(): the statements of the function + how to print them"""

    def __init__(self):
        self.states, self.inputs = [], []      # [(python name, default comment)]
        self.params = []                       # [(name, value node, default comment)]
        self.body, self.dx = [], []            # statements;  dx = [(i, expression)]
        self.dx_cm = {}                        # i -> comment lines above  dx[i] = ...
        self.outputs = []                      # per model output: expression node or None
        self.nx = 0

    def names(self):
        """every name a rename may touch -> kind"""
        d = {}
        for s in self.body:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and n.id not in RESERVED:
                    d.setdefault(n.id, 'signal')
        for n, _, _ in self.params:
            d[n] = 'parameter'
        for n, _ in self.states:
            d[n] = 'state'
        for n, _ in self.inputs:
            d[n] = 'input'
        return d

    def clean_rename(self, rename):
        """keep only the renames that are safe: known old name, valid new name, no two variables
        ending up with the same name"""
        known, out = self.names(), {}
        for old, new in (rename or {}).items():
            if isinstance(old, str) and isinstance(new, str) and old in known and new != old \
                    and re.fullmatch(r'[A-Za-z_]\w*', new) and new not in RESERVED:
                out[old] = new
        while True:
            final = {}
            for n in known:
                final.setdefault(out.get(n, n), []).append(n)
            clash = [o for grp in final.values() if len(grp) > 1 for o in grp if o in out]
            if not clash:
                return out
            for o in clash:
                del out[o]

    def adopt(self, old, new):
        """The state / input `old` takes over the name of the variable `new` that starts as its copy:

            Integrator_0 = x[0]                        pos = x[0]
            pos = Integrator_0               ->        if pos < 0.0001:
            if Integrator_0 < 0.0001:                      pos = 0.0001
                pos = 0.0001

        (the source of the model says  pos = x(1);  the code generator kept both). Returns the new
        Flat, or None when the two cannot be merged safely: `new` must start as that plain copy,
        and `old` must not be read any more once `new` has been given another value."""
        kinds = self.names()
        if kinds.get(old) not in ('state', 'input') or kinds.get(new) != 'signal':
            return None
        fl = copy.deepcopy(self)

        def uses(node, name, ctx=None):
            return any(isinstance(n, ast.Name) and n.id == name and (ctx is None or isinstance(n.ctx, ctx))
                       for n in ast.walk(node))
        at = next((i for i, s in enumerate(fl.body) if uses(s, new)), None)
        if at is None:
            return None
        cp = fl.body[at]
        if not (isinstance(cp, ast.Assign) and len(cp.targets) == 1 and isinstance(cp.targets[0], ast.Name)
                and cp.targets[0].id == new and isinstance(cp.value, ast.Name) and cp.value.id == old):
            return None
        st = {'dirty': False, 'ok': True}                # dirty: `new` no longer equals `old`

        def read(node):
            if st['dirty'] and uses(node, old, ast.Load):
                st['ok'] = False

        def scan(stmts):
            for s in stmts:
                if isinstance(s, ast.If):
                    read(s.test)
                    d0 = st['dirty']
                    scan(s.body)
                    d1, st['dirty'] = st['dirty'], d0
                    scan(s.orelse)
                    st['dirty'] = st['dirty'] or d1
                elif isinstance(s, (ast.For, ast.While)):
                    st['dirty'] = st['dirty'] or uses(s, new, ast.Store)        # a later pass of the loop
                    read(s)
                else:
                    read(s)                              # the right-hand side is evaluated first
                    st['dirty'] = st['dirty'] or uses(s, new, ast.Store)
        scan(fl.body[at + 1:])
        for _, e in fl.dx:
            read(e)
        if not st['ok'] or uses(cp, old, ast.Store) or any(uses(v, new) or uses(v, old) for _, v, _ in fl.params):
            return None
        grp = fl.states if kinds[old] == 'state' else fl.inputs
        k = next(i for i, (n, _) in enumerate(grp) if n == old)
        cm = list(getattr(cp, '_cm', ()))
        del fl.body[at]
        if cm and at < len(fl.body):
            fl.body[at]._cm = cm + [c for c in getattr(fl.body[at], '_cm', ()) if c not in cm]
        sub = _Subst({old: new})
        fl.body = [sub.visit(s) for s in fl.body]
        fl.dx = [(i, sub.visit(e)) for i, e in fl.dx]
        # (in the outputs the name stands for the state / input itself, as everywhere in the card)
        fl.outputs = [sub.visit(e) if e is not None else None for e in fl.outputs]
        grp[k] = (new, grp[k][1])
        return fl

    def emit(self, rename=None, meaning=None, comments=True):
        """-> python source of dynamics(t, x, u).  rename: {old: new} (see clean_rename),
        meaning: {name (old or new): text} written as the comment of the line that defines it."""
        rn = self.clean_rename(rename)
        mean = dict(meaning or {})

        def nm(n):
            return rn.get(n, n)

        def note(old, default):
            txt = mean.get(nm(old)) or mean.get(old) or default
            txt = ' '.join(str(txt).split()) if txt else ''
            return '  # ' + txt if txt else ''

        sub = _Subst(rn)

        def src(node):
            return ast.unparse(sub.visit(copy.deepcopy(node)))

        L = ['import numpy as np', '', 'def dynamics(t, x, u):']
        for i, (n, c) in enumerate(self.states):
            L.append('    %s = x[%d]%s' % (nm(n), i, note(n, c)))
        for n, v, c in self.params:
            L.append('    %s = %s%s' % (nm(n), src(v), note(n, c)))
        for i, (n, c) in enumerate(self.inputs):
            L.append('    %s = u[%d]%s' % (nm(n), i, note(n, c)))
        last, noted = [None], set()

        def block(stmts, ind):
            pad = '    ' * ind
            for s in stmts:
                if comments:
                    for c in getattr(s, '_cm', ()):
                        if c != last[0]:
                            L.append('%s# %s' % (pad, c))
                            last[0] = c
                if isinstance(s, ast.If):
                    kw = 'if'
                    while True:
                        L.append('%s%s %s:' % (pad, kw, src(s.test)))
                        block(s.body, ind + 1)
                        if len(s.orelse) == 1 and isinstance(s.orelse[0], ast.If):
                            s, kw = s.orelse[0], 'elif'
                            continue
                        if s.orelse:
                            L.append(pad + 'else:')
                            block(s.orelse, ind + 1)
                        break
                elif isinstance(s, ast.For):
                    L.append('%sfor %s in %s:' % (pad, src(s.target), src(s.iter)))
                    block(s.body, ind + 1)
                elif isinstance(s, ast.While):
                    L.append('%swhile %s:' % (pad, src(s.test)))
                    block(s.body, ind + 1)
                else:
                    line = pad + src(s)
                    if ind == 1 and isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                        old = s.targets[0].id
                        if old not in noted and (mean.get(nm(old)) or mean.get(old)):
                            line += note(old, '')
                            noted.add(old)
                    L.append(line)
        block(self.body, 1)
        L.append('    dx = np.zeros(%d)' % self.nx)
        for i, e in self.dx:
            if comments:
                for c in self.dx_cm.get(i, ()):
                    if c != last[0]:
                        L.append('    # %s' % c)
                        last[0] = c
            L.append('    dx[%d] = %s  # d(%s)/dt' % (i, src(e), nm(self.states[i][0])))
        L.append('    return dx')
        return '\n'.join(L) + '\n'

    def output_text(self, rename=None):
        sub = _Subst(self.clean_rename(rename))
        return [re.sub(r'\b(np|numpy)\.', '', ast.unparse(sub.visit(copy.deepcopy(e)))) if e is not None else None
                for e in self.outputs]


# ======================================================================================
class _Tx(ast.NodeTransformer):
    """expression translation (see module docstring)"""

    def __init__(self, fl):
        self.f = fl

    def visit_Name(self, node):
        if node.id in CONST_NAMES and isinstance(node.ctx, ast.Load):
            return ast.Attribute(ast.Name('np', ast.Load()), CONST_NAMES[node.id], ast.Load())
        if node.id == 'rtMinusInf':
            return ast.UnaryOp(ast.USub(), ast.Attribute(ast.Name('np', ast.Load()), 'inf', ast.Load()))
        if node.id in ('MAJOR_TIME_STEP', 'MINOR_TIME_STEP'):
            return ast.Constant(0 if node.id[1] == 'A' else 1)
        return node

    def _vec(self, sym, field, layout, idx, ctx):
        if field not in layout:
            raise Unsupported('unknown field %s of %s' % (field, sym))
        off, size, shape = layout[field]
        if idx is None:
            if not shape:
                return ast.Subscript(ast.Name(sym, ast.Load()), ast.Constant(off), ctx)
            return ast.Subscript(ast.Name(sym, ast.Load()),
                                 ast.Slice(ast.Constant(off), ast.Constant(off + size)), ctx)
        if len(shape) != 1:
            raise Unsupported('%s field %s is not a vector' % (sym, field))
        k = _fold(idx)
        i = ast.Constant(off + k) if isinstance(k, int) else \
            (idx if off == 0 else ast.BinOp(ast.Constant(off), ast.Add(), idx))
        return ast.Subscript(ast.Name(sym, ast.Load()), i, ctx)

    def _resolve(self, node, idx, ctx):
        """node: attribute chain (without the final subscript). Returns the new node or None."""
        f = self.f
        c = _chain(node)
        if c is None:
            r, parts = _root(node)
            if r in f.dx_alias and parts and len(parts) == 1:
                return self._vec('dx', parts[0], f.lay['x'], idx, ctx)
            return None
        grp = f.group(c[0])
        if grp in ('x', 'u', 'y'):
            if len(c) != 2:
                raise Unsupported('nested field in %s' % c[0])
            return self._vec({'x': 'x', 'u': 'u', 'y': '_y_'}[grp], c[1], f.lay[grp], idx, ctx)
        if grp == 'm':
            if c[1:] == ['Timing', 't'] and idx is not None:
                return ast.Name('t', ast.Load())
            if c[1:] == ['Timing', 'simTimeStep']:
                return ast.Constant(0)                   # MAJOR_TIME_STEP: every block is evaluated
            if not isinstance(ctx, ast.Load):
                raise _Drop()
            o = f.inst
            try:
                for a in c:
                    o = getattr(o, a)
                if idx is not None:
                    o = o[_fold(idx)]
            except Exception:
                raise _Drop()
            if isinstance(o, (int, float, np.number)):
                return _const(o)
            raise _Drop()
        if len(c) < 2:
            if grp == 'plain':
                name = f.local(c, 'const')
                return name if idx is None else ast.Subscript(name, idx, ctx)
            raise _Drop()
        kind = {'signal': 'signal', 'frozen': 'memory'}.get(grp, 'const')
        name = f.local(c, kind)
        if kind == 'const' and idx is not None and isinstance(_fold(idx), int):
            v = f.value(c)                               # index tables (sparse state-space ...) -> the number
            if isinstance(v, np.ndarray) and v.dtype.kind in 'iu' and v.ndim == 1:
                return _const(v[_fold(idx)])
        return name if idx is None else ast.Subscript(name, idx, ctx)

    def visit_Attribute(self, node):
        new = self._resolve(node, None, node.ctx)
        return new if new is not None else self.generic_visit(node)

    def visit_Subscript(self, node):
        idx = self.visit(node.slice)
        new = self._resolve(node.value, idx, node.ctx)
        if new is not None:
            return new
        node.value, node.slice = self.visit(node.value), idx
        return node

    def visit_Call(self, node):
        fn = node.func
        if isinstance(fn, ast.Attribute):
            c = _chain(fn)
            if c is not None and len(c) == 1:
                if c[0].startswith(('rtsi', 'rtm', 'rt_ertODE')):
                    raise _Drop()
                raise Unsupported('call of model function %s()' % c[0])
            raise Unsupported('call %s' % ast.unparse(fn))
        name = fn.id if isinstance(fn, ast.Name) else None
        if name == '_new' or name == '_init_val':
            base, dims = node.args[0].value, ast.literal_eval(node.args[1])
            if not dims or 0 in dims:
                raise Unsupported('%s of %r %r' % (name, base, dims))
            kw = [ast.keyword('dtype', ast.Name('int', ast.Load()))] if base in INT_TYPES else []
            if name == '_new':
                shape = ast.Constant(dims[0]) if len(dims) == 1 else ast.Tuple([ast.Constant(d) for d in dims], ast.Load())
                return ast.Call(ast.Attribute(ast.Name('np', ast.Load()), 'zeros', ast.Load()), [shape], kw)
            arr = np.zeros(dims)
            init = np.array(_fold(node.args[2]) if _fold(node.args[2]) is not None else 0, dtype=float)
            if init.ndim != len(dims):
                raise Unsupported('partial initialiser')
            arr[tuple(slice(0, n) for n in init.shape)] = init
            if base in INT_TYPES:
                arr = arr.astype(np.int64)
            v = _value_node(arr)
            return v
        if name == '_Ref':                               # pointer to frozen memory -> its (constant) value
            c = _chain(node.args[0])
            if c is not None and self.f.group(c[0]) == 'frozen' and isinstance(node.args[1], ast.Constant) \
                    and isinstance(node.args[1].value, str):
                return self.f.local(c + [node.args[1].value], 'memory')
            raise Unsupported('pointer %s' % ast.unparse(node)[:50])
        args = [self.visit(a) for a in node.args]
        if name == '_deref':
            if isinstance(node.args[0], ast.Name) and node.args[0].id in self.f.frozen_ptr:
                return args[0]
            raise Unsupported('pointer %s' % ast.unparse(node)[:50])
        if name == '_div':
            return ast.BinOp(args[0], ast.Div(), args[1])
        if name in CALLS:
            return _np(CALLS[name], *args)
        if name in ('_c_round', 'rt_roundd_snf'):
            return ast.BinOp(_np('sign', args[0]), ast.Mult(),
                             _np('floor', ast.BinOp(_np('abs', copy.deepcopy(args[0])), ast.Add(), ast.Constant(0.5))))
        if name == 'float':
            return args[0]
        if name == '_ci':
            return ast.Call(ast.Name('int', ast.Load()), [args[0]], [])
        if name == '_f32':
            return ast.Call(ast.Name('float', ast.Load()), [_np('float32', args[0])], [])
        if name in ('int', 'abs', 'range', 'min', 'max'):
            node.args = args
            return node
        raise Unsupported('call of %s()' % (name or ast.unparse(fn)))


class _Flattener(object):
    def __init__(self, Model, py_src):
        self.Model, self.inst = Model, Model()
        tree = ast.parse(py_src)
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == Model.__name__), None)
        if cls is None:
            raise Unsupported('class %s not found in the source' % Model.__name__)
        self.methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.comments = self._comments(py_src)
        m = self.inst
        self.gtype = dict(re.findall(r"self\.(\w+) = (?:_new|_init_val)\('([^']+)'", py_src))
        self.alias_of = dict(re.findall(r"^\s+self\.(\w+) = self\.(\w+)\s*$", py_src, re.M))      # self.M = self.M_
        self.lay = {'x': self._layout(getattr(m, m.X_ATTR, None)), 'u': self._layout(getattr(m, m.U_ATTR, None)),
                    'y': self._layout(getattr(m, m.Y_ATTR, None))}
        self.dx_alias, self.frozen_ptr, self.written = set(), set(), set()
        self.used = set(RESERVED)                       # identifiers that are taken
        self.loc = {}                                   # chain tuple -> python name
        self.kind = {}                                  # python name -> signal | const
        self.origin = {}                                # python name -> 'self.P.Gain_Gain'

    @staticmethod
    def _comments(src):
        """line number of a statement -> comment lines directly above it"""
        cm, pend, out = {}, [], {}
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                cm[tok.start[0]] = tok.string.lstrip('#').strip()
        lines = src.splitlines()
        for i, ln in enumerate(lines, 1):
            if i in cm and ln.lstrip().startswith('#'):
                c = re.sub(r'\s*incorporates:$', '', cm[i])
                if c and not NOISE_CM.match(c):
                    pend.append(c)
            elif ln.strip():
                if pend:
                    out[i] = pend
                pend = []
        return out

    @staticmethod
    def _layout(o):
        lay, off = {}, 0
        for k, v in getattr(o, '__dict__', {}).items():
            if isinstance(v, np.ndarray):
                lay[k] = (off, v.size, v.shape)
                off += v.size
            elif isinstance(v, (int, float, np.number)):
                lay[k] = (off, 1, ())
                off += 1
            else:
                raise Unsupported('field %s is not numeric' % k)
        lay['__n__'] = (off, 0, ())
        return lay

    def group(self, attr):
        m = self.inst
        attr = self.alias_of.get(attr, attr) if attr != 'M' else 'M'
        if attr == m.X_ATTR:
            return 'x'
        if attr == m.U_ATTR:
            return 'u'
        if attr == m.Y_ATTR:
            return 'y'
        ty = self.gtype.get(attr, '')
        if attr == 'M' or ty.startswith('tag_RTM_'):
            return 'm'
        if attr == 'Xdot' or ty.startswith(('XDot_', 'XDis_')):
            return 'm'
        if ty.startswith('DW_'):
            return 'frozen'
        if ty.startswith('B_') or attr in self.written:
            return 'signal'
        if not hasattr(getattr(m, attr, None), '__dict__'):
            return 'plain' if isinstance(getattr(m, attr, None), (np.ndarray, int, float, np.number)) else 'm'
        return 'const'

    def unique(self, want):
        want = re.sub(r'\W', '_', want)
        if not re.match(r'[A-Za-z_]', want):
            want = 'v_' + want
        name, k = want, 2
        while name in self.used:
            name, k = '%s_%d' % (want, k), k + 1
        self.used.add(name)
        return name

    def local(self, chain, kind):
        key = tuple(chain)
        if key not in self.loc:
            name = self.unique('_'.join(chain[1:]) if len(chain) > 1 else chain[0])
            self.loc[key], self.kind[name], self.origin[name] = name, kind, 'self.' + '.'.join(chain)
        return ast.Name(self.loc[key], ast.Load())

    def value(self, chain):
        o = self.inst
        for a in chain:
            o = getattr(o, a)
        return o

    # ---------------------------------------------------------------- statements
    def _is_droppable_method(self, name):
        """a model function that only touches the solver bookkeeping (e.g. rate_scheduler)"""
        fn = self.methods.get(name)
        if fn is None:
            return False
        for s in _stmts(fn.body):
            tg = s.targets if isinstance(s, ast.Assign) else ([s.target] if isinstance(s, ast.AugAssign) else [])
            for tt in tg:
                while isinstance(tt, ast.Subscript):
                    tt = tt.value
                c = _chain(tt)
                if c is not None and self.group(c[0]) != 'm':
                    return False
            if isinstance(s, (ast.Return,)) and s.value is not None:
                return False
        return True

    @staticmethod
    def _taint(names):
        """a memory update that cannot be translated: the memory becomes NaN. If nothing reads it
        afterwards the line is dead code and disappears; if something does, the validation fails."""
        return [ast.Assign([ast.Name(n, ast.Store())], ast.Attribute(ast.Name('np', ast.Load()), 'nan', ast.Load()))
                for n in sorted(set(names))]

    def _target_group(self, tgt):
        while isinstance(tgt, ast.Subscript):
            tgt = tgt.value
        c = _chain(tgt)
        if c is not None:
            return self.group(c[0])
        r, parts = _root(tgt)
        return 'dx' if r in self.dx_alias and parts else None

    def block(self, stmts):
        out, tx = [], _Tx(self)
        for s in stmts:
            cm = self.comments.get(getattr(s, 'lineno', -1), [])
            new = self.stmt(s, tx)
            if new and cm:
                new[0]._cm = cm + list(getattr(new[0], '_cm', []))
            out += new
        return out

    def stmt(self, s, tx):
        if isinstance(s, ast.Pass):
            return []
        if isinstance(s, (ast.Assign, ast.AugAssign)):
            tgts = s.targets if isinstance(s, ast.Assign) else [s.target]
            if any(isinstance(tt, ast.Name) and tt.id in self.dx_alias for tt in tgts):
                return []
            if isinstance(s.value, ast.Constant) and s.value.value is None:
                return []                                # pointer declaration
            if any(self._target_group(tt) == 'm' for tt in tgts):
                return []
            mem = [tt for tt in tgts if self._target_group(tt) == 'frozen']
            try:
                s.value = tx.visit(s.value)
                if isinstance(s, ast.Assign):
                    s.targets = [_as_store(tx.visit(tt)) for tt in tgts]
                else:
                    s.target = _as_store(tx.visit(s.target))
            except (_Drop, Unsupported):
                if not mem:
                    raise
                return self._taint([_store_base(tx.visit(tt)).id for tt in mem])     # an update we cannot translate
            except Exception:
                raise
            return [s]
        if isinstance(s, ast.Expr):
            v = s.value
            if isinstance(v, ast.Constant):
                return []
            if isinstance(v, ast.Call):
                c = _chain(v.func) if isinstance(v.func, ast.Attribute) else None
                if c is not None and len(c) == 1:
                    if c[0].startswith(('rtsi', 'rtm', 'rt_ertODE')) or self._is_droppable_method(c[0]):
                        return []
                    raise Unsupported('call of model function %s()' % c[0])
                name = v.func.id if isinstance(v.func, ast.Name) else None
                if name == '_store' and isinstance(v.args[0], ast.Name) and v.args[0].id in self.frozen_ptr:
                    return self._taint([self.local(c, 'memory').id for c in self.ptr_fields[v.args[0].id]])
                if name in ('_memcpy', '_memset'):
                    for k in (0, 1):                     # &a[0]  ->  a
                        a = v.args[k]
                        if isinstance(a, ast.Call) and getattr(a.func, 'id', None) == '_Ref' \
                                and isinstance(a.args[1], ast.Constant) and a.args[1].value == 0:
                            v.args[k] = a.args[0]
                    if self._target_group(v.args[0]) in ('m', 'frozen'):
                        return []
                    try:
                        dst, src = tx.visit(v.args[0]), tx.visit(v.args[1])
                    except _Drop:
                        return []
                    if name == '_memset' and _fold(src) != 0:
                        raise Unsupported('memset with a non-zero value')
                    if isinstance(dst, ast.Subscript) and isinstance(dst.slice, ast.Slice):
                        dst.ctx = ast.Store()
                    elif isinstance(dst, ast.Name):
                        dst = ast.Subscript(dst, ast.Slice(None, None), ast.Store())
                    else:
                        raise Unsupported('memcpy target %s' % ast.unparse(dst))
                    return [ast.Assign([dst], src)]
            raise Unsupported('statement %s' % ast.unparse(s)[:60])
        if isinstance(s, ast.If):
            try:
                s.test = tx.visit(s.test)
            except _Drop:
                raise Unsupported('condition on solver data: %s' % ast.unparse(s.test)[:60])
            body, orelse = self.block(s.body), self.block(s.orelse)
            v = _fold(s.test)
            if v is not None:
                return body if v else orelse
            if not body and not orelse:
                return []
            if not body:
                s.test, body, orelse = ast.UnaryOp(ast.Not(), s.test), orelse, []
            s.body, s.orelse = body, orelse
            return [s]
        if isinstance(s, (ast.For, ast.While)):
            if isinstance(s, ast.While):
                try:
                    s.test = tx.visit(s.test)
                except _Drop:
                    raise Unsupported('loop condition on solver data')
            else:
                s.iter = tx.visit(s.iter)
            s.body = self.block(s.body)
            if s.orelse:
                raise Unsupported('loop with else')
            return [s] if s.body else []
        if isinstance(s, (ast.Break, ast.Continue)):
            return [s]
        raise Unsupported('statement %s' % type(s).__name__)

    # ---------------------------------------------------------------- passes
    @staticmethod
    def unroll(body):
        """loops with a constant range whose body indexes x / u / dx / outputs with the loop variable"""
        out = []
        for s in body:
            for attr in ('body', 'orelse'):
                if isinstance(getattr(s, attr, None), list):
                    setattr(s, attr, _Flattener.unroll(getattr(s, attr)))
            if isinstance(s, ast.For) and isinstance(s.target, ast.Name) and isinstance(s.iter, ast.Call) \
                    and getattr(s.iter.func, 'id', None) == 'range':
                rng = [_fold(a) for a in s.iter.args]
                var = s.target.id
                touches = any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                              and n.value.id in ('x', 'u', 'dx', '_y_') and var in _names(n.slice)
                              for b in s.body for n in ast.walk(b))
                simple = not any(isinstance(n, (ast.Break, ast.Continue)) for b in s.body for n in ast.walk(b)) \
                    and var not in {n.id for b in s.body for n in ast.walk(b)
                                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                if touches and simple and all(isinstance(r, int) for r in rng) and len(range(*rng)) <= 64:
                    for k_i, k in enumerate(range(*rng)):
                        for b in s.body:
                            c = _FoldIdx().visit(_Subst({var: ast.Constant(k)}).visit(copy.deepcopy(b)))
                            if k_i:
                                for n in ast.walk(c):
                                    if hasattr(n, '_cm'):
                                        del n._cm
                            elif hasattr(s, '_cm') and b is s.body[0]:
                                c._cm = s._cm + list(getattr(c, '_cm', []))
                            out.append(c)
                    continue
            out.append(s)
        return out

    def scalarize(self, body):
        """an array that is only ever used element by element with constant indices (after the
        unrolling) becomes one variable per element: Integrator[1] -> Integrator_1"""
        use, bad, alloc = {}, set(), {}
        for s in _stmts(body):
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                    and isinstance(s.value, ast.Call) and getattr(s.value.func, 'attr', '') == 'zeros' \
                    and isinstance(s.value.args[0], ast.Constant) and not s.value.keywords:
                v = s.targets[0].id
                if v in alloc:
                    bad.add(v)
                alloc[v] = id(s)
                continue
            sub_bases = set()
            for n in ast.walk(s):
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                    sub_bases.add(id(n.value))
                    if isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, int):
                        use.setdefault(n.value.id, set()).add(n.slice.value)
                    else:
                        bad.add(n.value.id)
            if isinstance(s, (ast.If, ast.For, ast.While)):
                continue                                 # (its statements are visited on their own)
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and id(n) not in sub_bases:
                    bad.add(n.id)                        # used as a whole array
        for s in _stmts(body):                           # conditions / loop headers
            hdr = [s.test] if isinstance(s, (ast.If, ast.While)) else ([s.iter, s.target] if isinstance(s, ast.For) else [])
            for h in hdr:
                bases = {id(n.value) for n in ast.walk(h) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
                bad |= {n.id for n in ast.walk(h) if isinstance(n, ast.Name) and id(n) not in bases}
        todo = {v for v in alloc if v in use and v not in bad}
        todo |= {v for v in use if v not in bad and v not in alloc and self.kind.get(v) == 'signal'}
        if not todo:
            return body
        names = {(v, k): self.unique('%s_%d' % (v, k)) for v in todo for k in use[v]}
        for (v, k), n in names.items():
            self.kind[n] = 'element'

        class V(ast.NodeTransformer):
            def visit_Subscript(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id in todo:
                    return ast.Name(names[node.value.id, node.slice.value], node.ctx)
                return node
        drop = {alloc[v] for v in todo if v in alloc}

        def go(stmts):
            out = []
            for s in stmts:
                if id(s) in drop:
                    continue
                for attr in ('body', 'orelse'):
                    if isinstance(getattr(s, attr, None), list):
                        setattr(s, attr, go(getattr(s, attr)))
                out.append(V().visit(s))
            return out
        self.elem_init = {n: (v, k) for (v, k), n in names.items()}
        return go(body)

    @staticmethod
    def copy_prop(body):
        """v = x[0]; a = v; ...   ->   uses of v / a are replaced by x[0] (top level, straight-line)"""
        copies = {}

        def stored(s):
            st = _names(s, ast.Store)
            for q in _stmts([s]):
                if isinstance(q, ast.Assign):
                    st |= {b.id for b in (_store_base(tt) for tt in q.targets if isinstance(tt, ast.Subscript)) if b}
            return st

        def kill(names):
            for v in list(copies):
                if v in names or _names(copies[v]) & names:
                    del copies[v]

        def copyable(e):
            if isinstance(e, ast.Name):
                return e.id not in ('np',)
            return isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name) and e.value.id in ('x', 'u') \
                and isinstance(e.slice, ast.Constant)
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1:
                s.value = _Subst(dict(copies)).visit(s.value)
                tgt = s.targets[0]
                if isinstance(tgt, ast.Name):
                    kill({tgt.id})
                    if copyable(s.value) and tgt.id not in _names(s.value):
                        copies[tgt.id] = s.value
                else:
                    if isinstance(tgt, ast.Subscript):
                        tgt.slice = _Subst(dict(copies)).visit(tgt.slice)
                    kill(stored(s))
            else:
                st = stored(s)
                ok = {v: e for v, e in copies.items() if v not in st and not (_names(e) & st)}
                if ok:
                    sub = _Subst(ok)
                    for attr in s._fields:
                        val = getattr(s, attr)
                        if isinstance(val, list):
                            setattr(s, attr, [sub.visit(e) if isinstance(e, ast.AST) else e for e in val])
                        elif isinstance(val, ast.AST):
                            setattr(s, attr, sub.visit(val))
                kill(st)
        return body

    @staticmethod
    def drop_dead_inits(body):
        """a top-level assignment whose value is overwritten before anything reads it
        (`v = 0.0` C declarations, copies left over after the copy propagation)"""
        out, pend = [], []                               # pend: source lines of dropped statements
        for i, s in enumerate(body):
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and len(s.targets) == 1:
                v, dead = s.targets[0].id, False
                for nxt in body[i + 1:]:
                    if isinstance(nxt, ast.Assign) and len(nxt.targets) == 1 and isinstance(nxt.targets[0], ast.Name) \
                            and nxt.targets[0].id == v and v not in _names(nxt.value):
                        dead = True
                        break
                    if v in _names(nxt):
                        break
                if dead:
                    pend += [c for c in _src_cm(s) if c not in pend]
                    continue
            _give_cm(s, pend)
            out.append(s)
        return out

    @staticmethod
    def _reaching(body):
        """reaching definitions of the scalar variables over the (structured) statements.
        -> (uses: id(Name load node) -> set of definition ids,  defs: definition id -> (Name node, statement))
        A definition is `v = ...`, `v += ...` or a for-loop variable; ('undef', v) = no definition yet."""
        uses, defs, loops = {}, {}, []

        def use(node, st):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    uses.setdefault(id(n), set()).update(st.get(n.id) or {('undef', n.id)})

        def define(name, stmt, st):
            defs[id(name)] = (name, stmt)
            st[name.id] = frozenset([id(name)])

        def merge(*states):
            states = [x for x in states if x is not None]
            if not states:
                return None
            out = {}
            for v in set().union(*states):
                out[v] = frozenset().union(*[x.get(v) or frozenset([('undef', v)]) for x in states])
            return out

        def flow(stmts, st):
            for s in stmts:
                if st is None:
                    return None                          # after break / continue
                if isinstance(s, ast.Assign):
                    use(s.value, st)
                    for tt in s.targets:
                        if isinstance(tt, ast.Name):
                            define(tt, s, st)
                        else:
                            for n in ast.walk(tt):       # a[i] = ...: a and i are read
                                if isinstance(n, ast.Name):
                                    uses.setdefault(id(n), set()).update(st.get(n.id) or {('undef', n.id)})
                elif isinstance(s, ast.AugAssign):
                    use(s.value, st)
                    for n in ast.walk(s.target):
                        if isinstance(n, ast.Name):
                            uses.setdefault(id(n), set()).update(st.get(n.id) or {('undef', n.id)})
                    if isinstance(s.target, ast.Name):
                        define(s.target, s, st)
                elif isinstance(s, ast.If):
                    use(s.test, st)
                    st = merge(flow(s.body, dict(st)), flow(s.orelse, dict(st)))
                elif isinstance(s, (ast.For, ast.While)):
                    if isinstance(s, ast.For):
                        use(s.iter, st)
                    head = dict(st)
                    while True:
                        ctx = {'brk': [], 'cont': []}
                        loops.append(ctx)
                        inner = dict(head)
                        if isinstance(s, ast.For):
                            for n in ast.walk(s.target):
                                if isinstance(n, ast.Name):
                                    define(n, s, inner)
                        else:
                            use(s.test, inner)
                        out = flow(s.body, inner)
                        loops.pop()
                        new = merge(st, out, *ctx['cont'])
                        if new == head:
                            break
                        head = new
                    st = merge(head, *ctx['brk'])
                elif isinstance(s, ast.Break):
                    loops[-1]['brk'].append(st)
                    st = None
                elif isinstance(s, ast.Continue):
                    loops[-1]['cont'].append(st)
                    st = None
                elif isinstance(s, ast.Expr):
                    use(s.value, st)
            return st
        flow(body, {})
        return uses, defs

    def split_webs(self, body):
        """The C compiler re-uses one scratch variable for unrelated things (CL_wb is first the lift
        coefficient and later cos(theta)). Every group of definitions and uses that belong together
        gets its own variable, so that each can be named for what it is; definitions whose value
        is never read are removed."""
        arrays = {n.value.id for s in body for n in ast.walk(s)
                  if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)}
        skip = arrays | RESERVED
        while True:                                      # dead definitions, to a fixpoint
            uses, defs = self._reaching(body)
            live = set().union(*uses.values()) if uses else set()
            dead = {id(st) for d, (name, st) in defs.items()
                    if d not in live and name.id not in skip and isinstance(st, ast.Assign)
                    and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                    and not (isinstance(st.value, ast.Subscript) and getattr(st.value.value, 'id', None) in ('x', 'u'))}
                    # (`name = x[k]` stays: it is what gives state k its name)
            if not dead:
                break

            def clean(stmts):
                out = []
                for st in stmts:
                    if id(st) in dead:
                        continue
                    for attr in ('body', 'orelse'):
                        if isinstance(getattr(st, attr, None), list):
                            setattr(st, attr, clean(getattr(st, attr)))
                    if isinstance(st, ast.If) and not st.body and not st.orelse:
                        continue
                    if isinstance(st, ast.If) and not st.body:
                        st.test, st.body, st.orelse = ast.UnaryOp(ast.Not(), st.test), st.orelse, []
                    if isinstance(st, (ast.For, ast.While)) and not st.body:
                        continue
                    out.append(st)
                return out
            body = clean(body)
        # union-find over the definitions that reach a common use
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for d in defs:
            find(d)
        for ds in uses.values():
            ds = list(ds)
            for d in ds[1:]:
                parent[find(d)] = find(ds[0])
        loop_vars = {name.id for name, st in defs.values() if isinstance(st, (ast.For, ast.While))}
        order, webs = {}, {}                             # variable -> its webs in order of first definition
        pos = {id(n): k for k, n in enumerate(n for st in body for n in ast.walk(st) if isinstance(n, ast.Name))}
        for d, (name, _) in sorted(defs.items(), key=lambda kv: pos.get(kv[0], 0)):
            if name.id in skip or name.id in loop_vars:
                continue
            r = find(d)
            lst = order.setdefault(name.id, [])
            if r not in lst:
                lst.append(r)
        newname = {}
        for v, roots in order.items():
            if len(roots) < 2:
                continue
            undef = find(('undef', v)) if ('undef', v) in parent else None
            first = undef if undef in roots else roots[0]    # the web that reads the initial value keeps the name
            for r in roots:
                if r != first:
                    newname[r] = self.unique(v)
                    if v in self.kind:
                        self.kind[newname[r]] = 'local'
        if not newname:
            return body
        for d, (name, _) in defs.items():
            if find(d) in newname:
                name.id = newname[find(d)]
        by_id = {id(n): n for st in body for n in ast.walk(st) if isinstance(n, ast.Name)}
        for u, ds in uses.items():
            r = find(next(iter(ds)))
            if r in newname and u in by_id:
                by_id[u].id = newname[r]
        return body

    def split_versions(self, body):
        """a scratch variable that is assigned several times at the top level (alpha = sin(..); ...;
        alpha = sin(..)) becomes one variable per assignment, so each can get its own name"""
        top, inner = {}, set()
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                top[s.targets[0].id] = top.get(s.targets[0].id, 0) + 1
            else:
                inner |= _names(s, ast.Store)
                if isinstance(s, ast.Assign):
                    inner |= {b.id for b in [_store_base(tt) for tt in s.targets] if b is not None}
        for s in _stmts(body):
            if isinstance(s, ast.AugAssign):
                inner |= _names(s.target)
            for n in ast.walk(s):
                if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
                    inner.add(n.value.id)                # arrays keep their name
        multi = {v for v, c in top.items() if c > 1 and v not in inner}
        cur, seen = {}, set()
        for s in body:
            if cur:
                is_def = isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                tgt = s.targets[0] if is_def else None
                sub = _Subst(dict(cur))
                if is_def:
                    s.value = sub.visit(s.value)
                else:
                    for attr in s._fields:
                        val = getattr(s, attr)
                        if isinstance(val, list):
                            setattr(s, attr, [sub.visit(e) if isinstance(e, ast.AST) else e for e in val])
                        elif isinstance(val, ast.AST):
                            setattr(s, attr, sub.visit(val))
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                v = s.targets[0].id
                if v in multi:
                    if v in seen:
                        cur[v] = self.unique(v)
                        s.targets[0].id = cur[v]
                    seen.add(v)
        return body

    def name_vector(self, body, sym, lay, default_names):
        """x[k] / u[k] with a constant k -> a named variable. A line `name = x[k]` that is already in
        the code (the Integrator block output) gives the name. Returns [(name, default comment)]."""
        n = lay['__n__'][0]
        alias, drop, rep = {}, set(), {}
        counts = {}
        for s in _stmts(body):
            for v in _names(s, ast.Store):
                counts[v] = counts.get(v, 0) + 1
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                    and isinstance(s.value, ast.Subscript) and isinstance(s.value.value, ast.Name) \
                    and s.value.value.id == sym and isinstance(s.value.slice, ast.Constant) \
                    and counts.get(s.targets[0].id) == 1:
                k, v = s.value.slice.value, s.targets[0].id
                if k in alias:
                    rep[v] = alias[k][0]                 # second alias of the same entry
                else:
                    alias[k] = (v, (getattr(s, '_cm', None) or [''])[-1])
                drop.add(id(s))
        names = []
        for k in range(n):
            if k in alias:
                names.append(alias[k])
            else:
                names.append((self.unique(default_names[k]), ''))

        class V(ast.NodeTransformer):
            def visit_Subscript(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == sym and isinstance(node.ctx, ast.Load) \
                        and isinstance(node.slice, ast.Constant) and 0 <= node.slice.value < n:
                    return ast.Name(names[node.slice.value][0], ast.Load())
                return node
        kept, pend = [], []                               # the source lines of the dropped aliases stay
        for s in body:
            if id(s) in drop:
                pend += [c for c in _src_cm(s) if c not in pend]
            else:
                _give_cm(s, pend)
                kept.append(s)
        self.tail_cm += pend
        body = [V().visit(s) for s in kept]
        if rep:
            body = [_Subst(rep).visit(s) for s in body]
        return body, names

    @staticmethod
    def fold_flags(body):
        """tmp = (0 == 0); ... if tmp:   ->  the branch itself"""
        counts = {}
        for s in _stmts(body):
            for v in _names(s, ast.Store):
                counts[v] = counts.get(v, 0) + 1
        flags = {}
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                    and counts.get(s.targets[0].id) == 1 and isinstance(_fold(s.value), bool):
                flags[s.targets[0].id] = ast.Constant(_fold(s.value))
        if not flags:
            return body
        sub = _Subst(flags)

        def simp(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id in flags:
                    continue
                if isinstance(s, ast.If):
                    s.test = sub.visit(s.test)
                    s.body, s.orelse = simp(s.body), simp(s.orelse)
                    v = _fold(s.test)
                    if v is not None:
                        new = s.body if v else s.orelse
                        if new and hasattr(s, '_cm'):
                            new[0]._cm = s._cm + list(getattr(new[0], '_cm', []))
                        out += new
                        continue
                    if not s.body and not s.orelse:
                        continue
                    if not s.body:
                        s.test, s.body, s.orelse = ast.UnaryOp(ast.Not(), s.test), s.orelse, []
                    out.append(s)
                    continue
                if isinstance(s, (ast.For, ast.While)):
                    s.body = simp(s.body)
                    if not s.body:
                        continue
                    out.append(sub.visit(s))
                    continue
                out.append(sub.visit(s))
            return out
        return simp(body)

    @staticmethod
    def dce(body, keep, tail=None):
        """remove assignments to variables that are never read (to a fixpoint). The MATLAB source
        lines in the comments of a removed statement move to the next statement that stays
        (tail: list that receives the ones after the last statement)."""
        pend, acc = [], []
        while True:
            live = _loads(body) | keep
            changed = [False]

            def drop(s):
                changed[0] = True
                pend.extend(c for c in _src_cm(s) if c not in pend)

            def clean(stmts):
                out = []
                for s in stmts:
                    if isinstance(s, (ast.Assign, ast.AugAssign)):
                        tg = s.targets if isinstance(s, ast.Assign) else [s.target]
                        bases = [_store_base(tt) for tt in tg]
                        if all(b is not None and b.id not in live for b in bases):
                            drop(s)
                            continue
                        _give_cm(s, pend)
                    elif isinstance(s, ast.If):
                        mine = list(getattr(s, '_cm', ()))
                        s._cm = []
                        _give_cm(s, pend)
                        head = s._cm                     # what was carried up to this statement
                        s.body, s.orelse = clean(s.body), clean(s.orelse)
                        if not s.body and not s.orelse:
                            changed[0] = True
                            pend[:0] = head + [c for c in mine if _SRC_CM.match(c) and c not in head]
                            continue
                        s._cm = head + [c for c in mine if c not in head]
                        if not s.body:
                            s.test, s.body, s.orelse = ast.UnaryOp(ast.Not(), s.test), s.orelse, []
                    elif isinstance(s, (ast.For, ast.While)):
                        mine = list(getattr(s, '_cm', ()))
                        s._cm = []
                        _give_cm(s, pend)
                        head = s._cm
                        s.body = clean(s.body)
                        if not s.body:
                            changed[0] = True
                            pend[:0] = head + [c for c in mine if _SRC_CM.match(c) and c not in head]
                            continue
                        s._cm = head + [c for c in mine if c not in head]
                    else:
                        _give_cm(s, pend)
                    out.append(s)
                return out
            body = clean(body)
            acc[:0] = [c for c in pend if c not in acc]   # after the last statement of this pass
            del pend[:]
            if not changed[0]:
                if tail is not None:
                    tail.extend(acc)
                return body

    # ---------------------------------------------------------------- driver
    def run(self, webs=True):
        self.webs = webs
        self.tail_cm = []                                 # source lines that end up after the last statement
        m = self.inst
        step = self.methods.get(m.STEP_FN)
        der = self.methods.get('derivatives')
        if step is None or der is None:
            raise Unsupported('%s() / derivatives() not found' % m.STEP_FN)
        src = copy.deepcopy(step.body) + copy.deepcopy(der.body)
        # a C local variable called x / u / t / dx would shadow the arguments of dynamics()
        clash = set()
        for st in src:
            clash |= _names(st) & {'x', 'u', 't', 'dx', 'np', '_y_'}
        if clash:
            for st in src:
                self.used |= _names(st)
            ren = _Subst({v: self.unique(v + '_loc') for v in sorted(clash)})
            src = [ren.visit(st) for st in src]
        # names that are already used as local variables of the two functions
        for s in src:
            self.used |= _names(s)
        self.written = set()
        for s in _stmts(src):
            tg = s.targets if isinstance(s, ast.Assign) else ([s.target] if isinstance(s, ast.AugAssign) else [])
            for tt in tg:
                if isinstance(tt, ast.Name) and isinstance(getattr(s, 'value', None), ast.Attribute) \
                        and _chain(s.value) == ['M', 'derivs']:
                    self.dx_alias.add(tt.id)
                while isinstance(tt, ast.Subscript):
                    tt = tt.value
                c = _chain(tt)
                if c:
                    self.written.add(c[0])
        # local pointers that only ever point into frozen memory (Derivative block: lastU = &DW.LastUAtTimeA)
        ptr, bad, self.ptr_fields = set(), set(), {}
        for s in _stmts(src):
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                v, val = s.targets[0].id, s.value
                if isinstance(val, ast.Call) and getattr(val.func, 'id', None) == '_Ref':
                    c = _chain(val.args[0])
                    (ptr if c is not None and self.group(c[0]) == 'frozen' else bad).add(v)
                    if c is not None and isinstance(val.args[1], ast.Constant):
                        self.ptr_fields.setdefault(v, []).append(c + [val.args[1].value])
                elif not (isinstance(val, ast.Constant) and val.value is None):
                    bad.add(v)
        self.frozen_ptr = ptr - bad
        body = self.block(src)
        # memory (self.DW ...): ordinary variables that start at the frozen (initial) value
        def no_pass(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ast.Pass):
                    continue
                for attr in ('body', 'orelse'):
                    if isinstance(getattr(s, attr, None), list):
                        setattr(s, attr, no_pass(getattr(s, attr)))
                if isinstance(s, ast.If) and not s.body and not s.orelse:
                    continue
                if isinstance(s, ast.If) and not s.body:
                    s.test, s.body, s.orelse = ast.UnaryOp(ast.Not(), s.test), s.orelse, []
                if isinstance(s, (ast.For, ast.While)) and not s.body:
                    continue
                out.append(s)
            return out
        body = no_pass(body)
        init = []
        for key, name in self.loc.items():
            if self.kind[name] == 'memory':
                a = ast.Assign([ast.Name(name, ast.Store())], _value_node(self.value(key)))
                a._cm = ['%s (memory, frozen at its initial value)' % self.origin[name]]
                init.append(a)
        body = self.unroll(init + body)

        class Tab(ast.NodeTransformer):                   # index tables with a now constant index
            def visit_Subscript(tself, node):
                tself.generic_visit(node)
                if isinstance(node.value, ast.Name) and self.kind.get(node.value.id) == 'const' \
                        and isinstance(node.ctx, ast.Load) and isinstance(_fold(node.slice), int):
                    v = self.value(next(k for k, n in self.loc.items() if n == node.value.id))
                    if isinstance(v, np.ndarray) and v.dtype.kind in 'iu' and v.ndim == 1:
                        return _const(v[_fold(node.slice)])
                return node
        body = [_FoldIdx().visit(Tab().visit(s)) for s in body]

        fl = Flat()
        fl.nx = self.lay['x']['__n__'][0]
        ny = self.lay['y']['__n__'][0]
        # ---- outputs: _y_[k] = expr at the top level
        fl.outputs = [None] * ny
        for s in _stmts(body):
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and n.id == '_y_' and isinstance(n.ctx, ast.Load):
                    par = [p for p in ast.walk(s) if isinstance(p, ast.Subscript) and p.value is n]
                    if not par or not isinstance(par[0].ctx, ast.Store):
                        raise Unsupported('a root output is read back')
        def no_y(stmts):
            out = []
            for s in stmts:
                if isinstance(s, ast.Assign) and any(getattr(_store_base(tt), 'id', None) == '_y_' for tt in s.targets):
                    continue
                for attr in ('body', 'orelse'):
                    if isinstance(getattr(s, attr, None), list):
                        setattr(s, attr, no_y(getattr(s, attr)))
                out.append(s)
            return out
        body = self.scalarize(body)
        body = self.drop_dead_inits(body)
        body = self.copy_prop(body)
        body = self.drop_dead_inits(body)
        body = self.split_webs(body) if self.webs else self.split_versions(body)

        # ---- states / inputs get names
        def clean(n):
            n = re.sub(r'_CSTATE', '', n)
            n = re.sub(r'\[(\d+)\]', r'_\1', n)
            return n
        snames = [clean(n) for n in m.state_names]
        unames = [clean(n) for n in m.input_names]
        body, fl.states = self.name_vector(body, 'x', self.lay['x'], ['x_' + n if n in self.used else n for n in snames])
        body, fl.inputs = self.name_vector(body, 'u', self.lay['u'], ['u_' + n if n in self.used else n for n in unames])
        fl.states = [(n, c or 'state %s' % m.state_names[i]) for i, (n, c) in enumerate(fl.states)]
        fl.inputs = [(n, c or 'input %s' % m.input_names[i]) for i, (n, c) in enumerate(fl.inputs)]
        # the output expressions: _y_[k] = expr at the top level (read after the renaming passes)
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Subscript) \
                    and getattr(s.targets[0].value, 'id', None) == '_y_' and isinstance(s.targets[0].slice, ast.Constant):
                k = s.targets[0].slice.value
                if 0 <= k < ny:
                    fl.outputs[k] = s.value
        body = no_y(body)
        body = self.fold_flags(body)

        # ---- the trailing dx[k] = ... lines
        dx_expr, rest = {}, []
        tail = len(body)
        while tail > 0:
            s = body[tail - 1]
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Subscript) \
                    and getattr(s.targets[0].value, 'id', None) == 'dx' and isinstance(s.targets[0].slice, ast.Constant):
                tail -= 1
            else:
                break
        for s in body[tail:]:
            k, e = s.targets[0].slice.value, s.value
            if k in dx_expr:                              # dx[k] = dx[k] + ...  -> one expression

                class Sub(ast.NodeTransformer):
                    def visit_Subscript(self, node):
                        self.generic_visit(node)
                        if getattr(node.value, 'id', None) == 'dx' and isinstance(node.slice, ast.Constant) \
                                and node.slice.value == k:
                            return copy.deepcopy(dx_expr[k])
                        return node
                e = Sub().visit(e)
            if any(isinstance(n, ast.Name) and n.id == 'dx' for n in ast.walk(e)):
                raise Unsupported('derivative line reads another derivative')
            dx_expr[k] = e
        body = body[:tail]
        if any(isinstance(n, ast.Name) and n.id == 'dx' for s in body for n in ast.walk(s)):
            raise Unsupported('derivatives are not assigned in one block at the end')
        # dx[k] = Sum1 with Sum1 used nowhere else -> the expression of Sum1 itself (if it is the last
        # definition before the derivatives, so nothing it reads can have changed)
        uses = {}
        for s in body:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    uses[n.id] = uses.get(n.id, 0) + 1
        for e in dx_expr.values():
            for n in ast.walk(e):
                if isinstance(n, ast.Name):
                    uses[n.id] = uses.get(n.id, 0) + 1
        stores = {}
        for s in _stmts(body):
            for v in _names(s, ast.Store):
                stores[v] = stores.get(v, 0) + 1
        for k in sorted(dx_expr):
            e = dx_expr[k]
            if isinstance(e, ast.Name) and uses.get(e.id) == 1 and stores.get(e.id) == 1:
                d = next((s for s in body if isinstance(s, ast.Assign) and len(s.targets) == 1
                          and isinstance(s.targets[0], ast.Name) and s.targets[0].id == e.id), None)
                if d is None:
                    continue
                after = body[body.index(d) + 1:]
                written_after = set()
                for s in after:
                    written_after |= _names(s, ast.Store)
                    for q in _stmts([s]):
                        if isinstance(q, ast.Assign):
                            written_after |= {b.id for b in (_store_base(tt) for tt in q.targets) if b is not None}
                if not (_names(d.value) & written_after) and len(ast.unparse(d.value)) <= 160:
                    dx_expr[k] = d.value
                    fl.dx_cm[k] = list(getattr(d, '_cm', []))
                    body.remove(d)
        fl.dx = [(k, dx_expr.get(k, ast.Constant(0.0))) for k in range(fl.nx)]

        keep = set()
        for _, e in fl.dx:
            keep |= _names(e)
        tail_cm = []
        body = self.dce(body, keep, tail_cm)              # (the outputs do not keep code alive)
        tail_cm = self.tail_cm + [c for c in tail_cm if c not in self.tail_cm]
        if tail_cm and fl.nx:                             # source lines after the last statement: above the dx lines
            fl.dx_cm[0] = tail_cm + [c for c in fl.dx_cm.get(0, []) if c not in tail_cm]

        # ---- constants: self.P / frozen memory / constant local tables -> parameter lines
        used_names = set()
        for s in body:
            used_names |= _names(s)
        used_names |= keep
        params, defined = [], set()
        for key, name in self.loc.items():
            if self.kind[name] == 'const' and name in used_names:
                params.append((name, _value_node(self.value(key)), self.origin[name]))
                defined.add(name)
        counts = {}
        for s in _stmts(body):
            for v in _names(s, ast.Store):
                counts[v] = counts.get(v, 0) + 1
            if isinstance(s, ast.Assign):
                for tt in s.targets:
                    b = _store_base(tt) if isinstance(tt, ast.Subscript) else None
                    if b is not None:
                        counts[b.id] = counts.get(b.id, 0) + 2
        rest = []
        for s in body:
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                    and counts.get(s.targets[0].id) == 1 and _names(s.value) <= {'np', 'int'} | defined \
                    and not (isinstance(s.value, ast.Call) and getattr(s.value.func, 'attr', '') == 'zeros'):
                params.append((s.targets[0].id, s.value, (getattr(s, '_cm', None) or ['constant'])[-1]))
                defined.add(s.targets[0].id)
            else:
                rest.append(s)
        body = rest
        fl.params = params

        # ---- signals that may be read before they are assigned get their initial value
        pre = []
        known = defined | {n for n, _ in fl.states} | {n for n, _ in fl.inputs} | RESERVED
        need = self._maybe_unassigned(body, set(known))
        for name in sorted(need):
            if name in self.kind and self.kind[name] == 'signal':
                key = next(k for k, v in self.loc.items() if v == name)
                val = self.value(key)
                node = _np('zeros', ast.Constant(val.shape[0]) if val.ndim == 1 else
                           ast.Tuple([ast.Constant(d) for d in val.shape], ast.Load())) \
                    if isinstance(val, np.ndarray) and not np.any(val) else _value_node(val)
                pre.append(ast.Assign([ast.Name(name, ast.Store())], node))
            else:
                raise Unsupported('variable %s may be used before it is assigned' % name)
        fl.body = pre + body
        final = set(fl.names()) | {'np'}
        fl.outputs = [e if e is not None and _names(e) <= final else None for e in fl.outputs]
        for s in fl.body:
            ast.fix_missing_locations(s)
        return fl

    def _maybe_unassigned(self, body, defined):
        """names that are (or may be) read before an assignment that certainly happened"""
        need = set()

        def walk(stmts, defs):
            for s in stmts:
                if isinstance(s, ast.Assign):
                    need.update(_names(s.value, ast.Load) - defs)
                    for tt in s.targets:
                        if isinstance(tt, ast.Name):
                            defs.add(tt.id)
                        else:
                            need.update(_names(tt, ast.Load) - defs)
                elif isinstance(s, ast.AugAssign):
                    need.update(_names(s, None) - defs)
                elif isinstance(s, ast.If):
                    need.update(_names(s.test, ast.Load) - defs)
                    a, b = set(defs), set(defs)
                    walk(s.body, a)
                    walk(s.orelse, b)
                    defs |= (a & b)
                elif isinstance(s, ast.For):
                    need.update(_names(s.iter, ast.Load) - defs)
                    inner = set(defs) | _names(s.target)
                    walk(s.body, inner)
                elif isinstance(s, ast.While):
                    need.update(_names(s.test, ast.Load) - defs)
                    walk(s.body, set(defs))
        walk(body, defined)
        return need


def flatten(Model, py_src, webs=True):
    """-> Flat   (raises Unsupported).  webs=False: without the splitting of re-used scratch variables"""
    try:
        return _Flattener(Model, py_src).run(webs)
    except Unsupported:
        raise
    except Exception as ex:                              # an unforeseen shape of the generated code
        raise Unsupported('%s: %s' % (type(ex).__name__, ex))

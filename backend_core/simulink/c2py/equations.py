# -*- coding: utf-8 -*-
"""
equations.py  -  deterministic (NO LLM)   validated dynamics(t, x, u)  ->  equations of the JSON card

The validated function is a list of assignments. A reader of the JSON does not want
`dx[9] = Gain2 - Gain9 - Gain10`: he wants to know what Gain9 IS. That is a mechanical job - follow
every temporary back through the code until only states, inputs and parameters are left - so it
is done here by program (symbolic back-substitution), exactly and for free, instead of asking an
LLM to trace several hundred lines:

  * every assignment `v = expr` becomes a definition, with the variables in expr replaced by
    what they stand for
  * `if / elif / else` that only assigns variables becomes a conditional expression; the
    saturation pattern becomes  min(max(v, lower), upper)
  * a definition that is used once is written into the place where it is used, as long as the
    result stays readable (max_len); so `Gain9`, `Sum1`, ... disappear and what is left are the
    quantities that are really shared (they keep their name and are listed under "definitions")
  * what is computed by loops over arrays (matrix products, a linear solve ...) cannot be written
    as a formula: those variables are listed as "computed in the code" and used by name

verify() then evaluates every definition and equation at the sample points of the reference and
compares with the model, so each equation in the JSON carries a checked flag.
"""
import ast
import copy
import io
import keyword
import math
import re
import tokenize

import numpy as np

NA = 'not extractable'


class _No(Exception):
    """this expression cannot be written in the card language"""


# ---- the expression language of the card -------------------------------------------------
def _nanaware(fn):
    def w(*a):
        return float('nan') if any(v != v for v in a) else float(fn(a))
    return w


EXPR_FUNCS = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
    'atan2': np.arctan2, 'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh, 'exp': np.exp, 'log': np.log,
    'log10': np.log10, 'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign, 'floor': np.floor, 'ceil': np.ceil,
    'trunc': np.trunc, 'fmod': np.fmod, 'hypot': np.hypot, 'pow': np.power,
    'min': _nanaware(min), 'max': _nanaware(max), 'int': lambda v: float(int(v)), 'isnan': np.isnan,
    'isinf': np.isinf, 'pi': math.pi, 'inf': math.inf, 'nan': math.nan,
}
NP_TO_CARD = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
              'arctan2': 'atan2', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'exp': 'exp', 'log': 'log',
              'log10': 'log10', 'sqrt': 'sqrt', 'abs': 'abs', 'sign': 'sign', 'floor': 'floor', 'ceil': 'ceil',
              'trunc': 'trunc', 'fmod': 'fmod', 'hypot': 'hypot', 'fmin': 'min', 'fmax': 'max',
              'minimum': 'min', 'maximum': 'max', 'isnan': 'isnan', 'isinf': 'isinf', 'power': 'pow'}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.Call, ast.Name,
          ast.Constant, ast.Subscript, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
          ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
          ast.Tuple)


def compile_expr(text, names):
    """one expression of the card language -> code object (ValueError with a readable reason)"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty expression')
    src = re.sub(r'\b(np|numpy|math)\.', '', text.strip()).replace('^', '**')
    try:
        tree = ast.parse(src, mode='eval')
    except SyntaxError as ex:
        raise ValueError('not a valid expression (%s)' % ex.msg)
    for n in ast.walk(tree):
        if not isinstance(n, _NODES):
            raise ValueError('%s is not allowed in an expression' % type(n).__name__)
        if isinstance(n, ast.Name) and n.id not in names and n.id not in EXPR_FUNCS:
            raise ValueError('unknown name "%s"' % n.id)
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and callable(EXPR_FUNCS.get(n.func.id))):
            raise ValueError('only the listed functions can be called')
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError('only numbers are allowed as constants')
    return compile(tree, '<card>', 'eval')


# ---- small ast helpers -------------------------------------------------------------------
def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _stored(stmts):
    """every variable written somewhere in the statements (also a[i] = ..., loop variables)"""
    out = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                for tt in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    while isinstance(tt, ast.Subscript):
                        tt = tt.value
                    if isinstance(tt, ast.Name):
                        out.add(tt.id)
    return out


def _same(a, b):
    return ast.dump(a) == ast.dump(b)


def _call(fn, *args):
    return ast.Call(ast.Name(fn, ast.Load()), list(args), [])


def _text(node):
    return ast.unparse(node)


def _size(node):
    """size of an expression = number of names and numbers in it"""
    return sum(1 for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Constant))
               and not (isinstance(n, ast.Name) and n.id in EXPR_FUNCS))


def _comments(code):
    """line -> (trailing comment, comment line directly above)"""
    trail, own = {}, {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                txt = tok.string.lstrip('#').strip()
                (own if tok.line.lstrip().startswith('#') else trail)[tok.start[0]] = txt
    except (tokenize.TokenError, IndentationError):
        pass
    return trail, own


def _simplify_if(node):
    """saturation written with if / elif / else  ->  min(max(v, lo), hi)"""
    if not isinstance(node, ast.IfExp):
        return node

    def cmp(t):
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            return type(t.ops[0]), t.left, t.comparators[0]
        return None, None, None
    op, a, b = cmp(node.test)
    if op in (ast.Gt, ast.GtE) and _same(node.body, b):               # v > hi -> hi
        v, hi, rest = a, b, node.orelse
        if _same(rest, v):
            return _call('min', v, hi)
        if isinstance(rest, ast.Call) and getattr(rest.func, 'id', None) == 'max' and len(rest.args) == 2 \
                and _same(rest.args[0], v):
            return _call('min', rest, hi)
        op2, a2, b2 = cmp(rest.test) if isinstance(rest, ast.IfExp) else (None, None, None)
        if op2 in (ast.Lt, ast.LtE) and _same(a2, v) and _same(rest.body, b2) and _same(rest.orelse, v):
            return _call('min', _call('max', v, b2), hi)
    if op in (ast.Lt, ast.LtE) and _same(node.body, b):               # v < lo -> lo
        v, lo, rest = a, b, node.orelse
        if _same(rest, v):
            return _call('max', v, lo)
        if isinstance(rest, ast.Call) and getattr(rest.func, 'id', None) == 'min' and len(rest.args) == 2 \
                and _same(rest.args[0], v):
            return _call('min', _call('max', v, lo), rest.args[1])
        op2, a2, b2 = cmp(rest.test) if isinstance(rest, ast.IfExp) else (None, None, None)
        if op2 in (ast.Gt, ast.GtE) and _same(a2, v) and _same(rest.body, b2) and _same(rest.orelse, v):
            return _call('min', _call('max', v, lo), b2)
    return node


# ======================================================================================
class Equations(object):
    """result of build(): definitions (ordered), state equations, variables computed in the code"""

    def __init__(self):
        self.states, self.inputs = [], []
        self.params = {}                    # name -> value (primary parameters only)
        self.defs = []                      # [dict(name, node, meaning)]
        self.eqs = []                       # per state: expression node or None
        self.code_vars = {}                 # name -> dict(meaning, reads): computed by an algorithm in the code
        self.raw = []                       # per state: the right-hand side as written in the code

    def text(self, node):
        return re.sub(r'\s+', ' ', _text(node))

    def rename(self, mapping):
        """rename definitions {old: new}; unsafe entries are ignored. Returns the applied mapping."""
        taken = set(self.states) | set(self.inputs) | set(self.params) | set(self.code_vars) | set(EXPR_FUNCS) | {'t'}
        names = [d['name'] for d in self.defs]
        ok = {}
        for old, new in (mapping or {}).items():
            if old in names and isinstance(new, str) and re.fullmatch(r'[A-Za-z_]\w*', new) and new != old \
                    and not keyword.iskeyword(new) and new not in ('x', 'u', 'dx', 'np') \
                    and new not in taken and new not in names and new not in ok.values():
                ok[old] = new
        if not ok:
            return ok

        class R(ast.NodeTransformer):
            def visit_Name(self, node):
                node.id = ok.get(node.id, node.id)
                return node
        for d in self.defs:
            d['name'] = ok.get(d['name'], d['name'])
            d['node'] = R().visit(d['node'])
        self.eqs = [R().visit(e) if e is not None else None for e in self.eqs]
        return ok


class _Builder(object):
    def __init__(self, code, nx, nu, params, max_terms, small):
        self.code, self.nx, self.nu, self.max_terms, self.small = code, nx, nu, max_terms, small
        tree = ast.parse(code)
        self.fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
        self.trail, self.own = _comments(code)
        self.E = Equations()
        self.E.params = dict(params)
        self.env = {}                       # code variable -> node that stands for its current value
        self.defs = {}                      # definition name -> dict(node, meaning, order)
        self.opaque = {}                    # code variable -> description
        self.reads = {}                     # code variable -> names its algorithm reads
        self.used = set()
        self.last_store = {}

    # ---------------------------------------------------------------- helpers
    def meaning(self, stmt):
        """the comment of the line, else the nearest comment above it (the Simulink block / the
        MATLAB source line the statement comes from)"""
        ln = getattr(stmt, 'lineno', -1)
        if ln in self.trail:
            return self.trail[ln]
        if isinstance(stmt, (ast.If, ast.For, ast.While)) and ln - 1 not in self.own:
            for k in range(ln + 1, getattr(stmt, 'end_lineno', ln) + 1):      # the block comment is inside
                if k in self.own:
                    return self.own[k]
        for k in range(ln - 1, max(ln - 400, 0), -1):
            if k in self.own:
                return self.own[k]
        return ''

    def fresh(self, want):
        """name of a new definition: the variable's own name unless that is taken"""
        taken = set(self.defs) | set(self.opaque) | self.reserved
        if want in self.loopy:                           # the variable is (also) computed in a loop
            want += '_v'
        name, k = want, 2
        while name in taken or (name != want and name in self.used):
            name, k = '%s_%d' % (want, k), k + 1
        return name

    def define(self, var, node, stmt):
        node = _simplify_if(node)
        if isinstance(node, ast.Name) or (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))) \
                or (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and isinstance(node.slice, ast.Constant)):
            self.env[var] = node            # alias / plain number / element: no definition needed
            return
        name = self.fresh(var)
        self.defs[name] = dict(node=node, meaning=self.meaning(stmt), order=len(self.defs))
        self.env[var] = ast.Name(name, ast.Load())

    def make_opaque(self, var, stmt):
        reads = self.reads.setdefault(var, set())
        for n in ast.walk(stmt):                         # what the algorithm is computed from
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id != var:
                if n.id in self.env:
                    reads |= {m for m in _names(self.env[n.id]) if m not in EXPR_FUNCS}
                elif n.id in self.opaque:
                    reads.add(n.id)
        self.env.pop(var, None)
        if var in self.defs:                             # an earlier formula carries this name: move it away
            new = self.fresh(var + '_v')

            class R(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == var:
                        node.id = new
                    return node
            self.defs[new] = self.defs.pop(var)
            for k in self.reads:
                if var in self.reads[k]:
                    self.reads[k] = (self.reads[k] - {var}) | {new}
            for d in self.defs.values():
                d['node'] = R().visit(d['node'])
            for k in list(self.env):
                self.env[k] = R().visit(self.env[k])
            self.E.eqs = [R().visit(e) if e is not None else None for e in self.E.eqs]
        self.opaque.setdefault(var, self.meaning(stmt) or 'computed by an algorithm in the code')

    def subst(self, expr, idx, env=None):
        """expr of the code -> card expression in terms of states / inputs / parameters /
        definitions / variables computed in the code. Raises _No if that is not possible."""
        env = self.env if env is None else env
        b = self

        class S(ast.NodeTransformer):
            def visit_Name(self, node):
                v = node.id
                if v in env:
                    return copy.deepcopy(env[v])
                if v in b.opaque:
                    if b.last_store.get(v, -1) >= idx:
                        raise _No('%s changes again later in the code' % v)
                    b.code_used.add(v)
                    return node
                if v in ('np', 'int', 'abs', 'min', 'max', 'float', 't', 'True', 'False'):
                    return node
                raise _No('unknown variable %s' % v)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr in ('pi', 'inf', 'nan'):
                    return ast.Name(node.attr, ast.Load())
                raise _No('attribute %s' % ast.unparse(node))

            def visit_Call(self, node):
                f = node.func
                args = [self.visit(a) for a in node.args]
                if node.keywords:
                    raise _No('keyword arguments')
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'np':
                    if f.attr == 'power':
                        return ast.BinOp(args[0], ast.Pow(), args[1])
                    if f.attr in NP_TO_CARD:
                        return _call(NP_TO_CARD[f.attr], *args)
                    raise _No('np.%s' % f.attr)
                if isinstance(f, ast.Name) and f.id in ('abs', 'min', 'max', 'int'):
                    return _call(f.id, *args)
                if isinstance(f, ast.Name) and f.id == 'float' and len(args) == 1:
                    return args[0]
                raise _No('call %s' % ast.unparse(f))

            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name) and node.value.id in ('x', 'u'):
                    raise _No('raw x / u access')
                return ast.Subscript(self.visit(node.value), self.visit(node.slice), ast.Load())

            def generic_visit(self, node):
                if isinstance(node, (ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.List, ast.Dict, ast.Slice,
                                     ast.Starred, ast.JoinedStr)):
                    raise _No(type(node).__name__)
                return ast.NodeTransformer.generic_visit(self, node)
        return S().visit(copy.deepcopy(expr))

    @staticmethod
    def pure_if(s):
        """if / elif / else whose branches only assign plain variables (or contain such ifs)"""
        for st in s.body + s.orelse:
            if isinstance(st, ast.If):
                if not _Builder.pure_if(st):
                    return False
            elif not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                return False
        return True

    def run_if(self, s, idx, env):
        """symbolic execution of a pure if. Returns the new environment (var -> node)."""
        test = self.subst(s.test, idx, env)
        outs = []
        for branch in (s.body, s.orelse):
            e = dict(env)
            for st in branch:
                if isinstance(st, ast.If):
                    e = self.run_if(st, idx, e)
                else:
                    e[st.targets[0].id] = _simplify_if(self.subst(st.value, idx, e))
            outs.append(e)
        new = dict(env)
        for v in sorted(set(k for o in outs for k in o if o[k] is not env.get(k))):
            a, c = outs[0].get(v), outs[1].get(v)
            if a is None or c is None:
                raise _No('%s is only assigned in one branch' % v)
            new[v] = _simplify_if(ast.IfExp(test, a, c))
        return new

    # ---------------------------------------------------------------- driver
    def run(self):
        E, body = self.E, self.fn.body
        first_store = {}
        for i, s in enumerate(body):
            for v in _stored([s]):
                self.last_store[v] = i
                first_store.setdefault(v, i)
        self.used = set(n.id for n in ast.walk(self.fn) if isinstance(n, ast.Name)) | set(EXPR_FUNCS)
        self.loopy = set()
        for s in body:
            if isinstance(s, (ast.For, ast.While)) or (isinstance(s, ast.If) and not self.pure_if(s)):
                self.loopy |= _stored([s])
            elif isinstance(s, ast.Assign) and not isinstance(s.targets[0], ast.Name):
                self.loopy |= _stored([s])
        self.reserved = set(EXPR_FUNCS) | {'t', 'x', 'u'} | set(E.params)
        for s in body:                                   # state / input names
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Subscript) \
                    and getattr(s.value.value, 'id', None) in ('x', 'u'):
                self.reserved.add(s.targets[0].id)
        dx = next((n.value.id for n in ast.walk(self.fn) if isinstance(n, ast.Return)
                   and isinstance(n.value, ast.Name)), 'dx')
        snames, unames = [None] * self.nx, [None] * self.nu
        E.eqs, E.raw = [None] * self.nx, [NA] * self.nx
        self.code_used = set()
        for i, s in enumerate(body):
            if isinstance(s, ast.Expr) or isinstance(s, ast.Return):
                continue
            if isinstance(s, ast.Assign) and len(s.targets) == 1:
                tgt, val = s.targets[0], s.value
                if isinstance(tgt, ast.Name):
                    v = tgt.id
                    sub = val if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) else None
                    if sub is not None and sub.value.id in ('x', 'u') and isinstance(sub.slice, ast.Constant) \
                            and first_store.get(v) == i:
                        # (the variable may be given another value later - `if pos < 1e-4: pos = 1e-4` -
                        #  the NAME stands for the state itself, the new value becomes a definition)
                        if v in self.loopy:
                            raise ValueError('the state / input variable %s is re-assigned by an algorithm' % v)
                        (snames if sub.value.id == 'x' else unames)[sub.slice.value] = v
                        self.env[v] = ast.Name(v, ast.Load())
                        continue
                    if v in E.params and self.last_store.get(v) == i:
                        self.env[v] = ast.Name(v, ast.Load())
                        continue
                    if v == dx:
                        continue
                    try:
                        self.define(v, self.subst(val, i), s)
                    except _No:
                        self.make_opaque(v, s)
                    continue
                if isinstance(tgt, ast.Subscript) and getattr(tgt.value, 'id', None) == dx \
                        and isinstance(tgt.slice, ast.Constant) and 0 <= tgt.slice.value < self.nx:
                    k = tgt.slice.value
                    E.raw[k] = re.sub(r'\b(np|numpy|math)\.', '', ' '.join(ast.unparse(val).split()))
                    try:
                        E.eqs[k] = self.subst(val, i)
                    except _No:
                        E.eqs[k] = None
                    continue
            if isinstance(s, ast.If) and self.pure_if(s):
                try:
                    new = self.run_if(s, i, dict(self.env))
                    for v in new:
                        if new[v] is not self.env.get(v):
                            self.define(v, new[v], s)
                    continue
                except _No:
                    pass
            for v in _stored([s]):                        # loops, array code, anything else
                if v != dx:
                    self.make_opaque(v, s)
        E.states = [n or NA for n in snames]
        E.inputs = [n or NA for n in unames]
        self.reduce()
        return E

    def reduce(self):
        """drop unused definitions; write a definition into its user(s) when it is used once, or is
        tiny, as long as the result stays readable (size <= max_terms)"""
        E, defs = self.E, self.defs
        order = [n for n, _ in sorted(self.last_store.items(), key=lambda kv: kv[1])]

        def closure(v):
            """everything the algorithm behind code variable v reads, directly or through other
            code variables"""
            seen, todo = set(), [v]
            while todo:
                for r in self.reads.get(todo.pop(), ()):
                    if r not in seen:
                        seen.add(r)
                        if r in self.opaque:
                            todo.append(r)
            return seen

        def live_code():
            """the variables computed in the code that definitions / equations use by name"""
            live = set()
            for d in defs.values():
                live |= _names(d['node']) & set(self.opaque)
            for e in E.eqs:
                if e is not None:
                    live |= _names(e) & set(self.opaque)
            return live

        def uses(live):
            cnt = {}
            for d in defs.values():
                for n in ast.walk(d['node']):
                    if isinstance(n, ast.Name):
                        cnt[n.id] = cnt.get(n.id, 0) + 1
            for e in E.eqs:
                if e is not None:
                    for n in ast.walk(e):
                        if isinstance(n, ast.Name):
                            cnt[n.id] = cnt.get(n.id, 0) + 1
            for v in live:                               # an input of an algorithm keeps its name
                for r in closure(v):
                    cnt[r] = cnt.get(r, 0) + 1000
            return cnt

        class Inl(ast.NodeTransformer):
            def __init__(self, name, node):
                self.name, self.node = name, node

            def visit_Name(self, node):
                return copy.deepcopy(self.node) if node.id == self.name else node
        changed = True
        while changed:
            changed = False
            cnt = uses(live_code())
            dead = [n for n in defs if cnt.get(n, 0) == 0]
            for n in dead:
                del defs[n]
            if dead:
                changed = True
                continue
            for name in sorted(defs, key=lambda n: defs[n]['order']):
                d, c = defs[name], cnt.get(name, 0)
                if c >= 1000 or not (c == 1 or _size(d['node']) <= self.small):
                    continue
                users = [u for u in defs if u != name and name in _names(defs[u]['node'])]
                eqs = [k for k, e in enumerate(E.eqs) if e is not None and name in _names(e)]
                new_d = {u: Inl(name, d['node']).visit(copy.deepcopy(defs[u]['node'])) for u in users}
                new_e = {k: Inl(name, d['node']).visit(copy.deepcopy(E.eqs[k])) for k in eqs}
                if all(_size(v) <= self.max_terms for v in list(new_d.values()) + list(new_e.values())):
                    for u, v in new_d.items():
                        defs[u]['node'] = v
                    for k, v in new_e.items():
                        E.eqs[k] = v
                    del defs[name]
                    changed = True
                    break
        E.defs = [dict(name=n, node=defs[n]['node'], meaning=defs[n]['meaning'])
                  for n in sorted(defs, key=lambda n: defs[n]['order'])]
        live = live_code()
        keep = set(defs) | self.reserved
        E.code_vars = {v: dict(meaning=self.opaque[v], reads=sorted(r for r in closure(v) if r in keep and r not in EXPR_FUNCS))
                       for v in order if v in live}


def build(code, nx, nu, params, max_terms=30, small=4):
    """validated code -> Equations.  params: {name: value} of the PRIMARY parameters (the lines
    `name = value` that are model parameters); every other assignment is traced back.
    max_terms: an expression may grow to this many names + numbers when definitions are written
    into it; small: a definition of at most this size is always written out."""
    return _Builder(code, nx, nu, params, max_terms, small).run()


# ======================================================================================
def code_locals(code):
    """dynamics(t, x, u) of the code, returning its local variables (for the variables that are
    computed by an algorithm and therefore used by name in the card)"""
    src = re.sub(r'(?m)^(\s*)return (\w+)\s*$', r'\1return dict(locals())', code)
    ns = {'__name__': 'card_locals'}
    exec(compile(src, '<dynamics locals>', 'exec'), ns)
    return ns['dynamics']


def verify(E, code, pts, D, compare, tol):
    """evaluate the definitions and state equations at the sample points pts = [(t, x, u)] and
    compare with the reference derivatives D.  compare(ref, got, names, tol) -> {name: {'passed'}}
    Returns (eq_ok [bool per state], bad [names of definitions that could not be evaluated])."""
    nx = len(E.eqs)
    known = set(E.states) | set(E.inputs) | set(E.params) | set(E.code_vars) | {'t'}
    comp, bad = [], set()
    for d in E.defs:
        try:
            comp.append((d['name'], compile_expr(E.text(d['node']), known)))
            known.add(d['name'])
        except ValueError:
            bad.add(d['name'])
    eqs = []
    for e in E.eqs:
        try:
            eqs.append(compile_expr(E.text(e), known) if e is not None else None)
        except ValueError:
            eqs.append(None)
    fn = code_locals(code) if E.code_vars else None
    base = {k: (np.array(v, float) if isinstance(v, list) else float(v)) for k, v in E.params.items()}
    G = np.full((len(pts), nx), np.nan)
    err = np.seterr(all='ignore')
    try:
        for i, (tt, x, u) in enumerate(pts):
            ns = dict(base)
            if fn is not None:
                loc = fn(float(tt), np.array(x, float), np.array(u, float))
                ns.update({v: loc[v] for v in E.code_vars if v in loc})
            ns.update(zip(E.states, (np.float64(v) for v in x)))
            ns.update(zip(E.inputs, (np.float64(v) for v in u)))
            ns['t'] = np.float64(tt)
            for nm, co in comp:
                try:
                    ns[nm] = np.float64(eval(co, {'__builtins__': {}}, dict(EXPR_FUNCS, **ns)))
                except Exception:
                    bad.add(nm)
                    ns[nm] = np.float64('nan')
            env = dict(EXPR_FUNCS, **ns)
            for k, co in enumerate(eqs):
                if co is not None:
                    try:
                        G[i, k] = float(eval(co, {'__builtins__': {}}, env))
                    except Exception:
                        G[i, k] = np.nan
    finally:
        np.seterr(**err)
    ok = []
    for k in range(nx):
        ok.append(bool(eqs[k] is not None
                       and list(compare(D[:, [k]], G[:, [k]], ['v'], tol).values())[0]['passed']))
    return ok, sorted(bad)

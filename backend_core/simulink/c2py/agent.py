# -*- coding: utf-8 -*-
"""
agent.py  -  generated python model  ->  readable  dynamics(t, x, u)  +  JSON card

    1. reference      the generated <model>_py.py itself (class Reference)
    2. exact version  src/flatten.py turns the generated code into dynamics(t, x, u) by program
                      transformation - no LLM, exact by construction, validated against 1.
    3. readable       the LLM (OpenAI API) makes it readable, CONFIG agent_mode:
                        rewrite   small models: it rewrites the function as textbook equations;
                                  every draft is validated, a failed one goes back to it
                        annotate  big models: it only returns names + meanings, a program applies
                                  them to the exact version (one call, few output tokens)
                      a failed rewrite falls back to annotate; a model that flatten.py cannot
                      handle is rewritten by the LLM from the generated class (first design)
    4. validation     derivative check  dynamics(t,x,u) vs the model at sampled points
                      simulation check  states integrated with dynamics vs the model
       PASS -> <model>_dynamics.py
    5. JSON card      src/equations.py traces every temporary of the validated code back to
                      states / inputs / parameters (no LLM; saturations become min/max) and each
                      equation is checked numerically against the model.
       JSON agent     a SECOND, small LLM call reads that card (not the code) and gives the
                      remaining definitions engineering names and meanings; a program applies them.
                      -> <model>_dynamics.json

Everything in the JSON that CAN be read from the validated code (states, inputs, parameters,
state equations) is extracted from it with `ast`, so the JSON always describes the code that was
actually validated. What cannot be extracted is written as "not extractable".

API key: environment variable OPENAI_API_KEY (or a line OPENAI_API_KEY=... in <project>/.env).
"""
import ast
import copy
import io
import json
import math
import os
import re
import sys
import time
import tokenize
from pathlib import Path

import numpy as np

from . import validation
from .cost import CostCalculator
from . import equations
from .flatten import Unsupported, flatten
from ...prompts import load as _load_prompts

NA = 'not extractable'
STATUS_OK = 'Validated'
STATUS_PART = 'Validated (continuous part only, discrete memory frozen)'
STATUS_FAIL = 'Validation failed'

# ======================================================================================
# the prompts: backend_core/prompts/simulink/c2py_agent.yaml (edit them there)
# ======================================================================================
_PROMPTS = _load_prompts('simulink/c2py_agent')
HEAD_CLASS, HEAD_FLAT = _PROMPTS['head_class'], _PROMPTS['head_flat']
COMMON_CLASS, COMMON_FLAT = _PROMPTS['common_class'], _PROMPTS['common_flat']
RULE7 = _PROMPTS['rule7']                                   # rule 7 of the rewrite prompts, per cfg['agent_naming']
ANNOTATE_PROMPT, ANNOTATE_RULES = _PROMPTS['annotate_prompt'], _PROMPTS['annotate_rules']
FLAT_USER, IMPURE_FLAT = _PROMPTS['flat_user'], _PROMPTS['impure_flat']
USER_TEMPLATE, IMPURE_NOTE = _PROMPTS['user_template'], _PROMPTS['impure_note']
RETRY_TEMPLATE = _PROMPTS['retry_template']
DESCRIBE_PROMPT, DESCRIBE_PROMPT_MODEL = _PROMPTS['describe_prompt'], _PROMPTS['describe_prompt_model']


def rewrite_prompt(flat, naming):
    """system prompt of the rewrite agent (flat: from the exact translation, else from the class)"""
    text = (HEAD_FLAT + COMMON_FLAT) if flat else (HEAD_CLASS + COMMON_CLASS)
    if naming != 'free':
        pair = _PROMPTS['rewrite_model_naming']
        text = text.replace(pair['find'], pair['replace'])
    return text.replace('@RULE7@', RULE7['free' if naming == 'free' else 'model']).strip()


SYSTEM_PROMPT = rewrite_prompt(False, 'free')               # (kept for callers / tests)
SYSTEM_PROMPT_FLAT = rewrite_prompt(True, 'free')


def annotate_prompt(naming):
    text = ANNOTATE_PROMPT
    if naming != 'free':
        pair = _PROMPTS['annotate_model_naming']
        assert pair['find'] in text
        text = text.replace(pair['find'], pair['replace'])
    return text.replace('@RULES@', ANNOTATE_RULES['free' if naming == 'free' else 'model'])


# ======================================================================================
# 1. what the agent gets to see
# ======================================================================================
_NOISE_RE = re.compile(r'^\s*(self\.(rtsiSet|rtmSet)\w+\(|self\.M\.intgData\.|self\.P = _init_val\()')


_INFRA_STRUCT_RE = re.compile(r"^    '(c(real|u?int)\d*_T|ODE\d+_IntgData|__anon\d+|tag_RTM_\w+|XDis_\w+)': ")


def model_source(py_src):
    """model specific part of the generated module (STRUCTS + class), without the runtime
    library that is identical in every generated file, and without solver registration lines."""
    a = py_src.find('\nSTRUCTS = {')
    b = py_src.rfind('\nModel = ')
    body = py_src[a + 1 if a >= 0 else 0:b if b >= 0 else len(py_src)]
    keep = []
    for ln in body.splitlines():
        if _INFRA_STRUCT_RE.match(ln):
            continue
        if ln.lstrip().startswith('self.P = _init_val('):
            keep.append(ln[:len(ln) - len(ln.lstrip())] + '# self.P = ...   (values: see PARAMETERS in FACTS)')
        elif not _NOISE_RE.match(ln):
            keep.append(ln)
    return '\n'.join(keep).rstrip()


def _num(v):
    return repr(float(v)) if isinstance(v, (float, np.floating)) else repr(int(v))


def dump_struct(o, prefix):
    """'P.Gain_Gain = 2.0' lines for every numeric field of a generated _Struct."""
    out = []
    for k, v in getattr(o, '__dict__', {}).items():
        name = '%s.%s' % (prefix, k)
        if isinstance(v, np.ndarray):
            out.append('%s = %s' % (name, json.dumps(v.tolist())))
        elif isinstance(v, (int, float, np.number)):
            out.append('%s = %s' % (name, _num(v)))
        elif hasattr(v, '__dict__'):
            out += dump_struct(v, name)
    return out


def model_facts(Model, py_src='', solver='auto'):
    m = Model(solver=solver)
    dw = re.findall(r"self\.(\w+) = _new\('DW_", py_src) or [k for k in m.__dict__ if re.search(r'(^|_)DW$', k)]
    memory = []
    for a in dw:
        memory += dump_struct(getattr(m, a, None), 'self.' + a)
    return dict(states=list(m.state_names), inputs=list(m.input_names), outputs=list(m.output_names),
                h=m.step_size, solver=m._method(), params=dump_struct(getattr(m, 'P', None), 'self.P'),
                memory=memory, pure=True)


def build_messages(name, py_src, facts, naming='model'):
    def lines(names, sym):
        return '\n'.join('  %s[%d] = self.%s.%s' % (sym, i, attr, n) for i, (attr, n) in enumerate(names)) \
            or '  (none)'
    extra = []
    if facts['params']:
        extra.append('PARAMETERS (self.P):\n' + '\n'.join('  ' + p for p in facts['params']))
    if facts['memory'] and not facts['pure']:
        extra.append(IMPURE_NOTE + '\n' + '\n'.join('  ' + p for p in facts['memory']))
    user = USER_TEMPLATE.format(
        name=name, nx=len(facts['states']), nu=len(facts['inputs']), ny=len(facts['outputs']),
        states=lines([('X', n) for n in facts['states']], 'x'),
        inputs=lines([('U', n) for n in facts['inputs']], 'u'),
        outputs='\n'.join('  %d: self.Y.%s' % (i, n) for i, n in enumerate(facts['outputs'])) or '  (none)',
        h='%g' % facts['h'], solver=facts['solver'],
        extra=('\n'.join(extra) + '\n') if extra else '', code=model_source(py_src))
    return [{'role': 'system', 'content': rewrite_prompt(False, naming)}, {'role': 'user', 'content': user}]


# ======================================================================================
# 2. the LLM call
# ======================================================================================
def load_env(root):
    """KEY=VALUE lines of <root>/.env -> os.environ (existing variables win)."""
    f = Path(root) / '.env'
    if f.exists():
        for ln in f.read_text(encoding='utf-8', errors='replace').splitlines():
            m = re.match(r'\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$', ln)
            if m and not ln.lstrip().startswith('#'):
                os.environ.setdefault(m.group(1), m.group(2).strip('\'"'))


def openai_llm(model, timeout=300):
    """returns  llm(messages) -> text .  Key / endpoint: OPENAI_API_KEY, OPENAI_BASE_URL."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit('the agent needs the OpenAI SDK:  pip install openai   (or run with --no-agent)')
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set (environment variable, or a line '
                         'OPENAI_API_KEY=... in the .env file next to main.py)')
    client = OpenAI(timeout=timeout)

    def llm(messages):
        r = client.chat.completions.create(model=model, messages=messages,
                                           response_format={'type': 'json_object'})
        u = getattr(r, 'usage', None)                    # completion_tokens includes the reasoning tokens
        llm.usage.append((int(getattr(u, 'prompt_tokens', 0) or 0), int(getattr(u, 'completion_tokens', 0) or 0)))
        return r.choices[0].message.content
    llm.model, llm.usage = model, []                     # read by run() for the cost report
    return llm


def cost_report(model, usage, calc=None):
    """usage: [(input_tokens, output_tokens)] of every call -> dict with the USD cost (src/cost.py).
    cost_usd is None when the model has no entry in the price table."""
    price = (calc or CostCalculator()).resolve_price(model)
    calls = [dict(call=i + 1, input_tokens=a, output_tokens=b, cost_usd=price.cost(a, b) if price else None)
             for i, (a, b) in enumerate(usage)]
    tin, tout = sum(a for a, _ in usage), sum(b for _, b in usage)
    return dict(model=model, calls=calls, input_tokens=tin, output_tokens=tout,
                cost_usd=price.cost(tin, tout) if price else None,
                usd_per_million_tokens=dict(input=price.input_per_million, output=price.output_per_million)
                if price else None)


def _usd(v):
    return 'price unknown' if v is None else '$%.4f' % v


def parse_answer(text):
    """a rewrite answer: JSON object with a "python_code" string"""
    d = parse_json(text)
    if not isinstance(d.get('python_code'), str):
        raise ValueError('the answer has no "python_code" string')
    code = d['python_code'].strip()
    m = re.match(r'^```(?:python)?\s*\n(.*?)\n```$', code, re.S)
    d['python_code'] = (m.group(1) if m else code).strip() + '\n'
    return d


# ======================================================================================
# 3. loading the draft (it is LLM output: check it before running it)
# ======================================================================================
_FORBIDDEN = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'globals', 'locals', 'vars',
              'getattr', 'setattr', 'delattr', 'breakpoint', 'exit', 'quit', 'os', 'sys', 'subprocess'}


def check_code(code):
    """Static check of the draft. Raises ValueError with a message the agent can act on."""
    try:
        tree = ast.parse(code)
    except SyntaxError as ex:
        raise ValueError('syntax error in python_code, line %s: %s' % (ex.lineno, ex.msg))
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(a.name not in ('numpy', 'math') for a in node.names):
                raise ValueError('only "import numpy as np" (and math) is allowed')
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ('numpy', 'math'):
                raise ValueError('only "import numpy as np" (and math) is allowed')
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass                                             # docstring
        elif not isinstance(node, ast.FunctionDef):
            raise ValueError('only imports and function definitions are allowed at module level')
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics']
    if not fn:
        raise ValueError('python_code does not define dynamics(t, x, u)')
    if [a.arg for a in fn[0].args.args] != ['t', 'x', 'u']:
        raise ValueError('the signature must be exactly dynamics(t, x, u)')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            raise ValueError('imports inside functions are not allowed')
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN:
            raise ValueError('"%s" is not allowed in the dynamics code' % node.id)
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            raise ValueError('attribute "%s" is not allowed in the dynamics code' % node.attr)
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.With, ast.AsyncFunctionDef)):
            raise ValueError('%s is not allowed in the dynamics code' % type(node).__name__)
    return tree


def load_dynamics(code, nx):
    """exec the checked draft -> f(t, x, u) returning a float vector of length nx."""
    check_code(code)
    ns = {'__name__': 'dynamics_draft'}
    exec(compile(code, '<dynamics>', 'exec'), ns)
    dyn = ns['dynamics']

    def f(t, x, u):
        dx = np.asarray(dyn(float(t), np.array(x, float), np.array(u, float)), float).reshape(-1)
        if dx.size != nx:
            raise ValueError('dynamics() returned %d values, the model has %d states' % (dx.size, nx))
        return dx
    return f


# ======================================================================================
# 4. validation:  dynamics(t, x, u)  vs  the generated model
# ======================================================================================
ABS_TOL = 1e-9          # an error below this passes whatever the signal range is (signals that stay ~0)
BLOWUP = 1e9            # |state| above this = the reference simulation has diverged


def _compare(ref, got, names, tol):
    """per column NRMSE (validation.metrics) + pass flag. ref/got: N x n."""
    rmse, nrmse, mx, masked, bad = validation.metrics(ref, got)
    res = {}
    for j, n in enumerate(names):
        small = bool(np.isfinite(mx[j]) and mx[j] <= ABS_TOL)
        ok = bool(bad[j] == 0 and (small or (np.isfinite(nrmse[j]) and nrmse[j] <= tol)))
        if not np.isfinite(nrmse[j]) and masked[j] == ref.shape[0]:          # both non-finite everywhere
            ok = bad[j] == 0
        res[n] = dict(rmse=float(rmse[j]), nrmse=float(nrmse[j]), max_abs_error=float(mx[j]),
                      mismatched_nonfinite=int(bad[j]), passed=ok)
    return res


def _relation(ref, got):
    """(a, b) if  ref = a * got + b  at every sample point while got != ref, else None.
    Such a draft has the right form and a wrong constant - worth telling the agent."""
    ok = np.isfinite(ref) & np.isfinite(got)
    if ok.sum() < 8:
        return None
    r, g = ref[ok], got[ok]
    gs, rs = float(np.max(np.abs(g))), float(np.max(np.abs(r)))
    if gs == 0.0 or rs == 0.0 or np.ptp(g) <= 1e-12 * gs:
        return None
    (a, b), _, _, _ = np.linalg.lstsq(np.c_[g / gs, np.ones(len(g))], r, rcond=None)
    a = a / gs
    if np.max(np.abs(r - (a * g + b))) > 1e-8 * rs:
        return None
    return float(a), float(b)


def integrate(f, tab, x0, t, U):
    """the fixed-step solver of the generated runtime (CModel.rt_ertODEUpdateContinuousStates),
    applied to f: inputs held over a step, last stage evaluated at the end of the step."""
    c, A, b = tab
    h = float(t[1] - t[0])
    X = np.zeros((len(t), len(x0)))
    x = np.array(x0, float)
    for k in range(len(t)):
        X[k] = x
        if k == len(t) - 1 or not len(x0):
            continue
        u, tk = U[k], t[k]
        ks = [f(tk, x, u)]
        for i in range(1, len(b)):
            xs = x + h * sum(A[i][j] * ks[j] for j in range(i) if A[i][j] != 0)
            ks.append(f(t[k + 1] if c[i] == 1.0 else tk + c[i] * h, xs, u))
        x = x + h * sum(b[i] * ks[i] for i in range(len(b)) if b[i] != 0)
    return X


def _snapshot(roots):
    """every numeric field / array reachable from the given generated _Struct objects
    -> [(owner, key, saved value, is_array)]   (see _restore)"""
    items, seen = [], set()

    def walk(o):
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, list):
            pairs = list(enumerate(o))
        elif type(o).__name__ == '_Struct':
            pairs = list(o.__dict__.items())
        else:
            return
        for k, v in pairs:
            if isinstance(v, np.ndarray):
                items.append((o, k, v.copy(), True))
            elif isinstance(v, (bool, int, float, np.number)):
                items.append((o, k, v, False))
            else:
                walk(v)
    for r in roots:
        walk(r)
    return items


def _restore(items):
    for o, k, v, arr in items:
        if arr:                                          # in place: other fields may alias the array
            (o[k] if isinstance(o, list) else getattr(o, k))[...] = v
        elif isinstance(o, list):
            o[k] = v
        else:
            setattr(o, k, v)


class Reference(object):
    """The generated model as the reference for a dynamics(t, x, u) draft.

    f(t, x, u)   derivative of the generated model at an ARBITRARY point: one full (major step)
                 pass over all blocks without the solver / state update, on a scratch instance
                 whose memory (self.DW, block signals, counters) is put back to its initial
                 value before every call. So it is a pure function of (t, x, u).
    pure         True if integrating f reproduces Model.simulate(), i.e. the model really is
                 dx/dt = f(t, x, u). False if it has memory outside x (discrete states, Memory /
                 Derivative / Unit Delay blocks, random sources, ...): no function of (t, x, u)
                 can reproduce such a model exactly. Then the reference is the model with that
                 memory frozen at its initial value ("continuous part only").
    check(f)     compares a draft with the reference; all reference data is computed once here,
                 so every further draft only costs the evaluation of the draft itself."""

    def __init__(self, Model, cfg, solver='auto', log=print):
        mod = sys.modules[Model.__module__]
        self.tol = cfg.get('agent_tol', 1e-6)
        self.window = max(1, int(cfg.get('agent_window') or 25))
        m = Model(solver=solver)
        self.names = list(m.state_names)
        self.nx, self.nu = len(self.names), len(m.input_names)
        self.h, self.method = m.step_size, m._method()
        self.x0 = m.get_x()
        n = int(round(cfg['tf'] / self.h))
        self.n = n = max(2, min(n, int(cfg.get('agent_max_steps') or n)))
        self.t = t = np.arange(n + 1) * self.h
        tab = mod._TAB[self.method]

        s = self._s = Model(solver=solver)               # scratch instance behind f()
        s.rt_ertODEUpdateContinuousStates = lambda *a: None
        keep = (getattr(s, s.X_ATTR, None), getattr(s, s.U_ATTR, None))
        self._mem = _snapshot([v for v in s.__dict__.values() if not any(v is k for k in keep)])
        self._ticks = [k for k in getattr(s.M.Timing, '__dict__', {}) if re.fullmatch(r'clockTick[01]', k)]

        rng = np.random.default_rng(cfg.get('seed', 0))
        self.U, self.X, self.pure, pts, self.diverged = {}, {}, True, [], {}
        np_err = np.seterr(all='ignore')
        try:
            for sig in cfg['inputs']:
                U = validation.make_inputs(sig, t, self.nu, cfg['amp'], cfg['seed']) if self.nu \
                    else np.zeros((len(t), 0))
                X_sim = m.simulate(float(t[-1]), U if self.nu else None)['x'][:len(t)]
                X_f = integrate(self.f, tab, self.x0, t, U)
                self.U[sig], self.X[sig] = U, (X_sim, X_f)
                if not all(v['passed'] for v in _compare(X_sim, X_f, self.names, self.tol).values()):
                    self.pure = False
            for sig in cfg['inputs']:                    # reference trajectory: the real simulation if we can
                U, X = self.U[sig], self.X[sig][0 if self.pure else 1]
                # a reference that diverges (plant without its controller ...) is only compared up to there:
                # beyond it the numbers are astronomically large and any round-off difference explodes
                big = ~np.isfinite(X).all(axis=1) | (np.abs(np.nan_to_num(X)).max(axis=1) > BLOWUP)
                cut = int(np.argmax(big)) if big.any() else len(X)
                if cut < len(X):
                    cut = max(2, cut - self.window)
                    self.diverged[sig] = float(t[cut - 1])
                    X = X.copy()
                    X[cut:] = np.nan
                self.X[sig] = X
                # sample points of the derivative check: states the model really visits + a
                # perturbed copy of each (so a term that is ~0 on the trajectory is exercised too)
                fin = np.isfinite(X).all(axis=1)
                span = np.max(np.abs(X[fin]), axis=0) if fin.any() else np.ones(self.nx)
                for k in np.unique(np.linspace(0, n, min(n + 1, int(cfg.get('agent_samples', 150)))).astype(int)):
                    if fin[k]:
                        pts.append((t[k], X[k], U[k]))
                        pts.append((t[k], X[k] + 0.1 * rng.standard_normal(self.nx)
                                    * np.where(span > 0, span, 1.0), U[k]))
            self.pts = pts
            D, Y = [], []
            for tt, x, u in pts:
                D.append(self.f(tt, x, u))
                Y.append(self._s.get_y())                # the root outputs of the same evaluation
            self.D = np.array(D).reshape(len(pts), self.nx)
            self.Y = np.array(Y).reshape(len(pts), -1)
        finally:
            np.seterr(**np_err)
        self._tab = tab

    def f(self, t, x, u):
        s = self._s
        _restore(self._mem)
        s.set_x(x)
        if self.nu:
            s.set_u(u)
        Tm = s.M.Timing
        if hasattr(Tm, 'tArray'):
            Tm.tArray[...] = t
        else:
            Tm.t[0] = float(t)
        for k in self._ticks:
            setattr(Tm, k, int(round(t / self.h)))
        s._t_solver, s._tstop = float(t), None
        Tm.simTimeStep = 0                               # MAJOR_TIME_STEP: every block is executed
        getattr(s, s.STEP_FN)()
        s.derivatives()
        return s.get_xdot()

    def _windows(self, f, sig):
        """states simulated with f, restarted from the reference every `window` steps. An unstable
        system (inverted pendulum, plant without its controller) amplifies round-off differences
        without bound in one long open-loop run; over a short window a correct f stays at round-off
        and a wrong one does not."""
        X_ref, U, t = self.X[sig], self.U[sig], self.t
        out = np.full_like(X_ref, np.nan)
        out[0] = X_ref[0]
        for k0 in range(0, self.n, self.window):
            k1 = min(k0 + self.window, self.n)
            if np.all(np.isfinite(X_ref[k0])):
                out[k0 + 1:k1 + 1] = integrate(f, self._tab, X_ref[k0], t[k0:k1 + 1], U[k0:k1 + 1])[1:]
        out[~np.isfinite(X_ref).all(axis=1)] = np.nan    # rows that are not compared
        return out

    def check(self, f):
        """Compare a draft f(t, x, u) with the reference -> json-serialisable dict."""
        np_err = np.seterr(all='ignore')
        try:
            traj = {sig: _compare(self.X[sig], self._windows(f, sig), self.names, self.tol) for sig in self.U}
            D_got = np.array([f(tt, x, u) for tt, x, u in self.pts]).reshape(len(self.pts), self.nx)
        finally:
            np.seterr(**np_err)
        deriv = _compare(self.D, D_got, self.names, self.tol)
        passed = all(v['passed'] for v in deriv.values()) and \
            all(v['passed'] for sg in traj.values() for v in sg.values())
        worst = None
        bad = [j for j, nme in enumerate(self.names) if not deriv[nme]['passed']]
        if bad:
            err = np.abs(D_got - self.D)
            err = np.where(np.isfinite(err), err, np.where(np.isfinite(D_got) == np.isfinite(self.D), 0.0, np.inf))[:, bad]
            i, jj = np.unravel_index(np.argmax(err), err.shape)
            tt, x, u = self.pts[i]
            worst = dict(state=self.names[bad[jj]], index=int(bad[jj]), t=float(tt), x=[float(v) for v in x],
                         u=[float(v) for v in u], expected_dx=[float(v) for v in self.D[i]],
                         got_dx=[float(v) for v in D_got[i]])
            for j in bad:                                # "reference = a * yours + b": only a constant is wrong
                rel = _relation(self.D[:, j], D_got[:, j])
                if rel:
                    deriv[self.names[j]]['relation'] = rel
        return dict(passed=bool(passed), pure_continuous=bool(self.pure), tolerance=self.tol, steps=self.n,
                    dt=self.h, tf=float(self.t[-1]), solver=self.method, sample_points=len(self.pts),
                    derivative=deriv, trajectory=traj, worst=worst, reference_diverges_at=dict(self.diverged),
                    reference='generated python model' if self.pure else
                    'generated python model with its discrete memory (self.DW ...) frozen at the initial values',
                    tolerance_meaning='passed if RMSE / range(reference) <= tolerance (or max |error| <= %g) '
                                      'for every state, in the derivative check and in every simulation (restarted from the '
                                      'reference every %d steps)' % (ABS_TOL, self.window))


def _g(v):
    return '%.12g' % v


def constant_hint(given, code):
    """Constants in the equations of a draft whose value does not occur in the given code (as is,
    negated or inverted) -> text for the agent, or ''. Catches the typical slip of a rewrite:
    a folded number unfolded wrongly (0.002 written as 2*K/m instead of K/m)."""
    try:
        def fn_of(src):
            return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
        G = {float(n.value) for n in ast.walk(fn_of(given)) if isinstance(n, ast.Constant)
             and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
        G |= {float(v) for v in extract_card(given, 4096, 4096)['parameters'].values()
              if isinstance(v, (int, float))}
        par = {k: float(v) for k, v in extract_card(code, 4096, 4096)['parameters'].items()
               if isinstance(v, (int, float))}
        fn = fn_of(code)
        defs = {st.targets[0].id: ast.unparse(st.value) for st in fn.body if isinstance(st, ast.Assign)
                and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and st.targets[0].id in par}
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
               ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}

        def const(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                return float(n.value)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in par:
                return par[n.id]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                v = const(n.operand)
                return None if v is None else (-v if isinstance(n.op, ast.USub) else v)
            if isinstance(n, ast.BinOp) and type(n.op) in ops:
                a, b = const(n.left), const(n.right)
                if a is not None and b is not None:
                    try:
                        return float(ops[type(n.op)](a, b))
                    except Exception:
                        return None
            return None
        found = {}

        def visit(n):
            v = const(n)
            if v is not None:
                if not isinstance(n, ast.Constant):      # a bare number is a copy of the given one
                    found.setdefault(ast.unparse(n), v)
                return
            for c in ast.iter_child_nodes(n):
                visit(c)
        for st in fn.body:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id in par):
                visit(st)

        def known(v):
            c = [v, -v] + ([1.0 / v, -1.0 / v] if v else [])
            return any(math.isclose(a, g, rel_tol=1e-9, abs_tol=1e-300) for a in c for g in G)
        odd = ['%s = %s' % (k if k not in defs or defs[k] == _g(v) else '%s = %s' % (k, defs[k]), _g(v))
               for k, v in found.items() if math.isfinite(v) and not known(v)]
        if not odd:
            return ''
        nums = sorted(g for g in G if g not in (0.0, 1.0, 2.0, 3.0) and abs(g) < 1e15)
        return ('Constants in your equations whose value does not occur in the given code (not as is, not negated, '
                'not inverted) - check these first:\n' + '\n'.join('  ' + o for o in odd[:8]) +
                ('\nNumbers of the given code: ' + ', '.join(_g(g) for g in nums[:40]) if nums else ''))
    except Exception:
        return ''


def failure_report(val, facts, given=None, code=None):
    """what the agent is told when a draft fails the validation
    (given / code: the exact translation and the draft, for the hint about constants)"""
    L = ['The numerical comparison with the generated model FAILED (tolerance %g, relative to the '
         'range of the reference).' % val['tolerance'], '', 'Derivative check, dx[i] over %d sample points:'
         % val['sample_points']]
    for i, (n, d) in enumerate(val['derivative'].items()):
        L.append('  dx[%d] (%s): %s   RMSE/range = %.3g   max |error| = %.3g%s'
                 % (i, n, 'ok  ' if d['passed'] else 'WRONG', d['nrmse'], d['max_abs_error'],
                    '   (NaN/Inf where the reference is finite, or the reverse)' if d['mismatched_nonfinite'] else ''))
    w = val.get('worst')
    if w:
        L += ['', 'Worst sample (dx[%d], %s):' % (w['index'], w['state']), '  t = %r' % w['t'],
              '  x = %r' % w['x'], '  u = %r' % w['u'], '  reference dx = %r' % w['expected_dx'],
              '  your dx      = %r' % w['got_dx']]
        for i, d in enumerate(val['derivative'].values()):
            if d.get('relation'):
                a, b = d['relation']
                L += ['', 'dx[%d]: at EVERY sample point  reference = %s * (your dx[%d])%s.  So the form of your '
                      'equation is right and a constant in it is wrong%s.'
                      % (i, _g(a), i, '' if abs(b) < 1e-12 else ' %s %s' % ('+' if b > 0 else '-', _g(abs(b))),
                         ' (your variable term is %s times too large)' % _g(1.0 / a) if a else '')]
        hint = constant_hint(given, code) if given and code else ''
        if hint:
            L += ['', hint]
    elif not all(v['passed'] for s in val['trajectory'].values() for v in s.values()):
        L += ['', 'The derivatives agree at the sample points, but the simulated states drift apart:']
        for sig, s in val['trajectory'].items():
            for n, d in s.items():
                if not d['passed']:
                    L.append('  input "%s", state %s: RMSE/range = %.3g' % (sig, n, d['nrmse']))
        L.append('Look for a term that only matters away from the sampled points (a saturation / switch '
                 'branch, a time dependent block) or a rounded constant.')
    L += ['', 'Remember: x order = %s' % facts['states'], 'u order = %s' % facts['inputs']]
    if not val['pure_continuous']:
        L.append('Reference = the model with its memory frozen at the initial values (MEMORY section of FACTS).')
    return '\n'.join(L)


# ======================================================================================
# 5. JSON card: read back from the validated code
# ======================================================================================
def _comments(code):
    out = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string.lstrip('#').strip()
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def _index(node, var):
    """i of  var[i]  (None if node is something else)"""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == var:
        s = node.slice
        if isinstance(s, ast.Constant) and isinstance(s.value, int):
            return s.value
    return None


def _pretty(src):
    return re.sub(r'\b(np|numpy|math)\.', '', ' '.join(src.split())).replace('^', '**')


def _jsonable(v):
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, float, np.number)):
        v = float(v)
        return (int(v) if v.is_integer() and abs(v) < 1e15 else v) if math.isfinite(v) else None
    if isinstance(v, np.ndarray) and v.dtype.kind in 'fiu' and v.size <= 400 and np.all(np.isfinite(v)):
        return v.tolist()
    return None


def extract_card(code, nx, nu):
    """states / inputs / parameters / state equations as they are written in dynamics().
    Entries that are not written in the expected form come back as "not extractable"."""
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
    cm = _comments(code)
    states, s_mean = [NA] * nx, [NA] * nx
    inputs, i_mean = [NA] * nu, [NA] * nu
    eqs = [NA] * nx
    params, p_mean = {}, {}

    stores = {}                                         # name -> 1 if assigned exactly once and never modified
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            tg = [(tt, 1 if isinstance(tt, ast.Name) else 2) for tt in node.targets]
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            tg = [(node.target, 2)]
        else:
            continue
        for tt, w in tg:
            for e in ast.walk(tt):
                if isinstance(e, ast.Name):
                    stores[e.id] = stores.get(e.id, 0) + w
    dx_name = next((n.value.id for n in ast.walk(fn) if isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Name)), 'dx')
    ns = {'np': np, 'numpy': np, 'math': math}

    for st in fn.body:
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1):
            continue
        tgt, val, note = st.targets[0], st.value, cm.get(st.lineno, '')
        if isinstance(tgt, ast.Name):
            i, k = _index(val, 'x'), _index(val, 'u')
            if i is not None and 0 <= i < nx:
                states[i], s_mean[i] = tgt.id, note or NA
            elif k is not None and 0 <= k < nu:
                inputs[k], i_mean[k] = tgt.id, note or NA
            elif tgt.id == dx_name:
                if isinstance(val, ast.Call) and val.args and isinstance(val.args[0], (ast.List, ast.Tuple)) \
                        and len(val.args[0].elts) == nx:                   # dx = np.array([e0, e1, ...])
                    eqs = [_pretty(ast.get_source_segment(code, e) or NA) for e in val.args[0].elts]
            elif stores.get(tgt.id) == 1:
                used = {e.id for e in ast.walk(val) if isinstance(e, ast.Name)}
                if used <= set(ns) | set(params):       # only numbers and earlier parameters
                    try:
                        v = eval(compile(ast.Expression(val), '<param>', 'eval'), dict(ns), dict(params))
                    except Exception:
                        continue
                    if _jsonable(v) is not None:
                        params[tgt.id] = _jsonable(v)
                        p_mean[tgt.id] = note or NA
        else:
            i = _index(tgt, dx_name)
            if i is not None and 0 <= i < nx:
                eqs[i] = _pretty(ast.get_source_segment(code, val) or NA)
    return dict(states=states, state_meanings=s_mean, inputs=inputs, input_meanings=i_mean,
                state_equations=eqs, parameters=params, parameter_meanings=p_mean)


def unused_parameters(code):
    """parameters that are declared in dynamics() but never used: their value takes no part in
    the numerical validation, so it must not be reported as a validated parameter."""
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return [p for p in extract_card(code, 4096, 4096)['parameters'] if p not in used]


def style_report(code):
    """rules that the numerical test cannot see -> text for the agent (None if the draft is fine)"""
    unused = unused_parameters(code)
    if not unused:
        return None
    return ('The numbers are right, but the draft breaks rule 3: the parameter%s %s %s declared and never '
            'used - the equations contain bare numbers instead. Rewrite the intermediate and dx lines with '
            'the parameter names (unfold folded constants, e.g. K/m instead of 0.002, (v - R*i)/L instead of '
            '(v - i)/0.01), or delete a parameter that really does not occur in the dynamics. The result '
            'must still match the reference.' % ('' if len(unused) == 1 else 's', ', '.join(unused),
                                                 'is' if len(unused) == 1 else 'are'))


def model_names(text):
    """every identifier that occurs in the given code or its comments = the names of the model"""
    return set(re.findall(r'[A-Za-z_]\w*', text))


def foreign_names(code, vocab, nx, nu):
    """[(x / u, index, name)] of the states / inputs whose name is not a name of the model"""
    d = extract_card(code, nx, nu)
    return [(sym, i, n) for sym, names in (('x', d['states']), ('u', d['inputs']))
            for i, n in enumerate(names) if n != NA and n not in vocab]


def naming_report(code, vocab, nx, nu, given_names):
    """rule 7 (names of the model) -> text for the agent, None if the draft is fine.
    given_names: {'x': [...], 'u': [...]} the names the exact translation uses"""
    bad, L = foreign_names(code, vocab, nx, nu), []
    if bad:
        L.append('%s not %s of the model. Use the name the model itself has for it - the variable name in the '
                 'source lines of the MATLAB Function block if there is one, else the name in the given code (%s). '
                 'The meaning belongs in the comment.'
                 % (', '.join('`%s` (%s[%d])' % (n, sym, i) for sym, i, n in bad), 'a name' if len(bad) == 1 else 'names',
                    ', '.join('%s[%d] is `%s`' % (sym, i, given_names[sym][i]) for sym, i, n in bad)))
    helpers = helper_parameters(code, vocab)
    if helpers:
        L.append('%s: the model has no such quantity - write the expression where it is used (K/m, not k = K/m) '
                 'and delete the line.' % ', '.join('`%s = %s`' % h for h in helpers))
    if not L:
        return None
    return ('The numbers are right, but the draft breaks rule 7 (names of the model). ' + ' '.join(L) +
            ' Change nothing else; the result must still match the reference.')


def helper_parameters(code, vocab):
    """[(name, expression)] of parameter lines that combine other parameters under a name the
    model does not have (magnetic_accel_gain = K / m)"""
    fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'dynamics')
    par = extract_card(code, 4096, 4096)['parameters']
    return [(st.targets[0].id, ast.unparse(st.value)) for st in fn.body
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id in par and st.targets[0].id not in vocab
            and any(isinstance(n, ast.Name) and n.id in par for n in ast.walk(st.value))]


def use_model_names(code, vocab, nx, nu, given_names):
    """states / inputs with a made-up name get the name of the exact translation back
    -> (code, [(made-up, model name)])"""
    used = set(re.findall(r'[A-Za-z_]\w*', code))
    mp = {}
    for sym, i, n in foreign_names(code, vocab, nx, nu):
        new = given_names[sym][i]
        if new not in used and new not in mp.values():
            mp[n] = new
    if not mp:
        return code, []
    code = re.sub(r'\b(%s)\b' % '|'.join(re.escape(k) for k in mp), lambda m: mp[m.group(1)], code)
    return code, list(mp.items())


def system_type(nu, ny):
    if nu == 0:
        return 'autonomous (no inputs), %d output%s' % (ny, '' if ny == 1 else 's')
    return ('S' if nu == 1 else 'M') + 'I' + ('S' if ny == 1 else 'M') + 'O'


def _text(v):
    return v.strip() if isinstance(v, str) and v.strip() else NA


def build_card(name, answer, code, facts, drafts, val, outputs=None, meanings=None, method=None):
    """outputs: per output a text or None (None -> the answer of the LLM);
    meanings: {name: text} that replaces the meanings read from the code comments."""
    nx, nu, ny = len(facts['states']), len(facts['inputs']), len(facts['outputs'])
    warnings = []
    try:
        d = extract_card(code, nx, nu)
        unused = unused_parameters(code)
        if unused:                                      # not part of the validated equations -> not reported
            for p in unused:
                d['parameters'].pop(p, None)
                d['parameter_meanings'].pop(p, None)
            warnings.append('declared but unused in the equations, left out of "parameters": ' + ', '.join(unused))
    except Exception:
        d = dict(states=[NA] * nx, state_meanings=[NA] * nx, inputs=[NA] * nu, input_meanings=[NA] * nu,
                 state_equations=[NA] * nx, parameters={}, parameter_meanings={})
    outs = answer.get('outputs')
    outs = [_text(o) for o in outs] if isinstance(outs, list) and len(outs) == ny else [NA] * ny
    if outputs:
        outs = [o if o else outs[i] for i, o in enumerate(outputs)]
    if meanings is not None:
        def mean(n):
            return _text(meanings.get(n)) if isinstance(meanings, dict) else NA
        d['state_meanings'] = [mean(n) for n in d['states']]
        d['input_meanings'] = [mean(n) for n in d['inputs']]
        d['parameter_meanings'] = {n: mean(n) for n in d['parameters']}
    card = {
        'system': _text(answer.get('system')),
        'model': name,
        'drafts': drafts,
        'status': (STATUS_OK if val['pure_continuous'] else STATUS_PART) if val and val['passed'] else STATUS_FAIL,
        'artifact': 'dynamics(t, x, u) · Python',
        'method': method or NA,
        'model_details': {
            'states': d['states'],
            'state_meanings': d['state_meanings'],
            'inputs': d['inputs'],
            'input_meanings': d['input_meanings'],
            'outputs': outs,
            'state_equations': d['state_equations'],
            'parameters': d['parameters'] if d['parameters'] else NA,
            'parameter_meanings': d['parameter_meanings'] if d['parameters'] else NA,
            'system_type': system_type(nu, ny),
        },
        'source_names': {'states': facts['states'], 'inputs': facts['inputs'], 'outputs': facts['outputs']},
    }
    if _text(answer.get('notes')) != NA:
        card['notes'] = answer['notes'].strip()
    if warnings:
        card['warnings'] = warnings
    if val:
        card['validation'] = {
            'passed': val['passed'], 'pure_continuous': val['pure_continuous'], 'reference': val['reference'],
            'tolerance': val['tolerance'], 'tolerance_meaning': val['tolerance_meaning'],
            'solver': val['solver'], 'dt': val['dt'], 'tf': val['tf'], 'sample_points': val['sample_points'],
            'derivative_max_nrmse': _worst(val['derivative'].values()),
            'trajectory_max_nrmse': {s: _worst(v.values()) for s, v in val['trajectory'].items()},
        }
    return card


def _worst(items):
    v = [d['nrmse'] for d in items if math.isfinite(d['nrmse'])]
    return max(v) if v else None


# ======================================================================================
# 6. the agent loop
# ======================================================================================
def parse_json(text):
    """the JSON object of an answer (tolerates ```json fences / text around it)"""
    text = (text or '').strip()
    try:
        d = json.loads(text)
    except ValueError:
        m = re.search(r'\{.*\}', text, re.S)
        if not m:
            raise ValueError('the answer is not a JSON object')
        d = json.loads(m.group(0))
    if not isinstance(d, dict):
        raise ValueError('the answer is not a JSON object')
    return d


class _Session(object):
    """the LLM + the bookkeeping of one agent run (files, cost lines). The calls of the code agent
    are numbered draft_1, draft_2 ...; the calls of the JSON agent json_1, json_2 ..."""

    def __init__(self, llm, log, log_dir):
        self.llm, self.log, self.dir, self.n, self.count = llm, log, log_dir, 0, {}

    def ask(self, messages, what, tag='draft'):
        k = self.count[tag] = self.count.get(tag, 0) + 1
        if tag == 'draft':
            self.n = k                                   # number of code drafts
        self.tag = '%s_%d' % (tag, k)
        (self.dir / (self.tag + '_prompt.txt')).write_text(
            '\n\n'.join('### %s\n%s' % (m['role'], m['content']) for m in messages), encoding='utf-8')
        who = '[agent]' if tag == 'draft' else '[agent json]'
        self.log('%s %s %d (%s): asking the LLM ...' % (who, tag, k, what))
        t0 = time.time()
        try:
            text = self.llm(messages)
        except Exception as ex:                          # network / key / model name / quota ...
            raise SystemExit('%s the LLM request failed: %s: %s' % (who, type(ex).__name__, ex))
        usage = getattr(self.llm, 'usage', None)
        cost = cost_report(getattr(self.llm, 'model', None), usage) if usage else None
        self.log('%s %s %d: answer received (%.1fs)%s' % (who, tag, k, time.time() - t0, '' if not cost else
                 '   tokens: %d in + %d out   cost: %s' % (usage[-1][0], usage[-1][1], _usd(cost['calls'][-1]['cost_usd']))))
        (self.dir / (self.tag + '_answer.txt')).write_text(text or '', encoding='utf-8')
        return text


def _rewrite(ses, messages, ref, facts, nx, n_max, flat=None, vocab=None):
    """LLM writes the whole function; a failed validation goes back to it (n_max drafts).
    flat: the exact translation the LLM was given (for the hints of a failed draft);
    vocab: the names of the model (None: names are free) - a draft whose states / inputs have
    other names goes back too, and after the last draft the program puts the model's names back.
    -> (answer, code, val) of the best draft, or None if no draft could even be run"""
    best, first = None, list(messages)
    given = flat.emit() if flat is not None else None
    nu = len(facts['inputs'])
    gnames = {'x': [n for n, _ in flat.states], 'u': [n for n, _ in flat.inputs]} if flat is not None else None
    strict = vocab is not None and gnames is not None
    for k in range(1, n_max + 1):
        text = ses.ask(messages, 'rewrite %d/%d' % (k, n_max))
        val = None
        try:
            answer = parse_answer(text)
            code = answer['python_code']
            (ses.dir / (ses.tag + '_dynamics.py')).write_text(code, encoding='utf-8')
            val = ref.check(load_dynamics(code, nx))
            if not val['passed']:
                report = failure_report(val, facts, given, code)
            else:
                rs = [r for r in (style_report(code), naming_report(code, vocab, nx, nu, gnames) if strict else None) if r]
                report = '\n\n'.join(rs) or None
            if val['passed'] or best is None or not best[2]['passed']:
                best = (answer, code, val)               # a numerically correct draft is never replaced by a wrong one
        except Exception as ex:                          # not JSON / not runnable / wrong shape ...
            report = 'The draft could not be used: %s: %s' % (type(ex).__name__, ex)
        (ses.dir / (ses.tag + '_validation.json')).write_text(
            json.dumps(dict(val, style=report if val['passed'] else None) if val
                       else {'passed': False, 'error': report}, indent=2), encoding='utf-8')
        if report is None:
            ses.log('[agent] draft %d: PASS  (max RMSE/range: derivatives %.2e, simulations %.2e)'
                    % (ses.n, _worst(val['derivative'].values()) or 0.0,
                       max([_worst(v.values()) or 0.0 for v in val['trajectory'].values()] or [0.0])))
            break
        ses.log('[agent] draft %d: %s\n%s' % (ses.n, 'numerically OK, but sent back' if val and val['passed'] else 'FAIL',
                                              '\n'.join('        ' + ln for ln in report.splitlines()[:24])))
        # only the last draft goes back (the older ones would just be paid for again)
        messages = first + [{'role': 'assistant', 'content': text or ''},
                            {'role': 'user', 'content': RETRY_TEMPLATE.format(n=k, report=report)}]
    if strict and best is not None and best[2]['passed']:       # names the LLM made up after all
        try:
            code2, changed = use_model_names(best[1], vocab, nx, nu, gnames)
            if changed:
                val2 = ref.check(load_dynamics(code2, nx))
                if val2['passed']:
                    mp = dict(changed)
                    outs = best[0].get('outputs')
                    if isinstance(outs, list):
                        best[0]['outputs'] = [re.sub(r'\b(%s)\b' % '|'.join(re.escape(k) for k in mp),
                                                     lambda m: mp[m.group(1)], o) if isinstance(o, str) else o for o in outs]
                    best = (best[0], code2, val2)
                    ses.log('[agent] made-up names replaced by the names of the model: '
                            + ', '.join('%s -> %s' % c for c in changed))
        except Exception:
            pass
    return best


def _annotate(ses, flat, name, facts, ref, nx, extra, naming='model'):
    """LLM only chooses names and meanings; they are applied to the exact translation by program.
    naming 'model': a new name is only accepted if it is a name of the model (it occurs in the
    code or its comments); a state / input may take over the name of its copy (Flat.adopt).
    -> (answer, code, val, rename, flat). Always returns validated code (the plain translation at worst)."""
    code0, flat0 = flat.emit(), flat
    outs = flat.output_text()
    vocab = model_names(code0)
    messages = [{'role': 'system', 'content': annotate_prompt(naming)},
                {'role': 'user', 'content': FLAT_USER.format(
                    name=name, nx=nx, nu=len(facts['inputs']), extra=extra, code=code0,
                    outputs=', '.join(o or '?' for o in outs) or '(none)')}]
    first = list(messages)
    for k in (1, 2):
        text = ses.ask(messages, 'names and meanings%s' % ('' if k == 1 else ', 2nd try'))
        try:
            ans = parse_json(text)
            rename = ans.get('rename') if isinstance(ans.get('rename'), dict) else {}
            meaning = ans.get('meaning') if isinstance(ans.get('meaning'), dict) else {}
            ans['meaning'] = meaning = {k: v for k, v in meaning.items() if isinstance(k, str)}
            flat, kinds, keep, adopted, refused = flat0, flat0.names(), {}, [], []
            for old, new in rename.items():
                if not (isinstance(old, str) and isinstance(new, str)) or old == new:
                    continue
                if kinds.get(old) in ('state', 'input') and kinds.get(new) == 'signal':
                    f2 = flat.adopt(old, new)            # the state takes the name of its copy
                    if f2 is not None and ref.check(load_dynamics(f2.emit(), nx))['passed']:
                        flat = f2
                        adopted.append(new)
                        if old in meaning:
                            meaning.setdefault(new, meaning[old])
                        continue
                if naming != 'free' and new not in vocab:
                    refused.append(new)
                    if new in meaning:                   # the meaning it gave under its own name is kept
                        meaning.setdefault(old, meaning[new])
                    continue
                keep[old] = new
            rn = flat.clean_rename(keep)
            code = flat.emit(rn, meaning)
            (ses.dir / (ses.tag + '_dynamics.py')).write_text(code, encoding='utf-8')
            val = ref.check(load_dynamics(code, nx))
            if not val['passed']:
                raise ValueError('the renamed code does not pass the validation')
            ses.log('[agent] draft %d: PASS  (%d of %d renames applied, %d meanings)'
                    % (ses.n, len(rn) + len(adopted), len(rename), len(meaning)))
            if refused:
                ses.log('[agent] %d name%s refused - not %s of the model, the original name%s kept: %s'
                        % (len(refused), '' if len(refused) == 1 else 's', 'a name' if len(refused) == 1 else 'names',
                           ' is' if len(refused) == 1 else 's are', ', '.join(refused[:12])
                           + (' ...' if len(refused) > 12 else '')))
            return ans, code, val, rn, flat
        except Exception as ex:
            report = 'Your answer could not be used: %s: %s' % (type(ex).__name__, ex)
            ses.log('[agent] draft %d: FAIL\n        %s' % (ses.n, report))
            messages = first + [{'role': 'assistant', 'content': text or ''},
                                {'role': 'user', 'content': report + '\nAnswer again with the complete JSON object.'}]
    ses.log('[agent] no usable names from the LLM -> the exact translation is delivered with its original names')
    return {}, code0, ref.check(load_dynamics(code0, nx)), {}, flat0


# ======================================================================================
# 7. the JSON card: equations traced back by program, definitions named by the JSON agent
# ======================================================================================


def primary_parameters(code, nx, nu, flat=None, rename=None):
    """{name: value} of the parameters that belong in the card. With the exact translation (flat)
    these are the model's own parameters (self.P ..., frozen memory, constant tables); constants
    the code generator made up from them (Tsamp = Ppos2_N * Tsamp_WtEt, d_54 = -100) are NOT
    parameters - they are traced back like every other temporary."""
    params = extract_card(code, nx, nu)['parameters']
    if flat is None:
        return params
    rn = rename or {}
    keep = {rn.get(n, n) for n, v, c in flat.params
            if c.startswith('self.') or (isinstance(v, ast.Call) and getattr(v.func, 'attr', '') == 'array')}
    return {k: v for k, v in params.items() if k in keep}


def card_equations(card, code, ref, cfg, primary, log=print):
    """Trace the temporaries of the validated code back to states / inputs / parameters
    (src/equations.py, no LLM), check every equation numerically, and put the result in the card.
    Returns the Equations object (None if the code does not have the expected layout)."""
    md = card['model_details']
    nx, nu = len(md['states']), len(md['inputs'])
    try:
        E = equations.build(code, nx, nu, primary, int(cfg.get('json_max_terms', 30)))
        if NA in E.states or NA in E.inputs:
            raise ValueError('the code does not name every state / input')
        ok, bad = equations.verify(E, code, ref.pts, ref.D, _compare, cfg.get('agent_tol', 1e-6))
    except Exception as ex:
        log('[json] the equations could not be traced back (%s: %s) - the card shows the lines of the code'
            % (type(ex).__name__, ex))
        return None
    E.ok = ok
    put_equations(card, E, {})
    log('[json] equations traced back to states / inputs / parameters: %d of %d reproduce the model, '
        '%d definitions, %d quantities computed by an algorithm in the code'
        % (sum(ok), nx, len(E.defs), len(E.code_vars)))
    return E


def put_equations(card, E, meaning):
    md = card['model_details']
    md['parameters'] = dict(E.params) if E.params else NA
    if E.params:
        old = md.get('parameter_meanings') if isinstance(md.get('parameter_meanings'), dict) else {}
        md['parameter_meanings'] = {n: old.get(n, NA) for n in E.params}
    md['definitions'] = [dict(name=d['name'], expression=E.text(d['node']),
                              meaning=_text(meaning.get(d['name']) or d['meaning'])) for d in E.defs]
    md['computed_in_code'] = [dict(name=n, meaning=_text(meaning.get(n) or i['meaning']), computed_from=i['reads'])
                              for n, i in E.code_vars.items()]
    md['state_equations'] = [E.text(E.eqs[k]) if E.ok[k] else E.raw[k] for k in range(len(E.eqs))]
    md['state_equations_checked'] = [bool(v) for v in E.ok]
    card['json_method'] = ('state equations and definitions: derived by program from the validated code (every '
                           'temporary traced back to states / inputs / parameters) and checked numerically against '
                           'the model - %d of %d reproduce it (state_equations_checked); "computed_in_code" lists '
                           'what is computed by loops / matrix algorithms and has no formula'
                           % (sum(E.ok), len(E.ok)))


def describe_agent(ses, name, card, E, code, ref, cfg):
    """Second agent: reads the card (not the code) and explains the definitions. With
    cfg['agent_naming'] = 'free' it may also rename them; its names are applied by program and
    the equations are checked again, so it cannot break them."""
    md = card['model_details']

    def q(v):
        return '?' if v in (NA, '', None) else v
    pm = md.get('parameter_meanings') if isinstance(md.get('parameter_meanings'), dict) else {}
    L = ['MODEL NAME: %s' % name, '', 'STATES']
    L += ['  %s  -  %s' % (n, q(m)) for n, m in zip(md['states'], md['state_meanings'])]
    L += ['INPUTS'] + ['  %s  -  %s' % (n, q(m)) for n, m in zip(md['inputs'], md['input_meanings'])]
    L += ['PARAMETERS'] + ['  %s = %s  -  %s' % (n, json.dumps(v), q(pm.get(n))) for n, v in E.params.items()]
    L += ['DEFINITIONS (in order of evaluation;  name = formula   # hint: where it comes from in the model)']
    L += ['  %s = %s   # %s' % (d['name'], E.text(d['node']), d['meaning'] or '-') for d in E.defs]
    if E.code_vars:
        L += ['COMPUTED IN THE CODE (loops / matrix algorithms, no formula;  name  <- what it is computed from   # hint)']
        L += ['  %s  <- %s   # %s' % (n, ', '.join(i['reads']) or '-', i['meaning'] or '-') for n, i in E.code_vars.items()]
    L += ['STATE EQUATIONS']
    L += ['  d(%s)/dt = %s' % (n, e) for n, e in zip(md['states'], md['state_equations'])]
    free = cfg.get('agent_naming', 'model') == 'free'      # else: the card keeps the names of the code
    messages = [{'role': 'system', 'content': DESCRIBE_PROMPT if free else DESCRIBE_PROMPT_MODEL},
                {'role': 'user', 'content': '\n'.join(L)}]
    first = list(messages)
    for k in (1, 2):
        text = ses.ask(messages, 'names and meanings of the definitions%s' % ('' if k == 1 else ', 2nd try'), tag='json')
        try:
            ans = parse_json(text)
            rename = ans.get('rename') if free and isinstance(ans.get('rename'), dict) else {}
            meaning = ans.get('meaning') if isinstance(ans.get('meaning'), dict) else {}
            break
        except Exception as ex:
            ses.log('[agent json] json %d: FAIL - %s: %s' % (k, type(ex).__name__, ex))
            messages = first + [{'role': 'assistant', 'content': text or ''},
                                {'role': 'user', 'content': 'That was not a JSON object. Answer again with the JSON object only.'}]
    else:
        return card
    E2 = copy.deepcopy(E)
    applied = E2.rename(rename)
    ok2, _ = equations.verify(E2, code, ref.pts, ref.D, _compare, cfg.get('agent_tol', 1e-6))
    if applied and ok2 == E.ok:                          # (always true - the names do not touch the formulas)
        E2.ok = ok2
        E = E2
    else:
        applied = {}
    put_equations(card, E, meaning)
    for key, names in (('state_meanings', md['states']), ('input_meanings', md['inputs'])):
        md[key] = [_text(meaning.get(n)) if m == NA and n in meaning else m for n, m in zip(names, md[key])]
    if isinstance(md.get('parameter_meanings'), dict):
        for n, m in md['parameter_meanings'].items():
            if m == NA and n in meaning:
                md['parameter_meanings'][n] = _text(meaning[n])
    if _text(ans.get('system')) != NA:
        card['system'] = ans['system'].strip()
    if _text(ans.get('notes')) != NA:
        card['notes'] = (card.get('notes', '') + ' ' + ans['notes'].strip()).strip()
    card['json_drafts'] = ses.count.get('json', 0)
    ses.log('[agent json] %s%d meanings' % ('%d of %d definitions renamed, ' % (len(applied), len(E.defs)) if free else '',
                                            len(meaning)))
    return card


def run(name, py_path, out_dir, cfg, llm=None, log=print):
    """generated model  ->  <out_dir>/<name>_dynamics.py + <name>_dynamics.json

    1. reference: the generated model itself (Reference)
    2. exact translation without LLM (src/flatten.py), validated against the reference
    3. the LLM makes it readable, cfg['agent_mode']:
         'rewrite'   it rewrites the function as textbook equations; every draft is validated,
                     a failed one goes back to it (cfg['agent_max_attempts'] drafts)
         'annotate'  it only returns names + meanings, which are applied by program: one call,
                     few output tokens, the code stays exact
         'auto'      rewrite if the exact translation has <= cfg['agent_rewrite_max_lines'] lines,
                     else annotate
       If a rewrite fails, the annotate step is used; if step 2 is not possible for a model, the
       LLM has to rewrite the generated class itself (the old behaviour).
       cfg['agent_naming'] = 'model' (default): every name is a name of the model itself - the
       LLM adds meanings, it does not make up names (enforced by program); 'free': it may.
    Every prompt / answer / validation result is kept in <out_dir>/agent/.
    llm: callable(messages) -> text; default = OpenAI chat completions with cfg['agent_model'].
    Returns the JSON card; card['status'] is STATUS_OK / STATUS_PART on success, else STATUS_FAIL."""
    py_path, out_dir = Path(py_path), Path(out_dir)
    log_dir = out_dir / 'agent'
    log_dir.mkdir(parents=True, exist_ok=True)
    for old in list(log_dir.glob('draft_*')) + list(log_dir.glob('json_*')):      # files of an earlier run
        old.unlink()
    if llm is None:                                     # first: fail early if the key / SDK is missing
        llm = openai_llm(cfg.get('agent_model') or os.environ.get('OPENAI_MODEL') or 'gpt-5.4-mini')
    ses = _Session(llm, log, log_dir)
    Model = validation.load_model_class(py_path)
    solver = cfg.get('solver', 'auto')
    py_src = py_path.read_text(encoding='utf-8', errors='replace')
    facts = model_facts(Model, py_src, solver)
    nx = len(facts['states'])
    log('[agent] %d states, %d inputs, %d outputs' % (nx, len(facts['inputs']), len(facts['outputs'])))
    if nx == 0:
        raise SystemExit('[agent] the model has no continuous states -> there is no dx/dt = f(t, x, u) to write')
    log('[agent] preparing the reference (simulating the python model) ...')
    t0 = time.time()
    ref = Reference(Model, cfg, solver)
    facts['pure'] = ref.pure
    log('[agent] reference ready (%.1fs): %d steps of %g s, %d sample points' % (time.time() - t0, ref.n, ref.h, len(ref.pts)))
    if not ref.pure:
        log('[agent] NOTE: this model is not a pure dx/dt = f(t, x, u) - it has memory outside the continuous '
            'states\n        (%s).\n        No dynamics(t, x, u) can reproduce it exactly, so the continuous part is '
            'modelled with that memory frozen\n        at its initial value, and validated against the python model '
            'under the same convention.'
            % (', '.join(p.split(' = ')[0] for p in facts['memory'][:8]) + (' ...' if len(facts['memory']) > 8 else '')
               if facts['memory'] else 'blocks that hold their value between steps'))
    for sig, tt in ref.diverged.items():
        log('[agent] NOTE: with input "%s" the reference diverges (|x| > %g) - compared up to t = %g s' % (sig, BLOWUP, tt))

    # ---- 2. exact translation, no LLM
    flat, lines = None, None
    try:
        for webs in (True, False):                       # False: without splitting re-used scratch variables
            flat = flatten(Model, py_src, webs)
            base = flat.emit()
            v0 = ref.check(load_dynamics(base, nx))
            if v0['passed']:
                break
        (log_dir / 'exact_translation.py').write_text(base, encoding='utf-8')
        if v0['passed']:
            lines = len(flat.emit(comments=False).splitlines())
            log('[agent] exact translation (no LLM): PASS, %d lines' % lines)
        else:
            flat = None
            log('[agent] exact translation (no LLM): does not match the reference -> not used')
    except Unsupported as ex:
        log('[agent] exact translation (no LLM) not possible for this model: %s' % ex)
    except Exception as ex:
        flat = None
        log('[agent] exact translation (no LLM) failed: %s: %s' % (type(ex).__name__, ex))

    naming = 'free' if cfg.get('agent_naming', 'model') == 'free' else 'model'
    mode = cfg.get('agent_mode', 'auto')
    if flat is None:
        mode = 'rewrite'
    elif mode == 'auto':
        mode = 'rewrite' if lines <= int(cfg.get('agent_rewrite_max_lines', 60)) else 'annotate'
    log('[agent] mode: %s' % {'rewrite': 'rewrite - the LLM writes the equations, every draft is validated',
                               'annotate': 'annotate - the LLM only names things, the code stays the exact translation'}[mode])
    extra = '' if ref.pure else IMPURE_FLAT

    answer, code, val, method, outputs, meanings, report = {}, None, None, None, None, None, None
    if mode == 'rewrite':
        if flat is not None:
            messages = [{'role': 'system', 'content': rewrite_prompt(True, naming)},
                        {'role': 'user', 'content': FLAT_USER.format(
                            name=name, nx=nx, nu=len(facts['inputs']), extra=extra, code=flat.emit(),
                            outputs=', '.join(o or '?' for o in flat.output_text()) or '(none)')}]
        else:
            messages = build_messages(name, py_src, facts, naming)
            if len(messages[1]['content']) > 300000:
                log('[agent] WARNING: the model source is %d characters - it may not fit the context window of '
                    'the LLM' % len(messages[1]['content']))
        best = _rewrite(ses, messages, ref, facts, nx, int(cfg.get('agent_max_attempts', 3)), flat,
                        model_names(flat.emit()) if flat is not None and naming == 'model' else None)
        if best is not None and best[2]['passed']:
            answer, code, val = best
            method = 'LLM rewrite, validated against the generated python model'
        elif flat is not None:
            log('[agent] the rewrite did not pass -> keeping the exact translation and asking only for names')
            mode = 'annotate'
        elif best is not None:
            answer, code, val = best
            method = 'LLM rewrite (failed the validation)'
        else:
            report = 'no draft of the LLM could be run'
    if mode == 'annotate':
        answer, code, val, rn, flat = _annotate(ses, flat, name, facts, ref, nx, extra, naming)
        outputs = flat.output_text(rn)
        meanings = answer.get('meaning') if isinstance(answer.get('meaning'), dict) else {}
        method = 'exact translation of the generated code (no LLM)' + \
                 ('; names and meanings by the LLM' if (rn or meanings) else '; the LLM gave no usable names')

    py_out, json_out = out_dir / ('%s_dynamics.py' % name), out_dir / ('%s_dynamics.json' % name)
    if code is None:
        card = build_card(name, {}, 'def dynamics(t, x, u):\n    pass\n', facts, ses.n, None, method=method)
        card['error'] = report
    else:
        card = build_card(name, answer, code, facts, ses.n, val, outputs=outputs, meanings=meanings, method=method)
        card['naming'] = ('free: the LLM chooses the names' if naming == 'free' else
                          "model: every name is a name of the model (block / signal / parameter name, or the "
                          "variable name in the source of a MATLAB Function block); the LLM only adds the meanings")
    if card['status'] != STATUS_FAIL:                   # <name>_dynamics.py only ever holds validated code
        py_out.write_text(code, encoding='utf-8')
        log('[agent] %s' % py_out)
        # the card: equations traced back by program (free, exact), then named by the JSON agent
        prim = primary_parameters(code, nx, len(facts['inputs']), flat if mode == 'annotate' else None,
                                  rn if mode == 'annotate' else None)
        E = card_equations(card, code, ref, cfg, prim, log)
        if E is not None and cfg.get('agent_json', True) and (E.defs or E.code_vars or card['system'] == NA):
            describe_agent(ses, name, card, E, code, ref, cfg)
    else:
        if py_out.exists():
            py_out.unlink()
        log('[agent] no draft passed the validation - the drafts are in %s' % log_dir)
    for w in card.get('warnings', []):
        log('[agent] WARNING: ' + w)
    usage = getattr(llm, 'usage', None)
    if usage:
        cost = cost_report(getattr(llm, 'model', None), usage)
        (log_dir / 'cost.json').write_text(json.dumps(cost, indent=2), encoding='utf-8')
        p = cost['usd_per_million_tokens']
        log('[agent] COST: %s   (%d LLM call%s, %d input + %d output tokens, %s)'
            % (_usd(cost['cost_usd']), len(usage), '' if len(usage) == 1 else 's', cost['input_tokens'],
               cost['output_tokens'], '%s: $%g in / $%g out per 1M tokens' % (cost['model'], p['input'], p['output'])
               if p else 'no price for model "%s" in src/cost.py - add it to DEFAULT_PRICE_TABLE' % cost['model']))
    json_out.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding='utf-8')
    log('[agent] %s' % json_out)
    return card

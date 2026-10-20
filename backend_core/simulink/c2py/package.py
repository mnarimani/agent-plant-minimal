# -*- coding: utf-8 -*-
"""The upload package made by tools/c2py_export.m (on the user's computer, with MATLAB).

    <model>_c2py.zip
      <model>_c2py/
        c2py_manifest.json          model, solver, step size, list of files, test runs
        code/*.c, *.h               the C code of Embedded Coder (+ shared utilities)
        simulink/<model>_<sig>_input.csv      t, u1..um     the test input
        simulink/<model>_<sig>_simulink.csv   t, y1..yp     what Simulink answered

check(path)    -> Report: is it a complete package of ONE model? (errors / warnings, in words
                  the user can act on)
install(rep, out_dir)  -> copies the code to out_dir/codegen/<model>_ert_rtw and writes the
                  solver info for the validation; returns the Simulink data
                  {signal: (t, U, t_sim, Y_sim, error)} for validation.validate()
"""
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

FORMAT = 'c2py-package'
MANIFEST = 'c2py_manifest.json'
EXPORT_SCRIPT = 'c2py_export.m'

# headers that the generated code includes from the MATLAB installation, never generated
MATLAB_HEADERS = {
    'rtw_continuous.h', 'rtw_solver.h', 'solver_zc.h', 'zero_crossing_types.h', 'rt_logging.h',
    'rtw_extmode.h', 'ext_work.h', 'sysran_types.h', 'simstruc_types.h', 'rtw_matlogging.h',
    'tmwtypes.h', 'simstruc.h', 'rt_mxtypes.h', 'rtGetInf.h', 'rtGetNaN.h', 'rt_nonfinite.h',
}
C_STD_HEADERS = {'math.h', 'string.h', 'stddef.h', 'float.h', 'limits.h', 'stdlib.h', 'stdio.h',
                 'stdint.h', 'stdbool.h'}


class Report(object):
    def __init__(self, source):
        self.source = str(source)
        self.errors, self.warnings, self.info = [], [], []
        self.dir = None              # folder that holds the manifest
        self.manifest = None
        self.model = None
        self._tmp = None

    @property
    def ok(self):
        return not self.errors

    def text(self):
        L = ['package: %s' % self.source]
        if self.model:
            L.append('model  : %s' % self.model)
        L += ['  ' + s for s in self.info]
        L += ['  ERROR:   ' + s for s in self.errors]
        L += ['  WARNING: ' + s for s in self.warnings]
        L.append('  -> %s' % ('OK, the package can be used' if self.ok else
                              'NOT usable - fix the errors above and upload again'))
        return '\n'.join(L)

    def cleanup(self):
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


def _as_list(v):
    """jsonencode writes a struct array of one element as an object, not a list"""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _unpack(path, rep):
    """zip or folder -> the folder that holds the manifest (None if there is none)"""
    p = Path(path)
    if not p.exists():
        rep.errors.append('"%s" does not exist' % p)
        return None
    if p.is_file():
        if p.suffix.lower() in ('.slx', '.mdl'):
            rep.errors.append('this is the Simulink model itself. Put %s next to it, run it in MATLAB and upload '
                              'the <model>_c2py.zip it creates' % EXPORT_SCRIPT)
            return None
        if not zipfile.is_zipfile(p):
            rep.errors.append('"%s" is not a zip file. Upload the <model>_c2py.zip that %s creates'
                              % (p.name, EXPORT_SCRIPT))
            return None
        tmp = Path(tempfile.mkdtemp(prefix='c2py_pkg_'))
        rep._tmp = tmp
        with zipfile.ZipFile(p) as z:
            for m in z.infolist():
                target = (tmp / m.filename).resolve()
                if not str(target).startswith(str(tmp.resolve())):        # "../" or absolute paths
                    rep.errors.append('the zip contains an unsafe path: %s' % m.filename)
                    return None
            z.extractall(tmp)
        p = tmp
    found = sorted(p.rglob(MANIFEST), key=lambda f: len(f.parts))
    if not found:
        names = [f.name for f in p.rglob('*') if f.is_file()]
        if any(n.endswith('_ert_rtw') for n in (d.name for d in p.rglob('*') if d.is_dir())) \
                or any(n.lower().endswith(('.c', '.h')) for n in names):
            rep.errors.append('this looks like C code copied by hand: %s is missing. Use the <model>_c2py.zip '
                              'that %s creates - it also holds the Simulink results and the settings that '
                              'were used' % (MANIFEST, EXPORT_SCRIPT))
        else:
            rep.errors.append('%s not found - this is not a package made by %s' % (MANIFEST, EXPORT_SCRIPT))
        return None
    if len(found) > 1 and len(found[1].parts) == len(found[0].parts):
        rep.errors.append('the upload holds more than one package (%s) - upload one model at a time'
                          % ', '.join(sorted(f.parent.name for f in found)))
        return None
    return found[0].parent


def check(path, expected_model=None):
    """-> Report. expected_model: the model the user says this is (None = take it from the package)"""
    rep = Report(path)
    d = _unpack(path, rep)
    if d is None:
        return rep
    rep.dir = d
    try:
        man = json.loads((d / MANIFEST).read_text(encoding='utf-8'))
    except Exception as ex:
        rep.errors.append('%s cannot be read (%s) - the file was changed or cut off; run %s again'
                          % (MANIFEST, ex, EXPORT_SCRIPT))
        return rep
    if not isinstance(man, dict) or man.get('format') != FORMAT:
        rep.errors.append('%s is not a c2py package manifest' % MANIFEST)
        return rep
    rep.manifest = man
    model = man.get('model')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z]\w*', model):
        rep.errors.append('the manifest has no valid model name')
        return rep
    rep.model = model
    if expected_model and expected_model != model:
        rep.errors.append('this package is for the model "%s", not for "%s"' % (model, expected_model))
    rep.info.append('made with c2py_export %s, MATLAB %s, %s' % (man.get('version'), man.get('matlab_release'),
                                                                    man.get('created')))
    rep.info.append('solver %s, step size %s s, %s inputs, %s outputs'
                    % (man.get('solver'), man.get('fixed_step'), man.get('n_inputs'), man.get('n_outputs')))
    if man.get('settings_not_applied'):
        rep.warnings.append('these code generation settings could not be applied in MATLAB %s: %s'
                            % (man.get('matlab_release'), ', '.join(_as_list(man['settings_not_applied']))))

    # ---- the C code
    code = d / 'code'
    listed = {f.get('name'): f.get('bytes') for f in _as_list(man.get('files')) if isinstance(f, dict)}
    present = {f.name: f for f in code.glob('*') if f.is_file()} if code.is_dir() else {}
    if not code.is_dir():
        rep.errors.append('the folder "code" is missing')
    missing = sorted(n for n in listed if n not in present)
    if missing:
        rep.errors.append('file%s missing: %s' % ('s' if len(missing) > 1 else '', ', '.join(missing)))
    changed = sorted(n for n, b in listed.items() if n in present and b is not None
                     and present[n].stat().st_size != int(b))
    if changed:
        rep.errors.append('file%s changed after the export (size differs): %s'
                          % ('s' if len(changed) > 1 else '', ', '.join(changed)))
    extra = sorted(n for n in present if n not in listed)
    if extra:
        foreign = {}
        for n in extra:
            if n.lower().endswith('.c'):
                t = present[n].read_text(encoding='utf-8', errors='replace')
                for o in re.findall(r'\bvoid\s+(\w+)_initialize\s*\(\s*(?:void)?\s*\)\s*\{', t):
                    if o != model:
                        foreign[n] = o
        rep.warnings.append('not part of the export, ignored: %s'
                            % ', '.join('%s (code of the model "%s")' % (n, foreign[n]) if n in foreign else n
                                        for n in extra))
        for n in extra:
            present.pop(n)
    for need in ('%s.c' % model, '%s.h' % model, '%s_types.h' % model, 'rtwtypes.h'):
        if need not in present and need not in missing:
            rep.errors.append('%s is missing - the code of the model is incomplete' % need)
    texts = {}
    for n, f in present.items():
        if n.lower().endswith(('.c', '.h')):
            texts[n] = f.read_text(encoding='utf-8', errors='replace')
    others = sorted({m for n, t in texts.items() if n.lower().endswith('.c')
                     for m in re.findall(r'\bvoid\s+(\w+)_initialize\s*\(\s*(?:void)?\s*\)\s*\{', t)} - {model})
    if others:
        rep.errors.append('the code contains another model (%s) - export one model at a time' % ', '.join(others))
    mc = texts.get('%s.c' % model, '')
    if mc and not re.search(r'\bvoid\s+%s_initialize\s*\(' % re.escape(model), mc):
        rep.errors.append('%s.c does not define %s_initialize() - not Embedded Coder code of this model'
                          % (model, model))
    unresolved = set()
    for n, t in texts.items():
        for inc in re.findall(r'#\s*include\s+"([^"]+)"', t):
            base = inc.split('/')[-1]
            if base not in present and base not in MATLAB_HEADERS and base not in C_STD_HEADERS \
                    and base not in missing:
                unresolved.add(base)
    if unresolved:
        rep.errors.append('the code includes file%s that %s not in the package: %s'
                          % ('s' if len(unresolved) > 1 else '', 'are' if len(unresolved) > 1 else 'is',
                             ', '.join(sorted(unresolved))))
    m = re.search(r'rtsiSetSolverName\s*\([^,]*,\s*"(\w+)"', mc)
    if m and man.get('solver') and m.group(1) != man['solver']:
        rep.warnings.append('the code was generated with %s, the manifest says %s' % (m.group(1), man['solver']))
    rep.info.append('%d C/H files' % len(texts))

    # ---- the Simulink runs
    runs = [r for r in _as_list(man.get('runs')) if isinstance(r, dict)]
    usable = 0
    for r in runs:
        name = r.get('name', '?')
        files = [d / 'simulink' / str(r.get(k, '')) for k in ('input_file', 'simulink_file')]
        miss = [f.name for f in files if not f.is_file()]
        if miss:
            rep.warnings.append('test run "%s": %s missing - it cannot be compared with Simulink'
                                % (name, ', '.join(miss)))
            continue
        try:
            _read_run(d, r, man)
        except ValueError as ex:
            rep.warnings.append('test run "%s" cannot be used: %s' % (name, ex))
            continue
        if r.get('error'):
            rep.warnings.append('test run "%s": Simulink stopped at t = %s s (%s)' % (name, r.get('t_end'), r['error']))
        usable += 1
    if not usable:
        rep.errors.append('the package has no usable Simulink results (simulink/*.csv) - without them the python '
                          'model cannot be checked against Simulink. Upload the complete zip, or run %s again'
                          % EXPORT_SCRIPT)
    else:
        rep.info.append('%d Simulink test run%s: %s' % (usable, '' if usable == 1 else 's',
                                                       ', '.join(r.get('name', '?') for r in runs)))
    return rep


def _load_csv(f):
    a = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
    return a


def _read_run(d, r, man):
    """-> (t, U, t_sim, Y_sim)"""
    a = _load_csv(d / 'simulink' / r['input_file'])
    b = _load_csv(d / 'simulink' / r['simulink_file'])
    n_in, n_out = man.get('n_inputs'), man.get('n_outputs')
    if n_in is not None and a.shape[1] != int(n_in) + 1:
        raise ValueError('%s has %d columns, expected t + %s inputs' % (r['input_file'], a.shape[1], n_in))
    if n_out is not None and b.shape[1] != int(n_out) + 1:
        raise ValueError('%s has %d columns, expected t + %s outputs' % (r['simulink_file'], b.shape[1], n_out))
    if len(a) < 3 or len(b) < 2:
        raise ValueError('too few samples')
    return a[:, 0], a[:, 1:], b[:, 0], b[:, 1:]


def install(rep, out_dir):
    """copy the code into out_dir/codegen/<model>_ert_rtw, write the solver info for the
    validation -> {signal: (t, U, t_sim, Y_sim, error)}"""
    assert rep.ok
    model, man, d = rep.model, rep.manifest, rep.dir
    out_dir = Path(out_dir)
    cg = out_dir / 'codegen'
    if cg.exists():
        shutil.rmtree(cg)
    rtw = cg / ('%s_ert_rtw' % model)
    rtw.mkdir(parents=True)
    listed = {f.get('name') for f in _as_list(man.get('files')) if isinstance(f, dict)}
    for f in (d / 'code').glob('*'):
        if f.is_file() and f.name in listed:
            shutil.copy2(f, rtw / f.name)
    shutil.copy2(d / MANIFEST, cg / MANIFEST)
    val = out_dir / 'validation'
    val.mkdir(parents=True, exist_ok=True)
    info = dict(solver=man.get('solver_in_model'), solver_type=man.get('solver_type_in_model'),
                fixed_step=str(man.get('fixed_step')), interp=_as_list(man.get('inport_interpolation')),
                source='package made by %s %s (MATLAB %s)' % (EXPORT_SCRIPT, man.get('version'),
                                                            man.get('matlab_release')))
    (val / ('%s_info.json' % model)).write_text(json.dumps(info))
    data = {}
    for r in _as_list(man.get('runs')):
        try:
            t, U, ts, Y = _read_run(d, r, man)
        except Exception:
            continue
        data[r['name']] = (t, U, ts, Y, r.get('error') or '')
    return data

# -*- coding: utf-8 -*-
"""Simulink (MATLAB Engine) vs generated python model: same inputs, RMSE per output."""
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np

SOLVERS = ['ode1', 'ode2', 'ode3', 'ode4', 'ode5', 'ode8']
SENS_EPS = 1e-10            # relative input change used by the sensitivity ("twin") test


# ------------------------------------------------------------------ test signals
def make_inputs(kind, t, n_in, amp, seed):
    """N x n_in matrix; every channel has a different timing / frequency / seed."""
    tf = float(t[-1] - t[0]) or 1.0
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    U = np.zeros((len(t), n_in))
    for ch in range(n_in):
        if kind == 'step':
            U[:, ch] = amp * (t >= t[0] + 0.1 * tf * (1 + ch))
        elif kind == 'sin':
            U[:, ch] = amp * np.sin(2 * np.pi * (2.0 / tf) * (1 + 0.5 * ch) * (t - t[0]))
        elif kind == 'prbs':
            hold = max(dt, tf / 100.0)
            rng = np.random.default_rng(seed + ch)
            nblk = int(np.ceil(tf / hold)) + 1
            bits = rng.choice([-1.0, 1.0], size=nblk)
            U[:, ch] = amp * bits[np.minimum(((t - t[0]) / hold).astype(int), nblk - 1)]
        else:
            raise ValueError('unknown input signal "%s" (step | sin | prbs)' % kind)
    return U


# ------------------------------------------------------------------ python model
def load_model_class(py_path):
    py_path = Path(py_path)
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Model


def port_order(names, prefix):
    nums = []
    for n in names:
        m = re.fullmatch(prefix + r'(\d+)', n)
        if not m:
            return list(range(len(names)))
        nums.append(int(m.group(1)))
    return list(np.argsort(nums))


def run_python(model, t, U, interp):
    in_ord = port_order(model.input_names, 'In')
    out_ord = port_order(model.output_names, 'Out')
    U_py = np.zeros_like(U)
    for col, k in enumerate(in_ord):
        U_py[:, k] = U[:, col]
    kw = {}
    if 'interp_inputs' in inspect.signature(model.simulate).parameters:
        kw['interp_inputs'] = interp
    r = model.simulate(float(t[-1]), U_py, **kw)
    n = min(len(r['t']), len(t))
    return r['t'][:n], r['y'][:n][:, out_ord]


# ------------------------------------------------------------------ metrics / plots
def metrics(ref, py):
    """NaN/Inf-safe RMSE. A sample where BOTH sides are non-finite (e.g. a genuine 0/0 division
    at t=0, present identically in Simulink's own output) is dropped from the average rather than
    poisoning the whole score. A sample where only ONE side is non-finite is a real mismatch and
    is kept (reported separately as 'mismatched') so it still fails the comparison."""
    n = ref.shape[1]
    fin_ref, fin_py = np.isfinite(ref), np.isfinite(py)
    both_finite = fin_ref & fin_py
    mismatched = fin_ref != fin_py
    rmse = np.full(n, np.nan)
    nrmse = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    masked = np.zeros(n, dtype=int)
    bad = np.zeros(n, dtype=int)
    for j in range(n):
        bad[j] = int(mismatched[:, j].sum())
        m = both_finite[:, j]
        masked[j] = int((~m).sum() - bad[j])
        if not m.any():
            continue
        e = (py[m, j] - ref[m, j])
        rmse[j] = np.sqrt(np.mean(e ** 2))
        scale = np.ptp(ref[m, j])
        scale = scale if scale > 1e-12 else max(np.max(np.abs(ref[m, j])), 1e-12)
        nrmse[j] = rmse[j] / scale
        mx[j] = np.max(np.abs(e))
    return rmse, nrmse, mx, masked, bad


def _resample(t_ref, t_py, y_py):
    if len(t_ref) == len(t_py) and np.max(np.abs(t_ref - t_py)) < 1e-9:
        return y_py, False
    return np.column_stack([np.interp(t_ref, t_py, y_py[:, j]) for j in range(y_py.shape[1])]), True


_warned_mpl = [False]


def plot(path, name, sig, t, U, ref, py, labels=None):
    """One figure per input scenario: inputs on top, then every output (Simulink vs Python)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        if not _warned_mpl[0]:
            print('[plot] matplotlib is not installed -> no PNG files (pip install matplotlib)')
            _warned_mpl[0] = True
        return False
    try:
        n = ref.shape[1]
        labels = labels or ['Out%d' % (j + 1) for j in range(n)]
        ncols = 1 if n <= 4 else (2 if n <= 12 else 3)
        nrows = -(-n // ncols)
        fig = plt.figure(figsize=(9 if ncols == 1 else 6 * ncols, 2.4 * (nrows + 1)))
        gs = fig.add_gridspec(nrows + 1, ncols)
        ax_in = fig.add_subplot(gs[0, :])
        ax_in.plot(t, U)
        ax_in.set_ylabel('inputs')
        ax_in.set_title('%s - %s' % (name, sig))
        for j in range(n):
            ax = fig.add_subplot(gs[1 + j // ncols, j % ncols], sharex=ax_in)
            ax.plot(t, ref[:, j], label='Simulink')
            ax.plot(t, py[:, j], '--', label='Python')
            ax.set_ylabel(labels[j])
            if j == 0:
                ax.legend()
            if j // ncols == nrows - 1:
                ax.set_xlabel('t [s]')
        fig.tight_layout()
        fig.savefig(str(path), dpi=110)
        plt.close(fig)
        return True
    except Exception as ex:                      # never let a plot problem stop the validation
        print('[plot] could not draw %s: %s' % (path, ex))
        return False


def predictability_horizon(Model, solver, t, U, interp, y_p, tol):
    """Time up to which two simulations that differ only by round-off CAN agree within tol.

    For some inputs a model is unstable or chaotic (a controller driven into saturation, a plant
    without stabilising feedback ...): any tiny difference grows exponentially. Simulink and the
    python model do the same arithmetic but not bit for bit, so there they drift apart no matter
    how correct the translation is. Test: run the python model again with the inputs changed by
    1e-10 (relative). Where that twin run leaves the original by more than tol, a comparison
    with Simulink says nothing any more. Returns that time, or None if the twin stays within tol
    over the whole run (= the model is not sensitive; a mismatch with Simulink is then real)."""
    if U.shape[1] == 0 or len(t) < 3:
        return None
    U2 = U * (1.0 + SENS_EPS) + SENS_EPS * np.max(np.abs(U))
    y2 = run_python(Model(solver=solver), t, U2, interp)[1]
    n = min(len(y2), len(y_p))
    with np.errstate(all='ignore'):
        scale = np.maximum(np.maximum.accumulate(np.abs(np.nan_to_num(y_p[:n])), axis=0), 1e-12)
        ratio = np.abs(y2[:n] - y_p[:n]) / scale
    bad = ~(np.nan_to_num(ratio, nan=np.inf, posinf=np.inf) <= tol).all(axis=1)
    return float(t[int(np.argmax(bad))]) if bad.any() else None


# ------------------------------------------------------------------ the validation run
def validate(model_name, py_path, val_dir, cfg, session=None, package_data=None):
    """Run every input scenario on Simulink (session) and on the python model.
    session=None -> offline: use Simulink data cached in val_dir by a previous run.
    package_data: {signal: (t, U, t_sim, Y_sim, error)} from an upload package made by
    tools/c2py_export.m (src/package.py) - its test inputs and Simulink results are used.
    Returns a json-serialisable dict (also what main.py writes to rmse.json)."""
    val_dir = Path(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)
    Model = load_model_class(py_path)

    info_file = val_dir / ('%s_info.json' % model_name)
    if session:
        info = session.info(model_name)
        info_file.write_text(json.dumps(info))
    else:
        info = json.loads(info_file.read_text()) if info_file.exists() else {}
    # cfg['solver'] = one of SOLVERS: that solver is used everywhere (generated code, Simulink run,
    # python model). 'auto' = the solver stored in the model (the old behaviour).
    solver = cfg.get('solver') or 'auto'
    forced = solver if solver in SOLVERS else None
    print('[simulink] solver in the model=%s (%s)%s  fixed step=%s  Inport interpolation=%s'
          % (info.get('solver'), info.get('solver_type'), '  -> %s is used' % forced if forced else '',
             info.get('fixed_step'), info.get('interp')))
    if solver == 'auto' and info.get('solver') in SOLVERS:
        solver = info['solver']
    model = Model(solver=solver)
    dt = cfg.get('dt') or model.step_size
    n_in = len(model.input_names)
    interp = not cfg.get('no_interp', False)
    if 'off' in info.get('interp', []) and interp:
        print('[simulink] an Inport has "Interpolate data" OFF -> python holds the inputs')
        interp = False
    print('[py] %d inputs %s, %d outputs %s, step %g s, python solver %s (C code: %s)'
          % (n_in, model.input_names, len(model.output_names), model.output_names,
             model.step_size, model._method(), Model.SOLVER))
    warnings = []
    if forced and Model.SOLVER != forced:
        warnings.append('the C code was generated with %s, the validation uses %s (regenerate the code '
                        'without --skip-codegen to have the same solver in the C code)' % (Model.SOLVER, forced))
    if abs(dt - model.step_size) > 1e-12:
        warnings.append('dt (%g) differs from the C step size (%g)' % (dt, model.step_size))
    try:
        if abs(float(info.get('fixed_step')) - dt) > 1e-12:
            warnings.append('Simulink fixed step (%s) != input sample time (%g)'
                            % (info.get('fixed_step'), dt))
    except (TypeError, ValueError):
        pass

    t = np.arange(int(round(cfg['tf'] / dt)) + 1) * dt
    scenarios, sim_data, ok_all = {}, {}, True
    for sig in (list(package_data) if package_data is not None else cfg['inputs']):
        if package_data is not None:
            t, U, t_m, y_m, err = package_data[sig]
            if U.shape[1] != n_in:
                raise ValueError('the package has %d input columns, the python model %d inputs' % (U.shape[1], n_in))
            if len(t) > 1 and abs((t[1] - t[0]) - model.step_size) > 1e-9 * max(1.0, model.step_size):
                warnings.append('%s: the package was simulated with step %g, the C code has %g'
                                % (sig, t[1] - t[0], model.step_size))
            if err:
                warnings.append('%s: Simulink stopped at t=%g s of %g s, compared up to there (the model '
                                'diverges for this input): %s' % (sig, float(t_m[-1]), float(t[-1]), err[:200]))
        else:
            U = make_inputs(sig, t, n_in, cfg['amp'], cfg['seed'])
        cache = val_dir / ('%s_%s_matlab.npz' % (model_name, sig))
        if package_data is not None:
            np.savez(cache, t=t_m, y=y_m, u_in=U)
        elif session:
            t_m, _, y_m = session.run(model_name, t, U, fixed_step=dt, solver=forced)
            np.savez(cache, t=t_m, y=y_m, u_in=U)
            err = getattr(session, 'last_error', '')
            if err:                                      # compared up to where Simulink itself gave up
                warnings.append('%s: Simulink stopped at t=%g s of %g s, compared up to there (the model '
                                'diverges for this input - try a smaller amplitude): %s'
                                % (sig, float(t_m[-1]), float(t[-1]), err[:200]))
        else:
            if not cache.exists():
                raise FileNotFoundError('offline mode: %s not found (run once with MATLAB)' % cache)
            d = np.load(cache)
            if d['u_in'].shape != U.shape or not np.allclose(d['u_in'], U):
                raise ValueError('%s was made with other settings (tf/dt/amp/seed)' % cache)
            t_m, y_m = d['t'], d['y']
        t_p, y_p = run_python(model, t, U, interp)
        if y_m.shape[1] != y_p.shape[1]:
            raise ValueError('output count differs: Simulink %d columns, Python %d'
                             % (y_m.shape[1], y_p.shape[1]))
        y_c, resampled = _resample(t_m, t_p, y_p)
        if resampled:
            warnings.append('%s: time grids differ, python resampled on Simulink time' % sig)
        rmse, nrmse, mx, masked, bad = metrics(y_m, y_c)
        until = None
        if not all(bad[j] == 0 and not np.isnan(nrmse[j]) and nrmse[j] <= cfg['tol'] for j in range(len(rmse))):
            # mismatch: a wrong translation - or a model that amplifies round-off for this input?
            n_run = int(np.searchsorted(t, float(t_m[-1]) + 0.5 * dt)) if len(t_m) else 0
            print('[py] %s does not match - sensitivity test (python model again, inputs changed by %g) ...'
                  % (sig, SENS_EPS), flush=True)
            th = predictability_horizon(Model, solver, t[:n_run], U[:n_run], interp, y_p, cfg['tol'])
            keep = int(np.searchsorted(t_m, th)) if th is not None else 0
            if th is not None and keep >= 20:
                until = th
                rmse, nrmse, mx, masked, bad = metrics(y_m[:keep], y_c[:keep])
                warnings.append('%s: the model is unstable / chaotic for this input - changing the inputs by '
                                '%g already changes the python outputs by more than tol after t=%g s. '
                                'Simulink and python can therefore only be compared up to t=%g s (of %g s); '
                                'the result below is for that part.' % (sig, SENS_EPS, th, th, float(t[-1])))
            elif th is not None:
                warnings.append('%s: the model is too sensitive to this input to compare at all (a %g change '
                                'of the inputs exceeds tol after t=%g s)' % (sig, SENS_EPS, th))
        sim_data[sig] = (U, t_m, y_m, t)
        scenarios[sig] = {}
        for j in range(len(rmse)):
            ok = bool(bad[j] == 0 and not np.isnan(nrmse[j]) and nrmse[j] <= cfg['tol'])
            ok_all &= ok
            scenarios[sig]['Out%d' % (j + 1)] = dict(
                rmse=float(rmse[j]), nrmse=float(nrmse[j]), max_abs_error=float(mx[j]), passed=ok,
                masked_both_nonfinite=int(masked[j]), mismatched_nonfinite=int(bad[j]))
            if until is not None:
                scenarios[sig]['Out%d' % (j + 1)]['compared_until_s'] = until
            if masked[j]:
                warnings.append('%s Out%d: %d sample(s) skipped in RMSE (non-finite on both sides, '
                                'e.g. 0/0 at t=0)' % (sig, j + 1, masked[j]))
            if bad[j]:
                warnings.append('%s Out%d: %d sample(s) where only ONE side is non-finite - a real '
                                'mismatch, not masked' % (sig, j + 1, bad[j]))
        hdr = ','.join(['t'] + ['sim_Out%d' % (j + 1) for j in range(y_m.shape[1])]
                       + ['py_Out%d' % (j + 1) for j in range(y_c.shape[1])])
        np.savetxt(val_dir / ('%s_%s.csv' % (model_name, sig)), np.column_stack([t_m, y_m, y_c]),
                   delimiter=',', header=hdr, comments='')
        out_ord = port_order(model.output_names, 'Out')
        labels = [model.output_names[k] for k in out_ord] if len(out_ord) == y_c.shape[1] else None
        if len(t_m) == len(t):
            U_plot = U
        else:  # Simulink returned a different number of samples than requested - resample the
               # actual input onto t_m instead of silently showing a flat zero line
            U_plot = np.column_stack([np.interp(t_m, t, U[:, k]) for k in range(U.shape[1])])
        png = val_dir / ('%s_%s.png' % (model_name, sig))
        if plot(png, model_name, sig, t_m, U_plot, y_m, y_c, labels):
            print('[plot] %s' % png)

    sweep = {}
    if cfg.get('sweep'):
        for sig, (U, t_m, y_m, t) in sim_data.items():
            sweep[sig] = {}
            for n in SOLVERS:
                tp, yp = run_python(Model(solver=n), t, U, interp)
                yc, _ = _resample(t_m, tp, yp)
                sweep[sig][n] = float(np.nanmax(metrics(y_m, yc)[1]))

    return dict(model=model_name, passed=bool(ok_all), tolerance=cfg['tol'], tolerance_meaning=
                'passed if RMSE / peak-to-peak(Simulink output) <= tolerance', tf=float(t[-1]) if len(t) else cfg['tf'], dt=dt,
                python_solver=model._method(), c_code_solver=Model.SOLVER, simulink=info,
                simulink_solver_used=forced or info.get('solver'),
                inputs=list(package_data) if package_data is not None else cfg['inputs'],
                amplitude=cfg['amp'] if package_data is None else None,
                simulink_data='upload package (tools/c2py_export.m)' if package_data is not None else
                ('MATLAB session' if session else 'cached data of an earlier run'), scenarios=scenarios,
                solver_sweep_max_nrmse=sweep, warnings=warnings)


def print_report(res):
    print('\n%-8s %-6s %12s %12s %12s' % ('input', 'output', 'RMSE', 'RMSE/range', 'max|err|'))
    for sig, outs in res['scenarios'].items():
        for o, m in outs.items():
            note = ''
            if m.get('masked_both_nonfinite'):
                note += ' (%d sample(s) skipped, non-finite both sides)' % m['masked_both_nonfinite']
            if m.get('mismatched_nonfinite'):
                note += ' (%d MISMATCHED non-finite sample(s)!)' % m['mismatched_nonfinite']
            if m.get('compared_until_s') is not None:
                note += ' (first %g s only: unstable for this input)' % m['compared_until_s']
            print('%-8s %-6s %12.4e %12.4e %12.4e  %s%s' % (sig, o, m['rmse'], m['nrmse'],
                                                            m['max_abs_error'],
                                                            'PASS' if m['passed'] else 'FAIL', note))
    if res['solver_sweep_max_nrmse']:
        print('\nsolver sweep (max RMSE/range over outputs; matching solver -> ~1e-9)')
        print('%-8s ' % 'input' + ' '.join('%10s' % n for n in SOLVERS))
        for sig, d in res['solver_sweep_max_nrmse'].items():
            best = min(d, key=d.get)
            print('%-8s ' % sig + ' '.join('%10.2e' % d[n] for n in SOLVERS) + '   best: ' + best)
    for w in res['warnings']:
        print('[warn]', w)

"""Shared solver policy and sampling, with restarts at known discontinuities."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import SimulationSettings, Trace
from .inputs import InputSet
from .events import ContinuousEvents
from .solver_policy import ToleranceHistory, resolve_settings


def simulate(backend, settings: SimulationSettings, inputs: InputSet, *, transitions=()) -> Trace:
    if hasattr(backend, "transition_times"):
        transitions = (*transitions, *backend.transition_times(settings.start_time, settings.stop_time))
    state = np.asarray(backend.initial_state(settings.start_time, inputs.values(settings.start_time)), dtype=float)
    if state.shape != (backend.state_size,) or not np.all(np.isfinite(state)):
        raise ValueError("Backend returned invalid initial states.")
    events = sorted({float(t) for t in (*inputs.transitions(), *transitions)
                     if settings.start_time < t <= settings.stop_time})
    event_set = set(events)
    times = settings.time_grid(events)
    fixed_events = {}
    if settings.solver != "ode45":
        for event in events:
            index = (event - settings.start_time) / settings.step_size
            if not np.isclose(index, round(index), rtol=0, atol=1e-9):
                raise ValueError("For fixed-step comparison, Step transitions must lie on the simulation grid.")
            fixed_events[round(index)] = event
    states = np.empty((times.size, backend.state_size), dtype=float)
    states[0] = state
    outputs = np.empty((times.size, len(backend.output_names)), dtype=float)
    outputs[0] = backend.output(float(times[0]), state, inputs.values(float(times[0])))

    def sample(index, current_state, *, commit=False):
        time = float(times[index])
        values = inputs.values(time)
        if commit and hasattr(backend, "commit_event"):
            backend.commit_event(time, current_state, values)
        outputs[index] = backend.output(time, current_state, values)

    def derivative(time, current_state, right, is_event):
        # Integrate each pre-jump interval using its left limit at the endpoint.
        effective = float(np.nextafter(right, -np.inf)) if is_event and time >= right else float(time)
        value = np.asarray(backend.derivative(effective, current_state, inputs.values(effective)), dtype=float)
        if value.shape != state.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid state derivative at t={time:g}.")
        return value

    if settings.solver == "ode45" and backend.state_size and times.size > 1:
        # A direct API caller also receives a bounded automatic maximum step.
        settings = resolve_settings(settings, {"solver_hints": {"periodic_periods":
            [block.period for block in getattr(getattr(backend, "model", None), "blocks", {}).values()
             if type(block).__name__ == "PulseGenerator"]}})
        history = ToleranceHistory(settings, state)
        solver_class = history.solver_class()
        roots = ContinuousEvents(backend, settings, inputs)
        boundaries = sorted({settings.start_time, settings.stop_time, *events})
        integration_steps = 0
        evaluations = 0
        stationary_roots = 0
        for source_left, right in zip(boundaries[:-1], boundaries[1:]):
            left = source_left
            is_event = right in event_set
            while left < right:
                interval = right - left
                first_step = None
                if left == settings.start_time and settings.initial_step is not None:
                    first_step = min(settings.initial_step, settings.max_step, interval)
                elif events and interval <= settings.step_size:
                    first_step = min(interval, settings.max_step)
                maximum_before = history.maximum.copy()
                result = solve_ivp(lambda t, x: derivative(t, x, right, is_event),
                                   (left, right), state, method=solver_class, dense_output=True,
                                   rtol=settings.relative_tolerance, atol=history.absolute(),
                                   max_step=settings.max_step, first_step=first_step,
                                   events=roots.functions(left, state, right, is_event))
                if not result.success:
                    raise RuntimeError(f"ode45 failed: {result.message}")
                end = float(result.t[-1])
                integration_steps += len(result.t) - 1
                evaluations += result.nfev
                lower = int(np.searchsorted(times, left, side="right"))
                upper = int(np.searchsorted(times, end, side="right"))
                if lower < upper:
                    states[lower:upper] = result.sol(times[lower:upper]).T
                state = result.y[:, -1]
                for index in range(lower, upper):
                    if times[index] < right or not is_event:
                        sample(index, states[index])
                if result.status == 1:
                    # solve_ivp can accept a step past a terminal root before
                    # interpolating it. Discard its tail from tolerance history.
                    history.maximum = np.maximum(maximum_before, np.max(np.abs(result.y), axis=1))
                    roots.record(result)
                    if end <= left:
                        stationary_roots += 1
                        if stationary_roots > 8:
                            raise RuntimeError(f"Repeated zero crossings at t={left:g}; possible chattering.")
                    else:
                        stationary_roots = 0
                    left = end
                else:
                    left = right
            if is_event and hasattr(backend, "commit_event"):
                backend.commit_event(right, state, inputs.values(right))
            upper = int(np.searchsorted(times, right, side="right"))
            if upper and times[upper - 1] == right and is_event:
                sample(upper - 1, states[upper - 1])
        backend.simulation_diagnostics = {
            "internal_steps": integration_steps, "derivative_evaluations": evaluations,
            "known_source_events": len(events), "zero_crossing_signals": list(roots.names),
            "zero_crossing_count": roots.count, "first_zero_crossings": roots.hits,
            "final_absolute_tolerances": history.absolute().tolist(),
        }
    elif backend.state_size:
        for index in range(1, times.size):
            left, right = float(times[index - 1]), float(times[index])
            dt = right - left
            boundary = fixed_events.get(index, right)
            is_event = index in fixed_events
            k1 = derivative(left, state, boundary, is_event)
            if settings.solver == "ode1":
                state = state + dt * k1
            else:
                k2 = derivative(left + dt / 2, state + dt / 2 * k1, boundary, is_event)
                k3 = derivative(left + dt / 2, state + dt / 2 * k2, boundary, is_event)
                k4 = derivative(right, state + dt * k3, boundary, is_event)
                state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            states[index] = state
            sample(index, state, commit=is_event)
    else:
        states[:] = state
        event_index = 0
        for index in range(1, times.size):
            while event_index < len(events) and events[event_index] <= times[index]:
                event = events[event_index]
                if hasattr(backend, "commit_event"):
                    backend.commit_event(event, state, inputs.values(event))
                event_index += 1
            sample(index, state)
    trace = Trace(times, backend.output_names, outputs)
    trace.validate()
    return trace

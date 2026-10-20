"""Shared resolved limits and Simulink-style error scaling for Dormand-Prince.

The protected RK45 error hook is isolated here. SciPy owns integration stages,
adaptation and interpolation; this module changes only the acceptance norm.
"""
from dataclasses import replace
import numpy as np
from scipy.integrate import RK45


def resolve_settings(settings, contract):
    candidates = [max(settings.stop_time - settings.start_time, settings.step_size) / 50]
    periods = contract.get("solver_hints", {}).get("periodic_periods", [])
    candidates.extend(float(period) / 3 for period in periods if float(period) > 0)
    limit = settings.max_step if settings.max_step is not None else min(candidates)
    initial = min(settings.initial_step, limit) if settings.initial_step is not None else None
    return replace(settings, max_step=limit, initial_step=initial)


class ToleranceHistory:
    def __init__(self, settings, state):
        self.settings = settings
        self.maximum = np.abs(state).copy()
        self.has_accepted = False

    def absolute(self):
        floor = self.settings.absolute_tolerance
        if self.settings.auto_scale_absolute_tolerance and self.has_accepted:
            return np.maximum(floor, self.settings.relative_tolerance * self.maximum)
        return np.full(self.maximum.shape, floor)

    def accepted(self, state):
        self.has_accepted = True
        self.maximum = np.maximum(self.maximum, np.abs(state))

    def solver_class(self):
        if not hasattr(RK45, "_estimate_error_norm"):
            raise RuntimeError("This SciPy RK45 version does not expose the required error-control hook.")
        history = self
        class SharedRK45(RK45):
            def _estimate_error_norm(self, K, h, scale):
                candidate = self.y + h * np.dot(K[:-1].T, self.B)
                # Bound every state independently, using max rather than the
                # additive RMS scale used by SciPy's default RK45 policy.
                allowed = np.maximum(history.absolute(), self.rtol * np.maximum(np.abs(self.y), np.abs(candidate)))
                return float(np.max(np.abs(self._estimate_error(K, h)) / allowed))

            def step(self):
                message = super().step()
                if self.status != "failed":
                    history.accepted(self.y)
                    self.atol = history.absolute()
                return message
        return SharedRK45

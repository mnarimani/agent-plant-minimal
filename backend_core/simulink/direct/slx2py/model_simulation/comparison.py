"""Output identity checks, common-grid alignment and pairwise error metrics."""
from itertools import combinations

import numpy as np

from .contracts import Trace


def align(trace: Trace, time: np.ndarray, output_names: tuple[str, ...]) -> Trace:
    trace.validate()
    if set(trace.output_names) != set(output_names):
        raise ValueError("Simulation output names differ from the model contract.")
    tolerance = 1e-9 * max(1, abs(time[0]), abs(time[-1]))
    if trace.time[0] > time[0] + tolerance or trace.time[-1] < time[-1] - tolerance:
        raise ValueError("Simulation trace does not cover the requested time range; extrapolation is forbidden.")
    order = [trace.output_names.index(name) for name in output_names]
    if trace.time.shape == time.shape and np.allclose(trace.time, time, rtol=0, atol=tolerance):
        outputs = trace.outputs[:, order]
    else:
        outputs = np.column_stack([np.interp(time, trace.time, trace.outputs[:, column]) for column in order])
    result = Trace(time.copy(), output_names, outputs)
    result.validate()
    return result


def compare(traces: dict[str, Trace]) -> list[dict]:
    metrics = []
    for (left_name, left), (right_name, right) in combinations(traces.items(), 2):
        if left.output_names != right.output_names or not np.array_equal(left.time, right.time):
            raise ValueError("Compare only traces already aligned on the same output order/time grid.")
        for index, output in enumerate(left.output_names):
            error = right.outputs[:, index] - left.outputs[:, index]
            metrics.append({"output": output, "reference": left_name, "candidate": right_name,
                            "sample_count": len(error), "mse": float(np.mean(error ** 2)),
                            "rmse": float(np.sqrt(np.mean(error ** 2))), "max_absolute_error": float(np.max(np.abs(error)))})
    return metrics

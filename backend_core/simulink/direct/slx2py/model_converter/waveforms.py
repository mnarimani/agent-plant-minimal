"""Pure scalar waveforms used by runtime and independent equation rendering."""
import math


def pulse_value(time, amplitude, period, width, delay):
    if time < delay:
        return 0.0
    cycle = math.floor((time - delay) / period)
    # Compare the same representable edge times used by the scheduler. Do not
    # round a left-limit nextafter sample up to its discontinuity. The edges are
    # exact sums (fsum): delay + k*period + width can otherwise land one ulp past
    # the time Simulink switches at (0.2 + 3.0 + 0.45 > 3.65 in plain addition).
    while time < math.fsum((delay, cycle * period)):
        cycle -= 1
    while time >= math.fsum((delay, (cycle + 1) * period)):
        cycle += 1
    off = math.fsum((delay, cycle * period, period * width / 100))
    return float(amplitude) if time < off else 0.0

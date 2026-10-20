"""Inherited sample times (-1) of discrete blocks, resolved the way Simulink compiles them.

A discrete block with sample time -1 (the default of Unit Delay and Zero-Order Hold) takes the
rate of the signal that drives it. Simulink's rules, read from its compiled sample times
(matlab_block_tests/make_slx_block_reference.m, cases b_inh_*, fixed-step solver):

  - driven by discrete signals: the greatest common divisor of their rates (0.05 and 0.02 -> 0.01),
    also through blocks that only pass a rate on (Gain, Sum ...) and around loops
  - driven by a continuous signal (an Integrator, a source in time), by a root Inport, by a
    constant, or by nothing that decides: the fundamental step of the model - its fixed step
  - a block that passes a rate on (Gain, Sum ...) is continuous if one of its inputs is, else
    discrete with the gcd of the discrete inputs, else constant if all inputs are constant

The rates are propagated over the wired model until nothing changes; an input whose rate is not
known yet (a loop, a root Inport) does not decide. Every resolved block is rebuilt with its
sample time (``DiscreteBlock.inherit``) and named in the model's notes.
"""
from __future__ import annotations

import math
from fractions import Fraction

from .blocks import (Constant, DiscreteBlock, Inport, Integrator, MathBlock, PulseGenerator, SecondOrderIntegrator,
                     StateSpace, Step, TransferFcn)
from .graph import Endpoint

CONTINUOUS = "continuous"
CONSTANT = "constant"


class SampleTimeError(ValueError):
    pass


def _gcd(rates: list[float]) -> float:
    result = Fraction(rates[0]).limit_denominator(10 ** 9)
    for rate in rates[1:]:
        other = Fraction(rate).limit_denominator(10 ** 9)
        result = Fraction(math.gcd(result.numerator * other.denominator, other.numerator * result.denominator),
                          result.denominator * other.denominator)
    return float(result)


def _own_rate(block):
    """the rate a block has by itself: CONTINUOUS, CONSTANT, a period, or None (inherits)"""
    if isinstance(block, DiscreteBlock):
        if getattr(block, "inherit", None) is not None:
            return None
        return block.sample[0] if getattr(block, "sample", None) else CONTINUOUS   # Memory, Derivative ...
    if isinstance(block, Constant):
        return CONSTANT
    if isinstance(block, (Integrator, TransferFcn, StateSpace, SecondOrderIntegrator, Step, PulseGenerator)):
        return CONTINUOUS
    if isinstance(block, MathBlock) and block.input_count == 0:
        return CONSTANT if block.kind == "Ground" else CONTINUOUS       # sources in time
    return None                                                         # passes the rate of its inputs on


def _combine(rates: list) -> object:
    known = [r for r in rates if r is not None]
    if not known:
        return None
    if CONTINUOUS in known:
        return CONTINUOUS
    periods = [r for r in known if r != CONSTANT]
    return _gcd(periods) if periods else CONSTANT


def resolve(model, fixed_step: float | None) -> list[str]:
    """Give every discrete block that inherits its sample time the rate Simulink compiles for it
    (the block is replaced by one built with that rate). -> notes for the user."""
    inherited = [b for b in model.blocks.values() if getattr(b, "inherit", None) is not None]
    if not inherited:
        return []
    incoming = {c.destination: c.source for c in model.connections}
    rates = {block_id: _own_rate(block) for block_id, block in model.blocks.items()}
    for block_id, block in model.blocks.items():
        if isinstance(block, Inport):
            rates[block_id] = None                       # a root Inport takes its rate from where it goes
    resolved: dict[str, float] = {}
    for _ in range(4 * len(model.blocks) + 8):
        changed = False
        for block_id, block in model.blocks.items():
            if _own_rate(block) is not None or isinstance(block, Inport):
                continue
            sources = [incoming.get(Endpoint(block_id, port)) for port in range(block.input_count)]
            combined = _combine([rates.get(s.block_id) if s is not None else None for s in sources])
            if getattr(block, "inherit", None) is not None:
                # a discrete block: discrete inputs give their gcd; anything else the fundamental step
                new = combined if isinstance(combined, float) else None
            else:
                new = combined
            if new is not None and rates.get(block_id) != new:
                rates[block_id] = new
                changed = True
        if not changed:
            break
    notes = []
    for block in inherited:
        rate = rates.get(block.block_id)
        how = "the rate of the discrete signal that drives it"
        if not isinstance(rate, float):
            if fixed_step is None:
                raise SampleTimeError(
                    f"{block.kind} '{block.name}' inherits its sample time (-1) from a continuous or constant signal; "
                    "in a variable-step model this is not supported - set its sample time to a number.")
            rate, how = float(fixed_step), "the fixed step of the model (its input is not discrete)"
        if fixed_step is not None and abs(rate / fixed_step - round(rate / fixed_step)) > 1e-9:
            raise SampleTimeError(f"{block.kind} '{block.name}': sample time {rate:g} s is not a multiple of the "
                                  f"fixed step {fixed_step:g} s.")
        resolved[block.block_id] = rate
        new_block = block.inherit((rate, 0.0))
        new_block.inherited_rate = rate
        model.blocks[block.block_id] = new_block
        notes.append(f"{block.kind} '{block.name}' inherits its sample time (-1): {rate:g} s, {how}, as Simulink compiles it.")
    model._execution_order = None
    model._widths = None
    return notes

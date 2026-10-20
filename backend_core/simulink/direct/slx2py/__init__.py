"""The engine of route 0: a Simulink model (.slx) read directly from its XML, without MATLAB.

An .slx file is a zip of XML files (blocks, lines, parameters, solver settings). The engine
reads them, builds an executable block model and, for continuous models, the explicit
equations xdot = f(t, x, u), y = g(t, x, u). ``backend_core.simulink.direct.slx_direct`` turns those
into a draft of the plant-model agent.

Origin: the two packages below are copied from the SLX_TO_Python project (``model_engine/src/``)
and extended here (see "Added here" below):

- ``model_converter``    .slx -> block model (``model.py``) + explicit equations
                         (``dynamics.json`` / ``dynamics.py``); every module of the original
                         except ``cli`` / ``__main__`` (command line) and ``explorer_views``
                         (its Streamlit app)
    ``pipeline``         unzip safely, then topology -> block model -> export
    ``topology``         flatten subsystems, resolve lines and Goto/From, keep what reaches
                         an Outport (an unsupported block on that path is an error, never skipped)
    ``slx_reader``       XML block -> runtime block, solver settings
    ``blocks``, ``model``  the runtime blocks and the executable graph
    ``dynamics*``        the equations from the graph (rules per block type, expression IR)
    ``stateflow*``       MATLAB Function blocks and a subset of Stateflow
    ``dependency_report``  why a conversion failed (library blocks, referenced models ...)
    ``exporter``, ``viewer*``, ``diagram_layout``, ``source_*``  the files of a conversion
- ``model_simulation``   the simulation of the block model and of its equations with the
                         model's own solver (ode45 / ode4 / ode1), used to check them; only
                         the modules for that (not its runner, reports, MATLAB adapter, CLI)

Added here (2026-10):

- ``block_semantics``  memoryless blocks as expressions (a ``MathBlock``): sources (Clock, Ground,
                       Sine Wave, Signal Generator, Ramp, Chirp, Repeating Sequence), math (Abs, Sign,
                       Unary Minus, Bias, Polynomial, Sqrt, Math Function, Trigonometric Function,
                       Rounding, MinMax), discontinuities (Dead Zone, Quantizer, Coulomb & Viscous
                       Friction, Saturation / Dead Zone Dynamic, Wrap To Zero), logic (Relational and
                       Logical Operator, Compare To Constant / Zero), routing (Switch, Multiport Switch,
                       Data Type Conversion, IC, Signal Specification), 1-D / 2-D / direct lookup
                       tables with Simulink's interpolation and extrapolation methods
- ``blocks``           MathBlock, StateSpace (also the PID Controller), Second-Order Integrator,
                       limited Integrator; MATLAB Function with several outputs and vector inputs;
                       Fcn through the expression IR (more functions, C-like operators)
- ``stateflow``        MATLAB Function: the closing ``end``, ``for`` loops with constant bounds
                       (unrolled), u(1) indexing, MATLAB's math functions, ``~``
- ``dynamics_expressions``  the functions above (numpy semantics: NaN, not exceptions; MATLAB's
                       round / mod), table lookups, and the prelude a rendered function needs
- ``slx_reader``       library (Reference) blocks with their mask parameters (InstanceData),
                       MATLAB matrices / complex numbers in parameters, Zero-Pole, State-Space, the
                       defaults of omitted parameters (Sum, Product, Manual Switch)
- ``topology``         the ports of blocks at their defaults (no <PortCounts>), and of unknown
                       blocks from their lines (so they are reported as unsupported)
- vector signals (2026-10-14): ``Model.infer_widths`` resolves the width of every signal (each
  block gives its output widths from its input widths, repeated to a fixed point) before the states
  are bound; element-wise blocks expand scalars; Demux, Selector, Vector Concatenate, Gain (vector,
  K*u, u*K), Sum / Product / MinMax of the elements, vector Integrator, MIMO State-Space, vector
  Constant / Inport (PortDimensions) / Outport; MATLAB Function vector literals ([a b c]', [a; b]),
  ``...`` continuations. The elements of vector model inputs / outputs are named In1(1), In1(2) ...
- Simulink's reading of an unconnected input (0, with a note in model.json "warnings"), and a note
  when an Inport without dimensions is expanded to a vector
- MATLAB workspace variables (2026-10-15): ``workspace`` evaluates MATLAB scripts without MATLAB (its
  own interpreter: numbers, vectors, matrices, struct fields, operators, indexing, if / for, the common
  math / matrix / control functions such as eye, inv, lqr, care, are). The parameters are evaluated as
  in Simulink: the user's scripts (``pipeline.convert_slx(..., scripts=)``), then the model's own
  callbacks (PreLoadFcn / PostLoadFcn / InitFcn) and model workspace code. Block parameters may be any
  such expression (``[k 1]*m``, ``A(2,:)``, ``eye(2)``); a block names all its unknown variables at
  once; model.json "workspace" lists the variables, those used ("used") and those without a value
- discrete memory (2026-10-15): ``discrete_semantics`` + ``blocks.DiscreteBlock``: Memory, Derivative,
  Unit Delay, Zero-Order Hold, Discrete-Time Integrator (Forward Euler), Discrete Transfer Fcn,
  Discrete State-Space, Rate Limiter, Relay, Backlash. Simulink's two phases of a major step: the
  Output phase (``major``) and the Update phase (``Model.update``, the equations' ``update`` in
  dynamics.json / dynamics.py); the solver's stages in between see the memory unchanged. The memory
  is part of the state vector with derivative 0; a fixed-step major step is
  ``x = update(t, x, u)``, the outputs, then the stages from that x. Inherited sample times (-1,
  the default of Unit Delay / Zero-Order Hold / Discrete Transfer Fcn): ``sample_times`` gives each
  block the rate Simulink compiles for it (2026-10-16, checked against CompiledSampleTime). Not yet: Transport Delay,
  Hit Crossing, Integrator with external reset (refused, never approximated)
- fixes: Lookup extrapolates linearly (Simulink's default, was clipped); a pulse edge one ulp
  late (``waveforms``); Manual Switch takes the first input when nothing is stored

Every block is checked against Simulink itself: tests/test_slx_blocks.py (groups A, B, C), with the models and
results of matlab_block_tests/make_slx_block_reference.m (tests/fixtures/slx_blocks_reference.zip).
Whole models: matlab_block_tests/simulate_models_reference.m + tests/slx_models_check.py.

A new memoryless block is one rule in ``block_semantics`` (and its default ports in
``topology._DEFAULT_PORTS``). A block with state needs the XML parameters (``slx_reader``), the
runtime block (``blocks``, rendered by ``exporter``) and the rule for the equations
(``dynamics_rules``). Add a case for it to the MATLAB script, run it, and run the tests.

Needs Python 3.11+, numpy and scipy (already dependencies of this project).
"""

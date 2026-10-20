"""Simulink model -> validated dynamics(t, x, u), without MATLAB on this machine.

Two ways from an uploaded model to a draft of the plant-model agent (``backend_core.agent``),
one folder each:

    direct/   route 0, tried first for every uploaded .slx: the model read directly from its XML
              (no MATLAB, no LLM). Only the blocks checked against Simulink; anything else is
              refused with its reason.
    c2py/     the C-code routes, when route 0 is not possible: the C code Embedded Coder
              generates from the model (A: the script c2py_export.m on the user's computer, with
              Simulink's own test runs; B: the C files the user generates) -> Python.

Shared by both, here:

- ``intake``  what was attached in the chat (document / model / c2py package / C code / MATLAB
              script), what is still missing, and what the chat agent tells the user
- ``llm``     the LLM of the Simulink part (c2py's readable dynamics, values read from the chat)
              through this project's ``openai_client``

Both routes give a draft in the same format: ``{"system_name", "python_code", "metadata"}``.
"""

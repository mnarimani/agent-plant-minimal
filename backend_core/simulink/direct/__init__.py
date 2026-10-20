"""Route 0: a Simulink model (.slx) converted directly from its XML - no MATLAB, no LLM.

- ``slx_direct``   the app's side: runs the engine, checks the equations against its block model
                   (at random points and in simulation), picks the in-app simulation, writes the
                   draft and, when it fails, the reason - or the change the user makes in the model
- ``chat_values``  the values of missing MATLAB variables, read from a chat message (the MATLAB
                   parser first; an LLM only for values given in words, and checked)
- ``slx2py/``      the engine: .slx -> block model + explicit equations. From the SLX_TO_Python
                   project, extended here (its __init__ lists what was added)
"""

"""The C-code routes: Embedded Coder's C code of a Simulink model -> validated dynamics(t, x, u).

    A  the user runs ``c2py_export.m`` next to the model in their own MATLAB: C code + Simulink's
       answers to test inputs (step / sine / PRBS) in ``<model>_c2py.zip``
    B  the user generates the C code and uploads three files: <model>.c, <model>.h, <model>_data.c

    check the upload -> C code to Python -> Python vs Simulink (A) -> exact dynamics(t, x, u)
    -> readable with an LLM, every draft validated -> JSON card -> draft of the plant-model agent

From the C_to_Python project (``src/``), so improvements can be merged in both directions:

- ``c2py_export.m``  the MATLAB script the user downloads (route A)
- ``package``        check / unpack the zip of route A
- ``c2py``           rule-based C -> Python translator (no LLM)
- ``validation``     Python model vs the Simulink results of the package
- ``flatten``        generated Python model -> exact dynamics(t, x, u) (no LLM)
- ``agent``          the LLM makes it readable; every draft validated; JSON card
- ``equations``      equations of the card traced back by program, checked numerically
- ``cost``           token prices

New here:

- ``convert``        the pipeline for the app (route A or B), the card in this project's metadata
                     format, and the simulation of a converted model the way Simulink runs it
                     (also used by route 0)
"""

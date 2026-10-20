# Simulink example models

Models that the app converts directly from the `.slx` file (route 1, no MATLAB). Upload one in the
chat of the Streamlit app (`streamlit run app.py`, attach with 📎), or convert them all at once:

```bash
python -m pytest tests/test_example_models.py -q
```

Each conversion is checked against the engine's block model: `dx/dt` and the outputs y at 200 random
points, then a simulation. Together they take about 40 s (Longitudinal_Vehicle_Dynamics alone about
20 s).

| Model | States | Inputs u | Outputs y | What it shows |
|-------|-------:|---------:|----------:|---------------|
| `mimo_test1` ... `mimo_test4` | 2 | 2 | 1-2 | MIMO with a MATLAB Function in a subsystem; from `mimo_test2` on also Saturation, Mux / Selector |
| `mimo_test5` ... `mimo_test8` | 4 | 2 | 1 | MIMO with Transfer Fcn, Saturation, subsystems, MATLAB Function |
| `block_diagram_ode1` / `2` / `3` | 4 / 6 / 8 | 2 / 2 / 3 | 2 / 2 / 3 | ODEs drawn as block diagrams |
| `MSD_with_force` | 2 | 1 | 1 | mass-spring-damper |
| `InvertedP_test` | 2 | 1 | 2 | inverted pendulum (solver ode4) |
| `Quadcopter_Model` | 6 | 2 | 3 | quadcopter |
| `quad_six` | 12 | 1 | 12 | quadrotor, every state an output |
| `Cannon_Fire` | 8 | 1 | 1 | discrete memory blocks: the code also has `update(t, x, u)` |
| `PID_Controller` | 24 | 3 | 3 | discrete blocks; A, B, C, D come from the model's own InitFcn |
| `Longitidunal_Vehicle_Dynamics` | 8 | 1 | 3 | discrete blocks; 14 parameters come from the model's InitFcn |
| `mimo_test8_no_ports` | 4 | 0 | 0 | **no Inport and no Outport**: the app shows the block diagram to choose inputs and outputs (e.g. the free inputs ◆ of the two Transfer Fcn) before it converts |

Not here, because the direct route cannot read them (the app sends them to the C-code route):
models with library blocks whose contents are not stored in the `.slx` (e.g. Aerospace Blockset,
noise blocks), Stateflow charts, and Simscape. `tests/fixtures/` has `tst3` (library blocks) and
`tst4` (Stateflow) as examples of those.

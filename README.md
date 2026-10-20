# AgentPlant — Simulink conversion

The Streamlit app (`app.py`) converts a **Simulink model** into a runnable, validated Python model:

```python
def dynamics(t, x, u):   # dx/dt
def output(t, x, u):     # the outputs y
def update(t, x, u):     # discrete memory (only models with Unit Delay, Memory ...)
```

plus its metadata (states, inputs, outputs, equations). The model is attached in the chat; the
converted model is simulated and can be changed in the chat like any other draft.

---

## Run

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                   # then set OPENAI_API_KEY
streamlit run app.py
```

On Windows, keep the repository in a short path (or enable long paths), or `pip install` fails on
packages with very long file names.

| Variable | Default | |
|----------|---------|---|
| `OPENAI_API_KEY` | — | required |
| `AGENTPLANT_MODEL` | `gpt-4.1` | the chat agent |
| `AGENTPLANT_SIMULINK_MODEL` | `gpt-5.4-mini` | the LLM steps of the conversion |
| `AGENTPLANT_LOG_DIR` | `.logs` | prompt / response logs |

---

## How it works

### What can be uploaded

| Upload | Route |
|--------|-------|
| `.slx` | **direct** (no MATLAB): converted right away |
| `.m` script | the values of the model's MATLAB variables |
| zip of `c2py_export.m` | **C code, route A** |
| `<model>.c`, `<model>.h`, `<model>_data.c` (files or a zip) | **C code, route B** |

`backend_core/simulink/intake.py` classifies every upload, keeps what a conversion still waits for, and
tells the chat agent what to say to the user (texts in `backend_core/prompts/simulink/upload_guidance.yaml`).

### Direct route (`.slx`, no MATLAB, no LLM)

1. The `.slx` (a zip of XML) is read by the slx2py engine (`backend_core/simulink/direct/slx2py/`):
   subsystems are flattened, Goto / From are resolved, block parameters are evaluated (MATLAB
   expressions, with the values of the workspace variables).
2. The engine builds a numeric **block model** of the whole model and extracts its equations.
   Every block is converted, not only what reaches an Outport; sinks (Scope, Display ...) are left
   out. A block that cannot be converted and reaches no output is left out with a note.
3. `slx_direct.render_dynamics` writes `dynamics`, `output` and `update`.
4. Checks against the block model: `dx/dt`, the update and the outputs y at 200 random points
   (t, x, u), then both simulated with the model's solver (unit step on input 1).
5. The app simulates the result with the model's own fixed-step solver and plots the outputs y.

Discrete blocks (Memory, Unit Delay, Zero-Order Hold, discrete integrators and filters, Rate Limiter,
Relay, Backlash, Derivative) keep their memory as extra states with `dx/dt = 0`; `update(t, x, u)`
gives their new values at every major step, as Simulink's Update phase does.

### Missing values of MATLAB variables

Block parameters often are MATLAB variables (`m`, `K`, `A` ...) whose values are not in the `.slx`.
They are taken, in this order, from:

1. the model's own callbacks (PreLoadFcn, PostLoadFcn, InitFcn) — run by a MATLAB interpreter
   without MATLAB (`slx2py/model_converter/workspace.py`);
2. an attached `.m` script;
3. the chat: `m = 2` is read by a parser; values in words ("m is two", «جرم دو کیلوگرم») go to an LLM
   (`direct/chat_values.py`), and a value is taken only if every number in it is in the message.

While values are missing, the agent asks for exactly those; the model is converted as soon as they arrive.

### Models without Inport / Outport

When the model has no input u or no output y, the app does not convert it yet. It shows the model's
block diagram (`direct/diagram.py`, every system and subsystem), and the user clicks the signals:

- **input u** — the signal is replaced there by u (several inputs possible); ◆ marks an input port
  nothing is connected to, which can be an input too;
- **output y** — the signal is measured (several outputs possible);

then "Convert with these signals", or "Convert as it is". The choice becomes new root Inports /
Outports of the model (`topology.SignalChoice`) before it is converted.

### C-code route

For a model the direct route cannot read (blocks stored outside the `.slx`, Stateflow charts ...):

- **Route A** — the user runs `c2py_export.m` in MATLAB (attached by the app). Its zip has the
  generated C code and Simulink's own results for test inputs.
- **Route B** — the user generates the code with Embedded Coder and sends `<model>.c`, `.h`, `_data.c`.

`c2py/convert.py`: the C code is translated to Python (`c2py.py`), reduced to the exact
`dynamics(t, x, u)` (`flatten.py`), rewritten readable by an LLM (`agent.py`) and checked against the
exact translation; route A also against Simulink's results (`validation.py`).

### Prompts

Every LLM prompt of the app is in `backend_core/prompts/*.yaml` (chat agent, search, Simulink);
`backend_core/prompts/README.md` lists which file and key is used where.

---

## Code map

```
app.py                       # upload handling, conversions, the signal picker panel, plots
backend_core/
├── sandbox.py               # restricted exec of the code; output() / update() support
├── prompts/                 # every LLM prompt, as YAML
└── simulink/
    ├── intake.py            # uploads, what is missing, notes for the agent
    ├── llm.py               # the LLM of the conversion
    ├── direct/
    │   ├── slx_direct.py        # convert_slx(): conversion + checks -> DirectResult
    │   ├── diagram.py           # the block diagram and the signal choice
    │   ├── chat_values.py       # values of missing variables from a message
    │   └── slx2py/              # the slx2py engine (topology, blocks, equations, workspace)
    └── c2py/
        ├── convert.py           # convert_package() / convert_c_code(), simulate()
        ├── c2py.py, flatten.py, agent.py, equations.py, validation.py, package.py, cost.py
        └── c2py_export.m        # the MATLAB script of route A
```

| Function | Does |
|----------|------|
| `intake.classify(name, data)` | what an uploaded file is |
| `slx_direct.convert_slx(file_name, data, runs_dir, scripts=, signals=)` | the direct conversion -> `DirectResult` |
| `diagram.read_diagram(data)`, `figure(...)`, `choice(diagram, inputs, outputs)` | the block diagram; the user's choice -> `signals=` |
| `chat_values.read_values(text, missing, uses, llm)` | values of missing variables from a message |
| `convert.convert_package(...)`, `convert_c_code(...)`, `simulate(...)` | the C-code routes; the simulation |

Files of every conversion go to `simulink_runs/` (git-ignored).

---

## Example models and tests

`examples/simulink_models/` has 19 models that convert directly, with a README of what each shows.

```bash
pip install pytest
python -m pytest tests -q --ignore=tests/test_event_loop_nonblocking.py
```

About 5 minutes; no MATLAB and no API key needed. The block tests compare with Simulink's results
stored in `tests/fixtures/*_reference.zip`; `tests/test_example_models.py` converts every example model.

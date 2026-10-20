# AgentPlant — plant-model chat and Simulink conversion

AgentPlant is LabCD's plant-model assistant. A user describes a physical system — in words, equations,
a document, or a **Simulink model** — and gets a runnable, simulated Python model:

```python
def dynamics(t, x, u):   # dx/dt
def output(t, x, u):     # the outputs y   (converted Simulink models)
def update(t, x, u):     # discrete memory (models with Unit Delay, Memory ...)
```

plus metadata (states, inputs, outputs, parameters, equations).

This repository started as a minimal extraction of the AgentPlant / PlantModelChat subsystem of
[LabCD_Application](https://github.com/labcd-dev/LabCD_Application). Release v1 (branch
`release/v1-simulink`) adds the Streamlit app with Simulink conversion; see [What changed](#what-changed-since-the-extraction).

---

## The three parts

| Part | Start with | Uses | What it does |
|------|-----------|------|--------------|
| **Streamlit app** | `streamlit run app.py` | `backend_core/` (agent, rag, sandbox, websearch, prompts, **simulink**) | Chat agent, documents (RAG), web search, simulation lab, **Simulink conversion**. The most complete part. |
| **Minimal API** | `python run_api.py` | `minimal_api/` → `backend_core/AgentPlant/` (chat), `streamlit_core/` (upload, simulate, web search) | FastAPI without DB or real auth, for the React UI. |
| **React frontend** | `cd frontend && npm run dev` | the API above (Vite proxies `/api` → `http://localhost:8000`) | The "AgentPlant AI" chat on the Design page (`AgentPlantChat.tsx`). |

> **For the website:** the Simulink conversion lives only in the Streamlit app. The API and the React
> frontend do not call it yet — see [Wiring Simulink into the API](#wiring-simulink-into-the-api).

`backend_api/` (PlantModelChat + HTTP routers / services) is the upstream LabCD code, kept as a
reference; nothing in this repository imports it.

---

## Quick start

Python 3.10+ (tested with 3.11). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt        # includes ./packages/labcd_agents[all]
cp .env.example .env                   # then set OPENAI_API_KEY
streamlit run app.py
```

On Windows, keep the repository in a short path (or enable long paths): some packages pulled by
`labcd_agents[all]` have very long file names, and `pip install` fails deep in a folder tree.

Without `OPENAI_API_KEY` the app only shows how to set it. (The direct Simulink conversion itself
calls no LLM; the tests run without a key.)

### Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | everything that calls an LLM |
| `AGENTPLANT_MODEL` | `gpt-4.1` | the chat agent of the Streamlit app |
| `AGENTPLANT_SIMULINK_MODEL` | `gpt-5.4-mini` | the LLM steps of the Simulink conversion |
| `AGENTPLANT_LOG_DIR` | `.logs` | prompt / response logs (one `.jsonl` per day, git-ignored) |
| `AGENTPLANT_DEFAULT_MODEL` | `gpt-4o-mini` | the minimal API's chat |
| `AGENTPLANT_DEMO` | off | the minimal API: canned upload / simulation, no key needed |
| `HOST`, `PORT`, `RELOAD` | `0.0.0.0`, `8000`, `1` | `run_api.py` |

---

## The Streamlit app

A chat with the plant-model agent (`backend_core/agent.py`): one LLM call for the conversation and,
after every draft, a second one for its metadata (states, equations ...), checked against the code.
Attach files with 📎:

- **PDF / Markdown** — indexed and searched on every turn (RAG).
- **Simulink model** (`.slx`) and the MATLAB script (`.m`) of its parameters — converted (below).
- **Generated C code** (`.c` / `.h` or a zip) or the zip of `c2py_export.m` — converted (below).

Every draft is simulated in a restricted sandbox (`backend_core/sandbox.py`); the **Simulation Lab**
runs it with your own inputs. Web search is optional (toggle in the sidebar).

### Simulink conversion

| Upload | Route | Result |
|--------|-------|--------|
| `.slx` | **direct**: read from the model's XML — no MATLAB, no LLM | `dynamics`, `output` and, with discrete memory, `update`; checked against the engine's block model at 200 random points and in simulation |
| `.slx` it cannot read (library blocks, Stateflow charts ...) | **C code**: the zip of `c2py_export.m` (route A), or `<model>.c`, `.h`, `_data.c` from Embedded Coder (route B) | C translated to Python, rewritten readable by an LLM, validated (route A also against Simulink's own results) |

- **The whole model is converted** — every state, not only what reaches an Outport. Sinks (Scope,
  Display ...) are left out; an unsupported block that reaches no output is left out with a note.
- **Missing Inport / Outport:** the app shows the model's block diagram, and the user clicks the
  signals that are the inputs u and the outputs y (several of each; ◆ marks an input port nothing is
  connected to). Nothing is converted before that choice — or "Convert as it is".
- **MATLAB workspace variables:** from the model's own callbacks (InitFcn ...), an attached `.m`
  script, or the chat. `m = 2` is read by a parser; values in words ("m is two", «جرم دو کیلوگرم»)
  by an LLM, whose numbers must appear in the message.
- **Plots** show the outputs y (the states when a draft has no `output()`).

Not supported by the direct route: blocks stored outside the `.slx` (toolbox libraries, referenced
models, S-Functions), Stateflow charts, Simscape.

**Example models:** `examples/simulink_models/` — 19 models that convert directly, with a README.

### Code map

```
app.py                       # the Streamlit app (UI, turns, the signal picker panel)
backend_core/
├── agent.py                 # PlantModelAgent: conversation + metadata calls
├── sandbox.py               # restricted exec of drafted code, simulation, plots
├── rag.py, websearch.py     # document search, web search
├── openai_client.py, logging_utils.py, json_extract.py, steps.py
├── prompts/                 # every LLM prompt, as YAML — prompts/README.md maps file / key -> use
└── simulink/
    ├── intake.py            # what was uploaded, what is missing, the notes for the agent
    ├── llm.py               # the LLM of the conversion
    ├── direct/              # route 1: .slx -> equations
    │   ├── slx_direct.py        # convert_slx(): conversion + checks -> DirectResult
    │   ├── diagram.py           # the block diagram and the signal choice
    │   ├── chat_values.py       # values of missing variables from a chat message
    │   └── slx2py/              # the vendored SLX_TO_Python engine (topology, blocks, equations)
    └── c2py/                # route 2: generated C code -> Python
        ├── convert.py           # convert_package() / convert_c_code(), simulate()
        ├── c2py.py, flatten.py, agent.py, equations.py, validation.py, package.py, cost.py
        └── c2py_export.m        # the MATLAB script the user runs (attached by the app)
```

### Wiring Simulink into the API

Plain Python functions, no Streamlit inside:

| Function | Does |
|----------|------|
| `simulink.intake.classify(name, data)` | what an uploaded file is (model, script, C code, package ...) |
| `simulink.direct.slx_direct.convert_slx(file_name, data, runs_dir, scripts=, signals=)` | the direct conversion -> `DirectResult` (`ok`, `draft`, `reason`, `missing_variables`, `no_input` ...) |
| `simulink.direct.diagram.read_diagram(data)`, `figure(...)`, `choice(diagram, inputs, outputs)` | the block diagram, as data and as a plotly figure; the user's choice -> `signals=` |
| `simulink.direct.chat_values.read_values(text, missing, uses, llm)` | values of missing variables from a message |
| `simulink.c2py.convert.convert_package(...)`, `convert_c_code(...)`, `simulate(...)` | the C-code routes and the simulation of a converted model |

How `app.py` strings them together: `run_direct_conversion`, `run_conversion`, `submit` and
`render_signal_picker`. Files of every conversion go to `simulink_runs/` (git-ignored).

---

## Minimal API and React frontend

```bash
python run_api.py                      # http://localhost:8000, docs at /docs
cd frontend && npm install && npm run dev
```

| Endpoint | Behaviour |
|----------|-----------|
| `POST /api/v1/plant-model/chat` | one chat turn (`backend_core/AgentPlant`) |
| `GET/DELETE /api/v1/plant-model/conversations[/{id}]` | in-memory conversations |
| `POST /api/v1/plant-model/upload`, `GET .../attachments` | documents for RAG |
| `POST /api/v1/plant-model/simulate` | simulate a drafted `dynamics()` |
| `GET /api/v1/health`, `/models`, `/auth/me`, `/errors/config` | stubs (no DB, a fixed dev user) |

```bash
curl -s http://localhost:8000/api/v1/health
curl -s -X POST http://localhost:8000/api/v1/plant-model/chat -H "Content-Type: application/json" \
  -d '{"messages":[],"user_message":"Design a simple DC motor position plant","model":"gpt-4o-mini"}'
```

`AGENTPLANT_README.md` has more on the React chat and the API (demo mode, the event loop). The core
agent also runs as a CLI: `python backend_core/AgentPlant/run_cli.py` (see its `GUIDE.md`).

---

## Tests

```bash
pip install pytest                     # for the tests only
python -m pytest tests -q --ignore=tests/test_event_loop_nonblocking.py
```

About 5 minutes; no MATLAB and no API key needed. The Simulink results the block tests compare with
are stored in `tests/fixtures/*_reference.zip`; `tests/test_example_models.py` converts every example
model. `test_event_loop_nonblocking.py` needs a running API (see its docstring).

---

## What changed since the extraction

Unchanged from the original extraction: `backend_core/AgentPlant/`, `backend_api/`, `minimal_api/`,
`streamlit_core/`, `frontend/`, `packages/labcd_agents/`.

New or changed in release v1:

- `app.py` and `backend_core/` (agent, sandbox, rag, websearch ...): the Streamlit app.
- `backend_core/simulink/`: the Simulink conversion (direct and C-code routes).
- `backend_core/prompts/`: every prompt of the app as YAML.
- `tests/`, `examples/simulink_models/`, `.env.example`, `requirements.txt` (+ streamlit).

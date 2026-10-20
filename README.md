# AgentPlant Minimal Extraction

This is a **byte-for-byte identical** extraction of the AgentPlant / PlantModelChat subsystem from the [LabCD_Application](https://github.com/labcd-dev/LabCD_Application) repository.

**Guarantee:** No functional source code was modified. All original files retain original content and relative structure so that later improvements can be cleanly merged back into the main monorepo.

---

## What is AgentPlant?

AgentPlant is LabCD’s plant-model chatbot. Users describe physical dynamics in natural language (physics, transfer functions, or ODEs). The agent produces:

1. A Python `dynamics(t, x, u)` function (and related code)
2. Structured metadata (states, inputs, equations, parameters, assumptions)

It uses a **two-call architecture**:

| Call | Prompt | Role |
|------|--------|------|
| Primary agent | `plant_model_agent.yaml` | Conversation only. Emits one of `continue` / `draft` / `complete` with `system_name` + `python_code`. Never emits structured metadata. |
| Metadata agent | `plant_model_metadata.yaml` | After every successful draft or complete, reads the finished `python_code` and emits full metadata. A numerical verifier checks `state_equations` against `dynamics()`; one repair retry is allowed. |

The UI surfaces this as the “AgentPlant AI” chat (hosted on the Design page), with code preview, model picker, and hand-off into the rest of the control-design pipeline.

---

## Directory layout

```
agentplant-minimal/
├── README.md                          # this file
├── requirements.txt                   # minimal Python deps for R&D
├── .gitignore
│
├── backend_core/
│   └── AgentPlant/                    # ★ core agent (canonical)
│       ├── agent.py                   # PlantModelAgent + two-call logic
│       ├── plant_model_agent.yaml     # primary conversation prompt
│       ├── plant_model_metadata.yaml  # metadata extraction prompt
│       ├── run_cli.py                 # local CLI runner
│       ├── __init__.py
│       └── GUIDE.md                   # detailed agent behaviour guide
│
├── backend_api/
│   ├── PlantModelChat/                # ★ API-facing agent adapter
│   │   ├── agent.py                   # thinner wrapper used by HTTP layer
│   │   ├── prompts/
│   │   │   └── plant_model_agent.yaml
│   │   ├── __init__.py
│   │   └── GUIDE.md
│   └── http/                          # plant-model HTTP surface
│       ├── routers/plant_model.py     # FastAPI routes (/plant-model/*)
│       ├── schemas/plant_model.py     # request/response models
│       └── services/
│           ├── plant_model_service.py
│           ├── plant_model_chat_service.py
│           └── plant_artifact_service.py
│
├── packages/
│   └── labcd_agents/                  # shared LLM-agent foundation
│       ├── src/labcd_agents/          # BaseAgent, prompts, providers, …
│       ├── tests/
│       ├── pyproject.toml
│       └── README.md
│
└── frontend/                          # minimal React UI shell
    ├── package.json
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig*.json
    └── src/
        ├── App.tsx                    # ★ thin scaffolding (new) — mounts DesignPage only
        ├── main.tsx
        ├── index.css
        ├── pages/
        │   └── DesignPage.tsx         # hosts <PlantModelChat />
        ├── components/
        │   ├── PlantModelChat.tsx     # main chat UI (“AgentPlant AI”)
        │   ├── CodePreview.tsx
        │   ├── ComposerModelPicker.tsx
        │   ├── MarkdownContent.tsx
        │   ├── StatusMessage.tsx
        │   ├── PreLaunchModal.tsx
        │   └── landing/landing.css
        ├── api/
        │   ├── client.ts
        │   ├── endpoints.ts           # plantModelApi, plantArtifactApi, …
        │   ├── types.ts
        │   └── sse.ts
        ├── context/
        │   ├── AuthContext.tsx
        │   └── PipelineContext.tsx
        └── lib/
            ├── classes.ts
            └── modelPicker.ts
```

### Source mapping note

The upstream repository does **not** contain a directory named `backend_api/AgentPlant`.  
The API-side counterpart is `backend_api/PlantModelChat/`.  
Both the core agent (`backend_core/AgentPlant`) and the API adapter were extracted exactly as they exist upstream.

---

## Key files at a glance

| Path | Purpose |
|------|---------|
| `backend_core/AgentPlant/agent.py` | Full `PlantModelAgent` with primary + metadata calls, session state, draft/complete logic, numerical verification |
| `backend_core/AgentPlant/plant_model_agent.yaml` | System prompt for conversational plant modelling |
| `backend_core/AgentPlant/plant_model_metadata.yaml` | System prompt that forces metadata to match generated code |
| `backend_core/AgentPlant/run_cli.py` | Standalone CLI for interactive testing of the agent |
| `backend_core/AgentPlant/GUIDE.md` | Authoritative behaviour guide (conversation shapes, examples, edge cases) |
| `backend_api/PlantModelChat/agent.py` | Slightly thinner agent used by the HTTP service layer |
| `backend_api/http/routers/plant_model.py` | FastAPI routes: chat, conversations, artifacts, validation |
| `frontend/src/components/PlantModelChat.tsx` | React chat UI (archetypes, streaming, sidebar metadata, “Use model” hand-off) |
| `frontend/src/pages/DesignPage.tsx` | Page that mounts the chat and the pre-launch modal |
| `packages/labcd_agents/` | Shared `BaseAgent`, prompt library, provider adapters, token/pricing helpers |

---

## Setting up a new virtual environment

Use an isolated Python environment so AgentPlant R&D does not interfere with system packages or other projects.

### Prerequisites

- Python 3.10 or newer (`python3 --version`)
- `pip` and `venv` (usually bundled with Python)
- Optional but recommended: Node.js 18+ if you will run the frontend shell

### Create and activate the venv

**Linux / macOS (bash/zsh):**

```bash
# From the agentplant-minimal root
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Confirm you are inside the venv
which python   # should point to .../agentplant-minimal/.venv/bin/python
python --version
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

When the venv is active, the prompt typically shows `(.venv)`.

### Deactivate

```bash
deactivate
```

### Re-activate later

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Install Python dependencies inside the venv

With the venv **activated**:

```bash
# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install the shared agent foundation (editable)
pip install -e "./packages/labcd_agents[all]"

# Install the rest of the minimal requirements
pip install -r requirements.txt
```

Optional: if you only need a specific provider (smaller install):

```bash
pip install -e "./packages/labcd_agents[openai]"   # or [groq], [anthropic], etc.
```

### Environment variables (LLM keys)

The agent needs at least one provider key. Create a `.env` file in the repo root (it is git-ignored) or export variables in the shell:

```bash
# Example — adjust to the providers you use
export OPENAI_API_KEY="sk-..."
# export GROQ_API_KEY="..."
# export ANTHROPIC_API_KEY="..."
```

Or use a `.env` file that `python-dotenv` will load when present.

### Frontend (separate Node environment)

Node uses its own dependency isolation via `node_modules` (no Python venv required):

```bash
cd frontend
npm install
```

You can run the frontend while the Python venv is active or inactive; the two environments are independent.

### Quick “from zero” checklist

```bash
# 1. Create & activate Python venv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# 2. Install Python deps
python -m pip install --upgrade pip
pip install -e "./packages/labcd_agents[all]"
pip install -r requirements.txt

# 3. Set LLM keys
export OPENAI_API_KEY="sk-..."     # or write them into .env

# 4. (Optional) Frontend
cd frontend && npm install && cd ..
```

---

## Running the minimal API (uvicorn)

This tree includes a **minimal FastAPI app** that runs the core `PlantModelAgent` without DB or real auth.

### 1. Virtualenv + install

```bash
cd agentplant-minimal
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. API keys

```bash
export OPENAI_API_KEY="sk-..."
# optional: GROQ_API_KEY, ANTHROPIC_API_KEY, etc.
```

Or put keys in a `.env` file at the repo root (`python-dotenv` loads it).

### 3. Start uvicorn

```bash
# from agentplant-minimal root, venv active
python run_api.py
# equivalent:
# uvicorn minimal_api.app:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- OpenAPI docs: http://localhost:8000/docs  
- Chat: `POST /api/v1/plant-model/chat`

Stubs (no DB):

| Endpoint | Behaviour |
|----------|-----------|
| `GET /api/v1/health` | `{ "status": "ok" }` |
| `GET /api/v1/models` | list of model ids |
| `GET /api/v1/auth/me` | fixed dev user |
| `GET /api/v1/errors/config` | reporting disabled |
| `GET/DELETE /api/v1/plant-model/conversations` | in-memory only |

### 4. Frontend against this API

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` → `http://localhost:8000` by default (`vite.config.ts`).  
You should no longer see `ECONNREFUSED` once uvicorn is up.

### Quick smoke test (no UI)

```bash
curl -s http://localhost:8000/api/v1/health
curl -s http://localhost:8000/api/v1/models | head
curl -s -X POST http://localhost:8000/api/v1/plant-model/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[],"user_message":"Design a simple DC motor position plant","model":"gpt-4o-mini"}'
```

### What is intentionally omitted

- PostgreSQL / SQLAlchemy persistence  
- Real login, JWT, roles, credits  
- Artifacts, validation, pipeline hand-off services  

Core chat + draft/complete + metadata path is live for R&D / TDD (prompts, RAG experiments, sandbox simulation of drafted dynamics, etc.).

---

## Dependencies

### Python (backend)

See `requirements.txt`. After the venv is active and packages are installed as above, the agent depends on:

- `labcd_agents` (`BaseAgent`, `PromptLibrary`, `extract_json_from_response`)
- LangChain core + provider SDKs (via the `[all]` extra)
- FastAPI / SQLAlchemy / etc. for the HTTP surface (listed in `requirements.txt`)

The full LabCD stack also needs a database, auth, Redis, and LLM provider keys; those are **not** included in this minimal tree.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI expects a running backend that implements the `/plant-model/*` endpoints and normal auth. For pure front-end R&D you will need to mock `plantModelApi` / auth or point the API base URL at a full LabCD instance.

---

## Running for R&D

### Core agent (CLI)

With the venv activated and LLM keys set:

```bash
# from repo root
python -m backend_core.AgentPlant.run_cli
# or
python backend_core/AgentPlant/run_cli.py
```

See `backend_core/AgentPlant/GUIDE.md` for expected conversation shapes and test cases.

### Frontend shell only

```bash
cd frontend
npm install
npm run dev
```

Opens the Design page with the AgentPlant chat. Without a backend the chat will fail on network calls; the component itself still renders.

### Full stack

This tree is intentionally incomplete as a full application. To exercise the complete path (chat → artifact → pipeline hand-off) you need the rest of LabCD_Application (DB models, auth, job runners, etc.).

---

## Streamlit app and Simulink conversion

`app.py` is a Streamlit app around the same plant-model agent (`backend_core/agent.py`). Besides the
chat it converts **Simulink models** into a validated `dynamics(t, x, u)`:

```bash
pip install -r requirements.txt
cp .env.example .env               # then set OPENAI_API_KEY
streamlit run app.py
```

On Windows, keep the repo in a short path (or enable long paths): some packages pulled by
`labcd_agents[all]` have very long file names, and `pip install` fails deep in a folder tree.

> **Note for the website:** the Simulink features live only in the Streamlit app. `minimal_api/` and
> `frontend/` do not call them yet. The entry points to wire into the API are listed below.

### What it does

| Upload | Route | Result |
|--------|-------|--------|
| `.slx` | **direct**: read from the model's XML, no MATLAB, no LLM | `dynamics(t, x, u)`, `output(t, x, u)` (the outputs y) and, with discrete memory, `update(t, x, u)`, checked against the engine's block model at 200 random points and in simulation |
| `.slx` it cannot read (library blocks, Stateflow charts ...) | **C code**: the zip of `c2py_export.m`, or `<model>.c`, `.h`, `_data.c` from Embedded Coder | C translated to Python, rewritten readable by an LLM, validated |

- **The whole model is converted**: every state, not only what reaches an Outport. Sinks (Scope, Display ...) are left out. An unsupported block that reaches no output is left out with a note.
- **No Inport / Outport in the model**: the app shows the model's block diagram (`diagram.py`), and the user clicks the signals that are the inputs u and the outputs y. Nothing is converted before that choice. Unconnected input ports (◆) can be chosen as inputs too.
- **MATLAB workspace variables**: values come from the model's callbacks (InitFcn ...), an attached `.m` script or the chat. In the chat, `m = 2` is read by a parser. Values in words ("m is two", «جرم دو کیلوگرم») go to an LLM, and its answer is checked: every number must be in the message.
- **Plots** show the outputs y; the states when a draft has no `output()`.

**Example models** to try: `examples/simulink_models/` (19 models that convert directly, with a README
of what each shows).

### Layout

```
backend_core/
├── agent.py, sandbox.py, rag.py, websearch.py, openai_client.py   # chat agent, restricted exec + simulation
├── prompts/              # every LLM prompt, as YAML (see prompts/README.md)
└── simulink/
    ├── intake.py         # what was uploaded, what is missing, the notes for the agent
    ├── llm.py            # the LLM of the conversion (AGENTPLANT_SIMULINK_MODEL)
    ├── direct/           # route 1: .slx -> equations
    │   ├── slx_direct.py     # convert_slx(): conversion + checks -> DirectResult
    │   ├── diagram.py        # read_diagram(), figure(), choice(): the signal picker
    │   ├── chat_values.py    # values of missing variables from a chat message
    │   └── slx2py/           # the vendored SLX_TO_Python engine (topology, blocks, dynamics ...)
    └── c2py/             # route 2: generated C code -> Python
        ├── convert.py        # convert_package() / convert_c_code(), simulate()
        ├── c2py.py, flatten.py, agent.py, equations.py, validation.py, package.py, cost.py
        └── c2py_export.m     # the MATLAB script the user runs (served by the app)
```

Entry points for an API: `slx_direct.convert_slx(file_name, data, runs_dir, scripts=, signals=)`,
`diagram.read_diagram(data)` + `diagram.choice(diagram, inputs, outputs)`,
`chat_values.read_values(text, missing, uses, llm)`, `c2py.convert.convert_package(...)` /
`convert_c_code(...)`, `intake.classify(name, data)`. Conversion files go to `simulink_runs/` (git-ignored).

### Tests

```bash
pip install pytest                 # for the tests only
python -m pytest tests -q --ignore=tests/test_event_loop_nonblocking.py
```

`tests/test_example_models.py` converts every example model (about 40 s).
No MATLAB is needed: the Simulink results the block tests compare with are stored in
`tests/fixtures/*_reference.zip`. `test_event_loop_nonblocking.py` needs a running API (see its docstring).

---

## Design principles of this extraction

1. **Zero functional changes** — every original file is byte-identical to upstream.
2. **Minimal surface** — only the chat page and the smallest set of files required for that page to compile are present on the frontend.
3. **Clear ownership** — core logic lives in `backend_core/AgentPlant`; the HTTP adapter and UI are thin consumers.
4. **Merge-friendly** — relative paths match the monorepo, so improved files can be copied or cherry-picked back with minimal conflict.

---

## Suggested next steps for R&D

- Improve prompt quality / metadata verification inside `backend_core/AgentPlant`
- Add unit tests around the two-call flow and the numerical verifier
- Experiment with alternative providers via `labcd_agents` extras
- Iterate on the chat UX in `PlantModelChat.tsx` in isolation
- When ready, open focused PRs against the corresponding paths in LabCD_Application

---

## Suggested commit / PR message (for re-integration)

**Title:**  
`Extract AgentPlant / PlantModelChat subsystem as unmodified minimal tree for isolated R&D`

**Body:**  
This extraction was performed with zero functional code changes. All original sources retain identical content and relative paths so that later improvements can be cleanly merged back into LabCD_Application. Only thin scaffolding (README, a minimal App.tsx, requirements.txt) was added to make the tree a usable standalone starting point.

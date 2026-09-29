# AgentPlant (Streamlit)

A greenfield rebuild of the [AgentPlant / plant-model chat](https://github.com/mnarimani/agent-plant-minimal)
concept as a single-process Streamlit app. It keeps the core agent
contract — the primary agent emits **`continue` / `draft` / `complete`**,
and every draft/complete carries a runnable **`dynamics(t, x, u)`**
Python function — and adds OpenAI-native RAG, OpenAI-native web search,
a restricted-exec sandbox with an ODE simulator, and a staged chat UX.

No custom vector DB, no third-party search provider (no Tavily), no
FastAPI/React split — everything runs from `streamlit run app.py`.

---

## What it does

1. **Chat** with an agent that turns a description (prose, equations, or
   both) of a physical/engineered system into a Python `dynamics(t, x, u)`
   function, via the same three-shape JSON contract as the reference
   project:
   - `{"status": "continue", "reply": "..."}` — one focused question.
   - `{"status": "draft", "reply": "...", "system_name": "...", "python_code": "..."}`
     — a proposed model, open to revision.
   - `{"status": "complete", "system_name": "...", "python_code": "..."}`
     — finalized once you accept a draft (or paste a finished model).
2. **RAG** — upload a PDF/MD/TXT; it's indexed with OpenAI Files + a
   Vector Store and retrieved via the `file_search` hosted tool. The UI
   shows exactly what was retrieved (source file, score, snippet), not
   just the model's paraphrase.
3. **Web search** — off by default. When enabled, a small planning call
   decides whether a *specific* fact is actually needed (a datasheet
   value, a named standard, a real constant) and proposes one focused
   query; a heuristic filter then rejects generic/vague queries (e.g.
   "state variable definition", "plant model") before anything is sent
   to OpenAI's `web_search` tool. The query and a short brief are always
   shown.
4. **Sandbox** — every new/changed draft is restricted-exec'd (AST
   whitelist + minimal builtins — see [Sandbox security note](#sandbox-security-note))
   and auto-simulated once (step input) so you immediately see a plot.
   A "Simulation Lab" panel lets you re-run with step / ramp / sine on
   any input channel, your own initial state, and duration.
5. **Staged UX** — a live "circle + status" indicator narrates what's
   happening turn by turn, collapsing into an expandable step count once
   finished; the system's states/inputs/parameters appear as an editable
   sheet you can revise in one click.

---

## Staged UX, stage by stage

| Stage | Behavior |
|---|---|
| 0 | Bare chat. Sidebar: 🔍 **Search** toggle (default off) and a PDF/MD **upload** that enables RAG. |
| 1 | The agent asks focused questions to pin down the system; once a draft exists, an **editable "system sheet"** (states, parameters, inputs/outputs, assumptions) appears, with a button to send edits back as a revision request. |
| 2 | On the first draft: a live circle + status walks through what's happening (e.g. *"Inspecting uploaded pdf file datasheet.pdf…"*), then a **sandbox simulation card** (Plotly) appears automatically. |
| 3 | On later turns: the same circle shows **one status line** while working; once done it collapses under **"N steps"**. |
| 4 | Every past turn keeps its step log as an **expandable** element in the transcript — nothing disappears after the run finishes. |

This maps directly onto Streamlit's `st.status(...)` widget: it renders
the spinner/checkmark ("circle"), its `label=` is the single status line,
and `expanded=False` after completion gives the collapsed "N steps" that
stays clickable in history.

---

## Project layout

```
agentplant_streamlit/
├── app.py                          # Streamlit UI — stages 0-4, wires everything together
├── backend_core/
│   ├── agent.py                    # PlantModelAgent: continue/draft/complete + metadata call
│   ├── rag.py                      # OpenAI Files + Vector Store + file_search
│   ├── websearch.py                # OpenAI web_search + focused-query gate
│   ├── sandbox.py                  # restricted exec of dynamics() + ODE sim + Plotly figure
│   ├── openai_client.py            # logged Responses-API wrapper (get_client, responses_call)
│   ├── json_extract.py             # permissive JSON extraction from LLM text
│   ├── steps.py                    # Step dataclass for the staged step log
│   ├── logging_utils.py            # prompt/response logging under .logs/
│   └── prompts/
│       ├── plant_model_agent.yaml       # primary agent system/user prompt
│       └── plant_model_metadata.yaml    # metadata-extraction system/user prompt
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

Requires **Python 3.10+**.

```bash
cd agentplant_streamlit
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then edit .env and set OPENAI_API_KEY
# or: export OPENAI_API_KEY=sk-...

streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Used for every call: agent, metadata, RAG, web search. |
| `AGENTPLANT_MODEL` | `gpt-4.1` | Model used for all Responses API calls. |
| `AGENTPLANT_LOG_DIR` | `.logs` | Where prompt/response JSONL logs are written. |

---

## Try it

- *"a DC motor driving a beam with a ball rolling on it, on the moon"* →
  a few clarifying questions, then a draft with `g = 1.62 m/s^2` labelled
  as an assumption.
- Paste a finished set of state equations directly → the agent moves
  straight to a draft (or complete, if you say so).
- Upload a datasheet PDF, then ask *"what's the damping ratio for this
  actuator?"* → watch the RAG step show the retrieved passage.
- Turn on 🔍 **Web search** and ask about a specific real part number →
  watch the planner propose a focused query and the search step show the
  brief + links. Ask something generic like "what's a state variable?"
  and note the search is skipped.
- Once you have a draft, open **🧪 Simulation Lab**, pick `sine` on
  `u[0]`, and re-run.

---

## Fixes from real-run feedback

Two issues surfaced from actual `.logs/` traces and were fixed at the root cause, not just papered over:

**1. Sandbox failures now get an automatic diagnosis-and-repair pass, and the underlying numerics are more robust in the first place.**
`scipy`'s adaptive RK45 can fail with a cryptic `Required step size is less than spacing between numbers` in two distinct situations, both now handled:
- A **state that runs away** (an unstable/insufficiently-damped pole) is caught by a terminal integration event at a fixed magnitude threshold, well before float64 overflow — the simulation stops cleanly and a genuinely useful *partial* trajectory is still returned and plotted, labelled as `diverged`.
- A **near-singularity** (e.g. `1/cos(theta)` close to +/-90deg) is a different failure shape: the state itself stays bounded while the derivative grows enormous. Empirically, this needed a check *inside* the dynamics call itself (reached on every trial evaluation, not just once per accepted step) — a scipy event checked only post-step did not catch it in reasonable wall-clock time, because the accepted-state derivative grows far more slowly than the solver's own unaccepted trial-stage evaluations that actually balloon the search.
- Only a genuine **hard failure** (the singularity case, or a solver error) triggers automatic diagnose-and-repair: the failure is fed back to the agent in a synthetic follow-up turn asking it to diagnose the likely modelling bug and revise `python_code`, which is then re-simulated once automatically — all shown as extra steps in that turn's step log. A **diverged-but-successful** run (state ran away, partial trajectory returned) is deliberately *not* auto-repaired, since some real plants are open-loop unstable by design; the sim card explains the divergence and leaves the call to the user instead of risking the agent "fixing" a correct model into an incorrect one just to make the smoke test converge.

**2. The web-search query planner is now given whatever RAG already retrieved this turn, and no longer mistakes a plain sentence for a proper noun.**
Without the retrieved context, a message like *"refer to the attached content"* gave the planner nothing to work with; it replied with a request back to the user ("Please upload the content...") instead of a query or `NONE`. Two fixes: `plan_query` now receives `retrieved_context` and is told explicitly not to search when the user is just pointing at an already-retrieved file; and `looks_focused`'s proper-noun heuristic no longer credits a query's *first* word for being capitalized (that's just English sentence-capitalization, not evidence of a real proper noun) — which is what let that request-shaped sentence slip through the focus filter in the first place.

## Design notes

**Why RAG/web-search retrieval is a separate call from the agent's JSON
turn.** The primary agent's turn must reliably return a single JSON
object; OpenAI's hosted tools (`file_search`, `web_search`) are instead
called in their own plain-text (non-JSON) request, and the result is
handed to the agent as plain text (`retrieved_context` / `web_context`
placeholders in the prompt). This sidesteps a known rough edge where
combining `text.format=json_schema` with `file_search`/`web_search` in
one call can occasionally return malformed JSON — and it keeps each
concern in its own step for the UI's step log.

**Why the agent object lives in `st.session_state`, not a stateless
API.** The reference project's `PlantModelAgent` was designed to be
re-hydrated per HTTP request (`apply_session_state` / `export_session_state`)
because a FastAPI worker doesn't persist Python objects between requests.
This app is a single continuous Streamlit session per user, so the
`PlantModelAgent` instance (with `draft_count` / `latest_draft`) simply
lives in `st.session_state` for the life of the session — no
serialize/rehydrate step needed.

**Metadata verification.** The reference project runs a numeric verifier
that checks the metadata's `state_equations` against the actual code
(`backend_core/plant_compiler`), which isn't part of the public minimal
extraction. This rebuild instead does shape validation (right number of
states, required keys present) with one repair retry, and simply omits
metadata if it still doesn't validate — the system sheet degrades
gracefully rather than blocking the conversation.

### Sandbox security note

`backend_core/sandbox.py` restricts drafted `dynamics()` code with (1)
an AST whitelist — only `numpy`/`math`/`scipy` imports, no dunder
attribute access, no `with`/`try`/`global`/`nonlocal` — and (2) execution
with a minimal, hand-picked `__builtins__` (no `open`, `exec`, `eval`,
unrestricted `__import__`, `os`, `sys`, etc.). This is solid
defense-in-depth against an accidental or lightly-adversarial draft (a
stray `import os`, a placeholder `exec(...)`) but it is **process-level
Python restriction, not a hardened sandbox** — it does not protect
against a determined attacker with arbitrary compute/DoS intent (e.g. an
infinite loop) the way a subprocess/container/resource-limited boundary
would. If you expose this app beyond trusted users, put the sandbox
behind a real OS-level boundary (subprocess with a timeout + memory cap,
or a container) in front of `compile_dynamics`/`simulate`.

### Logging

Every LLM call (agent turns, metadata extraction, RAG grounding, web
query planning, web search) is appended as one JSON line to
`.logs/YYYY-MM-DD.jsonl` — timestamp, call kind, model, prompts, and
response text (each field truncated at 20k characters). Logging never
raises; a logging failure is swallowed so it can't break the app.

---

## What's intentionally not included

- A separate HTTP API / React frontend (the original repo's split) — this
  app is a single Streamlit process by design.
- A custom vector database — RAG is OpenAI Files + Vector Store only.
- Any non-OpenAI search provider.
- Multi-user auth, persistence across processes, or a production-grade
  sandbox boundary (see the security note above).

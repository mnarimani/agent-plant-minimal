# AgentPlant — mockup implementation (cld + streamlit_core)

Implements the **approved ChatGPT-style mockup** (`artifacts/chatgpt-style-ui.html`) on top of the `cld` frontend and `agentplant_streamlit/backend_core` domain logic.

## Quick start

```bash
cd agentplant-app

# API
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Demo mode (no OpenAI key): canned sim + local “upload” bookkeeping
export AGENTPLANT_DEMO=1
# Or real mode:
# export OPENAI_API_KEY=sk-...
# unset AGENTPLANT_DEMO
python run_api.py
# → http://0.0.0.0:8000  docs at /docs

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → Vite proxies /api to the backend (see vite.config.ts)
```

### Demo vs real mode

| Mode | When | Behavior |
|------|------|----------|
| **Demo** | `AGENTPLANT_DEMO=1` or no `OPENAI_API_KEY` for upload/sim shortcuts | Upload records filename only; simulate returns canned trajectories; chat still needs a real key unless you stub the agent separately |
| **Real** | `OPENAI_API_KEY` set, `AGENTPLANT_DEMO` unset | RAG indexes via OpenAI Files + vector store; sandbox executes drafted `dynamics()` with scipy |

## Mockup surface → component → backend

| Mockup surface | Frontend | Backend |
|----------------|----------|---------|
| Chat bubbles + continue/draft/complete pills | `AgentPlantChat.tsx` | `POST /api/v1/plant-model/chat` → `backend_core.AgentPlant.PlantModelAgent.step` |
| Tool-trace (collapsed by default) | `AgentPlantChat` steps expander | Additive `steps[]` on chat response (`kind` / `label` / `detail`) — aligned with `streamlit_core/steps.py` |
| PY artifact + code panel | Artifact card → panel `code` | `draft` / `final_result.python_code` |
| Sandbox artifact + Simulation Lab (collapsed by default) | Artifact card → panel `sandbox` | `POST /api/v1/plant-model/simulate` → `streamlit_core/sandbox.py` |
| Sim success / diverged / fail | Status line + plot | `SimResult.success`, `diverged`, `message`, `attempts` |
| Search toggle | Composer ◎ | `web_search_enabled` on chat body (focus-gate lives in `streamlit_core/websearch.py`; wire deeper in a follow-up) |
| File attach | Composer 📎 | `POST /api/v1/plant-model/upload` → `streamlit_core/rag.py` |
| Hover-only metas | CSS `.ap-meta` | Frontend only |

**Not in this UI:** the discarded stage-tab harness (0–5). Progressive disclosure matches the approved mockup instead.

## Event-loop / blocking OpenAI

Blocking work (OpenAI SDK, scipy simulate, artificial sleeps) is offloaded with `run_in_threadpool` / a shared `ThreadPoolExecutor` in `minimal_api/extras.py`. Chat `agent.step` is also run via `run_in_threadpool`.

### Verification

```bash
# terminal 1
AGENTPLANT_DEMO=1 uvicorn minimal_api.app:app --port 8765

# terminal 2
python tests/test_event_loop_nonblocking.py
# expects: OK: /health stayed responsive during slow probe
```

The test starts a **slow** request (`/_probe/slow?seconds=2`, sleeps in a worker thread) and, while it is in flight, calls `/api/v1/health` with a **1.5s** client timeout. If the event loop were blocked by a sync sleep on the main thread, health would stall and the assertion would fail.

Demo-mode unit tests alone are **not** treated as proof; this concurrent probe is.

## New / touched files (scoped)

- `minimal_api/extras.py` — upload, simulate, probes
- `minimal_api/app.py` — include router; additive chat fields; async chat + threadpool
- `streamlit_core/` — copy of Streamlit `backend_core` (rag, sandbox, websearch, steps, …)
- `frontend/src/components/AgentPlantChat.tsx` + `agentplant-chat.css`
- `frontend/src/pages/DesignPage.tsx` — mounts `AgentPlantChat`
- `frontend/src/api/types.ts`, `endpoints.ts` — upload / simulate / additive chat fields
- `tests/test_event_loop_nonblocking.py`
- `AGENTPLANT_README.md` (this file)

## Done criteria checklist

- [x] Mockup layout: chat left, artifact panel right, lab collapsed by default, metas on hover  
- [x] Chat continue/draft/complete surface  
- [x] Code + sandbox artifacts  
- [x] Upload + simulate API (demo + real paths)  
- [x] Event-loop probe test documented  
- [ ] Full websearch focus-gate + RAG chunk injection into agent.step context (plumbing present; deepen when keys available)  
- [ ] E2E multi-page PDF against live OpenAI (requires key + network in the target environment)

"""AgentPlant — Streamlit chat app.

Staged UX (see README for the full table):

  Stage 0  Bare chat; Search toggle; PDF/MD upload enables RAG.
  Stage 1  Agent asks for system context; an editable "system sheet"
           (states / inputs / parameters) appears once a draft exists.
  Stage 2  First draft: circle + live status ("Inspecting uploaded pdf
           file X.pdf...") -> automatic sandbox simulation card.
  Stage 3  Later turns: same circle, one status line while running; the
           full step list collapses under "N steps" once finished.
  Stage 4  Every past turn's step log stays expandable in the transcript.

Run: `streamlit run app.py` (see README.md for setup).
"""

from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from backend_core import agent as agent_mod
from backend_core import openai_client
from backend_core import rag as rag_mod
from backend_core import sandbox as sandbox_mod
from backend_core import websearch as websearch_mod
from backend_core.steps import Step

st.set_page_config(page_title="AgentPlant", page_icon="🧭", layout="wide")

_ACK_RE = re.compile(
    r"^\s*(yes|yep|yeah|sure|ok|okay|finish|done|looks?\s+good|good|perfect|"
    r"ship\s*it|confirm|approved|that'?s\s+(good|fine|ok|okay))\W*$",
    re.IGNORECASE,
)


def _is_bare_ack(message: str) -> bool:
    return bool(_ACK_RE.match(message.strip()))


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return slug or "dynamics"


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------


def _bootstrap() -> None:
    if "client" not in st.session_state:
        try:
            st.session_state.client = openai_client.get_client()
            st.session_state.api_key_error = None
        except Exception as exc:
            st.session_state.client = None
            st.session_state.api_key_error = str(exc)

    if st.session_state.get("client") is not None:
        if "pm_agent" not in st.session_state:
            st.session_state.pm_agent = agent_mod.PlantModelAgent(st.session_state.client)
        if "rag" not in st.session_state:
            st.session_state.rag = rag_mod.OpenAIRag(st.session_state.client)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("web_search_enabled", False)
    st.session_state.setdefault("last_simmed_code", None)
    st.session_state.setdefault("lab_last_sim", None)


_bootstrap()

if st.session_state.api_key_error:
    st.title("🧭 AgentPlant")
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "Export it, or put it in a `.env` file in the project root:\n\n"
        "```bash\nexport OPENAI_API_KEY=sk-...\n```\n\n"
        "Then restart `streamlit run app.py`."
    )
    st.stop()

client = st.session_state.client
pm_agent: agent_mod.PlantModelAgent = st.session_state.pm_agent
rag: rag_mod.OpenAIRag = st.session_state.rag


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_step_log(steps: List[Step], *, expanded: bool) -> None:
    n = len(steps)
    label = f"{n} step{'s' if n != 1 else ''}"
    with st.status(label, state="complete", expanded=expanded):
        for s in steps:
            icon = "✅" if s.ok else "⚠️"
            st.markdown(f"{icon} **{s.label}**")
            if s.detail:
                st.caption(s.detail)


def render_sim_card(sim: sandbox_mod.SimResult, *, key_prefix: str) -> None:
    with st.container(border=True):
        if not sim.success:
            st.error(f"Sandbox simulation failed: {sim.message}")
            return
        st.markdown(
            f"**🧪 Sandbox simulation** — `{sim.input_kind}` input on `u[{sim.input_channel}]`, "
            f"{sim.n_states} state(s), t ∈ [0, {sim.t[-1]:.2f}] s"
        )
        if sim.diverged:
            st.warning(
                f"{sim.message}\n\nShowing the trajectory up to that point. If this "
                f"instability is intentional (e.g. an open-loop-unstable plant), no "
                f"action is needed — just say so. Otherwise, describe what looks wrong "
                f"and I'll revise the draft."
            )
        fig = sandbox_mod.build_figure(sim)
        st.plotly_chart(fig, use_container_width=True, key=f"simfig_{key_prefix}")


def render_completion_banner(final_payload: Dict[str, Any], *, key_prefix: str) -> None:
    st.success(f"✅ Model ready — **{final_payload.get('system_name', 'system')}**")
    slug = _slugify(final_payload.get("system_name", "dynamics"))
    st.download_button(
        "⬇️ Download dynamics.py",
        data=final_payload.get("python_code", ""),
        file_name=f"{slug}.py",
        mime="text/x-python",
        key=f"dl_{key_prefix}",
    )


def render_system_sheet(latest_draft: Dict[str, Any], draft_count: int) -> None:
    meta = latest_draft.get("metadata") or {}
    if not meta:
        return
    import pandas as pd

    with st.expander("📋 System sheet — states, inputs, parameters (editable)", expanded=(draft_count <= 1)):
        st.caption(f"System type: {meta.get('system_type', '—')}")

        states = meta.get("states", []) or []
        meanings = meta.get("state_meanings", []) or []
        eqs = meta.get("state_equations", []) or []
        states_df = pd.DataFrame({"state": states, "meaning": meanings, "equation": eqs})
        st.markdown("**States**")
        edited_states = st.data_editor(
            states_df, use_container_width=True, hide_index=True, num_rows="fixed",
            disabled=["state", "equation"], key="states_editor",
        )

        params = meta.get("parameters", {}) or {}
        params_df = pd.DataFrame({"parameter": list(params.keys()), "value": list(params.values())})
        st.markdown("**Parameters**")
        edited_params = st.data_editor(
            params_df, use_container_width=True, hide_index=True, num_rows="dynamic", key="params_editor",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Inputs**")
            for item in meta.get("inputs", []) or []:
                st.markdown(f"- {item}")
        with col2:
            st.markdown("**Outputs**")
            for item in meta.get("outputs", []) or []:
                st.markdown(f"- {item}")

        if meta.get("assumptions"):
            st.markdown("**Assumptions**")
            for a in meta["assumptions"]:
                st.markdown(f"- {a}")

        if st.button("✏️ Send edits as a revision request", key="send_edits_btn"):
            diffs: List[str] = []
            for _, row in edited_states.iterrows():
                if row["state"] in states:
                    idx = states.index(row["state"])
                    if idx < len(meanings) and meanings[idx] != row["meaning"]:
                        diffs.append(f"meaning of {row['state']}: '{meanings[idx]}' -> '{row['meaning']}'")
            for _, row in edited_params.iterrows():
                orig_val = params.get(row["parameter"])
                if orig_val is None:
                    diffs.append(f"add parameter {row['parameter']} = {row['value']}")
                elif str(orig_val) != str(row["value"]):
                    diffs.append(f"{row['parameter']} = {row['value']} (was {orig_val})")
            if not diffs:
                st.info("No edits detected yet — change a value above first.")
            else:
                revision_msg = "Please update the draft with these changes:\n- " + "\n- ".join(diffs)
                submit_user_turn(revision_msg)
                st.rerun()


def render_simulation_lab(python_code: str, metadata: Optional[Dict[str, Any]]) -> None:
    with st.expander("🧪 Simulation Lab — run your own scenario", expanded=False):
        try:
            n_states, n_inputs, _scalar_u = sandbox_mod.infer_dims(python_code)
        except Exception as exc:
            st.error(f"Could not analyze dynamics code: {exc}")
            return

        state_names = (metadata or {}).get("state_meanings") or (metadata or {}).get("states") or [
            f"x{i}" for i in range(n_states)
        ]

        c1, c2, c3 = st.columns(3)
        channel = c1.number_input(
            "Input channel (u index)", min_value=0, max_value=max(n_inputs - 1, 0), value=0, step=1, key="lab_channel"
        )
        kind = c2.selectbox("Waveform", ["step", "ramp", "sine"], key="lab_kind")
        duration = c3.number_input("Duration (s)", min_value=1.0, max_value=300.0, value=10.0, step=1.0, key="lab_duration")

        params: Dict[str, float] = {}
        if kind == "step":
            cA, cB = st.columns(2)
            params["t0"] = cA.number_input("Step time t0 (s)", value=1.0, key="lab_step_t0")
            params["amplitude"] = cB.number_input("Amplitude", value=1.0, key="lab_step_amp")
        elif kind == "ramp":
            cA, cB = st.columns(2)
            params["t0"] = cA.number_input("Ramp start t0 (s)", value=1.0, key="lab_ramp_t0")
            params["slope"] = cB.number_input("Slope (units/s)", value=1.0, key="lab_ramp_slope")
        else:
            cA, cB, cC = st.columns(3)
            params["amplitude"] = cA.number_input("Amplitude", value=1.0, key="lab_sine_amp")
            params["freq_hz"] = cB.number_input("Frequency (Hz)", value=0.5, key="lab_sine_freq")
            params["bias"] = cC.number_input("Bias", value=0.0, key="lab_sine_bias")

        st.caption("Initial state x0")
        x0_cols = st.columns(min(n_states, 6) or 1)
        x0_vals = []
        for i in range(n_states):
            col = x0_cols[i % len(x0_cols)]
            label = state_names[i] if i < len(state_names) else f"x{i}"
            x0_vals.append(col.number_input(str(label), value=0.0, key=f"lab_x0_{i}"))

        if st.button("▶️ Run simulation", type="primary", key="lab_run_btn"):
            sim = sandbox_mod.simulate(
                python_code,
                x0=x0_vals,
                t_final=float(duration),
                dt=max(float(duration) / 500.0, 0.005),
                input_channel=int(channel),
                input_kind=kind,
                input_params=params,
                state_names=state_names,
            )
            st.session_state.lab_last_sim = sim

        if st.session_state.get("lab_last_sim") is not None:
            render_sim_card(st.session_state.lab_last_sim, key_prefix="lab")


# ---------------------------------------------------------------------------
# Turn pipeline (RAG -> web search -> agent -> metadata -> sandbox)
# ---------------------------------------------------------------------------


def _history_for_agent() -> List[Dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m["role"] in ("user", "assistant")]


def submit_user_turn(user_message: str) -> None:
    """Append a user message, run the full pipeline, append the assistant turn."""
    st.session_state.messages.append({"role": "user", "content": user_message, "id": str(uuid.uuid4())})
    with st.chat_message("user"):
        st.markdown(user_message)
    run_turn(user_message)


def run_turn(user_message: str) -> None:
    history_before = _history_for_agent()[:-1]  # exclude the message we just appended
    history_text = agent_mod.PlantModelAgent.format_history(history_before)
    is_first_draft_turn = pm_agent.draft_count == 0

    with st.chat_message("assistant"):
        with st.status(
            "Reading the room…" if is_first_draft_turn else "Working…",
            expanded=is_first_draft_turn,
        ) as box:
            steps: List[Step] = []
            retrieved_context = ""
            web_context = ""

            # -- RAG ------------------------------------------------------
            if rag.has_files and not _is_bare_ack(user_message):
                file_label = rag.file_names[-1]
                box.update(label=f"Inspecting uploaded pdf file {file_label}…")
                st.write(f"🔎 Inspecting **{file_label}** for relevant context…")
                try:
                    summary, chunks = rag.retrieve(user_message)
                    retrieved_context = summary
                    if chunks:
                        lines = "\n".join(
                            f"- **{c.file_name}**"
                            + (f" (score {c.score:.2f})" if c.score is not None else "")
                            + f": {c.text[:180]}…"
                            for c in chunks
                        )
                        st.markdown(f"Retrieved {len(chunks)} passage(s):\n\n{lines}")
                        steps.append(Step("rag", f"Searched {file_label}", lines, True))
                    else:
                        st.write("No relevant passages found in the uploaded file.")
                        steps.append(Step("rag", f"Searched {file_label} — nothing relevant found", None, True))
                except Exception as exc:
                    st.write(f"⚠️ File search failed: {exc}")
                    steps.append(Step("rag", "File search failed", str(exc), False))

            # -- Web search -------------------------------------------------
            if st.session_state.web_search_enabled:
                box.update(label="Checking whether a web search is needed…")
                try:
                    query = websearch_mod.plan_query(
                        client, history_text, user_message, retrieved_context=retrieved_context
                    )
                except Exception as exc:
                    query = None
                    steps.append(Step("web", "Could not plan a web query", str(exc), False))
                if query and websearch_mod.looks_focused(query):
                    box.update(label=f"Searching web: {query}")
                    try:
                        brief, links = websearch_mod.search(client, query)
                        web_context = f"Query: {query}\n{brief}"
                        st.markdown(f'🌐 **Searched:** _{query}_\n\n{brief}')
                        if links:
                            st.caption(" · ".join(f"[{l['title']}]({l['url']})" for l in links[:5]))
                        steps.append(Step("web", f'Searched: "{query}"', brief, True))
                    except Exception as exc:
                        st.write(f"⚠️ Web search failed: {exc}")
                        steps.append(Step("web", "Web search failed", str(exc), False))
                elif query is not None:
                    st.write(f"Skipped web search — query too generic: \"{query}\"")
                    steps.append(Step("web", "Skipped web search (query too generic)", query, True))
                else:
                    st.write("No focused web query was needed for this turn.")
                    steps.append(Step("web", "Skipped web search (not needed)", None, True))

            # -- Primary agent -----------------------------------------------
            box.update(label="Drafting the plant model…" if is_first_draft_turn else "Updating the model…")
            display_text, final_payload = pm_agent.step(
                history_before, user_message, retrieved_context, web_context
            )
            steps.append(Step("agent", f"Agent turn — status: {pm_agent.last_status}", None, True))
            if pm_agent.last_metadata_generated:
                steps.append(Step("metadata", "Extracted structured metadata (system sheet)", None, True))

            # -- Sandbox: auto-run on a new/changed draft ---------------------
            sim_result: Optional[sandbox_mod.SimResult] = None
            code_now = pm_agent.latest_code
            if code_now and code_now != st.session_state.last_simmed_code:
                box.update(label="Running sandbox simulation…")
                state_names = None
                if pm_agent.latest_draft and pm_agent.latest_draft.get("metadata"):
                    meta_now = pm_agent.latest_draft["metadata"]
                    state_names = meta_now.get("state_meanings") or meta_now.get("states")
                sim_result = sandbox_mod.run_default_simulation(code_now, state_names=state_names)
                st.session_state.last_simmed_code = code_now
                steps.append(
                    Step(
                        "sandbox",
                        "Ran default sandbox simulation" + ("" if sim_result.success else " — failed"),
                        None if sim_result.success else sim_result.message,
                        sim_result.success,
                    )
                )

                # -- Auto diagnose-and-repair: one attempt, hard failures only.
                # A *diverged-but-successful* run (state ran away but we still
                # got a partial trajectory) is deliberately NOT auto-repaired:
                # some real plants are open-loop unstable by design, and
                # "fixing" the draft just to make the smoke test converge
                # could silently turn a correct model into an incorrect one.
                # Only a hard failure (non-finite/singular derivative, or a
                # solver error) is unambiguously a bug worth auto-fixing.
                if not sim_result.success:
                    box.update(label="Diagnosing the simulation failure…")
                    st.write(f"⚠️ {sim_result.message}")
                    steps.append(Step("sandbox", "Simulation failure flagged for diagnosis", sim_result.message, False))

                    box.update(label="Asking the agent to diagnose and fix the draft…")
                    repair_history = history_before + [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": display_text},
                    ]
                    repair_message = (
                        f"The default sandbox simulation of the draft above failed:\n"
                        f"{sim_result.message}\n\n"
                        "Diagnose the likely modelling bug (e.g. a sign error in a "
                        "restoring/damping term, a division that can hit zero, an "
                        "unrealistic parameter value, or a kinematic singularity) and "
                        "revise python_code to fix it, keeping the same states/inputs/"
                        "system intent unless the bug requires changing them. Explain "
                        "the fix briefly in reply."
                    )
                    try:
                        repair_text, repair_final = pm_agent.step(repair_history, repair_message, "", "")
                        steps.append(
                            Step("agent", f"Agent diagnosed and revised the draft — status: {pm_agent.last_status}", None, True)
                        )
                        st.markdown(repair_text)

                        repaired_code = pm_agent.latest_code
                        if repaired_code and repaired_code != code_now:
                            box.update(label="Re-running sandbox simulation on the revised draft…")
                            state_names2 = None
                            if pm_agent.latest_draft and pm_agent.latest_draft.get("metadata"):
                                meta2 = pm_agent.latest_draft["metadata"]
                                state_names2 = meta2.get("state_meanings") or meta2.get("states")
                            sim_result = sandbox_mod.run_default_simulation(repaired_code, state_names=state_names2)
                            st.session_state.last_simmed_code = repaired_code
                            steps.append(
                                Step(
                                    "sandbox",
                                    "Re-ran sandbox simulation on the revised draft"
                                    + ("" if sim_result.success else " — still failing"),
                                    None if sim_result.success else sim_result.message,
                                    sim_result.success,
                                )
                            )
                        display_text = display_text + "\n\n---\n" + repair_text
                        if repair_final is not None:
                            final_payload = repair_final
                    except Exception as exc:
                        st.write(f"⚠️ Automatic diagnosis failed: {exc}")
                        steps.append(Step("agent", "Automatic diagnosis failed", str(exc), False))

            box.update(
                label=f"{len(steps)} step{'s' if len(steps) != 1 else ''}",
                state="complete",
                expanded=False,
            )

        st.markdown(display_text)
        if sim_result is not None:
            render_sim_card(sim_result, key_prefix=f"turn_{len(st.session_state.messages)}")
        if final_payload is not None:
            render_completion_banner(final_payload, key_prefix=f"turn_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": display_text,
            "id": str(uuid.uuid4()),
            "steps": steps,
            "sim": sim_result,
            "final_payload": final_payload,
        }
    )


# ---------------------------------------------------------------------------
# Sidebar — Stage 0 controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧭 AgentPlant")
    st.caption(f"Model: `{openai_client.DEFAULT_MODEL}`")

    st.session_state.web_search_enabled = st.toggle(
        "🔍 Web search",
        value=st.session_state.web_search_enabled,
        help="Off by default. When on, the agent may issue a focused web_search "
        "query for specific facts (datasheet values, standards, recent info). "
        "Generic textbook queries are skipped.",
    )

    uploaded = st.file_uploader(
        "📎 Upload PDF or Markdown for RAG",
        type=["pdf", "md", "markdown", "txt"],
        help="Indexed with OpenAI Files + a Vector Store, retrieved via file_search.",
    )
    if uploaded is not None and uploaded.name not in rag.file_names:
        with st.spinner(f"Indexing {uploaded.name}…"):
            tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{uploaded.name}"
            tmp_path.write_bytes(uploaded.getvalue())
            try:
                rag.add_file(str(tmp_path), display_name=uploaded.name)
                st.success(f"{uploaded.name} indexed for retrieval.")
            except Exception as exc:
                st.error(f"Failed to index file: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)

    if rag.has_files:
        st.caption("RAG source(s): " + ", ".join(f"**{n}**" for n in rag.file_names))

    st.divider()
    if st.button("🔄 New conversation"):
        for key in ("messages", "pm_agent", "rag", "last_simmed_code", "lab_last_sim"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption("Prompt/response logs are written under `.logs/`.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🧭 AgentPlant")
st.caption(
    "Describe a physical or engineered system in plain language, in equations, "
    "or both. The agent will draft a runnable `dynamics(t, x, u)` you can simulate."
)

if not st.session_state.messages:
    st.info(
        "Try: *\"a DC motor driving a beam with a ball rolling on it, on the "
        "moon\"* — or paste equations you already have."
    )

# Stage 4: replay full history, each turn's step log expandable.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("steps"):
            render_step_log(msg["steps"], expanded=False)
        st.markdown(msg["content"])
        if msg.get("sim") is not None:
            render_sim_card(msg["sim"], key_prefix=f"hist_{msg['id']}")
        if msg.get("final_payload"):
            render_completion_banner(msg["final_payload"], key_prefix=f"hist_{msg['id']}")

# Stage 1: system sheet, once a draft with metadata exists.
if pm_agent.latest_draft and pm_agent.latest_draft.get("metadata"):
    render_system_sheet(pm_agent.latest_draft, pm_agent.draft_count)

# Manual simulation controls, once any code exists.
if pm_agent.latest_code:
    render_simulation_lab(
        pm_agent.latest_code,
        pm_agent.latest_draft.get("metadata") if pm_agent.latest_draft else None,
    )

prompt = st.chat_input("Describe the system, answer a question, or ask for a change…")
if prompt:
    submit_user_turn(prompt)
    st.rerun()

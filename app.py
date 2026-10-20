"""AgentPlant — Streamlit chat app.

Staged UX (see README for the full table):

  Stage 0  Bare chat; Search toggle; files are attached in the chat input.
  Stage 1  Agent asks for system context; an editable "system sheet"
           (states / inputs / parameters) appears once a draft exists.
  Stage 2  First draft: circle + live status ("Inspecting uploaded pdf
           file X.pdf...") -> automatic sandbox simulation card.
  Stage 3  Later turns: same circle, one status line while running; the
           full step list collapses under "N steps" once finished.
  Stage 4  Every past turn's step log stays expandable in the transcript.

One chat for every kind of file (backend_core/simulink/intake.py sorts them):

  - PDF / MD / TXT          indexed for RAG, searched on every later turn
  - <model>_c2py.zip        made by c2py_export.m: converted right away
  - .c / .h (or a zip)      generated C code: collected over several messages until it
                            is complete, then converted (not compared with Simulink)
  - .slx / .mdl             an .slx is first converted directly from its XML (no MATLAB,
                            backend_core/simulink/direct/slx_direct.py). If that is not possible
                            (library blocks, Stateflow, workspace variables, .mdl ...),
                            the agent says why and guides the user in the conversation,
                            step by step, to one of the two uploads above; when it
                            recommends the script it attaches c2py_export.m to its reply

A converted model (backend_core/simulink) lands in the chat as a draft, so the rest
of the flow is the same as for a model drafted from a description.

Run: `streamlit run app.py` (see README.md for setup).
"""

from __future__ import annotations

import copy
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from backend_core import agent as agent_mod
from backend_core import openai_client
from backend_core import rag as rag_mod
from backend_core import sandbox as sandbox_mod
from backend_core import websearch as websearch_mod
from backend_core import prompts
from backend_core.simulink import intake
from backend_core.steps import Step

st.set_page_config(page_title="AgentPlant", page_icon="🧭", layout="wide")

APP_DIR = Path(__file__).resolve().parent
SIMULINK_RUNS_DIR = APP_DIR / "simulink_runs"          # files of every conversion (git-ignored)
SIMULINK_EXPORT_SCRIPT = APP_DIR / "backend_core" / "simulink" / "c2py" / "c2py_export.m"

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
    st.session_state.setdefault("simulink_origin", None)   # the converted draft, while one is in use
    st.session_state.setdefault("intake_model", None)      # Simulink model waiting for its C code
    st.session_state.setdefault("code_stage", intake.CodeStage())   # C files received so far
    st.session_state.setdefault("workspace_scripts", [])   # (source, MATLAB code): variables of Simulink models
    st.session_state.setdefault("diagram_model", None)     # Simulink model whose inputs / outputs are chosen on its diagram


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
        drive = ("no input u (the model runs from its own sources)" if sim.n_inputs == 0 and sim.u is not None
                 else f"`{sim.input_kind}` input on `u[{sim.input_channel}]`")
        st.markdown(f"**🧪 Sandbox simulation** — {drive}, {sim.n_states} state(s), t ∈ [0, {sim.t[-1]:.2f}] s")
        if sim.diverged:
            st.warning(
                f"{sim.message}\n\nShowing the trajectory up to that point. If this "
                f"instability is intentional (e.g. an open-loop-unstable plant), no "
                f"action is needed — just say so. Otherwise, describe what looks wrong "
                f"and I'll revise the draft."
            )
        fig = sandbox_mod.build_figure(sim)
        st.plotly_chart(fig, width="stretch", key=f"simfig_{key_prefix}")


def render_completion_banner(final_payload: Dict[str, Any], *, key_prefix: str) -> None:
    st.success(f"✅ Model ready — **{final_payload.get('system_name', 'system')}**")
    sim_info = (final_payload.get("metadata") or {}).get("simulink")
    if isinstance(sim_info, dict):
        st.caption(f"Simulink model `{sim_info.get('model')}`: {sim_info.get('status')}")
    slug = _slugify(final_payload.get("system_name", "dynamics"))
    st.download_button(
        "⬇️ Download dynamics.py",
        data=final_payload.get("python_code", ""),
        file_name=f"{slug}.py",
        mime="text/x-python",
        key=f"dl_{key_prefix}",
    )


# files the agent may attach to a reply ("attach" in its JSON)
SERVED_FILES = {intake.EXPORT_SCRIPT: (SIMULINK_EXPORT_SCRIPT, "text/plain")}


def render_attachments(names: List[str], *, key_prefix: str) -> None:
    for i, name in enumerate(names):
        path, mime = SERVED_FILES[name]
        st.download_button(f"📎 {name}", data=path.read_bytes(), file_name=name, mime=mime,
                           key=f"att_{key_prefix}_{i}", on_click="ignore")


def render_user_message(msg: Dict[str, Any]) -> None:
    if msg.get("files"):
        st.caption("📎 " + intake.describe_files(msg["files"]))
    if msg.get("content"):
        st.markdown(msg["content"])


def render_system_sheet(latest_draft: Dict[str, Any], draft_count: int, key_suffix: str = "") -> None:
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
            states_df, width="stretch", hide_index=True, num_rows="fixed",
            disabled=["state", "equation"], key=f"states_editor_{key_suffix}",
        )

        params = meta.get("parameters", {}) or {}
        params_df = pd.DataFrame({"parameter": list(params.keys()), "value": list(params.values())})
        st.markdown("**Parameters**")
        edited_params = st.data_editor(
            params_df, width="stretch", hide_index=True, num_rows="dynamic", key=f"params_editor_{key_suffix}",
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
                # sent at the end of the script, below the transcript (the sheet sits inside it)
                st.session_state.pending_revision = "Please update the draft with these changes:\n- " + "\n- ".join(diffs)
                st.rerun()


def render_current_draft() -> None:
    """The Simulink badge, the system sheet and the Simulation Lab of the current draft. They sit
    under the assistant message that made the draft (session_state.draft_msg_id), so later
    messages of the chat come below them instead of above."""
    render_simulink_badge()
    if pm_agent.latest_draft and pm_agent.latest_draft.get("metadata"):
        render_system_sheet(pm_agent.latest_draft, pm_agent.draft_count,
                            key_suffix=st.session_state.get("draft_msg_id") or "draft")
    if pm_agent.latest_code:
        render_simulation_lab(
            pm_agent.latest_code,
            pm_agent.latest_draft.get("metadata") if pm_agent.latest_draft else None,
        )


def render_simulation_lab(python_code: str, metadata: Optional[Dict[str, Any]]) -> None:
    with st.expander("🧪 Simulation Lab — run your own scenario", expanded=False):
        try:
            n_states, n_inputs, _scalar_u = sandbox_mod.infer_dims(python_code)
        except Exception as exc:
            st.error(f"Could not analyze dynamics code: {exc}")
            return
        converted = _converted_simulation(python_code)
        if converted:
            n_states, n_inputs = len(converted["initial_state"]), max(converted["n_inputs"], 1)
            st.caption(f"Simulated like the Simulink model: {converted['solver']} fixed-step, dt = "
                       f"{converted['fixed_step']:g} s, starting from the model's own initial state.")

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
            start = float(converted["initial_state"][i]) if converted else 0.0
            x0_vals.append(col.number_input(str(label), value=start, format="%g", key=f"lab_x0_{i}"))

        if st.button("▶️ Run simulation", type="primary", key="lab_run_btn"):
            if converted:
                sim = _simulate_converted(
                    python_code, t_final=float(duration), input_channel=int(channel), input_kind=kind,
                    input_params=params, x0=x0_vals, state_names=state_names,
                )
            else:
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
# Converted Simulink models
# ---------------------------------------------------------------------------


AS_IS = "as it is"        # submit(signals=AS_IS): the model waiting for its inputs, converted without them


@st.cache_data(show_spinner=False, max_entries=4)
def _model_diagram(data: bytes, file_name: str):
    """the block diagram of an .slx (backend_core/simulink/direct/diagram.py)"""
    from backend_core.simulink.direct import diagram as diagram_mod
    return diagram_mod.read_diagram(data, file_name)


def render_signal_picker(info: intake.ModelInfo) -> None:
    """The block diagram of a Simulink model without root Inports / Outports: the user clicks the
    signals that are its inputs and outputs, then converts it with them (submit(signals=...))."""
    from backend_core.simulink.direct import diagram as diagram_mod
    ss = st.session_state
    try:
        diagram = _model_diagram(info.data, info.file)
    except Exception as exc:                  # noqa: BLE001 - the chat still works without it
        st.caption(f"The diagram of {info.file} could not be drawn ({type(exc).__name__}: {exc}).")
        return
    if not diagram.signals:
        return
    for name in ("sig_inputs", "sig_outputs"):     # the choice so far (of this model only)
        ss[name] = [k for k in ss.get(name) or [] if k in diagram.signals]
    if info.signals is not None and not ss.sig_inputs and not ss.sig_outputs:
        ss.sig_inputs, ss.sig_outputs = list(info.signals.inputs), list(info.signals.outputs)
    with st.container(border=True):
        st.markdown(f"**🔌 Inputs and outputs of {info.name}** — click the dot on a signal")
        st.caption("● a signal · ◆ an input port nothing is connected to (it reads 0). 🔵 input u: what the "
                   "signal fed now reads u instead · 🔴 output y: the signal is measured (it is plotted) · 🟢 an "
                   "Inport / Outport the model already has. A second click removes it. Scroll to zoom, drag to move.")
        systems = list(diagram.systems)
        cols = st.columns([3, 2]) if len(systems) > 1 else [st.container()]
        with cols[0]:
            mode = st.radio("A click marks the signal as", ["input", "output"], horizontal=True, key="sig_mode",
                            index=1 if not diagram.model_outputs and not ss.sig_outputs else 0)
        view = systems[0]
        if len(systems) > 1:
            with cols[1]:
                view = st.selectbox("System", systems, key="sig_view", format_func=lambda p: diagram.systems[p].title)
        keys = diagram_mod.marker_keys(diagram, view)
        fig = diagram_mod.figure(diagram, view, ss.sig_inputs, ss.sig_outputs)
        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", width="stretch",
                                key="sig_fig_%d" % ss.get("sig_clicks", 0),
                                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove":
                                        ["select2d", "lasso2d", "autoScale2d"]})
        clicked = diagram_mod.selected_keys(event, keys, curve=len(fig.data) - 1)
        if clicked:
            target = "sig_inputs" if mode == "input" else "sig_outputs"
            chosen = list(ss[target])
            for key in clicked:
                if target == "sig_outputs" and diagram_mod.is_input_port(diagram, key):
                    st.toast("An input port nothing is connected to can only be an input u.")
                    continue
                if key in (diagram.model_outputs if target == "sig_outputs" else diagram.model_inputs):
                    st.toast("%s is already an %s of the model." % (diagram.label(key), mode))
                    continue
                chosen.remove(key) if key in chosen else chosen.append(key)
            ss[target] = chosen
            ss.sig_clicks = ss.get("sig_clicks", 0) + 1      # a new chart: its selection starts empty
            st.rerun()

        # a signal the model already has as an input / output is not offered again
        options = {"sig_inputs": [k for k in diagram.signals if k not in diagram.model_inputs],
                   "sig_outputs": [k for k in diagram.signals if not diagram_mod.is_input_port(diagram, k)
                                   and k not in diagram.model_outputs]}
        for name in options:
            ss[name] = [k for k in ss[name] if k in options[name]]
        for name, label in (("sig_inputs", "Inputs u"), ("sig_outputs", "Outputs y")):
            ss[name + "_w"] = ss[name]
            st.multiselect(label, options[name], key=name + "_w", format_func=diagram.label,
                           on_change=lambda n=name: ss.__setitem__(n, list(ss[n + "_w"])))
        ready = bool(ss.sig_outputs or (diagram.model_outputs and ss.sig_inputs))
        cols = st.columns([1, 1, 2])
        if cols[0].button("Convert with these signals", type="primary", disabled=not ready, key="sig_convert"):
            ss.pending_signals = (list(ss.sig_inputs), list(ss.sig_outputs))
            st.rerun()
        if info.no_inputs and cols[1].button("Convert as it is", key="sig_as_is",
                                             help="The model has no input u: it runs from its own sources"):
            ss.pending_signals = AS_IS
            st.rerun()
        if not ready:
            st.caption("Choose at least one output." if not diagram.model_outputs else "Choose at least one signal.")


def _converted_origin_matches(code: Optional[str]) -> bool:
    """True if `code` is still the converted Simulink model, unchanged."""
    origin = st.session_state.get("simulink_origin")
    if not origin or not code:
        return False
    from backend_core.simulink.c2py import convert as simulink_convert
    return simulink_convert.same_code(code, origin["python_code"])


def _converted_simulation(code: Optional[str]) -> Optional[Dict[str, Any]]:
    """How to simulate `code` if it comes from a converted Simulink model (unchanged, or changed in
    the chat but with the same number of states): the model's solver, step size and initial state.
    None for every other draft (they keep the general sandbox)."""
    origin = st.session_state.get("simulink_origin")
    if not origin or not code or not origin.get("tableau"):
        return None
    sim = origin["metadata"]["simulink"].get("simulation")
    if not sim:
        return None
    try:
        n_states = sandbox_mod.infer_dims(code)[0]
    except Exception:
        return None
    return sim if n_states == len(sim["initial_state"]) else None


def _simulate_converted(code: str, *, t_final: float = 10.0, input_channel: int = 0, input_kind: str = "step",
                        input_params: Optional[Dict[str, float]] = None, x0: Optional[List[float]] = None,
                        state_names: Optional[List[str]] = None) -> sandbox_mod.SimResult:
    """converted Simulink model: its own fixed-step solver, step size and initial state"""
    from backend_core.simulink.c2py import convert as simulink_convert
    origin = st.session_state.simulink_origin
    sim = origin["metadata"]["simulink"]["simulation"]
    if state_names is None:
        state_names = _state_labels(origin["metadata"])
    output_names = [o.split(" = ")[0] for o in origin["metadata"].get("outputs") or []]
    return simulink_convert.simulate(
        code, tableau=origin["tableau"], fixed_step=sim["fixed_step"],
        x0=sim["initial_state"] if x0 is None else x0, n_inputs=sim["n_inputs"], t_final=t_final,
        input_channel=input_channel, input_kind=input_kind,
        input_params=input_params if input_params is not None else {"t0": 1.0, "amplitude": 1.0},
        state_names=state_names, solver_name=sim["solver"], output_names=output_names or None,
    )


def _state_labels(meta: Dict[str, Any]) -> List[str]:
    """plot labels: the meaning of a state when the model gives one, else its name"""
    names = list(meta.get("states") or [])
    meanings = list(meta.get("state_meanings") or [])
    return [m if isinstance(m, str) and m and m != "not extractable" else n
            for n, m in zip(names, meanings + [None] * (len(names) - len(meanings)))]


def _verified_against(info: Dict[str, Any]) -> str:
    if info.get("verified_against"):
        return info["verified_against"]
    return "Simulink" if info.get("compared_with_simulink", True) else "its generated C code"


def render_simulink_badge() -> None:
    origin = st.session_state.get("simulink_origin")
    if not origin or not pm_agent.latest_draft:
        return
    info = origin["metadata"]["simulink"]
    if _converted_origin_matches(pm_agent.latest_draft.get("python_code")):
        extra = ("" if info.get("compared_with_simulink", True)
                 else " (not compared with Simulink itself)" if info.get("verified_against")
                 else " (checked against %s; not compared with Simulink)" % _verified_against(info))
        st.success(f"✅ Converted from the Simulink model `{info['model']}` — {info['status']}{extra}.")
    else:
        st.warning(f"⚠️ Converted from the Simulink model `{info['model']}`, then changed in the chat — "
                   f"no longer verified against {_verified_against(info)}.")


def _upload_context() -> str:
    """What the chat agent should know about Simulink uploads this turn (see intake.agent_context)."""
    origin = st.session_state.get("simulink_origin")
    note = None
    if origin and pm_agent.latest_draft:
        info = origin["metadata"]["simulink"]
        unchanged = _converted_origin_matches(pm_agent.latest_draft.get("python_code"))
        note = prompts.fill(prompts.text("chat/agent_notes", "converted_draft"), model=info["model"],
                            source=info.get("source", "c2py package"), verified_against=_verified_against(info),
                            state=prompts.text("chat/agent_notes", "state_unchanged" if unchanged else "state_changed"))
    return intake.agent_context(st.session_state.get("intake_model"), st.session_state.get("code_stage"), note)


# ---------------------------------------------------------------------------
# Turn pipeline (RAG -> web search -> agent -> metadata -> sandbox)
# ---------------------------------------------------------------------------


def _agent_text(msg: Dict[str, Any]) -> str:
    """A message as the agent sees it: attached files are named in front of the text."""
    files = msg.get("files") or []
    text = msg.get("content") or ""
    if msg["role"] == "user" and files:
        return f"[Attached: {', '.join(files)}]" + (f"\n{text}" if text else "")
    return text


def _history_for_agent() -> List[Dict[str, str]]:
    return [{"role": m["role"], "content": _agent_text(m)} for m in st.session_state.messages
            if m["role"] in ("user", "assistant")]


def _agent_step(history, message, retrieved_context, web_context, upload_context=""):
    """pm_agent.step, then keep the Simulink metadata of a converted model honest:
    unchanged code keeps the validated metadata, changed code is marked as no longer
    verified against Simulink (backend_core.simulink.c2py.convert.reconcile)."""
    text, final = pm_agent.step(history, message, retrieved_context, web_context, upload_context)
    origin = st.session_state.get("simulink_origin")
    if origin:
        from backend_core.simulink.c2py import convert as simulink_convert
        simulink_convert.reconcile(pm_agent.latest_draft, origin)
        if final is not None and final is not pm_agent.latest_draft:
            simulink_convert.reconcile(final, origin)
    return text, final


def run_agent(agent_message: str, query_text: str, history_before: List[Dict[str, str]], box,
              steps: List[Step], turn_note: str = "",
              ) -> Tuple[str, Optional[sandbox_mod.SimResult], Optional[Dict[str, Any]]]:
    """One agent turn inside the open assistant message: RAG + web search on ``query_text`` (the
    user's own words, if any), the agent on ``agent_message``, then the sandbox on a new draft.
    -> (display_text, sim_result, final_payload)"""
    history_text = agent_mod.PlantModelAgent.format_history(history_before)
    is_first_draft_turn = pm_agent.draft_count == 0
    retrieved_context = ""
    web_context = ""

    # -- RAG ------------------------------------------------------------
    if rag.has_files and query_text and not _is_bare_ack(query_text):
        file_label = rag.file_names[-1]
        box.update(label=f"Inspecting uploaded file {file_label}…")
        st.write(f"🔎 Inspecting **{file_label}** for relevant context…")
        try:
            summary, chunks = rag.retrieve(query_text)
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

    # -- Web search -------------------------------------------------------
    if st.session_state.web_search_enabled and query_text:
        box.update(label="Checking whether a web search is needed…")
        try:
            query = websearch_mod.plan_query(
                client, history_text, query_text, retrieved_context=retrieved_context
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

    # -- Primary agent ----------------------------------------------------
    box.update(label="Drafting the plant model…" if is_first_draft_turn else "Updating the model…")
    upload_context = "\n".join(p for p in (turn_note, _upload_context()) if p)
    display_text, final_payload = _agent_step(
        history_before, agent_message, retrieved_context, web_context, upload_context
    )
    steps.append(Step("agent", f"Agent turn — status: {pm_agent.last_status}", None, True))
    if pm_agent.last_metadata_generated:
        steps.append(Step("metadata", "Extracted structured metadata (system sheet)", None, True))

    # -- Sandbox: auto-run on a new/changed draft --------------------------
    sim_result: Optional[sandbox_mod.SimResult] = None
    code_now = pm_agent.latest_code
    if code_now and code_now != st.session_state.last_simmed_code:
        box.update(label="Running sandbox simulation…")
        state_names = None
        if pm_agent.latest_draft and pm_agent.latest_draft.get("metadata"):
            meta_now = pm_agent.latest_draft["metadata"]
            state_names = meta_now.get("state_meanings") or meta_now.get("states")
        if _converted_simulation(code_now):
            sim_result = _simulate_converted(code_now)
        else:
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
        if not sim_result.success and _converted_origin_matches(code_now):
            st.write(f"⚠️ {sim_result.message}")
            steps.append(Step("sandbox", "Not auto-repaired: this is the converted Simulink model, "
                              "validated against Simulink", sim_result.message, True))
        elif not sim_result.success:
            box.update(label="Diagnosing the simulation failure…")
            st.write(f"⚠️ {sim_result.message}")
            steps.append(Step("sandbox", "Simulation failure flagged for diagnosis", sim_result.message, False))

            box.update(label="Asking the agent to diagnose and fix the draft…")
            repair_history = history_before + [
                {"role": "user", "content": agent_message},
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
                repair_text, repair_final = _agent_step(repair_history, repair_message, "", "")
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

    return display_text, sim_result, final_payload


def run_conversion(kind: str, upload: Optional[intake.Upload], box,
                   steps: List[Step]) -> Tuple[str, Optional[sandbox_mod.SimResult], bool]:
    """A c2py package (kind PACKAGE, ``upload`` = the zip) or the complete C code collected in
    the CodeStage (kind CODE) -> a validated draft of the agent. -> (display_text, sim_result, ok)"""
    from backend_core.simulink.c2py import convert as simulink_convert
    from backend_core.simulink import llm as simulink_llm

    log_lines: List[str] = []

    def log(msg: Any) -> None:
        text = str(msg).strip()
        if text:
            log_lines.append(text)
            box.update(label=text.splitlines()[0][:120])

    llm = simulink_llm.make_llm(client)
    if kind == intake.PACKAGE:
        upload_dir = SIMULINK_RUNS_DIR / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        zip_path = upload_dir / f"{uuid.uuid4().hex}_{upload.name}"
        zip_path.write_bytes(upload.data)
        box.update(label="Converting the Simulink model…")
        result = simulink_convert.convert_package(zip_path, SIMULINK_RUNS_DIR, llm, log=log)
    else:
        box.update(label="Converting the generated C code…")
        result = simulink_convert.convert_c_code(dict(st.session_state.code_stage.files), SIMULINK_RUNS_DIR,
                                                 llm, log=log)
    steps.extend(result.steps)
    if result.out_dir is not None:
        steps.append(Step("simulink", "Conversion log", "\n".join(log_lines + [f"files: {result.out_dir}"]), True))
    if not result.ok:
        return result.message, None, False
    return _adopt_converted(result.draft, result.tableau, result.message, box, steps)


def run_direct_conversion(upload: intake.Upload, box, steps: List[Step],
                          scripts=()) -> Tuple[Optional[str], Optional[sandbox_mod.SimResult], bool]:
    """Route 0: the uploaded .slx converted directly from its XML (no MATLAB, no LLM).
    ``scripts``: (source, MATLAB code) of the user that set workspace variables (attached .m files,
    values typed in the chat). -> (display_text, sim_result, ok). Not ok: nothing is shown yet - the
    reason is kept in ``upload.model.direct_error``; the agent tells the user and asks for the values
    of missing workspace variables or guides them to the C-code routes."""
    from backend_core.simulink.direct import slx_direct

    def log(msg: Any) -> None:
        text = str(msg).strip()
        if text:
            box.update(label=text.splitlines()[0][:120])

    info = upload.model
    box.update(label="Converting the Simulink model directly from the model file…")
    result = slx_direct.convert_slx(info.file, upload.model_data or b"", SIMULINK_RUNS_DIR, log=log,
                                    scripts=list(scripts), signals=info.signals)
    steps.extend(result.steps)
    if not result.ok:
        info.no_inputs = False
        info.direct_error = result.reason
        info.missing_variables, info.only_variables = result.missing_variables, result.only_variables
        info.model_fix = result.model_fix
        info.variable_uses = result.variable_uses
        info.data = upload.model_data
        # no root Outport / inputs from MATLAB data: the user chooses the signals on the diagram
        info.diagram = slx_direct.chosen_on_diagram(result.model_fix) or info.signals is not None
        st.session_state.diagram_model = info if info.diagram else None
        return None, None, False
    if result.no_input and info.signals is None and not info.as_is:
        # it converts, but without an input u or an output y: the user first chooses them on the diagram
        # (or converts it as it is) - nothing is converted or simulated before
        info.data, info.diagram, info.no_inputs = upload.model_data, True, True
        info.direct_error, info.model_fix, info.missing_variables, info.only_variables = result.no_input, None, [], False
        st.session_state.diagram_model = info
        return None, None, False
    info.no_inputs = False
    return _adopt_converted(result.draft, result.tableau, result.message, box, steps)


def _adopt_converted(draft: Dict[str, Any], tableau, message: str, box,
                     steps: List[Step]) -> Tuple[str, sandbox_mod.SimResult, bool]:
    """A converted model (any route) becomes the current draft of the conversation and is simulated
    the way its route says (solver, step, initial state). -> (display_text, sim_result, True)"""
    pm_agent.reset()
    pm_agent.latest_draft = copy.deepcopy(draft)
    pm_agent.draft_count = 1
    pm_agent.latest_code = draft["python_code"]
    pm_agent.last_status = "draft"
    st.session_state.simulink_origin = dict(copy.deepcopy(draft), tableau=tableau)
    st.session_state.intake_model = None
    st.session_state.diagram_model = None
    st.session_state.code_stage.clear()
    st.session_state.lab_last_sim = None

    box.update(label="Running the simulation…")
    sim_result = _simulate_converted(draft["python_code"])
    st.session_state.last_simmed_code = draft["python_code"]
    sim_info = draft["metadata"]["simulink"]["simulation"]
    stopped = not sim_result.success or sim_result.diverged
    steps.append(Step("sandbox", "Ran a unit-step simulation (%s fixed-step, dt = %g s, from the "
                      "model's initial state)%s" % (sim_info["solver"], sim_info["fixed_step"],
                                                    " — stopped early" if stopped else ""),
                      sim_result.message if stopped else None, True))
    reply = message
    if stopped:
        reply += ("\n\nThe quick unit-step simulation below stopped early. The model is validated "
                  "against %s, so this is how the model itself behaves for that input, not a "
                  "translation error - try other inputs in the Simulation Lab."
                  % _verified_against(draft["metadata"]["simulink"]))
    display_text = agent_mod.PlantModelAgent._format_draft(
        {"reply": reply, "system_name": draft["system_name"], "python_code": draft["python_code"]}
    )
    return display_text, sim_result, True


def _reply_attachments(reply: str) -> List[str]:
    """Files to show under the agent's reply: what it asked for ("attach"), and the export script
    whenever it tells the user about it while a Simulink conversion is waiting for files."""
    names = [n for n in pm_agent.last_attachments if n in SERVED_FILES]
    pending = st.session_state.get("intake_model") is not None or bool(st.session_state.get("code_stage"))
    if pending and intake.EXPORT_SCRIPT in (reply or "") and intake.EXPORT_SCRIPT not in names:
        names.append(intake.EXPORT_SCRIPT)
    return list(dict.fromkeys(names))


def _index_document(upload: intake.Upload, steps: List[Step]) -> bool:
    if upload.name in rag.file_names:
        steps.append(Step("rag", f"{upload.name} is already indexed", None, True))
        return True
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{upload.name}"
    tmp_path.write_bytes(upload.data)
    try:
        rag.add_file(str(tmp_path), display_name=upload.name)
        steps.append(Step("rag", f"Indexed {upload.name} for retrieval", None, True))
        return True
    except Exception as exc:
        steps.append(Step("rag", f"Could not index {upload.name}", str(exc), False))
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def submit(text: str, files: List[Any], signals: Any = None) -> None:
    """One user turn: text and/or attached files -> one assistant turn.

    Files are handled first (documents indexed, a c2py zip or complete C code converted, a
    Simulink model inspected and, if it is an .slx, converted directly; C code collected; MATLAB
    scripts kept for the workspace variables of models). While a model waits for the values of
    workspace variables, the MATLAB assignments typed in the message are taken too and the model
    is converted again.
    ``signals``: (inputs, outputs) chosen on the diagram of the model in ``diagram_model`` - it is
    converted with them; AS_IS: converted as it is, without inputs.
    A conversion is the answer of the turn;
    otherwise the agent answers, knowing what was uploaded and what is still missing."""
    text = (text or "").strip()
    chosen_model = st.session_state.get("diagram_model") if signals is not None else None
    if chosen_model is not None and signals == AS_IS:
        chosen_model.signals, chosen_model.as_is = None, True
        text = "Convert %s as it is, without inputs" % chosen_model.file
    elif chosen_model is not None:
        from backend_core.simulink.direct import diagram as diagram_mod
        chosen_model.signals = diagram_mod.choice(_model_diagram(chosen_model.data, chosen_model.file), *signals)
        names = intake.chosen_names(chosen_model.signals)
        text = "Signals chosen on the diagram of %s - inputs: %s; outputs: %s" % (
            chosen_model.file, names["inputs"], names["outputs"])
    uploads = [intake.classify(f.name, f.getvalue()) for f in files]
    user_msg = {"role": "user", "content": text, "files": [u.name for u in uploads], "id": str(uuid.uuid4())}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        render_user_message(user_msg)

    history_before = _history_for_agent()[:-1]
    by_kind = intake.split_by_kind(uploads)
    steps: List[Step] = []
    attachments: List[str] = []
    sim_result: Optional[sandbox_mod.SimResult] = None
    final_payload: Optional[Dict[str, Any]] = None
    display_text: Optional[str] = None
    is_first_draft_turn = pm_agent.draft_count == 0
    draft_before = (pm_agent.draft_count, pm_agent.latest_code)   # a new draft moves the sheet here

    with st.chat_message("assistant"):
        with st.status("Reading the upload…" if uploads else ("Reading the room…" if is_first_draft_turn else "Working…"),
                       expanded=bool(uploads) or is_first_draft_turn) as box:
            indexed = [u.name for u in by_kind[intake.DOCUMENT] if _index_document(u, steps)]
            for u in by_kind[intake.OTHER]:
                steps.append(Step("upload", f"Cannot use {u.name}: {u.note}", None, False))

            # MATLAB workspace variables: attached .m scripts, and values typed while a model waits for them
            from backend_core.simulink.direct import chat_values, slx_direct
            pending = st.session_state.get("intake_model")
            waiting_for_values = bool(pending is not None and pending.missing_variables and pending.data)
            new_scripts = [s for u in by_kind[intake.SCRIPT] + by_kind[intake.MODEL] for s in u.scripts]
            chat = None
            if text and by_kind[intake.MODEL] and slx_direct.chat_assignments(text):
                # values typed in the same message as the model (`m = 2`): its workspace too
                new_scripts.append((chat_values.PARSER_SOURCE, slx_direct.chat_assignments(text)))
            if text and waiting_for_values and not by_kind[intake.MODEL] and chosen_model is None:
                # the parser first; an LLM only when the message names no variable (chat_values)
                from backend_core.simulink import llm as simulink_llm
                box.update(label="Reading the values in your message…")
                chat = chat_values.read_values(text, pending.missing_variables, pending.variable_uses,
                                               llm=simulink_llm.make_llm(client))
                if chat.script:
                    new_scripts.append((chat.source, chat.script))
                if chat.rejected:
                    steps.append(Step("simulink", "Values the LLM gave that were not taken (%d)" % len(chat.rejected),
                                      "\n".join(chat.rejected), False))
            if new_scripts:
                st.session_state.workspace_scripts.extend(new_scripts)
                steps.append(Step("simulink", "MATLAB variables received (%s)" % ", ".join(s for s, _ in new_scripts),
                                  "\n".join(chat.read) if chat is not None and chat.read
                                  else "\n\n".join(code for _, code in new_scripts), True))
            turn_notes: List[str] = []

            if chosen_model is not None:
                steps.append(Step("simulink", "Converting %s %s" % (chosen_model.file, "as it is" if chosen_model.as_is
                                  else "with the signals chosen on its diagram"), text, True))
                retry = intake.Upload(intake.MODEL, chosen_model.file, chosen_model.data, model=chosen_model,
                                      model_data=chosen_model.data)
                display_text, sim_result, converted = run_direct_conversion(
                    retry, box, steps, st.session_state.workspace_scripts)
                if not converted:
                    st.session_state.intake_model = chosen_model
                    turn_notes.append(intake.signals_failed_note(chosen_model))
            elif by_kind[intake.PACKAGE]:
                pkg = by_kind[intake.PACKAGE][0]
                if len(by_kind[intake.PACKAGE]) > 1:
                    steps.append(Step("simulink", "Several c2py packages attached — converting only %s" % pkg.name,
                                      None, False))
                display_text, sim_result, _converted = run_conversion(intake.PACKAGE, pkg, box, steps)
            else:
                if by_kind[intake.MODEL]:
                    model_upload = by_kind[intake.MODEL][-1]
                    info = model_upload.model
                    label, detail = intake.summarize_model(info)
                    steps.append(Step("simulink", label, detail, True))
                    display_text, sim_result, converted = run_direct_conversion(
                        model_upload, box, steps, st.session_state.workspace_scripts)
                    if not converted:
                        st.session_state.intake_model = info     # the agent asks for values / guides to the C code
                elif waiting_for_values and new_scripts:
                    steps.append(Step("simulink", "Converting %s again with the values received" % pending.file,
                                      None, True))
                    retry = intake.Upload(intake.MODEL, pending.file, pending.data, model=pending,
                                          model_data=pending.data)
                    display_text, sim_result, converted = run_direct_conversion(
                        retry, box, steps, st.session_state.workspace_scripts)
                    if not converted:
                        turn_notes.append(intake.values_received_note(
                            pending, [s for s, _ in new_scripts], chat.read if chat else None,
                            chat.unclear if chat else None))
                    elif chat is not None and chat.source == chat_values.LLM_SOURCE:
                        display_text = ("Values read from your message - please check them:\n"
                                        + "\n".join("- `%s`" % r for r in chat.read) + "\n\n" + display_text)
                elif waiting_for_values and chat is not None and not chat.script:
                    turn_notes.append(intake.values_not_found_note(pending, chat.unclear, chat.rejected))
                elif by_kind[intake.SCRIPT]:
                    turn_notes.append(intake.script_kept_note([u.name for u in by_kind[intake.SCRIPT]]))
                if by_kind[intake.CODE] and display_text is None:
                    stage: intake.CodeStage = st.session_state.code_stage
                    for u in by_kind[intake.CODE]:
                        if stage.add(u.code_files):
                            steps.append(Step("simulink", "These files are the code of another model — "
                                              "the C files received before were dropped", None, True))
                    chk = stage.check()
                    steps.append(Step("simulink", "Generated C code: %d file(s)%s" % (
                        len(chk.files), " of %s" % chk.model if chk.model else ""),
                        "\n".join(chk.files + ["missing: " + m for m in chk.missing] + chk.problems), chk.ok))
                    if chk.ok:
                        display_text, sim_result, _converted = run_conversion(intake.CODE, None, box, steps)

            if display_text is None:
                if text or by_kind[intake.MODEL] or by_kind[intake.CODE] or by_kind[intake.OTHER] \
                        or not by_kind[intake.DOCUMENT]:
                    turn_note = "\n".join(
                        [prompts.fill(prompts.text("chat/agent_notes", "unusable_file"), file=u.name, why=u.note,
                                      accepted=", ".join(intake.ACCEPTED_TYPES)) for u in by_kind[intake.OTHER]]
                        + [intake.just_uploaded_note(u.model) for u in by_kind[intake.MODEL][-1:]
                           if st.session_state.get("intake_model") is u.model]
                        + turn_notes)
                    display_text, sim_result, final_payload = run_agent(
                        _agent_text(user_msg), text, history_before, box, steps, turn_note)
                    attachments = _reply_attachments(display_text)
                elif indexed:
                    display_text = (f"Indexed **{', '.join(indexed)}** — I'll search it on every turn. "
                                    "What system should I model from it?")
                else:
                    display_text = "I could not index the attached file(s) — see the step log above."

            box.update(
                label=f"{len(steps)} step{'s' if len(steps) != 1 else ''}",
                state="complete",
                expanded=False,
            )

        key = f"turn_{len(st.session_state.messages)}"
        st.markdown(display_text)
        render_attachments(attachments, key_prefix=key)
        if sim_result is not None:
            render_sim_card(sim_result, key_prefix=key)
        if final_payload is not None:
            render_completion_banner(final_payload, key_prefix=key)

    msg_id = str(uuid.uuid4())
    if pm_agent.latest_code and (pm_agent.draft_count, pm_agent.latest_code) != draft_before:
        st.session_state.draft_msg_id = msg_id      # the sheet and the lab now sit under this message
    if st.session_state.get("diagram_model") is not None:
        st.session_state.diagram_msg_id = msg_id    # the diagram to choose signals on, under the latest reply
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": display_text,
            "id": msg_id,
            "steps": steps,
            "attachments": attachments,
            "sim": sim_result,
            "final_payload": final_payload,
        }
    )


# ---------------------------------------------------------------------------
# Sidebar
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

    st.markdown("**Files in this chat**")
    if rag.has_files:
        st.caption("RAG source(s): " + ", ".join(f"**{n}**" for n in rag.file_names))
    pending_model = st.session_state.get("intake_model")
    if pending_model is not None and pending_model.diagram:
        st.caption(f"Simulink model **{pending_model.file}** — choose its inputs and outputs on its diagram")
    elif pending_model is not None and pending_model.model_fix:
        st.caption(f"Simulink model **{pending_model.file}** — needs a change in Simulink, then upload it again")
    elif pending_model is not None and pending_model.missing_variables and pending_model.only_variables:
        st.caption(f"Simulink model **{pending_model.file}** — waiting for the values of "
                   + ", ".join(f"`{n}`" for n in pending_model.missing_variables[:12])
                   + (" …" if len(pending_model.missing_variables) > 12 else ""))
    elif pending_model is not None:
        st.caption(f"Simulink model **{pending_model.file}** — waiting for its generated C code or c2py zip")
    if st.session_state.get("workspace_scripts"):
        st.caption("MATLAB variables from: " + ", ".join(dict.fromkeys(s for s, _ in st.session_state.workspace_scripts)))
    stage_now: intake.CodeStage = st.session_state.code_stage
    if stage_now:
        chk_now = stage_now.check()
        st.caption(f"C code{' of **' + chk_now.model + '**' if chk_now.model else ''}: {len(chk_now.files)} file(s)"
                   + (f", {len(chk_now.missing)} missing" if chk_now.missing else ""))
    if not (rag.has_files or pending_model is not None or stage_now):
        st.caption("Attach files with 📎 in the chat box: a PDF / Markdown document, a Simulink model "
                   "(.slx / .mdl) and the MATLAB script (.m) of its parameters, its generated C code "
                   "(.c / .h or a zip), or the zip of c2py_export.m.")
    from backend_core.simulink import llm as simulink_llm
    st.caption(f"LLM of Simulink conversions: `{simulink_llm.model_name()}` (`AGENTPLANT_SIMULINK_MODEL`)")

    st.divider()
    if st.button("🔄 New conversation"):
        for key in ("messages", "pm_agent", "rag", "last_simmed_code", "lab_last_sim", "simulink_origin",
                    "intake_model", "code_stage", "draft_msg_id", "pending_revision", "workspace_scripts",
                    "diagram_model", "diagram_msg_id", "pending_signals", "sig_inputs", "sig_outputs"):
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
    "or both — or attach a document or a Simulink model. The agent will draft a "
    "runnable `dynamics(t, x, u)` you can simulate."
)

if not st.session_state.messages:
    st.info(
        "Try: *\"a DC motor driving a beam with a ball rolling on it, on the "
        "moon\"* — paste equations you already have — or attach a Simulink model "
        "(.slx): I'll convert it directly, or walk you through it if it needs MATLAB."
    )

# Stage 4: replay full history, each turn's step log expandable.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            render_user_message(msg)
            continue
        if msg.get("steps"):
            render_step_log(msg["steps"], expanded=False)
        st.markdown(msg["content"])
        render_attachments(msg.get("attachments") or [], key_prefix=f"hist_{msg['id']}")
        if msg.get("sim") is not None:
            render_sim_card(msg["sim"], key_prefix=f"hist_{msg['id']}")
        if msg.get("final_payload"):
            render_completion_banner(msg["final_payload"], key_prefix=f"hist_{msg['id']}")
        if msg["id"] == st.session_state.get("draft_msg_id"):
            render_current_draft()
        if msg["id"] == st.session_state.get("diagram_msg_id") and st.session_state.get("diagram_model") is not None:
            render_signal_picker(st.session_state.diagram_model)

# a draft whose message is not in the transcript (should not happen): below it, as before
if pm_agent.latest_code and st.session_state.get("draft_msg_id") not in {m["id"] for m in st.session_state.messages}:
    render_current_draft()

pending_revision = st.session_state.pop("pending_revision", None)
pending_signals = st.session_state.pop("pending_signals", None)
prompt = st.chat_input(
    "Describe the system, ask a question, or attach files…",
    accept_file="multiple",
    file_type=intake.ACCEPTED_TYPES,
)
if pending_revision:
    submit(pending_revision, [])
    st.rerun()
if pending_signals:
    submit("", [], signals=pending_signals)
    st.rerun()
if prompt and (prompt.text or prompt.files):
    submit(prompt.text, list(prompt.files))
    st.rerun()

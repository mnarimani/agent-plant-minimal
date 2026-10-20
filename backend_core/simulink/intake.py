"""What did the user upload into the chat, and what is still missing for a conversion?

The chat has one file input for everything. This module sorts each upload into one kind
and keeps the state of a Simulink conversion that is still waiting for files:

    document   .pdf / .md / .markdown / .txt        -> RAG (backend_core.rag)
    package    the <model>_c2py.zip of c2py_export.m -> convert.convert_package
    code       generated C code (.c / .h, loose or zipped, without a manifest)
               -> collected in a CodeStage over several messages until it is complete,
                  then convert.convert_c_code
    script     a MATLAB script (.m) that sets workspace variables of the model (its parameters);
               also an .m file inside the zip of a model. Used by the direct conversion
    model      the Simulink model itself (.slx / .mdl, or a zip holding one). inspect_model()
               reads what it can from the file; an .slx is first converted directly from its
               XML (slx_direct, no MATLAB). Only if that fails (ModelInfo.direct_error) is the
               user guided to one of the two routes below
    other      anything else (the reason is in Upload.note)

Two routes from a Simulink model to an upload that can be converted, when the direct
conversion of the .slx (route 0, slx_direct) is not possible:

    A  automatic   c2py_export.m next to the model -> <model>_c2py.zip (C code + Simulink test
                   runs, so the Python translation is also compared with Simulink)
    B  manual      the user generates the C code with Embedded Coder and uploads the .c / .h
                   files (only checked against the C code, not against Simulink)

Pure functions and plain data only (no Streamlit), so the app can keep the state in
st.session_state and the tests can run it directly.
"""

from __future__ import annotations

import html
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .. import prompts
from .c2py import package

DOCUMENT, PACKAGE, CODE, MODEL, SCRIPT, OTHER = "document", "package", "code", "model", "script", "other"

DOC_EXT = {".pdf", ".md", ".markdown", ".txt"}
MODEL_EXT = {".slx", ".mdl"}
CODE_EXT = {".c", ".h"}
SCRIPT_EXT = {".m"}
MAX_SCRIPT_BYTES = 1024 * 1024
# what the chat input accepts (without the dot)
ACCEPTED_TYPES = sorted(e[1:] for e in DOC_EXT | MODEL_EXT | CODE_EXT | SCRIPT_EXT | {".zip"})

MAX_CODE_BYTES = 50 * 1024 * 1024         # C/H files taken out of one zip, in total

_INIT_DEF_RE = re.compile(r"\bvoid\s+(\w+)_initialize\s*\(\s*(?:void)?\s*\)\s*\{")
_INIT_DECL_RE = re.compile(r"\bextern\s+void\s+(\w+)_initialize\s*\(")
_EXTERN_VAR_RE = re.compile(r"^\s*extern\s+(?:const\s+)?[\w\s]+?[\s*]+(?:const\s+)?(\w+)\s*;", re.M)


# ---------------------------------------------------------------------------------------------
# one upload
# ---------------------------------------------------------------------------------------------
@dataclass
class Upload:
    kind: str
    name: str
    data: bytes
    code_files: Dict[str, bytes] = field(default_factory=dict)   # kind code: file name -> content
    model: Optional["ModelInfo"] = None                          # kind model
    model_data: Optional[bytes] = None                           # kind model: the model file itself
    scripts: List[Tuple[str, str]] = field(default_factory=list)  # (source, MATLAB code): kind script, or in a model zip
    note: Optional[str] = None                                   # why it is "other", or a remark


def classify(name: str, data: bytes) -> Upload:
    """-> Upload. Never raises: a file that cannot be read is kind "other" with a note."""
    base = PurePosixPath(name.replace("\\", "/")).name
    ext = PurePosixPath(base).suffix.lower()
    if ext in DOC_EXT:
        return Upload(DOCUMENT, base, data)
    if ext in MODEL_EXT:
        return Upload(MODEL, base, data, model=inspect_model(base, data), model_data=data)
    if ext in SCRIPT_EXT:
        if len(data) > MAX_SCRIPT_BYTES:
            return Upload(OTHER, base, data, note="a MATLAB script larger than %d MB" % (MAX_SCRIPT_BYTES // 2 ** 20))
        return Upload(SCRIPT, base, data, scripts=[_script(base, data)])
    if ext in CODE_EXT:
        return Upload(CODE, base, data, code_files={base: data})
    if ext == ".zip":
        return _classify_zip(base, data)
    return Upload(OTHER, base, data, note="unsupported file type%s" % (" (%s)" % ext if ext else ""))


def _classify_zip(name: str, data: bytes) -> Upload:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        infos = [i for i in z.infolist() if not i.is_dir() and not _junk(i.filename)]
    except zipfile.BadZipFile:
        return Upload(OTHER, name, data, note="not a valid zip file")
    names = [i.filename for i in infos]
    if any(PurePosixPath(n).name == package.MANIFEST for n in names):
        return Upload(PACKAGE, name, data)
    code = [i for i in infos if PurePosixPath(i.filename).suffix.lower() in CODE_EXT]
    if code:
        if sum(i.file_size for i in code) > MAX_CODE_BYTES:
            return Upload(OTHER, name, data, note="the C/H files in this zip are larger than %d MB"
                          % (MAX_CODE_BYTES // 2 ** 20))
        files: Dict[str, bytes] = {}
        for i in code:
            files[PurePosixPath(i.filename).name] = z.read(i)
        return Upload(CODE, name, data, code_files=files)
    models = [i for i in infos if PurePosixPath(i.filename).suffix.lower() in MODEL_EXT]
    if models:
        first = models[0]
        model_data = z.read(first)
        info = inspect_model(PurePosixPath(first.filename).name, model_data)
        if len(models) > 1:
            info.notes.insert(0, "The zip holds %d models (%s); I looked at %s. Convert one model at a time."
                              % (len(models), ", ".join(PurePosixPath(m.filename).name for m in models),
                                 PurePosixPath(first.filename).name))
        scripts = [_script(i.filename, z.read(i)) for i in infos
                   if PurePosixPath(i.filename).suffix.lower() in SCRIPT_EXT and i.file_size <= MAX_SCRIPT_BYTES]
        return Upload(MODEL, name, data, model=info, model_data=model_data, scripts=scripts)
    return Upload(OTHER, name, data, note="the zip holds no c2py package, no C code and no Simulink model")


def _script(name: str, data: bytes) -> Tuple[str, str]:
    """(source, MATLAB code) of an .m file"""
    text = data.decode("utf-8", errors="replace").lstrip("\ufeff")
    return "the attached script %s" % PurePosixPath(name.replace("\\", "/")).name, text


def _junk(path: str) -> bool:
    p = PurePosixPath(path)
    return "__MACOSX" in p.parts or p.name.startswith("._") or p.name == ".DS_Store"


# ---------------------------------------------------------------------------------------------
# the Simulink model file (best effort: it cannot be run here)
# ---------------------------------------------------------------------------------------------
CONTINUOUS_BLOCKS = {
    "Integrator", "SecondOrderIntegrator", "TransferFcn", "StateSpace", "ZeroPole",
    "TransportDelay", "VariableTransportDelay", "VariableTimeDelay",
}


@dataclass
class ModelInfo:
    name: str
    file: str
    fmt: str                                      # "slx" | "mdl"
    read: bool = False                            # False: the file could not be read at all
    solver: Optional[str] = None
    solver_type: Optional[str] = None
    fixed_step: Optional[str] = None
    n_inports: Optional[int] = None               # root level; None = unknown
    n_outports: Optional[int] = None
    block_types: Dict[str, int] = field(default_factory=dict)
    continuous: List[str] = field(default_factory=list)       # continuous-time blocks found
    library_blocks: List[str] = field(default_factory=list)   # links to toolbox libraries (not simulink/...)
    model_refs: int = 0
    sfunctions: int = 0
    simscape: bool = False
    notes: List[str] = field(default_factory=list)            # what the user should know / fix
    direct_error: Optional[str] = None            # why the direct conversion of the file (slx_direct) failed
    missing_variables: List[str] = field(default_factory=list)   # MATLAB variables it had no value for
    only_variables: bool = False                  # ... and nothing else stopped it: their values are enough
    model_fix: Optional[str] = None               # a change of the model the user must make (slx_direct)
    variable_uses: Dict[str, List[str]] = field(default_factory=dict)   # missing variable -> blocks using it
    data: Optional[bytes] = field(default=None, repr=False)      # the model file, to convert it again
    diagram: bool = False                         # the app shows its diagram to choose inputs / outputs on
    signals: Any = None                           # ... the SignalChoice made there (topology.SignalChoice)
    no_inputs: bool = False                       # it converts, but has no input u: waits for the choice first
    as_is: bool = False                           # the user wants it converted without inputs


def inspect_model(file_name: str, data: bytes) -> ModelInfo:
    stem = PurePosixPath(file_name).stem
    fmt = PurePosixPath(file_name).suffix.lower().lstrip(".") or "slx"
    info = ModelInfo(name=stem, file=file_name, fmt=fmt)
    try:
        if fmt == "mdl":
            _read_mdl(info, data.decode("utf-8", errors="replace"))
        else:
            _read_slx(info, data)
        info.read = True
    except Exception:                             # noqa: BLE001 - best effort only
        info.read = False
    info.notes += _model_notes(info)
    return info


def _read_slx(info: ModelInfo, data: bytes) -> None:
    z = zipfile.ZipFile(io.BytesIO(data))
    texts = {n: z.read(n).decode("utf-8", errors="replace") for n in z.namelist()
             if n.startswith("simulink/") and n.endswith(".xml")}
    if not texts:
        raise ValueError("no simulink/*.xml in the file")
    allx = "\n".join(texts.values())
    _scan(info, allx, r'BlockType="(\w+)"', r'<P Name="SourceBlock"[^>]*>([^<]*)<',
          lambda p: r'<P Name="%s"[^>]*>([^<]*)<' % p)
    root = texts.get("simulink/systems/system_root.xml")
    system = None
    if root is not None:
        system = ET.fromstring(root)
        if system.tag != "System":
            system = system.find(".//System")
    elif "simulink/blockdiagram.xml" in texts:
        system = ET.fromstring(texts["simulink/blockdiagram.xml"]).find(".//System")
    if system is not None:
        blocks = system.findall("Block")
        info.n_inports = sum(1 for b in blocks if b.get("BlockType") == "Inport")
        info.n_outports = sum(1 for b in blocks if b.get("BlockType") == "Outport")


def _read_mdl(info: ModelInfo, text: str) -> None:
    if "BlockType" not in text:
        raise ValueError("not a Simulink .mdl file")
    _scan(info, text, r'\bBlockType\s+"?(\w+)"?', r'\bSourceBlock\s+"([^"]*)"',
          lambda p: r'\b%s\s+"([^"]*)"' % p)


def _scan(info: ModelInfo, text: str, block_re: str, source_re: str, param_re) -> None:
    for t in re.findall(block_re, text):
        info.block_types[t] = info.block_types.get(t, 0) + 1
    sources = re.findall(source_re, text)
    for p, attr in (("Solver", "solver"), ("SolverType", "solver_type"), ("FixedStep", "fixed_step")):
        m = re.search(param_re(p), text)
        if m and m.group(1).strip():
            setattr(info, attr, m.group(1).strip())
    if not info.solver:
        m = re.search(param_re("SolverName"), text)
        info.solver = m.group(1).strip() if m and m.group(1).strip() else None
    info.continuous = sorted(t for t in info.block_types if t in CONTINUOUS_BLOCKS)
    info.continuous += sorted({s.split("/")[-1] for s in sources if "/Continuous/" in s})
    info.library_blocks = sorted({" ".join(html.unescape(s).split()) for s in sources if s and not s.startswith("simulink/")})
    info.model_refs = info.block_types.get("ModelReference", 0)
    info.sfunctions = info.block_types.get("S-Function", 0)
    info.simscape = any(t.startswith(("Simscape", "PMIO", "PMComponent")) for t in info.block_types) or \
        any(s.lower().startswith(("fl_lib", "ee_lib", "sm_lib", "nesl_utility", "simscape")) for s in sources)


def _model_notes(info: ModelInfo) -> List[str]:
    if not info.read:
        return ["I could not read the inside of %s, so I cannot check it in advance." % info.file]
    notes = []
    if info.n_outports == 0:
        notes.append("The model has no Outport at the top level. Add at least one root Outport: without "
                     "outputs there is nothing to compare between Simulink and the converted model.")
    # blocks of toolbox libraries (e.g. the Aerospace Blockset 6DOF) hide their integrators: no claim then
    if not info.continuous and not info.simscape and not info.library_blocks:
        notes.append("I found no continuous-time block (Integrator, Transfer Fcn, State-Space ...). The "
                     "conversion writes dx/dt = f(t, x, u), so the model needs at least one continuous state.")
    if info.model_refs:
        notes.append("The model uses %d Model Reference block(s). The code of referenced models is generated "
                     "separately and is not converted - the conversion may fail." % info.model_refs)
    if info.sfunctions:
        notes.append("The model has %d S-Function block(s). Code generation only works for S-Functions that "
                     "support it (inlined / with a TLC file)." % info.sfunctions)
    if info.simscape:
        notes.append("The model uses Simscape blocks. Their generated code can be large and is less tested "
                     "with this converter.")
    if info.solver_type and info.solver_type.lower().startswith("variable"):
        notes.append("The model uses a variable-step solver (%s). Code generation needs a fixed-step solver; "
                     "c2py_export.m switches to ode8 in memory, without changing your file."
                     % (info.solver or "variable-step"))
    return notes


# ---------------------------------------------------------------------------------------------
# generated C code, collected over several messages
# ---------------------------------------------------------------------------------------------
@dataclass
class CodeCheck:
    model: Optional[str]
    files: List[str]
    missing: List[str] = field(default_factory=list)      # files the user still has to send
    problems: List[str] = field(default_factory=list)     # cannot be fixed by sending more files
    ignored: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.model) and not self.missing and not self.problems


class CodeStage:
    """The .c / .h files of one model, sent in one or more messages."""

    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}

    def __bool__(self) -> bool:
        return bool(self.files)

    def clear(self) -> None:
        self.files = {}

    def add(self, files: Dict[str, bytes]) -> bool:
        """Add files (a file with the same name is replaced). If they are the code of another
        model than the one collected so far, the old files are dropped first. -> True then."""
        new = {n: d for n, d in files.items() if PurePosixPath(n).suffix.lower() in CODE_EXT}
        mine, theirs = _models_in(self.files), _models_in(new)
        restarted = bool(mine and theirs and not (mine & theirs))
        if restarted:
            self.files = {}
        self.files.update(new)
        return restarted

    def check(self) -> CodeCheck:
        return check_code(self.files)


def _texts(files: Dict[str, bytes]) -> Dict[str, str]:
    return {n: d.decode("utf-8", errors="replace") for n, d in files.items()}


def _models_in(files: Dict[str, bytes]) -> set:
    out = set()
    for n, t in _texts(files).items():
        out.update(_INIT_DEF_RE.findall(t) if n.lower().endswith(".c") else _INIT_DECL_RE.findall(t))
    return out


REQUIRED_CODE = ("{m}.c", "{m}.h")     # + {m}_data.c when the model has parameters (checked below)


def check_code(files: Dict[str, bytes]) -> CodeCheck:
    """Is this the complete Embedded Coder code of ONE model with continuous states?

    Three files are enough: <model>.c (the code), <model>.h (its structs) and <model>_data.c (the
    parameter values; a model without tunable parameters has none). The translator writes the two
    typedefs of <model>_types.h itself and replaces rtwtypes.h, rt_nonfinite.* and the other
    MATLAB headers - checked on five models: the same result to the last bit with or without
    them. A function the code calls from shared utility code (slprj/ert/_sharedutils) is asked
    for when the translation finds it missing (convert.py)."""
    texts = _texts(files)
    names = sorted(texts)
    ignored = [n for n in names if re.search(r"(^|_)main\w*\.c$", n, re.I)]   # sample main: not needed
    for n in ignored:
        texts.pop(n)
    defined = sorted({m for n, t in texts.items() if n.lower().endswith(".c") for m in _INIT_DEF_RE.findall(t)})
    declared = sorted({m for n, t in texts.items() if n.lower().endswith(".h") for m in _INIT_DECL_RE.findall(t)})
    chk = CodeCheck(model=None, files=names, ignored=ignored)
    if len(defined) > 1:
        chk.problems.append("These files hold the code of more than one model (%s). Send the code of one model "
                            "at a time; model references are not supported." % ", ".join(defined))
        return chk
    model = defined[0] if defined else (declared[0] if len(declared) == 1 else _model_from_names(texts))
    chk.model = model
    if not model:
        chk.missing.append("<model>.c - the main C file of the model (it defines <model>_initialize)")
        return chk
    for need in REQUIRED_CODE:
        if need.format(m=model) not in texts:
            chk.missing.append(need.format(m=model))
    # globals of the model declared in a header but defined in no C file: usually <model>_data.c
    # (the parameter values). Without it the model would silently run with zero parameters.
    c_text = "\n".join(t for n, t in texts.items() if n.lower().endswith(".c"))
    for n, t in texts.items():
        if not n.lower().endswith(".h"):
            continue
        for var in _EXTERN_VAR_RE.findall(t):
            if not var.startswith(model + "_"):
                continue
            if not re.search(r"^(?!\s*extern)[^;\n]*\b%s\s*(=|;|\[)" % re.escape(var), c_text, re.M):
                need = "%s_data.c" % model
                if need not in chk.missing and need not in texts:
                    chk.missing.append(need)
                elif need in texts and "the definition of %s" % var not in chk.missing:
                    chk.missing.append("the C file that defines %s" % var)
    main_c = texts.get("%s.c" % model, "")
    if main_c:
        if not _INIT_DEF_RE.search(main_c):
            chk.problems.append("%s.c does not define %s_initialize() - this is not Embedded Coder code of the "
                                "model" % (model, model))
        elif not re.search(r"\b%s_(step\w*|output)\s*\(" % re.escape(model), main_c):
            chk.problems.append("%s.c has no %s_step() - generate the code with a fixed-step solver and the "
                                "Embedded Coder target ert.tlc" % (model, model))
        elif not re.search(r"\b%s_derivatives\s*\(" % re.escape(model), main_c):
            chk.problems.append("The code has no continuous states (no %s_derivatives()): either the model has none "
                                "- then there is no dx/dt = f(t, x, u) to write - or it was generated without "
                                "'Support: continuous time'." % model)
    return chk


def _model_from_names(texts: Dict[str, str]) -> Optional[str]:
    for n in texts:
        m = re.fullmatch(r"(\w+?)_(types|private|data)\.(h|c)", n)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------------------------
# guidance for the user - given by the chat agent, in the conversation
# ---------------------------------------------------------------------------------------------
EXPORT_SCRIPT = package.EXPORT_SCRIPT       # the file the agent can attach to its reply ("attach")


# the texts: backend_core/prompts/simulink/upload_guidance.yaml
GUIDANCE = "simulink/upload_guidance"


def _text(key: str, **values) -> str:
    return prompts.fill(prompts.text(GUIDANCE, key), **values)


def export_steps(model: Optional[str]) -> str:
    call = _text("export_call", model=model) if model else _text("export_call_unnamed")
    return _text("export_steps", call=call, model=model or _text("unnamed_model"))


def manual_steps(model: Optional[str]) -> str:
    return _text("manual_steps", model=model or _text("unnamed_model"))


def agent_context(model: Optional[ModelInfo], stage: Optional[CodeStage], origin_note: Optional[str] = None) -> str:
    """The state of the Simulink conversion for the prompt of the chat agent ('' = nothing pending).
    The agent tells the user everything in the conversation; this is its reference."""
    lines: List[str] = []
    if origin_note:
        lines.append(origin_note)
    if model is None and not stage:
        return "\n".join(lines)
    chk = stage.check() if stage else None
    name = model.name if model is not None else (chk.model if chk else None)
    if model is not None and model.diagram:
        lines.extend(_diagram_context(model))
    if model is not None and model.no_inputs:
        lines.append(_text("context_no_inputs", file=model.file, name=model.name, reason=model.direct_error or ""))
        return "\n".join(lines)
    if model is not None and model.model_fix:
        lines.append(_text("context_model_fix", file=model.file, name=model.name, fix=model.model_fix))
        return "\n".join(lines)
    if model is not None and model.missing_variables and model.only_variables:
        lines.append(_text("context_missing_variables", file=model.file, name=model.name,
                           variables=", ".join(model.missing_variables), details=model.direct_error or ""))
    elif model is not None:
        lines.append(_text("context_failed", file=model.file, name=model.name,
                           reason=model.direct_error or _text("context_failed_no_reason")))
        facts = []
        if model.solver or model.solver_type:
            facts.append(_text("fact_solver", solver=model.solver or "?", type=model.solver_type or "?"))
        if model.n_inports is not None:
            facts.append(_text("fact_ports", inports=model.n_inports, outports=model.n_outports or 0))
        if model.continuous:
            facts.append(_text("fact_continuous", blocks=", ".join(model.continuous[:10])))
        if facts:
            lines.append(_text("facts", facts="; ".join(facts)))
        for n in model.notes:
            lines.append(_text("found_in_file", note=n))
    if chk:
        lines.append(_text("code_received", count=len(chk.files), files=", ".join(chk.files[:20]),
                           of_model=_text("code_received_of_model", model=chk.model) if chk.model else ""))
        if chk.missing:
            lines.append(_text("code_missing", files="; ".join(chk.missing)))
        for p in chk.problems:
            lines.append(_text("code_problem", problem=p))
        if chk.ok:
            lines.append(_text("code_complete"))
    lines.append(_text("routes", export_steps=export_steps(name), manual_steps=manual_steps(name)))
    return "\n".join(lines)


def _diagram_context(model: ModelInfo) -> List[str]:
    lines = [_text("context_diagram", name=model.name)]
    if model.signals is not None:
        lines.append(_text("context_signals", **chosen_names(model.signals)))
    return lines


def chosen_names(choice) -> Dict[str, str]:
    """the signals of a SignalChoice by the names of their new ports -> {"inputs", "outputs"}"""
    names = dict(choice.names)
    return {kind: ", ".join(names.get(k, k) for k in keys) or "none"
            for kind, keys in (("inputs", choice.inputs), ("outputs", choice.outputs))}


def signals_failed_note(info: ModelInfo) -> str:
    """For the turn in which the conversion with the signals chosen on the diagram failed."""
    return _text("signals_failed", file=info.file, **chosen_names(info.signals),
                 reason=info.direct_error or _text("signals_failed_no_reason"))


def just_uploaded_note(info: ModelInfo) -> str:
    """For the turn in which the Simulink model arrives: the first reply only opens the conversation."""
    if info.no_inputs:
        return _text("no_inputs_note", file=info.file, name=info.name, reason=info.direct_error or "")
    if info.model_fix and info.diagram:
        return _text("just_uploaded_model_fix_diagram", file=info.file, name=info.name, fix=info.model_fix)
    if info.model_fix:
        return _text("just_uploaded_model_fix", file=info.file, name=info.name, fix=info.model_fix)
    if info.missing_variables and info.only_variables:
        return _text("just_uploaded_missing_variables", file=info.file, name=info.name,
                     variables=", ".join(info.missing_variables), details=info.direct_error or "")
    return _text("just_uploaded_failed", file=info.file, name=info.name,
                 reason=info.direct_error or _text("just_uploaded_failed_no_reason"))


def values_received_note(info: ModelInfo, sources: List[str], read: Optional[List[str]] = None,
                         unclear: Optional[List[str]] = None) -> str:
    """For the turn in which the user sent values of workspace variables but the model still did not convert."""
    if info.no_inputs:
        return _text("no_inputs_note", file=info.file, name=info.name, reason=info.direct_error or "")
    if info.missing_variables and info.only_variables:
        return _text("values_received_missing", sources=", ".join(sources),
                     taken=_text("values_taken", read="; ".join(read)) if read else "",
                     asked=_text("values_unclear", unclear="; ".join(unclear)) if unclear else "",
                     file=info.file, variables=", ".join(info.missing_variables), details=info.direct_error or "")
    return _text("values_received_failed", sources=", ".join(sources), file=info.file,
                 reason=info.direct_error or _text("values_received_failed_no_reason"))


def values_not_found_note(info: ModelInfo, unclear: List[str], rejected: Optional[List[str]] = None) -> str:
    """For a turn of a model that waits for values, when the message gave none that could be taken."""
    return _text("values_not_found", file=info.file, variables=", ".join(info.missing_variables),
                 unclear=_text("values_not_found_unclear", unclear="; ".join(unclear)) if unclear else "",
                 rejected=_text("values_not_found_rejected", rejected="; ".join(rejected)) if rejected else "")


def script_kept_note(sources: List[str]) -> str:
    """For the turn in which a MATLAB script arrives while no model waits for values."""
    return _text("script_kept", sources=", ".join(sources))


def describe_files(names: Iterable[str]) -> str:
    names = list(names)
    return ", ".join(names[:8]) + (" ... (%d files)" % len(names) if len(names) > 8 else "")


def split_by_kind(uploads: Iterable[Upload]) -> Dict[str, List[Upload]]:
    out: Dict[str, List[Upload]] = {k: [] for k in (DOCUMENT, PACKAGE, CODE, MODEL, SCRIPT, OTHER)}
    for u in uploads:
        out[u.kind].append(u)
    return out


def summarize_model(info: ModelInfo) -> Tuple[str, Optional[str]]:
    """(one line for the step log, detail)"""
    if not info.read:
        return "Received the Simulink model %s (could not look inside)" % info.file, None
    detail = []
    if info.solver or info.solver_type:
        detail.append("solver: %s (%s)" % (info.solver or "?", info.solver_type or "?"))
    if info.n_inports is not None:
        detail.append("root ports: %d in, %d out" % (info.n_inports, info.n_outports or 0))
    detail.append("continuous blocks: %s" % (", ".join(info.continuous) or "none found"))
    detail.append("%d blocks" % sum(info.block_types.values()))
    return "Inspected the Simulink model %s" % info.file, "\n".join(detail)

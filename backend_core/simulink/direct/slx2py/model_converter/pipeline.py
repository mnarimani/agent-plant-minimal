"""One SLX extraction/conversion pipeline shared by the app and CLI."""
from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from .blocks import Constant
from .exporter import export_model
from .graph import Endpoint
from .dependency_report import write_failure_report
from .slx_reader import load_model
from .topology import SignalChoice, build_topology
from . import workspace as matlab_workspace

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_BYTES = 250 * 1024 * 1024
MAX_ARCHIVE_FILES = 10000


def validate_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    stem = Path(name).stem
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (Path(name).suffix.lower() != ".slx" or any(c in name for c in '<>:"|?*')
            or any(ord(c) < 32 for c in name) or stem in {"", ".", ".."}
            or stem.endswith((".", " ")) or stem.split(".")[0].upper() in reserved):
        raise ValueError("Select an SLX file with a valid model name.")
    return name


def extract_slx(source: Path, destination: Path) -> Path:
    if not source.is_file() or source.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("SLX must exist and be no larger than 50 MB.")
    destination = destination.resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".x-{uuid4().hex[:8]}"
    stage.mkdir()
    try:
        with ZipFile(source) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_FILES or sum(e.file_size for e in entries) > MAX_EXTRACTED_BYTES:
                raise ValueError("SLX exceeds extraction limits.")
            total = 0
            for entry in entries:
                target = (stage / entry.filename.replace("\\", "/")).resolve()
                if not target.is_relative_to(stage) or ":" in entry.filename:
                    raise ValueError("Unsafe path in SLX archive.")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as input_file, target.open("wb") as output_file:
                    while chunk := input_file.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_EXTRACTED_BYTES:
                            raise ValueError("SLX exceeds extraction limits.")
                        output_file.write(chunk)
        extracted = stage
        if not (stage / "simulink").is_dir():
            candidates = [p for p in stage.iterdir() if p.is_dir() and (p / "simulink").is_dir()]
            if len(candidates) != 1:
                raise ValueError("SLX must contain exactly one Simulink model.")
            extracted = candidates[0]
        # Destination must be fresh: model versions are immutable, never overwritten.
        if destination.exists():
            raise FileExistsError(f"Extraction destination already exists: {destination}")
        extracted.replace(destination)
        return destination
    finally:
        if stage.parent == parent and stage.name.startswith(".x-"):
            shutil.rmtree(stage, ignore_errors=True)


def convert_slx(source: Path, extracted: Path, output: Path, *, parameters: dict | None = None,
                scripts: list[tuple[str, str]] | tuple = (), signals: SignalChoice | None = None) -> Path:
    validate_filename(source.name)
    extract_slx(source, extracted)
    return convert_directory(extracted, output, parameters=parameters, scripts=scripts, signals=signals)


def model_workspace(extracted: Path, *, parameters: dict | None = None,
                    scripts: list[tuple[str, str]] | tuple = ()) -> matlab_workspace.Workspace:
    """The MATLAB workspace the block parameters are evaluated in, as Simulink builds it: the
    user's scripts (``(source, MATLAB code)``, the base workspace), then the model's own model
    workspace and callbacks (PreLoadFcn, PostLoadFcn, InitFcn). ``parameters`` given explicitly
    (name -> number / array) win."""
    base = matlab_workspace.Workspace()
    for source, code in scripts:
        base.run(code, source)
    ws = matlab_workspace.model_workspace(extracted, base)
    for name, value in (parameters or {}).items():
        ws.set(name, matlab_workspace._matlab_value(value), "given explicitly")
    return ws


def convert_directory(extracted: Path, output: Path, *, parameters: dict | None = None,
                      scripts: list[tuple[str, str]] | tuple = (), signals: SignalChoice | None = None) -> Path:
    """``signals``: inputs / outputs the user chose on the diagram (new root Inports / Outports)."""
    stage, active_ids = "topology", None
    staging = None
    ws = None
    try:
        topology = build_topology(extracted, signals)
        active_ids = {b.block_id for b in topology.active_blocks}
        stage = "workspace"
        ws = model_workspace(extracted, parameters=parameters, scripts=scripts)
        stage = "runtime_loading"
        used = matlab_workspace.UsedParameters(ws.parameters())
        loaded = load_model(extracted, topology, parameters=used)
        loaded.model.workspace = {**ws.report(), "used": sorted(used.used)}
        stage = "runtime_validation"
        # Simulink reads an unconnected input as 0 (and warns): ground it the same way
        loaded.model.warnings = (list(getattr(loaded.model, "not_converted", []))
                                 + list(getattr(loaded.model, "sample_time_notes", [])))
        for block in list(loaded.model.blocks.values()):
            connected = {edge.destination.port for edge in loaded.model.connections if edge.destination.block_id == block.block_id}
            for port in sorted(set(range(block.input_count)) - connected):
                ground = Constant(f"{block.block_id}::ground{port + 1}", f"{block.name} (unconnected input {port + 1})", 0.0)
                loaded.model.add_block(ground)
                loaded.model.connect(Endpoint(ground.block_id, 0), Endpoint(block.block_id, port))
                loaded.model.warnings.append(f"Input {port + 1} of block '{block.name}' is not connected; it is 0, as in Simulink.")
        loaded.model._compile_if_needed()
        # a root Inport without dimensions is taken as a scalar; Simulink may give it the width of
        # the vector it is combined with (back propagation), so say so where that matters
        for inport in (b for b in loaded.model.blocks.values() if getattr(b, "inherited", False)):
            for edge in loaded.model.connections:
                if edge.source.block_id != inport.block_id:
                    continue
                target = loaded.model.blocks[edge.destination.block_id]
                if max(getattr(target, "output_widths", [1]) or [1]) > 1:
                    loaded.model.warnings.append(
                        f"Inport '{inport.name}' has no dimensions set (-1); it is taken as a scalar, which block "
                        f"'{target.name}' expands to a vector. If it is a vector in Simulink, set its Port dimensions.")
                    break
        stage = "export"
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.parent / (".export-" + uuid4().hex[:12])
        path = export_model(loaded, staging, topology)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Conversion output must be empty; previous outputs cannot be mixed with a new conversion.")
        output.mkdir(exist_ok=True)
        for artifact in staging.iterdir():
            artifact.replace(output / artifact.name)
        return output / path.name
    except (ValueError, OSError) as error:
        try:
            write_failure_report(extracted, output, error, stage, active_ids,
                                 workspace=ws.report() if ws is not None else None)
        except OSError as report_error:
            error.add_note(f"Conversion diagnostics could not be saved: {report_error}")
        raise

    finally:
        if staging is not None and staging.resolve().parent == Path(output).resolve().parent and staging.name.startswith(".export-"):
            shutil.rmtree(staging, ignore_errors=True)

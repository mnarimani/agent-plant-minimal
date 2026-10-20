from __future__ import annotations

import ast
import json
from pathlib import Path

from .blocks import Block, Concatenate, Constant, DiscreteBlock, Demux, Fcn, Gain, Selector, Inport, Integrator, Lookup1D, ManualSwitch, MathBlock, MatlabFunction, Mux, Outport, Product, PulseGenerator, Saturate, SecondOrderIntegrator, StateSpace, StateflowChart, Step, Sum, TransferFcn
from .dynamics_export import export_dynamics
from .model import Model
from .parameters import extract_model_parameters
from .slx_reader import LoadedModel
from .topology import ModelTopology, write_skipped_blocks_log, write_skipped_blocks_report
from .viewer import export_model_view, ModelViewError


class ModelExportError(ValueError):
    pass


def export_model(
    loaded_model: LoadedModel,
    output_directory: str | Path,
    topology: ModelTopology | None = None,
) -> Path:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.py"
    path.write_text(_render_model(loaded_model), encoding="utf-8")
    contract = {
        "schema_version": 1,
        "model_name": loaded_model.model.name,
        "model_file": "model.py",
        "inputs": [element for block in loaded_model.model.blocks.values() if isinstance(block, Inport)
                   for element in block.element_names],
        "outputs": [element for block in _resolved(loaded_model.model).blocks.values() if isinstance(block, Outport)
                    for element in block.element_names],
        "parameters": extract_model_parameters(loaded_model.model),
        "warnings": list(getattr(loaded_model.model, "warnings", [])),
        # the MATLAB workspace variables the parameters were evaluated with (model callbacks, user scripts)
        "workspace": getattr(loaded_model.model, "workspace", None) or {"variables": {}, "unresolved": {}, "notes": []},
        "dynamics": export_dynamics(loaded_model.model, directory),
        "source_solver_options": loaded_model.solver_options.get("raw", loaded_model.solver_options),
        "solver_hints": {"periodic_periods": [block.period for block in loaded_model.model.blocks.values()
                                                if isinstance(block, PulseGenerator)]},
        "simulation": {
            "solver": loaded_model.solver_name,
            "start_time": loaded_model.start_time,
            "stop_time": loaded_model.stop_time,
            "step_size": loaded_model.fixed_step,
            "relative_tolerance": loaded_model.relative_tolerance,
            "absolute_tolerance": loaded_model.absolute_tolerance,
            "max_step": loaded_model.solver_options.get("MaxStep", "auto"),
            "initial_step": loaded_model.solver_options.get("InitialStep", "auto"),
            "min_step": loaded_model.solver_options.get("MinStep", "auto"),
            "auto_scale_absolute_tolerance": (loaded_model.solver_options.get("AbsTol") == "auto"
                                               or loaded_model.solver_options.get("AutoScaleAbsTol") == "on"),
            "zero_crossing_control": loaded_model.solver_options.get("ZeroCrossControl", "UseLocalSettings"),
        },
    }
    (directory / "model.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    report_topology = ModelTopology(
        loaded_model.model.name,
        tuple(loaded_model.model.blocks.values()),
        tuple(loaded_model.model.connections),
        loaded_model.skipped_blocks,
    )
    write_skipped_blocks_report(report_topology, directory / "skipped_blocks.json")
    write_skipped_blocks_log(report_topology, directory / "skipped_blocks.log")
    try:
        export_model_view(path, topology)
    except ModelViewError as error:
        for name in ("model.html", "model_graph.json"):
            directory.joinpath(name).unlink(missing_ok=True)
        (directory / "model_view_report.json").write_text(json.dumps(
            {"status": "error", "message": str(error)}, indent=2), encoding="utf-8")
    return path


def _render_model(loaded_model: LoadedModel) -> str:
    model = loaded_model.model
    lines = [
        _standalone_runtime_source(),
        "",
        "inf, nan = math.inf, math.nan    # repr() of non-finite parameters of the blocks below",
        "",
        "",
        "def build_model() -> Model:",
        f"    model = Model({model.name!r})",
    ]

    for block in model.blocks.values():
        lines.append(f"    model.add_block({_block_expression(block)})")
    lines.append("")
    for connection in model.connections:
        lines.append(
            "    model.connect(Endpoint(" + repr(connection.source.block_id) + f", {connection.source.port}), "
            "Endpoint(" + repr(connection.destination.block_id) + f", {connection.destination.port}))"
        )
    for block_id, source in model._enable_sources.items():
        lines.append(
            "    model.set_enable(" + repr(block_id) + ", Endpoint(" + repr(source.block_id) + f", {source.port}))"
        )
    for block_id, source in model._trigger_sources.items():
        lines.append(
            "    model.set_trigger(" + repr(block_id) + ", Endpoint(" + repr(source.block_id) + f", {source.port}))"
        )
    lines.extend(["    return model", ""])
    return "\n".join(lines)


def _standalone_runtime_source() -> str:
    """Embed the runtime required by an exported model into one Python file."""
    header = """from __future__ import annotations

from abc import ABC, abstractmethod
import ast
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol, TypeAlias
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray
from scipy.signal import tf2ss"""
    module_names = (
        "graph",
        "waveforms",
        "dynamics_expressions",
        "stateflow",
        "stateflow_machine",
        "blocks",
        "model",
    )
    sections = [header]
    for module_name in module_names:
        source = Path(__file__).with_name(f"{module_name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        definitions: list[str] = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef)):
                continue
            first_line = min(
                [node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", []))]
            )
            definitions.append("\n".join(lines[first_line - 1 : node.end_lineno]))
        sections.append("\n\n".join(definitions))
    return "\n\n\n".join(sections)


def _resolved(model: Model) -> Model:
    model.infer_widths()
    return model


def _plain(value):
    """numbers and arrays as plain Python (lists) for model.py"""
    import numpy as np
    return value.tolist() if isinstance(value, np.ndarray) else value


def _block_expression(block: Block) -> str:
    if isinstance(block, PulseGenerator):
        return f"PulseGenerator({block.block_id!r}, {block.name!r}, amplitude={block.amplitude!r}, period={block.period!r}, width={block.width!r}, delay={block.delay!r})"
    if isinstance(block, Inport):
        return f"Inport({block.block_id!r}, {block.name!r}, input_key={block.input_key!r}, width={block.width!r})"
    if isinstance(block, Demux):
        return f"Demux({block.block_id!r}, {block.name!r}, {block.sizes if block.sizes is not None else block.output_count!r})"
    if isinstance(block, Selector):
        return f"Selector({block.block_id!r}, {block.name!r}, {block.indices!r})"
    if isinstance(block, Concatenate):
        return f"Concatenate({block.block_id!r}, {block.name!r}, {block.input_count!r})"
    if isinstance(block, Outport):
        return f"Outport({block.block_id!r}, {block.name!r})"
    if isinstance(block, MatlabFunction):
        return f"MatlabFunction({block.block_id!r}, {block.name!r}, {block.script!r})"
    if isinstance(block, StateflowChart):
        return f"StateflowChart({block.block_id!r}, {block.name!r}, {block.chart_xml!r})"
    if isinstance(block, Step):
        return (
            f"Step({block.block_id!r}, {block.name!r}, step_time={block.step_time!r}, "
            f"initial_value={block.initial_value!r}, final_value={block.final_value!r})"
        )
    if isinstance(block, Constant):
        return f"Constant({block.block_id!r}, {block.name!r}, value={_plain(block.value)!r})"
    if isinstance(block, Gain):
        return f"Gain({block.block_id!r}, {block.name!r}, gain={_plain(block.gain)!r}, multiplication={block.multiplication!r})"
    if isinstance(block, Sum):
        return f"Sum({block.block_id!r}, {block.name!r}, signs={block.signs.tolist()!r})"
    if isinstance(block, Integrator):
        limits = (f", lower_limit={block.lower_limit!r}, upper_limit={block.upper_limit!r}"
                  if block.limited else "").replace("inf", "math.inf")
        return (
            f"Integrator({block.block_id!r}, {block.name!r}, initial_value={_plain(block.initial_value)!r}, "
            f"external_initial_condition={block.external_initial_condition!r}{limits})"
        )
    if isinstance(block, SecondOrderIntegrator):
        return (f"SecondOrderIntegrator({block.block_id!r}, {block.name!r}, initial_x={block.initial_x!r}, "
                f"initial_dxdt={block.initial_dxdt!r}, show_output={block.show_output!r})")
    if isinstance(block, StateSpace):
        return (f"StateSpace({block.block_id!r}, {block.name!r}, {block.a.tolist()!r}, {block.b.tolist()!r}, "
                f"{block.c.tolist()!r}, {block.d.tolist()!r}, initial_condition={block.initial_conditions.tolist()!r})")
    if isinstance(block, DiscreteBlock):
        return (f"DiscreteBlock({block.block_id!r}, {block.name!r}, {block.kind!r}, {block.input_count!r}, "
                f"{[t.to_dict() for t in block.templates]!r}, {[t.to_dict() for t in block.update_templates]!r}, "
                f"{[_plain(v) for v in block.initial]!r}, "
                f"major={[t.to_dict() for t in block.major_templates] if block.major_templates else None!r}, "
                f"feedthrough={block.direct_feedthrough!r}, vector={block.vector!r}, labels={block.labels!r})")
    if isinstance(block, MathBlock):
        return (f"MathBlock({block.block_id!r}, {block.name!r}, {block.kind!r}, {block.input_count!r}, "
                f"{[t.to_dict() for t in block.templates]!r}, reduce={block.reduce!r})")
    if isinstance(block, TransferFcn):
        return (
            f"TransferFcn({block.block_id!r}, {block.name!r}, numerator={block.numerator.tolist()!r}, "
            f"denominator={block.denominator.tolist()!r}, initial_condition={block.initial_conditions.tolist()!r})"
        )
    if isinstance(block, Product):
        return f"Product({block.block_id!r}, {block.name!r}, {block.operations!r})"
    if isinstance(block, Saturate):
        return f"Saturate({block.block_id!r}, {block.name!r}, {block.lower_limit!r}, {block.upper_limit!r}, zero_crossing={block.zero_crossing!r})"
    if isinstance(block, Lookup1D):
        return f"Lookup1D({block.block_id!r}, {block.name!r}, {block.breakpoints.tolist()!r}, {block.table.tolist()!r})"
    if isinstance(block, Mux):
        return f"Mux({block.block_id!r}, {block.name!r}, {block.input_count!r})"
    if isinstance(block, ManualSwitch):
        return f"ManualSwitch({block.block_id!r}, {block.name!r}, {block.current_setting!r})"
    if isinstance(block, Fcn):
        return f"Fcn({block.block_id!r}, {block.name!r}, {block.expression!r})"
    raise ModelExportError(f"Unsupported block type '{type(block).__name__}' for export.")

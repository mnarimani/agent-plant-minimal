from __future__ import annotations

import ast
import json
from pathlib import Path

from .topology import ModelTopology
from .viewer_source import source_hints
from .source_diagram import xml_views


class ModelViewError(ValueError):
    pass


BLOCK_PARAMETERS = {
    "Outport": (),
    "Inport": ("input_key",),
    "MatlabFunction": ("script",),
    "StateflowChart": ("chart_xml",),
    "Step": ("step_time", "initial_value", "final_value"),
    "PulseGenerator": ("amplitude", "period", "width", "delay"),
    "Constant": ("value",),
    "Gain": ("gain",),
    "Sum": ("signs",),
    "Integrator": ("initial_value", "external_initial_condition"),
    "TransferFcn": ("numerator", "denominator", "initial_condition"),
    "Product": ("operations",),
    "Saturate": ("lower_limit", "upper_limit"),
    "Lookup1D": ("breakpoints", "table"),
    "Mux": ("input_count",),
    "ManualSwitch": ("current_setting",),
    "Fcn": ("expression",),
}


def read_generated_graph(source_path: str | Path) -> dict:
    """Read exported declarations without importing or executing the model."""
    path = Path(source_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    settings = {}
    builder = None
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == "build_model":
            builder = statement
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith(("SIMULATION_", "MODEL_INPUT_")):
                    settings[target.id] = _literal(statement.value)
    if builder is None:
        raise ModelViewError(f"No build_model() declaration in {path}.")
    graph = {
        "schema_version": 2,
        "model_name": path.stem,
        "source_file": path.name,
        "settings": settings,
        "nodes": [],
        "edges": [],
    }
    for statement in builder.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == "Model" and call.args:
                graph["model_name"] = _literal(call.args[0])
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
            continue
        if call.func.value.id != "model":
            continue
        if call.func.attr == "add_block":
            graph["nodes"].append(_read_block(call.args[0]))
        elif call.func.attr == "connect":
            graph["edges"].append({
                "source": _endpoint(call.args[0]),
                "destination": _endpoint(call.args[1]),
                "kind": "signal",
            })
        elif call.func.attr in ("set_enable", "set_trigger"):
            graph["edges"].append({
                "source": _endpoint(call.args[1]),
                "destination": {"block_id": _literal(call.args[0]), "port": call.func.attr[4:]},
                "kind": call.func.attr[4:],
            })
    node_ids = {node["id"] for node in graph["nodes"]}
    if not node_ids or len(node_ids) != len(graph["nodes"]):
        raise ModelViewError("The generated graph must contain unique block IDs.")
    for edge in graph["edges"]:
        if edge["source"]["block_id"] not in node_ids or edge["destination"]["block_id"] not in node_ids:
            raise ModelViewError("A connection refers to a block missing from the generated model.")
    return graph


def export_model_view(source_path: str | Path, topology: ModelTopology | None = None) -> Path:
    """Write a versioned graph sidecar and an offline, read-only model viewer."""
    path = Path(source_path)
    graph = read_generated_graph(path)
    contract_path = path.with_name("model.json")
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        graph["inputs"] = contract["inputs"]
        graph["outputs"] = contract["outputs"]
        graph["parameters"] = contract.get("parameters", [])
        graph["settings"] = contract["simulation"]
        graph["dynamics"] = contract.get("dynamics")
    hints = source_hints(topology) if topology is not None else {}
    graph["source_preview"] = hints.get("thumbnail")
    for node in graph["nodes"]:
        hint = hints.get("nodes", {}).get(node["id"], {})
        node["position"] = hint.get("bounds")
        node["source_layout"] = hint
    try:
        graph["views"] = xml_views(topology, graph)
    except ValueError as error:
        raise ModelViewError(str(error)) from error
    content = render_model_view(graph)
    path.with_name(f"{path.stem}_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    viewer_path = path.with_name("model.html")
    viewer_path.write_text(content, encoding="utf-8")
    return viewer_path


def render_model_view(graph: dict) -> str:
    """Render stored geometry with current presentation assets, without conversion."""
    payload = json.dumps(graph, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    script = Path(__file__).with_name("viewer.js").read_text(encoding="utf-8")
    return template.replace("__VIEWER_SCRIPT__", script, 1).replace("__MODEL_GRAPH__", payload, 1)


def _literal(node: ast.AST | None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ModelViewError("Viewer requires literal exported declarations; no model code is executed.") from error


def _endpoint(node: ast.AST) -> dict:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Endpoint":
        raise ModelViewError("Expected an exported Endpoint declaration.")
    block_id, port = (_literal(argument) for argument in node.args)
    if not isinstance(port, int) or port < 0:
        raise ModelViewError("Endpoint ports must be nonnegative integers.")
    return {"block_id": block_id, "port": port}


def _read_block(node: ast.AST) -> dict:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or len(node.args) < 2:
        raise ModelViewError("Expected an exported block constructor with ID and name.")
    block_type = node.func.id
    if block_type not in BLOCK_PARAMETERS:
        raise ModelViewError(f"No diagram definition is registered for {block_type!r}; diagram generation was stopped.")
    parameter_names = BLOCK_PARAMETERS[block_type]
    parameters = {}
    for index, argument in enumerate(node.args[2:]):
        name = parameter_names[index] if index < len(parameter_names) else f"argument_{index + 1}"
        parameters[name] = _literal(argument)
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ModelViewError("Unpacking block parameters is not supported by the viewer.")
        parameters[keyword.arg] = _literal(keyword.value)
    return {
        "id": _literal(node.args[0]),
        "name": _literal(node.args[1]),
        "type": block_type,
        "parameters": parameters,
    }

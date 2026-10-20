"""Explain conversion failures using evidence in the extracted SLX only.

An XML reference is evidence of a dependency, not proof that it is active.
Callbacks are recorded without running them or guessing workspace values.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET


def inspect_dependencies(directory: Path, active_ids: set[str] | None = None) -> dict:
    directory = directory.resolve()
    systems = directory / "simulink" / "systems"
    dependencies, callbacks = [], []

    def parameter(element, name, default=""):
        return element.findtext(f"P[@Name='{name}']", default=default) or default

    def stored(name):
        target = (directory / name.replace("\\", "/")).resolve()
        return target.is_relative_to(directory) and target.is_file()

    def visit(system, ids=(), names=(), ancestors=(), stateflow=False):
        for block in system.findall("Block"):
            sid, name = block.get("SID", ""), block.get("Name", "")
            block_id = "::".join((*ids, sid))
            full_name = "/".join((*names, name))
            kind = block.get("BlockType")
            sf_type = parameter(block, "SFBlockType")
            child = block.find("System")
            reference = child.get("Ref") if child is not None else None
            has_definition = child is not None and (not reference or stored(f"simulink/systems/{reference}.xml"))
            active = block_id in active_ids if active_ids is not None else None
            item = {"block_id": block_id, "block_name": full_name,
                    "block_type": kind, "on_output_path": active}
            source = parameter(block, "SourceBlock")
            if source:
                dependencies.append({**item, "kind": "library", "reference": source,
                                     "status": "embedded_definition" if has_definition else "external_definition",
                                     "detail": "Library implementation is cached in XML." if has_definition else
                                     "XML contains a library link; its implementation is outside the extracted SLX."})
            if kind == "ModelReference":
                model = parameter(block, "ModelNameDialog") or parameter(block, "ModelName")
                embedded = stored(model) or stored(f"{model}.slx") if model else False
                dependencies.append({**item, "kind": "referenced_model", "reference": model,
                                     "status": "stored_file" if embedded else "external_definition",
                                     "detail": "Referenced model requires a separate conversion."})
            # Stateflow's generated sf_sfun is backed by chart XML, not a missing user S-function.
            if kind == "S-Function" and not stateflow:
                dependencies.append({**item, "kind": "s_function", "reference": parameter(block, "FunctionName"),
                                     "status": "external_definition",
                                     "detail": "S-function executable/source is external; a Python rule is also required."})
            for p in block.findall("P"):
                if p.get("Name") in {"InitFcn", "LoadFcn", "PreLoadFcn", "PostLoadFcn", "StartFcn", "StopFcn"} and p.text:
                    callbacks.append({**item, "callback": p.get("Name"), "code": p.text})
            if child is not None:
                nested = child
                if reference:
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", reference) or reference in ancestors:
                        continue
                    path = systems / f"{reference}.xml"
                    if not path.is_file():
                        dependencies.append({**item, "kind": "system_xml", "reference": reference,
                                             "status": "missing_archive_file", "detail": "Referenced subsystem XML is missing."})
                        continue
                    nested = ET.parse(path).getroot()
                visit(nested, (*ids, sid), (*names, name), (*ancestors, reference), stateflow or bool(sf_type))

    visit(ET.parse(systems / "system_root.xml").getroot(), ancestors=("system_root",))
    diagram_path = directory / "simulink" / "blockdiagram.xml"
    workspace = {}
    if diagram_path.is_file():
        diagram = ET.parse(diagram_path).getroot()
        for p in diagram.iter("P"):
            if p.get("Name") in {"PreLoadFcn", "PostLoadFcn", "InitFcn", "StartFcn", "StopFcn"} and p.text:
                callbacks.append({"block_name": "Model", "callback": p.get("Name"), "code": p.text})
        node = diagram.find(".//ModelWorkspace")
        if node is not None:
            workspace = {p.get("Name"): p.text or "" for p in node.findall("P")}
    workspace["stored_files"] = [p.relative_to(directory).as_posix() for p in sorted((directory / "simulink").glob("modelWorkspace.*"))]
    workspace["values_decoded"] = False
    workspace["detail"] = "Stored binary workspace data is present but this converter does not decode it. Saved initialization code may be inactive."
    return {"dependencies": dependencies, "callbacks": callbacks, "workspace": workspace,
            "scope": "Source references only; active relevance is unknown when topology construction fails. Callback presence does not prove execution or missing values."}


def write_failure_report(directory: Path, output: Path, error: Exception, stage: str,
                         active_ids: set[str] | None = None, workspace: dict | None = None) -> dict:
    # Diagnostics must never replace the original error if inspection itself fails.
    try:
        information = inspect_dependencies(directory, active_ids)
    except (OSError, ValueError, ET.ParseError) as inspection_error:
        information = {"dependencies": [], "callbacks": [], "workspace": {},
                       "inspection_error": str(inspection_error)}
    missing = [d for d in information["dependencies"]
               if d["status"] in {"external_definition", "missing_archive_file"} and d["on_output_path"] is not False]
    message = "Model conversion could not be completed."
    if missing:
        message += " The extracted SLX does not include some referenced external definitions; see Dependency details."
    else:
        message += " See the conversion error for the unsupported feature or invalid data."
    report = {"schema_version": 1, "model_name": directory.name, "status": "error",
              "executable": False, "message": message,
              "error": {"stage": stage, "code": type(error).__name__, "message": str(error)},
              **information}
    if workspace is not None:
        # the MATLAB variables found (model callbacks, the user's scripts) and those without a value
        report["workspace_variables"] = workspace
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

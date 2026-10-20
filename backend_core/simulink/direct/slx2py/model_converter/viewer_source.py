"""Presentation hints from the original SLX; never change numerical topology."""
from __future__ import annotations

import base64
import math
import re
from xml.etree import ElementTree

from .topology import ModelTopology


def _parameter(element, name, default=""):
    item = element.find(f"P[@Name='{name}']")
    return item.text if item is not None and item.text else default


def _numbers(text):
    values = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", text)]
    return values if all(math.isfinite(value) for value in values) else []


def source_hints(topology: ModelTopology) -> dict:
    nodes = {}
    for block in topology.active_blocks:
        nodes[block.block_id] = {"hierarchy": list(block.hierarchy), "display_name": block.element.get("Name", block.name)}
        bounds = _numbers(_parameter(block.element, "Position"))
        if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        rotation = _numbers(_parameter(block.element, "BlockRotation", "0"))
        counts = block.element.find("PortCounts")
        nodes[block.block_id] = {
            "hierarchy": list(block.hierarchy), "display_name": block.element.get("Name", block.name),
            "bounds": bounds, "rotation": int(rotation[0]) % 360 if rotation else 0,
            "mirror": _parameter(block.element, "BlockMirror", "off") == "on",
            "name_location": _parameter(block.element, "NameLocation", "bottom"),
            "sum_inputs": _parameter(block.element, "Inputs"), "is_root": block.is_root,
            "input_count": int(counts.get("in", "0")) if counts is not None else 0,
            "output_count": int(counts.get("out", "0")) if counts is not None else 0,
        }
    result = {"nodes": nodes, "routes": [], "thumbnail": None}
    directory = topology.source_directory
    if directory is None:
        return result
    thumbnail = directory / "metadata/thumbnail.png"
    if thumbnail.is_file() and thumbnail.stat().st_size <= 2 * 1024 * 1024:
        data = thumbnail.read_bytes()
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            result["thumbnail"] = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    root_path = directory / "simulink/systems/system_root.xml"
    if not root_path.is_file():
        return result
    root = ElementTree.parse(root_path).getroot()
    by_sid = {block.element.get("SID"): block.block_id for block in topology.active_blocks if block.is_root}

    def endpoint(text):
        match = re.fullmatch(r"(.+)#(in|out):(\d+)", text)
        if match is None or match[1] not in by_sid:
            return None
        return {"block_id": by_sid[match[1]], "port": int(match[3]) - 1}

    def branches(element, source, points, label):
        values = _numbers(_parameter(element, "Points"))
        if len(values) % 2:
            values = []
        offsets = points + [values[i:i + 2] for i in range(0, len(values), 2)]
        destination = endpoint(_parameter(element, "Dst"))
        if source and destination:
            result["routes"].append({"source": source, "destination": destination, "offsets": offsets, "label": label})
        for branch in element.findall("Branch"):
            branches(branch, source, offsets, label)

    for line in root.findall("Line"):
        branches(line, endpoint(_parameter(line, "Src")), [], _parameter(line, "Name"))
    return result

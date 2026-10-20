"""The block diagram of an .slx, drawn for the user to choose its inputs and outputs on it.

A model whose author did not put Inport / Outport blocks at its top level has no inputs u and
no outputs y the conversion can use (slx_direct.NO_OUTPORT). Instead of asking the user to change
the model in Simulink, the app shows its diagram - every block and line of every system, at the
positions of the .slx - and the user clicks the signals that are the model's inputs and outputs
(as many as they like). The choice (a ``SignalChoice``) goes to ``slx_direct.convert_slx``: an
output gets a new root Outport, an input replaces the signal by a new root Inport (whatever
computed it stays in the model, but no longer drives it), and the conversion goes on as for a model
that had them.

    read_diagram(data)            the .slx bytes -> Diagram (geometry and signals, no plotly)
    choice(diagram, ins, outs)    the chosen signal keys -> SignalChoice for the conversion
    figure(diagram, system, ...)  one system of it as a plotly figure, the signals as clickable markers

A signal is the output port of the block that computes it, by its key "<flat block id>#<port>"
(topology.signal_key); a line in a subsystem that comes from a port of the subsystem carries the
signal of the block outside that really computes it.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

from .slx2py.model_converter import pipeline, topology

Point = Tuple[float, float]
ROOT: Tuple[str, ...] = ()

# colors of the figure: a chosen input, a chosen output, a port the model already has
INPUT_COLOR, OUTPUT_COLOR, PLAIN_COLOR, MODEL_PORT_COLOR = "#1e88e5", "#e53935", "#8a8a8a", "#43a047"


@dataclass
class DiagramBlock:
    sid: str
    name: str
    block_type: str
    rect: Tuple[float, float, float, float]       # left, top, right, bottom (Simulink: y grows down)
    glyph: str                                     # the short text inside the block
    child: Optional[Tuple[str, ...]] = None        # a subsystem: the path of its system


@dataclass
class DiagramLine:
    polylines: List[List[Point]]                   # the line and each of its branches
    key: Optional[str]                             # the signal it carries (None: not a signal)
    anchor: Point                                  # where its marker sits
    destinations: List[str] = field(default_factory=list)   # its Dst ports, 'SID#in:1' ...


@dataclass
class DiagramSystem:
    path: Tuple[str, ...]                          # the SIDs of the subsystems from the root
    title: str
    blocks: List[DiagramBlock] = field(default_factory=list)
    lines: List[DiagramLine] = field(default_factory=list)


@dataclass
class Diagram:
    model: str
    systems: Dict[Tuple[str, ...], DiagramSystem]
    signals: Dict[str, Dict]                       # key -> {"block_id", "block", "type", "port", "label"}
    model_inputs: List[str]                        # signals of the root Inports the model has
    model_outputs: List[str]                       # signals its root Outports show

    def label(self, key: str) -> str:
        info = self.signals.get(key)
        return info["label"] if info else key


# ---------------------------------------------------------------------------------------------
# the .slx -> Diagram
# ---------------------------------------------------------------------------------------------
def read_diagram(data: bytes, file_name: str = "model.slx") -> Diagram:
    """The diagram of the .slx ``data``. Raises ValueError for a file that is not a readable .slx."""
    name = Path(file_name.replace("\\", "/")).stem or "model"
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "model.slx"
        source.write_bytes(data)
        model_dir = pipeline.extract_slx(source, Path(tmp) / "model")
        found = topology.model_signals(model_dir)
        systems: Dict[Tuple[str, ...], DiagramSystem] = {}
        _read_system(model_dir / "simulink" / "systems", "system_root", ROOT, name, found["lines"],
                     found["signals"], systems)
        try:                                       # the blocks that reach a root Outport
            active = {b.block_id for b in topology.build_topology(model_dir).active_blocks}
        except Exception:                          # noqa: BLE001 - no Outport: nothing reaches one
            active = set()
    signals = found["signals"]
    for info in signals.values():                  # a subsystem by its name, not by its SIDs
        system = systems.get(tuple(info["block_id"].split("::"))) if info.get("kind") == "input" else None
        if system is not None and info["type"] == "SubSystem":
            info["block"] = system.title.split("/", 1)[-1]
    _label_signals(signals)
    root = systems[ROOT]
    # a root Inport that reaches no output is not an input of the converted model
    model_inputs = [key for key, info in signals.items()
                    if info["type"] == "Inport" and "::" not in info["block_id"] and info["block_id"] in active]
    outports = {block.sid for block in root.blocks if block.block_type == "Outport"}
    model_outputs = [line.key for line in root.lines
                     if line.key and any(d.split("#")[0] in outports for d in line.destinations)]
    return Diagram(name, systems, signals, model_inputs, list(dict.fromkeys(model_outputs)))


def choice(diagram: Diagram, inputs: Sequence[str], outputs: Sequence[str]) -> topology.SignalChoice:
    """The signals chosen on the diagram -> the SignalChoice of the conversion (the new ports are
    named after the block of their signal, as the user saw it on the diagram)."""
    names = tuple((key, _port_name(diagram, key)) for key in list(inputs) + list(outputs))
    return topology.SignalChoice(tuple(inputs), tuple(outputs), names)


def is_input_port(diagram: Diagram, key: str) -> bool:
    """True for an input port nothing is connected to (it can only be an input u, not an output)"""
    return (diagram.signals.get(key) or {}).get("kind") == "input"


def _port_name(diagram: Diagram, key: str) -> str:
    info = diagram.signals.get(key)
    if not info:
        return key
    name = " ".join(info["block"].split("/")[-1].split())
    if info.get("kind") == "input":
        return name if info["port"] == 1 else "%s in%d" % (name, info["port"])
    return name if not info.get("several") else "%s %d" % (name, info["port"])


def _label_signals(signals: Dict[str, Dict]) -> None:
    per_block: Dict[str, int] = {}
    for info in signals.values():
        info.setdefault("kind", "output")
        if info["kind"] == "output":
            per_block[info["block_id"]] = per_block.get(info["block_id"], 0) + 1
    for info in signals.values():
        if info["kind"] == "input":
            info["label"] = "%s (%s), input %d - not connected" % (info["block"], info["type"], info["port"])
            continue
        info["several"] = per_block[info["block_id"]] > 1 or info["port"] > 1
        info["label"] = "%s (%s)%s" % (info["block"], info["type"],
                                       ", output %d" % info["port"] if info["several"] else "")


def _read_system(directory: Path, reference: str, path: Tuple[str, ...], title: str,
                 line_keys: Dict, signals: Dict, systems: Dict[Tuple[str, ...], DiagramSystem]) -> None:
    root = ElementTree.parse(directory / ("%s.xml" % reference)).getroot()
    system = systems[path] = DiagramSystem(path, title)
    wired = topology._wired_ports(root)
    ports: Dict[str, Tuple] = {}                  # sid -> (input points, output points, top centre, rect)
    children = []
    for element in root.findall("Block"):
        sid, block_type = element.get("SID", ""), element.get("BlockType", "")
        rect = _position(element)
        if not sid or rect is None:
            continue
        name = element.get("Name", "")
        child = None
        if block_type == "SubSystem" and topology._parameter(element, "SFBlockType") not in {"Chart", "MATLAB Function"}:
            system_ref = element.find("System")
            if system_ref is not None and system_ref.get("Ref"):
                child = path + (sid,)
                children.append((system_ref.get("Ref"), child, "%s/%s" % (title, " ".join(name.split()))))
        try:
            n_in, n_out = topology._port_counts(element, wired.get(sid, (0, 0)))
        except Exception:                 # noqa: BLE001 - a block without a type: draw it without ports
            n_in, n_out = wired.get(sid, (0, 0))
        ports[sid] = _port_points(element, rect, n_in, n_out)
        system.blocks.append(DiagramBlock(sid, " ".join(name.split()), block_type, rect,
                                          _glyph(element, block_type), child))
    wired_out, wired_in = set(), set()
    for index, line in enumerate(root.findall("Line")):
        source = topology._parameter(line, "Src")
        if source:
            wired_out.add(source)
        wired_in.update(topology._line_destinations(line))
        start = _end_point(source, ports, "out") if source else None
        if start is None:
            continue
        polylines: List[List[Point]] = []
        _trace(line, start, ports, polylines)
        trunk = polylines[0]
        anchor = ((trunk[0][0] + trunk[1][0]) / 2, (trunk[0][1] + trunk[1][1]) / 2) if len(trunk) > 1 \
            else (trunk[0][0] + 8, trunk[0][1])
        system.lines.append(DiagramLine(polylines, line_keys.get((path, index)), anchor,
                                        topology._line_destinations(line)))
    # an output port no line leaves (a model without Outports often ends in one): a short stub to click
    for block in system.blocks:
        if block.child is not None:
            continue                               # its signals are inside it
        left, top, right, bottom = block.rect
        for k, (x, y) in enumerate(ports[block.sid][1]):
            key = topology.signal_key(topology.Endpoint("::".join(path + (block.sid,)), k))
            if "%s#out:%d" % (block.sid, k + 1) in wired_out or key not in signals:
                continue
            dx, dy = (x - (left + right) / 2, y - (top + bottom) / 2)
            step = 18.0 / max(1e-9, (dx * dx + dy * dy) ** 0.5)
            end = (x + dx * step, y + dy * step) if abs(dx) + abs(dy) > 0 else (x + 18.0, y)
            system.lines.append(DiagramLine([[(x, y), end]], key, end))
    # an input port nothing is connected to (it reads 0): a short stub to click, to make it an input u
    for block in system.blocks:
        left, top, right, bottom = block.rect
        for k, (x, y) in enumerate(ports[block.sid][0]):
            key = topology.input_key("::".join(path + (block.sid,)), k)
            if "%s#in:%d" % (block.sid, k + 1) in wired_in or key not in signals:
                continue
            dx, dy = (x - (left + right) / 2, y - (top + bottom) / 2)
            step = 18.0 / max(1e-9, (dx * dx + dy * dy) ** 0.5)
            end = (x + dx * step, y + dy * step) if abs(dx) + abs(dy) > 0 else (x - 18.0, y)
            system.lines.append(DiagramLine([[end, (x, y)]], key, end))
    for child_reference, child_path, child_title in children:
        if (directory / ("%s.xml" % child_reference)).is_file():
            _read_system(directory, child_reference, child_path, child_title, line_keys, signals, systems)


def _position(element) -> Optional[Tuple[float, float, float, float]]:
    text = topology._parameter(element, "Position")
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text or "")]
    return tuple(numbers[:4]) if len(numbers) >= 4 else None


def _port_points(element, rect, n_in: int, n_out: int):
    """(input port points, output port points, top centre) of a block, with its rotation / mirror"""
    left, top, right, bottom = rect
    rotation = int(float(topology._parameter(element, "BlockRotation") or 0)) % 360
    mirror = (topology._parameter(element, "BlockMirror") or "off") == "on"
    orientation = topology._parameter(element, "Orientation")          # older files
    if orientation in {"left", "up", "down"} and not topology._parameter(element, "BlockRotation"):
        rotation, mirror = {"left": (0, True), "up": (270, False), "down": (90, False)}[orientation]

    def spread(n, a, b):
        return [a + (b - a) * (2 * i + 1) / (2 * n) for i in range(n)]

    if rotation in (90, 270):
        down = (rotation == 90) != mirror
        ins = [(x, top if down else bottom) for x in spread(n_in, left, right)]
        outs = [(x, bottom if down else top) for x in spread(n_out, left, right)]
    else:
        flipped = (rotation == 180) != mirror
        ins = [(right if flipped else left, y) for y in spread(n_in, top, bottom)]
        outs = [(left if flipped else right, y) for y in spread(n_out, top, bottom)]
    return ins, outs, ((left + right) / 2, top), rect


def _end_point(text: Optional[str], ports, direction: str) -> Optional[Point]:
    """'SID#out:2' / 'SID#in:1' / 'SID#enable' ... -> the point of that port"""
    if not text:
        return None
    sid, _, port = text.partition("#")
    if sid not in ports:
        return None
    ins, outs, top, _rect = ports[sid]
    kind, _, number = port.partition(":")
    if kind in {"in", "out"} and number.isdigit():
        points = ins if kind == "in" else outs
        k = int(number) - 1
        return points[k] if 0 <= k < len(points) else top
    return top                                     # enable, trigger, ifaction ... ports sit on top


def _trace(element, start: Point, ports, polylines: List[List[Point]]) -> None:
    """a <Line> or <Branch> from ``start``: its Points are offsets, one after the other; its branches
    start where its own points end"""
    points = [start]
    for dx, dy in _offsets(topology._parameter(element, "Points")):
        points.append((points[-1][0] + dx, points[-1][1] + dy))
    corner = points[-1]
    end = _end_point(topology._parameter(element, "Dst"), ports, "in")
    if end is not None:
        points.extend(_into_port(points[-1], end, _sideways(topology._parameter(element, "Dst"), ports)))
    polylines.append(points)
    for branch in element.findall("Branch"):
        _trace(branch, corner, ports, polylines)


# a line ends this close to the port computed from the block's size: it is that port (Simulink rounds
# the place of its ports; the line shows where it really is)
SNAP = 6.0


def _into_port(last: Point, end: Point, sideways: bool) -> List[Point]:
    """the points from the last point of a line into its port, at right angles as in Simulink: a port
    on the side of a block is entered horizontally, one on the top or bottom vertically"""
    across, along = (1, 0) if sideways else (0, 1)          # the coordinate that must match, the other
    if abs(end[across] - last[across]) <= SNAP:
        snapped = list(end)
        snapped[across] = last[across]
        return [tuple(snapped)]
    if abs(end[along] - last[along]) <= 0.5:
        return [end]
    elbow = list(last)
    elbow[across] = end[across]
    return [tuple(elbow), end]


def _sideways(text: Optional[str], ports) -> bool:
    """True when the destination port is on the left or right side of its block"""
    sid = (text or "").partition("#")[0]
    if sid not in ports or not text:
        return True
    end = _end_point(text, ports, "in")
    left, top, right, bottom = ports[sid][3]
    return end is not None and (abs(end[0] - left) < 0.5 or abs(end[0] - right) < 0.5) \
        and not (abs(end[1] - top) < 0.5 or abs(end[1] - bottom) < 0.5)


def _offsets(text: Optional[str]) -> List[Point]:
    if not text:
        return []
    out = []
    for row in text.strip(" []").split(";"):
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", row)]
        if len(numbers) >= 2:
            out.append((numbers[0], numbers[1]))
    return out


_GLYPHS = {"Integrator": "1/s", "Sum": "Σ", "Product": "×", "TransferFcn": "TF", "StateSpace": "ẋ=Ax+Bu",
           "Saturate": "sat", "Scope": "scope", "Step": "step", "Sin": "sin", "Mux": "mux", "Demux": "demux",
           "Terminator": "⊣", "UnitDelay": "1/z", "ZeroOrderHold": "ZOH", "Display": "disp", "Clock": "t",
           "Abs": "|u|", "Switch": "switch", "RelationalOperator": "rel", "Logic": "logic",
           "DiscreteIntegrator": "K/(z-1)", "DiscreteTransferFcn": "TF(z)", "Memory": "mem",
           "FromWorkspace": "workspace", "FromFile": "file", "Derivative": "du/dt", "Fcn": "f(u)"}


def _glyph(element, block_type: str) -> str:
    if block_type in {"Inport", "Outport"}:
        return "%s %s" % ("In" if block_type == "Inport" else "Out", topology._parameter(element, "Port") or "1")
    if block_type == "Gain":
        return _short(topology._parameter(element, "Gain") or "1")
    if block_type == "Constant":
        return _short(topology._parameter(element, "Value") or "1")
    if block_type == "SubSystem":
        kind = topology._parameter(element, "SFBlockType")
        return {"Chart": "chart", "MATLAB Function": "fcn"}.get(kind, "subsystem")
    if block_type in {"Goto", "From"}:
        return "[%s]" % _short(topology._parameter(element, "GotoTag") or "")
    return _GLYPHS.get(block_type, _short(block_type))


def _short(text: str, n: int = 10) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------------------------
# a system of the Diagram -> plotly figure
# ---------------------------------------------------------------------------------------------
def figure(diagram: Diagram, path: Tuple[str, ...], inputs: Sequence[str], outputs: Sequence[str]):
    """One system of the diagram: blocks, lines (a chosen input blue, a chosen output red, the
    model's own ports green) and one marker per signal. The marker trace is the last one; its
    points carry the signal keys as customdata, in the order of ``marker_keys``."""
    import plotly.graph_objects as go

    system = diagram.systems[path]
    inputs, outputs = set(inputs), set(outputs)
    model_ports = set(diagram.model_inputs) | set(diagram.model_outputs)

    def color(key):
        if key in inputs:
            return INPUT_COLOR
        if key in outputs:
            return OUTPUT_COLOR
        return MODEL_PORT_COLOR if key in model_ports else PLAIN_COLOR

    fig = go.Figure()
    by_color: Dict[str, Tuple[List, List]] = {}
    for line in system.lines:
        xs, ys = by_color.setdefault(color(line.key), ([], []))
        for polyline in line.polylines:
            xs.extend([p[0] for p in polyline] + [None])
            ys.extend([p[1] for p in polyline] + [None])
    for c, (xs, ys) in by_color.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=c, width=3 if c != PLAIN_COLOR else 1.5),
                                 hoverinfo="skip", showlegend=False))

    shapes, notes = [], []
    for block in system.blocks:
        left, top, right, bottom = block.rect
        shapes.append(dict(type="rect", x0=left, y0=top, x1=right, y1=bottom, layer="below",
                           line=dict(color="#555", width=2 if block.child else 1),
                           fillcolor="rgba(120,144,156,0.18)" if block.child else "rgba(255,255,255,0.0)"))
        notes.append(dict(x=(left + right) / 2, y=(top + bottom) / 2, text=block.glyph, showarrow=False,
                          font=dict(size=10)))
        notes.append(dict(x=(left + right) / 2, y=bottom + 4, text=block.name, showarrow=False, yanchor="top",
                          font=dict(size=10, color="#666")))

    keys, xs, ys, colors, texts = marker_keys(diagram, path), [], [], [], []
    anchors = {}
    for line in system.lines:
        anchors.setdefault(line.key, line.anchor)
    for key in keys:
        line = DiagramLine([], key, anchors[key])
        xs.append(line.anchor[0])
        ys.append(line.anchor[1])
        colors.append(color(line.key))
        role = ("input" if line.key in inputs else "output" if line.key in outputs else
                "input of the model" if line.key in diagram.model_inputs else
                "output of the model" if line.key in diagram.model_outputs else
                "not connected (reads 0) - click to make it an input u" if is_input_port(diagram, line.key) else
                "click to choose")
        texts.append("%s<br><i>%s</i>" % (diagram.label(line.key), role))
    symbols = ["diamond" if is_input_port(diagram, key) else "circle" for key in keys]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", customdata=keys, hovertext=texts, hoverinfo="text",
                             marker=dict(size=13, color=colors, symbol=symbols, line=dict(color="white", width=1.5)),
                             showlegend=False))

    all_x = [v for b in system.blocks for v in (b.rect[0], b.rect[2])] + [p[0] for l in system.lines for pl in l.polylines for p in pl]
    all_y = [v for b in system.blocks for v in (b.rect[1], b.rect[3])] + [p[1] for l in system.lines for pl in l.polylines for p in pl]
    pad = 30
    x_range = [min(all_x, default=0) - pad, max(all_x, default=100) + pad]
    y_range = [max(all_y, default=100) + pad + 14, min(all_y, default=0) - pad]     # y grows down, as in Simulink
    height = max(380, min(760, int((y_range[0] - y_range[1]) / max(1.0, x_range[1] - x_range[0]) * 1100)))
    fig.update_layout(shapes=shapes, annotations=notes, height=height, margin=dict(l=4, r=4, t=4, b=4),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", dragmode="pan",
                      clickmode="event+select", hovermode="closest")
    fig.update_xaxes(visible=False, range=x_range)
    fig.update_yaxes(visible=False, range=y_range, scaleanchor="x", scaleratio=1)
    return fig


def marker_keys(diagram: Diagram, path: Tuple[str, ...]) -> List[str]:
    """the signals of a system, in the order of the markers of ``figure``"""
    return list(dict.fromkeys(line.key for line in diagram.systems[path].lines if line.key))


def selected_keys(event, keys: Sequence[str], curve: Optional[int] = None) -> List[str]:
    """the signal keys of the points a plotly selection event of ``figure`` holds (``curve``: the
    index of the marker trace, used when a point has no customdata)"""
    found = []
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError, AttributeError):
        try:
            points = event.selection.points
        except AttributeError:
            return []
    for point in points or []:
        data = point.get("customdata")
        if isinstance(data, (list, tuple)):
            data = data[0] if data else None
        if isinstance(data, str) and data in keys:
            found.append(data)
        elif curve is not None and point.get("curve_number") == curve \
                and isinstance(point.get("point_index"), int) and 0 <= point["point_index"] < len(keys):
            found.append(keys[point["point_index"]])
    return list(dict.fromkeys(found))

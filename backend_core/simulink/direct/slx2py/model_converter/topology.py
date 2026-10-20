"""Model-topology extraction independent from executable block support.

This module deliberately understands XML connections before it asks the
runtime whether a block has an implementation.  It lets the loader discard
visualization, documentation, and otherwise disconnected portions of a model
without hiding unsupported blocks that actually influence a root Outport.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from xml.etree import ElementTree

from .graph import Connection, Endpoint


class TopologyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkippedBlock:
    block_id: str
    name: str
    block_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class FlatBlock:
    block_id: str
    name: str
    block_type: str
    element: ElementTree.Element
    is_root: bool
    hierarchy: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _InputRef:
    path: tuple[str, ...]
    port: int


@dataclass(frozen=True, slots=True)
class _OutputRef:
    path: tuple[str, ...]
    port: int


@dataclass(frozen=True, slots=True)
class _GotoRef:
    path: tuple[str, ...]
    tag: str


@dataclass(frozen=True, slots=True)
class _SystemInterface:
    inputs: dict[int, _InputRef]
    outputs: dict[int, _OutputRef]


SignalRef = Endpoint | _InputRef | _OutputRef | _GotoRef


@dataclass(frozen=True, slots=True)
class SignalChoice:
    """Signals the user chose as the model's inputs and outputs, for a model without (enough) root
    Inports / Outports.

    A signal is an output port of a block, by its key "<flat block id>#<port>" (port from 1, see
    ``signal_key``). An input replaces the signal: whatever it fed now reads a new root Inport. An
    input can also be an input port nothing is connected to - of a block, or of a subsystem - by its
    key "<flat block id or subsystem path>#in<port>" (``input_key``): it reads the new root Inport
    instead of 0. An output gets a new root Outport. The new Inports / Outports are numbered in the
    order of ``inputs`` / ``outputs``. ``names``: (key, name) of the new ports."""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    names: tuple[tuple[str, str], ...] = ()

    def name(self, key: str, default: str) -> str:
        return dict(self.names).get(key) or default


def signal_key(endpoint: Endpoint) -> str:
    return "%s#%d" % (endpoint.block_id, endpoint.port + 1)


def input_key(block_id: str, port: int) -> str:
    """the key of input ``port`` (from 0) of a block or a subsystem (its path joined by '::')"""
    return "%s#in%d" % (block_id, port + 1)


def parse_input_key(key: str) -> Endpoint | None:
    """an input key -> Endpoint(block id or subsystem path, port from 0); None for a signal key"""
    block_id, _, port = key.rpartition("#")
    if block_id and port.startswith("in") and port[2:].isdigit() and int(port[2:]) >= 1:
        return Endpoint(block_id, int(port[2:]) - 1)
    return None


def parse_signal_key(key: str) -> Endpoint:
    block_id, _, port = key.rpartition("#")
    if not block_id or not port.isdigit() or int(port) < 1:
        raise TopologyError(f"Invalid signal {key!r}.")
    return Endpoint(block_id, int(port) - 1)


@dataclass(frozen=True, slots=True)
class ModelTopology:
    """The flat model. Every block that computes something is active - the whole model, as Simulink
    simulates it, not only what reaches an output. Left out: virtual signal routing (compiled away)
    and sinks (Scope, Display, Terminator, To Workspace ...: no output, no dynamics).
    ``output_ids``: the active blocks on a path to a root Outport (or to an enable / trigger)."""
    model_name: str
    active_blocks: tuple[FlatBlock, ...]
    active_connections: tuple[Connection, ...]
    skipped_blocks: tuple[SkippedBlock, ...]
    enable_sources: tuple[tuple[str, Endpoint], ...] = ()
    trigger_sources: tuple[tuple[str, Endpoint], ...] = ()
    source_directory: Path | None = None
    output_ids: frozenset = frozenset()


def build_topology(model_directory: str | Path, choice: SignalChoice | None = None) -> ModelTopology:
    directory = Path(model_directory)
    systems_directory = directory / "simulink" / "systems"
    root_system_path = systems_directory / "system_root.xml"
    if not root_system_path.is_file():
        raise TopologyError(f"Missing Simulink system file: {root_system_path}")

    blocks: list[FlatBlock] = []
    virtual_blocks: list[SkippedBlock] = []
    raw_connections: list[tuple[SignalRef, SignalRef]] = []
    raw_enable_connections: list[tuple[SignalRef, tuple[str, ...]]] = []
    raw_trigger_connections: list[tuple[SignalRef, str]] = []
    _flatten_system(
        root_system_path,
        systems_directory,
        blocks,
        virtual_blocks,
        raw_connections,
        raw_enable_connections,
        raw_trigger_connections,
        path=(),
        hierarchy=(),
        is_root=True,
    )
    new_ports = _new_ports(choice, blocks, raw_connections) if choice is not None else {}
    if new_ports:
        _wire_free_inputs(choice, new_ports, blocks, raw_connections)
    connections, enable_connections, trigger_connections = _resolve_connections(
        raw_connections,
        raw_enable_connections,
        raw_trigger_connections,
    )
    if new_ports:
        connections, enable_connections, trigger_connections = _apply_choice(
            choice, new_ports, blocks, connections, enable_connections, trigger_connections)
    output_ids = _outport_ancestors(blocks, connections)
    control_sources = [source.block_id for _, source in enable_connections + trigger_connections]
    output_ids.update(_upstream_block_ids(control_sources, connections))
    # the whole model: every block but the sinks (a block that only shows or stores a signal)
    sinks = {block.block_id for block in blocks if _is_sink(block)}
    active_ids = {block.block_id for block in blocks} - sinks
    output_ids &= active_ids
    active_blocks = tuple(block for block in blocks if block.block_id in active_ids)
    active_connections = tuple(
        connection
        for connection in connections
        if connection.source.block_id in active_ids and connection.destination.block_id in active_ids
    )
    skipped = [
        SkippedBlock(block.block_id, block.name, block.block_type, "sink_without_dynamics")
        for block in blocks
        if block.block_id not in active_ids
    ]
    skipped.extend(virtual_blocks)
    active_enable_sources = tuple(
        (block.block_id, source)
        for block in active_blocks
        for enabled_path, source in enable_connections
        if block.block_id.startswith("::".join(enabled_path) + "::")
    )
    active_trigger_sources = tuple(
        (block_id, source)
        for block_id, source in trigger_connections
        if block_id in active_ids and source.block_id in active_ids
    )
    return ModelTopology(
        directory.name,
        active_blocks,
        active_connections,
        tuple(skipped),
        active_enable_sources,
        active_trigger_sources,
        directory.resolve(),
        frozenset(output_ids),
    )


# blocks that only show, store or check a signal: no output, no dynamics (Stop / Assertion act on the
# simulation run, not on dx/dt)
_SINKS = {"Scope", "Display", "Terminator", "ToWorkspace", "ToFile", "Stop", "Assertion", "XYGraph",
          "FloatingScope", "SignalViewerScope", "DataStoreWrite"}


def _is_sink(block: FlatBlock) -> bool:
    if block.is_root and block.block_type == "Outport":
        return False
    if block.block_type in _SINKS:
        return True
    try:
        return _port_counts(block.element)[1] == 0 and block.element.find("PortCounts") is not None
    except TopologyError:
        return False


def model_signals(model_directory: str | Path) -> dict:
    """Every signal of the model, for choosing inputs and outputs on its diagram.

    -> {"signals": {key: {"block_id", "block": name, "type", "port"}},
        "lines": {(system path, line index): key}}  - the signal each <Line> of each system XML
    carries (the system path is the SIDs of the subsystems from the root; a line from a subsystem
    port, an Inport or a From carries the signal of the block that really computes it). An output
    port no line leaves is a signal too: a model without Outports often ends in one."""
    directory = Path(model_directory)
    systems_directory = directory / "simulink" / "systems"
    blocks: list[FlatBlock] = []
    raw_connections: list[tuple[SignalRef, SignalRef]] = []
    lines: list[tuple[tuple[str, ...], int, SignalRef]] = []
    _flatten_system(systems_directory / "system_root.xml", systems_directory, blocks, [], raw_connections, [], [],
                    path=(), hierarchy=(), is_root=True, line_sources=lines)
    bindings = {destination: source for source, destination in raw_connections if not isinstance(destination, Endpoint)}
    by_id = {block.block_id: block for block in blocks}
    signals: dict[str, dict] = {}
    carried: dict[tuple[tuple[str, ...], int], str] = {}
    for path, index, source in lines:
        try:
            endpoint = _resolve_source(source, bindings)
        except TopologyError:
            continue                      # a route that ends nowhere: nothing to choose
        block = by_id.get(endpoint.block_id)
        if block is None:
            continue
        key = signal_key(endpoint)
        signals.setdefault(key, {"block_id": block.block_id, "block": " ".join(block.name.split()),
                                 "type": block.block_type, "port": endpoint.port + 1})
        carried[(path, index)] = key
    for block in blocks:
        try:
            outputs = _port_counts(block.element)[1]
        except TopologyError:
            continue
        for port in range(outputs):
            signals.setdefault(signal_key(Endpoint(block.block_id, port)), {
                "block_id": block.block_id, "block": " ".join(block.name.split()), "type": block.block_type,
                "port": port + 1})
    for key, info in _free_inputs(blocks, raw_connections).items():
        signals[key] = info
    return {"signals": signals, "lines": carried}


def _free_inputs(blocks: list[FlatBlock], raw_connections) -> dict[str, dict]:
    """the input ports nothing is connected to (they read 0): of blocks that compute something, and of
    subsystems -> {input key: {"block_id", "block", "type", "port", "kind": "input"}}"""
    bindings = {destination for _, destination in raw_connections if not isinstance(destination, Endpoint)}
    taken = {destination for _, destination in raw_connections if isinstance(destination, Endpoint)}
    free: dict[str, dict] = {}
    for block in blocks:
        try:
            inputs, outputs = _port_counts(block.element)
        except TopologyError:
            continue
        if not outputs:
            continue                              # a sink: an input there changes nothing
        for port in range(inputs):
            if Endpoint(block.block_id, port) not in taken:
                free[input_key(block.block_id, port)] = {
                    "block_id": block.block_id, "block": " ".join(block.name.split()), "type": block.block_type,
                    "port": port + 1, "kind": "input"}
    for source, _ in raw_connections:
        if isinstance(source, _InputRef) and source not in bindings:
            path = "::".join(source.path)
            free.setdefault(input_key(path, source.port), {
                "block_id": path, "block": path, "type": "SubSystem", "port": source.port + 1, "kind": "input"})
    return free


def _new_ports(choice: SignalChoice, blocks: list[FlatBlock], raw_connections) -> dict:
    """the new root Inports / Outports of a SignalChoice, in its order -> {("Inport" | "Outport", key): FlatBlock}"""
    by_id = {block.block_id: block for block in blocks}
    subsystems = {"::".join(ref.path) for pair in raw_connections for ref in pair if isinstance(ref, _InputRef)}
    used_names = {block.name for block in blocks if block.is_root and block.block_type in {"Inport", "Outport"}}
    count = {kind: sum(1 for block in blocks if block.is_root and block.block_type == kind) for kind in ("Inport", "Outport")}
    ports = {}
    for kind, keys in (("Inport", choice.inputs), ("Outport", choice.outputs)):
        for key in keys:
            free = parse_input_key(key)
            if free is not None and kind == "Outport":
                raise TopologyError(f"{key!r} is an input port; only a signal can be an output.")
            endpoint = free or parse_signal_key(key)
            block = by_id.get(endpoint.block_id)
            if block is None and not (free is not None and endpoint.block_id in subsystems):
                raise TopologyError(f"The chosen signal {key!r} is not a signal of this model.")
            default = " ".join((block.name if block else endpoint.block_id).split("/")[-1].split())
            name = base = choice.name(key, default)
            k = 2
            while name in used_names:
                name, k = "%s_%d" % (base, k), k + 1
            used_names.add(name)
            count[kind] += 1
            element = ElementTree.Element("Block", {"BlockType": kind, "Name": name,
                                                    "SID": "chosen%s%d" % (kind, count[kind])})
            ElementTree.SubElement(element, "P", {"Name": "Port"}).text = str(count[kind])
            position = (_parameter(block.element, "Position") if block else None) or "[0, 0, 30, 14]"
            ElementTree.SubElement(element, "P", {"Name": "Position"}).text = position   # the exporter's view needs one
            flat = FlatBlock("chosen::%s%d" % (kind.lower(), count[kind]), name, kind, element, True)
            blocks.append(flat)
            ports[(kind, key)] = flat
    return ports


def _wire_free_inputs(choice: SignalChoice, ports: dict, blocks: list[FlatBlock], raw_connections) -> None:
    """a chosen input port nothing was connected to now reads its new root Inport"""
    by_id = {block.block_id: block for block in blocks}
    taken = {destination for _, destination in raw_connections}
    for key in choice.inputs:
        endpoint = parse_input_key(key)
        if endpoint is None:
            continue
        block = by_id.get(endpoint.block_id)
        if block is not None:
            target, name = endpoint, block.name
            if endpoint.port >= _port_counts(block.element)[0]:
                raise TopologyError(f"Block '{block.name}' has no input {endpoint.port + 1}.")
        else:
            target, name = _InputRef(tuple(endpoint.block_id.split("::")), endpoint.port), endpoint.block_id
        if target in taken:
            raise TopologyError(f"Input {endpoint.port + 1} of '{name}' is connected already: choose the signal "
                                f"that feeds it instead.")
        raw_connections.append((Endpoint(ports[("Inport", key)].block_id, 0), target))


def _apply_choice(choice, ports, blocks, connections, enable_connections, trigger_connections):
    """the chosen signals: an input replaces the signal by its new root Inport, an output gets a new
    root Outport (see SignalChoice)"""
    by_id = {block.block_id: block for block in blocks}
    sources = {connection.source for connection in connections}

    def endpoint_of(key):
        endpoint = parse_signal_key(key)
        block = by_id[endpoint.block_id]
        if endpoint not in sources and endpoint.port >= _port_counts(block.element)[1]:
            raise TopologyError(f"Block '{block.name}' has no output {endpoint.port + 1}.")
        return endpoint

    connections = list(connections)
    replaced: dict[Endpoint, Endpoint] = {}
    for key in choice.inputs:
        if parse_input_key(key) is not None:
            continue                              # wired before the routes were resolved
        endpoint = endpoint_of(key)
        replacement = replaced[endpoint] = Endpoint(ports[("Inport", key)].block_id, 0)
        connections = [Connection(replacement, c.destination) if c.source == endpoint else c for c in connections]
        enable_connections = tuple((path, replacement if source == endpoint else source)
                                   for path, source in enable_connections)
        trigger_connections = tuple((target, replacement if source == endpoint else source)
                                    for target, source in trigger_connections)
    chosen_outports = {ports[("Outport", key)].block_id for key in choice.outputs}
    shown = {c.source for c in connections                # signals a root Outport of the model already shows
             if c.destination.block_id in by_id and by_id[c.destination.block_id].is_root
             and by_id[c.destination.block_id].block_type == "Outport" and c.destination.block_id not in chosen_outports}
    dropped = []
    for key in choice.outputs:
        endpoint = endpoint_of(key)
        source = replaced.get(endpoint, endpoint)         # an output of a signal that is also an input shows that input
        outport = ports[("Outport", key)]
        if source in shown:
            dropped.append(outport)                       # the model shows it already: no second output y
            continue
        shown.add(source)
        connections.append(Connection(source, Endpoint(outport.block_id, 0)))
    if dropped:
        for outport in dropped:
            blocks.remove(outport)
        first = sum(1 for b in blocks if b.is_root and b.block_type == "Outport" and b.block_id not in chosen_outports)
        kept = [b for b in blocks if b.block_id in chosen_outports]
        for number, outport in enumerate(kept, start=first + 1):    # the new Outports numbered without gaps
            next(p for p in outport.element.findall("P") if p.get("Name") == "Port").text = str(number)
    return tuple(connections), enable_connections, trigger_connections


def write_skipped_blocks_report(topology: ModelTopology, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": topology.model_name,
        "active_block_count": len(topology.active_blocks),
        "active_connection_count": len(topology.active_connections),
        "skipped_block_count": len(topology.skipped_blocks),
        "skipped_blocks": [asdict(block) for block in topology.skipped_blocks],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_skipped_blocks_log(topology: ModelTopology, output_path: str | Path) -> Path:
    """Write a compact, human-readable companion to the JSON skip report.

    The topology pass intentionally keeps unsupported blocks when they are on a
    path to a root Outport.  This log therefore documents only blocks that were
    proven irrelevant to the exported input-to-output computation, or virtual
    signal-routing blocks that were compiled away.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Model: {topology.model_name}",
        f"Active blocks: {len(topology.active_blocks)}",
        f"Active connections: {len(topology.active_connections)}",
        f"Skipped blocks: {len(topology.skipped_blocks)}",
        "",
    ]
    if not topology.skipped_blocks:
        lines.append("No blocks were skipped.")
    else:
        lines.append("SID | Type | Name | Reason")
        lines.append("-" * 88)
        for block in topology.skipped_blocks:
            name = " ".join(block.name.split())
            lines.append(f"{block.block_id} | {block.block_type} | {name} | {block.reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _flatten_system(
    system_path: Path,
    systems_directory: Path,
    blocks: list[FlatBlock],
    virtual_blocks: list[SkippedBlock],
    raw_connections: list[tuple[SignalRef, SignalRef]],
    raw_enable_connections: list[tuple[SignalRef, tuple[str, ...]]],
    raw_trigger_connections: list[tuple[SignalRef, str]],
    path: tuple[str, ...],
    hierarchy: tuple[str, ...],
    is_root: bool,
    line_sources: list | None = None,
) -> _SystemInterface:
    try:
        system_root = ElementTree.parse(system_path).getroot()
    except ElementTree.ParseError as error:
        raise TopologyError(f"Invalid Simulink system XML: {system_path}") from error

    elements = list(system_root.findall("Block"))
    input_ports = _port_indices(elements, "Inport")
    output_ports = _port_indices(elements, "Outport")
    sources: dict[Endpoint, SignalRef] = {}
    destinations: dict[Endpoint, SignalRef] = {}
    interface_inputs: dict[int, _InputRef] = {}
    interface_outputs: dict[int, _OutputRef] = {}
    subsystem_paths: dict[str, tuple[str, ...]] = {}
    trigger_targets: dict[str, str] = {}
    wired = _wired_ports(system_root)

    for element in elements:
        sid = _required_attribute(element, "SID", "block")
        block_type = _required_attribute(element, "BlockType", "block")
        name = element.get("Name", "")

        if block_type == "Inport":
            port = input_ports[sid]
            if is_root:
                block_id = _flat_block_id(path, sid)
                blocks.append(FlatBlock(block_id, name, block_type, element, True))
                sources[Endpoint(sid, 0)] = Endpoint(block_id, 0)
            else:
                reference = _InputRef(path, port)
                interface_inputs[port] = reference
                sources[Endpoint(sid, 0)] = reference
            continue

        if block_type == "Outport":
            port = output_ports[sid]
            if is_root:
                block_id = _flat_block_id(path, sid)
                blocks.append(FlatBlock(block_id, name, block_type, element, True))
                destinations[Endpoint(sid, 0)] = Endpoint(block_id, 0)
            else:
                reference = _OutputRef(path, port)
                interface_outputs[port] = reference
                destinations[Endpoint(sid, 0)] = reference
            continue

        if block_type == "SubSystem":
            # Stateflow blocks are executable subsystems, not transparent
            # signal-routing containers. Preserve their public interface and
            # deliberately do not expose generated S-Functions, triggers,
            # Demuxes, or Terminators as ordinary model blocks.
            stateflow_type = _parameter(element, "SFBlockType")
            if stateflow_type in {"Chart", "MATLAB Function"}:
                block_id = _flat_block_id(path, sid)
                flat_type = "StateflowChart" if stateflow_type == "Chart" else "MatlabFunction"
                blocks.append(FlatBlock(block_id, _flat_name(hierarchy, name, is_root), flat_type, element, is_root, hierarchy))
                input_count, output_count = _port_counts(element)
                _register_block_ports(sources, destinations, sid, block_id, input_count, output_count)
                if stateflow_type == "Chart":
                    trigger_targets[sid] = block_id
                continue
            reference = _subsystem_reference(element, name)
            child_interface = _flatten_system(
                systems_directory / f"{reference}.xml",
                systems_directory,
                blocks,
                virtual_blocks,
                raw_connections,
                raw_enable_connections,
                raw_trigger_connections,
                path + (sid,),
                hierarchy + (name,),
                False,
                line_sources,
            )
            _register_subsystem_ports(sources, destinations, sid, child_interface)
            subsystem_paths[sid] = path + (sid,)
            continue

        if block_type == "Goto":
            visibility = _parameter(element, "TagVisibility") or "local"
            if visibility != "local":
                raise TopologyError(f"Goto '{name}' uses unsupported {visibility!r} visibility; cross-system tag resolution is not implemented.")
            tag = _parameter(element, "GotoTag")
            if tag is None:
                tag = "A"  # Documented Simulink library default, omitted from SLX when unchanged.
            if not tag:
                raise TopologyError(f"Goto block '{name}' has no GotoTag.")
            destinations[Endpoint(sid, 0)] = _GotoRef(path, tag)
            virtual_blocks.append(SkippedBlock(_flat_block_id(path, sid), _flat_name(hierarchy, name, is_root), block_type, "compiled_as_signal_routing"))
            continue

        if block_type == "From":
            tag = _parameter(element, "GotoTag")
            if tag is None:
                tag = "A"  # Documented Simulink library default, omitted from SLX when unchanged.
            if not tag:
                raise TopologyError(f"From block '{name}' has no GotoTag.")
            sources[Endpoint(sid, 0)] = _GotoRef(path, tag)
            virtual_blocks.append(SkippedBlock(_flat_block_id(path, sid), _flat_name(hierarchy, name, is_root), block_type, "compiled_as_signal_routing"))
            continue

        block_id = _flat_block_id(path, sid)
        blocks.append(FlatBlock(block_id, _flat_name(hierarchy, name, is_root), block_type, element, is_root, hierarchy))
        input_count, output_count = _port_counts(element, wired.get(sid, (0, 0)))
        _register_block_ports(sources, destinations, sid, block_id, input_count, output_count)

    for line_index, line in enumerate(system_root.findall("Line")):
        source_text = _parameter(line, "Src")
        if source_text is None:
            continue
        source = sources.get(_parse_endpoint(source_text, "out"))
        if source is not None and line_sources is not None:
            line_sources.append((path, line_index, source))
        if source is None:
            # Enable, trigger, reset, and action ports are control semantics;
            # they are intentionally not treated as ordinary signal edges.
            continue
        for destination_text in _line_destinations(line):
            if destination_text.endswith("#enable"):
                subsystem_id = destination_text.removesuffix("#enable")
                enabled_path = subsystem_paths.get(subsystem_id)
                if enabled_path is not None:
                    raw_enable_connections.append((source, enabled_path))
                continue
            if destination_text.endswith("#trigger"):
                target_id = trigger_targets.get(destination_text.removesuffix("#trigger"))
                if target_id is not None:
                    raw_trigger_connections.append((source, target_id))
                else:
                    raise TopologyError(f"Triggered subsystem port {destination_text!r} is not implemented; execution control cannot be discarded.")
                continue
            try:
                destination_endpoint = _parse_endpoint(destination_text, "in")
            except TopologyError as error:
                raise TopologyError(f"Unsupported execution-control destination {destination_text!r}; no approximation was exported.") from error
            destination = destinations.get(destination_endpoint)
            if destination is not None:
                raw_connections.append((source, destination))

    return _SystemInterface(interface_inputs, interface_outputs)


def _resolve_connections(
    raw_connections: list[tuple[SignalRef, SignalRef]],
    raw_enable_connections: list[tuple[SignalRef, tuple[str, ...]]],
    raw_trigger_connections: list[tuple[SignalRef, str]],
) -> tuple[
    tuple[Connection, ...],
    tuple[tuple[tuple[str, ...], Endpoint], ...],
    tuple[tuple[str, Endpoint], ...],
]:
    bindings: dict[_InputRef | _OutputRef | _GotoRef, SignalRef] = {}
    for source, destination in raw_connections:
        if isinstance(destination, Endpoint):
            continue
        if destination in bindings:
            raise TopologyError(f"Signal-routing target '{destination}' has more than one source.")
        bindings[destination] = source

    connections: list[Connection] = []
    for source, destination in raw_connections:
        if not isinstance(destination, Endpoint):
            continue
        try:
            resolved = _resolve_source(source, bindings)
        except _UnconnectedPort:
            continue                              # an unconnected subsystem port: the input reads 0, as in Simulink
        connections.append(Connection(resolved, destination))
    enable_connections = tuple(
        (enabled_path, _resolve_source(source, bindings))
        for source, enabled_path in raw_enable_connections
    )
    trigger_connections = tuple(
        (target, _resolve_source(source, bindings))
        for source, target in raw_trigger_connections
    )
    return tuple(connections), enable_connections, trigger_connections


def _resolve_source(source: SignalRef, bindings: dict[_InputRef | _OutputRef | _GotoRef, SignalRef]) -> Endpoint:
    visited: set[_InputRef | _OutputRef | _GotoRef] = set()
    while not isinstance(source, Endpoint):
        if source in visited:
            raise TopologyError(f"Circular virtual signal route at '{source}'.")
        visited.add(source)
        try:
            source = bindings[source]
        except KeyError as error:
            kind = _UnconnectedPort if isinstance(source, (_InputRef, _OutputRef)) else TopologyError
            raise kind(f"Unconnected virtual signal route at '{source}'.") from error
    return source


class _UnconnectedPort(TopologyError):
    """a port of a subsystem that nothing is connected to"""


def _upstream_block_ids(seed_ids: list[str], connections: tuple[Connection, ...]) -> set[str]:
    """Return every data source needed to produce the requested controls."""
    predecessors: dict[str, set[str]] = defaultdict(set)
    for connection in connections:
        predecessors[connection.destination.block_id].add(connection.source.block_id)
    active = set(seed_ids)
    pending = deque(seed_ids)
    while pending:
        block_id = pending.popleft()
        for predecessor in predecessors.get(block_id, ()):
            if predecessor not in active:
                active.add(predecessor)
                pending.append(predecessor)
    return active


def _outport_ancestors(blocks: list[FlatBlock], connections: tuple[Connection, ...]) -> set[str]:
    root_outports = {block.block_id for block in blocks if block.is_root and block.block_type == "Outport"}
    upstream: dict[str, set[str]] = defaultdict(set)
    for connection in connections:
        upstream[connection.destination.block_id].add(connection.source.block_id)

    active = set(root_outports)
    queue = deque(root_outports)
    while queue:
        block_id = queue.popleft()
        for source_id in upstream[block_id]:
            if source_id not in active:
                active.add(source_id)
                queue.append(source_id)
    return active


def _register_block_ports(
    sources: dict[Endpoint, SignalRef],
    destinations: dict[Endpoint, SignalRef],
    sid: str,
    block_id: str,
    input_count: int,
    output_count: int,
) -> None:
    for port in range(output_count):
        sources[Endpoint(sid, port)] = Endpoint(block_id, port)
    for port in range(input_count):
        destinations[Endpoint(sid, port)] = Endpoint(block_id, port)


def _register_subsystem_ports(
    sources: dict[Endpoint, SignalRef],
    destinations: dict[Endpoint, SignalRef],
    sid: str,
    interface: _SystemInterface,
) -> None:
    for port, source in interface.outputs.items():
        sources[Endpoint(sid, port)] = source
    for port, destination in interface.inputs.items():
        destinations[Endpoint(sid, port)] = destination


# the ports of blocks whose parameters are at their defaults: Simulink then omits <PortCounts>
_DEFAULT_PORTS = {
    **{name: (1, 1) for name in (
        "Abs", "Signum", "UnaryMinus", "Bias", "Polyval", "Sqrt", "Math", "Trigonometry", "Rounding", "MinMax",
        "DeadZone", "Quantizer", "DataTypeConversion", "InitialCondition", "SignalSpecification", "ZeroPole",
        "StateSpace", "Selector", "UnitDelay", "ZeroOrderHold", "Memory", "DiscreteIntegrator",
        "DiscreteTransferFcn", "DiscreteStateSpace", "TransportDelay", "Derivative", "RateLimiter", "Relay",
        "Backlash", "HitCross")},
    **{name: (0, 1) for name in ("Clock", "Ground", "Sin", "SignalGenerator")},
    "Switch": (3, 1), "RelationalOperator": (2, 1), "Logic": (2, 1), "Product": (2, 1), "MultiPortSwitch": (4, 1),
    "Demux": (1, 2), "Concatenate": (2, 1), "Lookup_n-D": (2, 1), "LookupNDDirect": (2, 1),
    "SecondOrderIntegrator": (1, 2),
}


def _wired_ports(system_root: ElementTree.Element) -> dict[str, tuple[int, int]]:
    """block SID -> (inputs, outputs) that lines connect: the ports of a block when nothing else says"""
    wired: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in system_root.findall("Line"):
        source = _parameter(line, "Src")
        ends = ([(source, "out")] if source else []) + [(d, "in") for d in _line_destinations(line)]
        for text, direction in ends:
            sid, _, port = text.partition(f"#{direction}:")
            if port.isdigit():
                slot = 0 if direction == "in" else 1
                wired[sid][slot] = max(wired[sid][slot], int(port))
    return {sid: (counts[0], counts[1]) for sid, counts in wired.items()}


def _port_counts(element: ElementTree.Element, wired: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    block_type = _required_attribute(element, "BlockType", "block")
    port_counts = element.find("PortCounts")
    if port_counts is not None:
        return int(port_counts.get("in", "0")), int(port_counts.get("out", "0"))
    if block_type in _DEFAULT_PORTS:
        return _DEFAULT_PORTS[block_type]

    fixed_counts = {
        "Constant": (0, 1), "Step": (0, 1), "Gain": (1, 1), "Integrator": (1, 1), "TransferFcn": (1, 1),
        "Fcn": (1, 1), "Saturate": (1, 1), "ManualSwitch": (2, 1), "Scope": (1, 0),
        "Terminator": (1, 0), "EnablePort": (0, 0), "TriggerPort": (0, 1),
        "DiscretePulseGenerator": (0, 1), "From": (0, 1), "Goto": (1, 0), "Inport": (0, 1), "Outport": (1, 0),
    }
    if block_type == "Sum":
        signs = (_parameter(element, "Inputs") or "++").replace("|", "")
        return (int(signs) if signs.strip().isdigit() else len(signs)), 1
    if block_type == "Mux":
        return int(_parameter(element, "Inputs") or "1"), 1
    return fixed_counts.get(block_type, wired)


def _port_indices(elements: list[ElementTree.Element], block_type: str) -> dict[str, int]:
    indices: dict[str, int] = {}
    for fallback_index, element in enumerate(block for block in elements if block.get("BlockType") == block_type):
        port_text = _parameter(element, "Port")
        try:
            port = int(port_text) - 1 if port_text is not None else fallback_index
        except ValueError as error:
            raise TopologyError(f"Invalid port number '{port_text}' on block '{element.get('Name', '')}'.") from error
        indices[_required_attribute(element, "SID", "block")] = port
    return indices


def _subsystem_reference(element: ElementTree.Element, name: str) -> str:
    system = element.find("System")
    reference = system.get("Ref") if system is not None else None
    if not reference:
        raise TopologyError(f"Subsystem '{name}' has no system reference.")
    return reference


def _flat_block_id(path: tuple[str, ...], sid: str) -> str:
    return "::".join(path + (sid,))


def _flat_name(hierarchy: tuple[str, ...], name: str, is_root: bool) -> str:
    return name if is_root or not hierarchy else "/".join(hierarchy + (name,))


def _line_destinations(element: ElementTree.Element) -> list[str]:
    destinations: list[str] = []
    destination = _parameter(element, "Dst")
    if destination is not None:
        destinations.append(destination)
    for branch in element.findall("Branch"):
        destinations.extend(_line_destinations(branch))
    return destinations


def _parse_endpoint(value: str, direction: str) -> Endpoint:
    separator = f"#{direction}:"
    try:
        block_id, port_text = value.split(separator, maxsplit=1)
        port = int(port_text) - 1
    except ValueError as error:
        raise TopologyError(f"Invalid {direction} endpoint '{value}'.") from error
    if port < 0:
        raise TopologyError(f"Invalid {direction} endpoint '{value}'.")
    return Endpoint(block_id, port)


def _parameter(element: ElementTree.Element, name: str) -> str | None:
    for parameter in element.findall("P"):
        if parameter.get("Name") == name:
            return parameter.text or ""
    return None


def _required_attribute(element: ElementTree.Element, name: str, element_name: str) -> str:
    value = element.get(name)
    if value is None:
        raise TopologyError(f"A {element_name} is missing {name}.")
    return value

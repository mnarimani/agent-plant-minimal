from __future__ import annotations

import ast
from dataclasses import dataclass, field
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from . import block_semantics, discrete_semantics, sample_times, workspace
from .blocks import Concatenate, Constant, Demux, Fcn, Gain, Selector, Inport, Integrator, Lookup1D, ManualSwitch, MathBlock, MatlabFunction, Mux, Outport, Product, PulseGenerator, Saturate, SecondOrderIntegrator, StateSpace, StateflowChart, Step, Sum, TransferFcn
from .graph import Endpoint
from .model import Model
from .topology import ModelTopology, SkippedBlock, TopologyError, build_topology


class ModelLoaderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: Model
    solver_name: str
    fixed_step: float
    start_time: float
    stop_time: float
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-6
    skipped_blocks: tuple[SkippedBlock, ...] = ()
    solver_options: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _InputRef:
    path: tuple[str, ...]
    port: int


@dataclass(frozen=True, slots=True)
class _OutputRef:
    path: tuple[str, ...]
    port: int


@dataclass(frozen=True, slots=True)
class _SystemInterface:
    inputs: dict[int, _InputRef]
    outputs: dict[int, _OutputRef]


SignalRef = Endpoint | _InputRef | _OutputRef


def load_model(
    model_directory: str | Path,
    topology: ModelTopology | None = None,
    parameters: dict[str, float] | None = None,
) -> LoadedModel:
    directory = Path(model_directory)
    systems_directory = directory / "simulink" / "systems"
    system_path = systems_directory / "system_root.xml"
    config_path = directory / "simulink" / "configSet0.xml"
    if not system_path.is_file():
        raise ModelLoaderError(f"Missing Simulink system file: {system_path}")
    if not config_path.is_file():
        raise ModelLoaderError(f"Missing Simulink configuration file: {config_path}")

    try:
        config_root = ElementTree.parse(config_path).getroot()
    except ElementTree.ParseError as error:
        raise ModelLoaderError(f"Invalid model XML in '{directory}'.") from error

    try:
        topology = topology or build_topology(directory)
    except TopologyError as error:
        raise ModelLoaderError(str(error)) from error

    model = Model(directory.name)
    chart_directory = directory / "simulink" / "stateflow"
    chart_sources = _load_stateflow_chart_sources(chart_directory)
    matlab_function_sources = _load_stateflow_charts(chart_directory)
    parameters = parameters if parameters is not None else {}
    model_start_time = _numeric_expression(_required_nested_parameter(config_root, "StartTime"), "StartTime", parameters)
    step_text = _required_nested_parameter(config_root, "FixedStep")
    model_step = None if step_text == "auto" else _numeric_expression(step_text, "FixedStep", parameters)
    parsed = {}
    failures: dict[str, str] = {}
    for flat_block in topology.active_blocks:
        try:
            block = _parse_flat_block(
                flat_block.element,
                flat_block.block_id,
                flat_block.name,
                parameters,
                chart_sources,
                matlab_function_sources,
                flat_block.block_type,
                model_start_time,
                model_step,
            )
        except (ModelLoaderError, ValueError) as error:
            failures[flat_block.block_id] = (
                f"{flat_block.block_type} '{flat_block.name}' (SID {flat_block.block_id}): {error}"
                + _other_symbols(str(error), flat_block, parameters, chart_sources, matlab_function_sources,
                                 model_start_time, model_step))
            continue
        parsed[flat_block.block_id] = block
    # The whole model is converted. A block that cannot be is left out only when nothing it computes
    # reaches an output and it is not a missing value (that the user can give): then it goes, with
    # every block after it, and the model says so. Anywhere else it stops the conversion.
    fatal = [text for block_id, text in failures.items()
             if block_id in topology.output_ids or "workspace symbol '" in text]
    if fatal:
        details = "\n- ".join(fatal + [t for t in failures.values() if t not in fatal])
        raise ModelLoaderError(
            "Unsupported block(s) on a path to a root Outport. They were not skipped because doing so would change the model:\n- "
            + details
        )
    left_out = _downstream(set(failures), topology.active_connections)
    names = {b.block_id: b.name for b in topology.active_blocks}
    model.not_converted = (
        ["Not converted - %s. Nothing it computes reaches an output; it is left out together with the blocks after it%s."
         % (text, (": " + ", ".join("'%s'" % names[i] for i in sorted(left_out - set(failures)))) if left_out - set(failures) else "")
         for text in failures.values()])
    for block_id, block in parsed.items():
        if block_id not in left_out:
            model.add_block(block)

    for connection in topology.active_connections:
        if connection.source.block_id in left_out or connection.destination.block_id in left_out:
            continue
        model.connect(connection.source, connection.destination)
    try:
        model.sample_time_notes = sample_times.resolve(model, model_step)
    except sample_times.SampleTimeError as error:
        raise ModelLoaderError(str(error)) from error
    for block_id, source in topology.enable_sources:
        model.set_enable(block_id, source)
    for block_id, source in topology.trigger_sources:
        model.set_trigger(block_id, source)

    solver_name = _required_nested_parameter(config_root, "SolverName")
    start_time = _numeric_expression(_required_nested_parameter(config_root, "StartTime"), "StartTime", parameters)
    stop_time = _numeric_expression(_required_nested_parameter(config_root, "StopTime"), "StopTime", parameters)
    fixed_step_text = _required_nested_parameter(config_root, "FixedStep")
    if fixed_step_text != "auto":
        fixed_step = _numeric_expression(fixed_step_text, "FixedStep", parameters)
    elif solver_name == "ode45":
        # This is an output/reporting interval only.  Ode45 chooses its own
        # adaptive internal integration steps from the configured tolerances.
        fixed_step = (stop_time - start_time) / 1000.0
    else:
        raise ModelLoaderError(f"Unsupported automatic-step solver '{solver_name}'.")

    return LoadedModel(
        model=model,
        solver_name=solver_name,
        fixed_step=fixed_step,
        start_time=start_time,
        stop_time=stop_time,
        relative_tolerance=_tolerance_parameter(config_root, "RelTol", default=1e-3),
        absolute_tolerance=_tolerance_parameter(
            config_root, "AbsTol", default=min(1e-6, 1e-3 * _tolerance_parameter(config_root, "RelTol", default=1e-3))),
        solver_options=_solver_options(config_root, parameters),
        skipped_blocks=topology.skipped_blocks,
    )


def _downstream(start: set, connections) -> set:
    """the blocks in ``start`` and every block their signals reach"""
    after: dict[str, set] = {}
    for connection in connections:
        after.setdefault(connection.source.block_id, set()).add(connection.destination.block_id)
    seen, stack = set(start), list(start)
    while stack:
        for nxt in after.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _other_symbols(message, flat_block, parameters, chart_sources, matlab_function_sources, start_time, step=None) -> str:
    """A block stops at its first unknown workspace variable. Read it once more with every unknown
    name counted as 1, to name all of them at once (so the user is asked for them in one go)."""
    if "workspace symbol '" not in message or not isinstance(parameters, workspace.UsedParameters):
        return ""
    parameters.missing = set()
    try:
        _parse_flat_block(flat_block.element, flat_block.block_id, flat_block.name, parameters, chart_sources,
                          matlab_function_sources, flat_block.block_type, start_time, step)
    except Exception:                         # noqa: BLE001 - only the names are wanted
        pass
    finally:
        found, parameters.missing = parameters.missing, None
    known = set(re.findall(r"workspace symbol '([^']+)'", message))
    return "".join(f" It also references workspace symbol '{name}'." for name in sorted(found - known))


def _flatten_system(
    system_path: Path,
    systems_directory: Path,
    charts: dict[str, str],
    model: Model,
    raw_connections: list[tuple[SignalRef, SignalRef]],
    path: tuple[str, ...],
    hierarchy: tuple[str, ...],
    is_root: bool,
) -> _SystemInterface:
    try:
        system_root = ElementTree.parse(system_path).getroot()
    except ElementTree.ParseError as error:
        raise ModelLoaderError(f"Invalid Simulink system file: {system_path}") from error

    blocks = list(system_root.findall("Block"))
    input_ports = _port_indices(blocks, "Inport")
    output_ports = _port_indices(blocks, "Outport")
    sources: dict[Endpoint, SignalRef] = {}
    destinations: dict[Endpoint, SignalRef] = {}
    interface_inputs: dict[int, _InputRef] = {}
    interface_outputs: dict[int, _OutputRef] = {}

    for element in blocks:
        sid = _block_sid(element)
        block_type = _block_type(element)
        name = _block_name(element)

        if block_type == "Inport":
            port = input_ports[sid]
            if is_root:
                block_id = _flat_block_id(path, sid)
                model.add_block(Inport(block_id, name))
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
                model.add_block(Outport(block_id, name))
                destinations[Endpoint(sid, 0)] = Endpoint(block_id, 0)
            else:
                reference = _OutputRef(path, port)
                interface_outputs[port] = reference
                destinations[Endpoint(sid, 0)] = reference
            continue

        if block_type == "SubSystem":
            block_id = _flat_block_id(path, sid)
            if _parameter(element, "SFBlockType") == "MATLAB Function":
                function = _matlab_function_block(element, block_id, _flat_block_name(hierarchy, name, is_root), hierarchy, charts)
                model.add_block(function)
                _register_block_ports(sources, destinations, sid, block_id, function.input_count, function.output_count)
                continue

            reference = _subsystem_reference(element, name)
            child_interface = _flatten_system(
                systems_directory / f"{reference}.xml",
                systems_directory,
                charts,
                model,
                raw_connections,
                path=path + (sid,),
                hierarchy=hierarchy + (name,),
                is_root=False,
            )
            _register_subsystem_ports(sources, destinations, sid, child_interface)
            continue

        block_id = _flat_block_id(path, sid)
        block = _parse_block(element, block_id, _flat_block_name(hierarchy, name, is_root))
        model.add_block(block)
        _register_block_ports(sources, destinations, sid, block_id, block.input_count, block.output_count)

    for line in system_root.findall("Line"):
        source_text = _parameter(line, "Src")
        if source_text is None:
            raise ModelLoaderError("A line is missing its source endpoint.")
        source_endpoint = _parse_endpoint(source_text, "out")
        try:
            source = sources[source_endpoint]
        except KeyError as error:
            raise ModelLoaderError(f"Unknown source endpoint '{source_text}'.") from error
        for destination_text in _line_destinations(line):
            destination_endpoint = _parse_endpoint(destination_text, "in")
            try:
                destination = destinations[destination_endpoint]
            except KeyError as error:
                raise ModelLoaderError(f"Unknown destination endpoint '{destination_text}'.") from error
            raw_connections.append((source, destination))

    return _SystemInterface(interface_inputs, interface_outputs)


def _connect_flattened_model(model: Model, raw_connections: list[tuple[SignalRef, SignalRef]]) -> None:
    bindings: dict[_InputRef | _OutputRef, SignalRef] = {}
    for source, destination in raw_connections:
        if isinstance(destination, Endpoint):
            continue
        if destination in bindings:
            raise ModelLoaderError(f"Interface port '{destination}' has more than one source.")
        bindings[destination] = source

    for source, destination in raw_connections:
        if isinstance(destination, Endpoint):
            model.connect(_resolve_source(source, bindings), destination)


def _resolve_source(source: SignalRef, bindings: dict[_InputRef | _OutputRef, SignalRef]) -> Endpoint:
    visited: set[_InputRef | _OutputRef] = set()
    while not isinstance(source, Endpoint):
        if source in visited:
            raise ModelLoaderError(f"Circular subsystem interface connection at '{source}'.")
        visited.add(source)
        try:
            source = bindings[source]
        except KeyError as error:
            raise ModelLoaderError(f"Unconnected subsystem interface port '{source}'.") from error
    return source


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


def _matlab_function_block(
    element: ElementTree.Element,
    block_id: str,
    display_name: str,
    hierarchy: tuple[str, ...],
    charts: dict[str, str],
) -> MatlabFunction:
    name = _block_name(element)
    chart_name = "/".join(hierarchy + (name,))
    try:
        script = charts[chart_name]
    except KeyError as error:
        raise ModelLoaderError(f"Missing Stateflow chart for MATLAB Function '{chart_name}'.") from error

    try:
        return MatlabFunction(block_id, display_name, script)
    except ValueError as error:
        raise ModelLoaderError(f"Unsupported MATLAB Function '{chart_name}': {error}") from error


def _load_stateflow_charts(directory: Path) -> dict[str, str]:
    charts: dict[str, str] = {}
    if not directory.is_dir():
        return charts
    for chart_path in directory.glob("chart_*.xml"):
        try:
            chart = ElementTree.parse(chart_path).getroot()
        except ElementTree.ParseError as error:
            raise ModelLoaderError(f"Invalid Stateflow chart: {chart_path}") from error
        name = _parameter(chart, "name")
        script = next((parameter.text or "" for parameter in chart.iter("P") if parameter.get("Name") == "script"), None)
        if name and script:
            charts[name] = script
    return charts


def _load_stateflow_chart_sources(directory: Path) -> dict[str, str]:
    charts: dict[str, str] = {}
    if not directory.is_dir():
        return charts
    for chart_path in directory.glob("chart_*.xml"):
        try:
            chart = ElementTree.parse(chart_path).getroot()
        except ElementTree.ParseError as error:
            raise ModelLoaderError(f"Invalid Stateflow chart: {chart_path}") from error
        name = _parameter(chart, "name")
        if name:
            charts[name] = ElementTree.tostring(chart, encoding="unicode")
    return charts


def _subsystem_reference(element: ElementTree.Element, name: str) -> str:
    system = element.find("System")
    reference = system.get("Ref") if system is not None else None
    if not reference:
        raise ModelLoaderError(f"Subsystem '{name}' has no system reference.")
    return reference


def _port_indices(blocks: list[ElementTree.Element], block_type: str) -> dict[str, int]:
    indices: dict[str, int] = {}
    matching_blocks = [block for block in blocks if _block_type(block) == block_type]
    for fallback_index, element in enumerate(matching_blocks):
        port_text = _parameter(element, "Port")
        try:
            port = int(port_text) - 1 if port_text is not None else fallback_index
        except ValueError as error:
            raise ModelLoaderError(f"Invalid port number '{port_text}' on block '{_block_name(element)}'.") from error
        if port < 0 or port in indices.values():
            raise ModelLoaderError(f"Invalid or duplicate port number on block '{_block_name(element)}'.")
        indices[_block_sid(element)] = port
    return indices


def _flat_block_id(path: tuple[str, ...], sid: str) -> str:
    return "::".join(path + (sid,))


def _flat_block_name(hierarchy: tuple[str, ...], name: str, is_root: bool) -> str:
    return name if is_root or not hierarchy else "/".join(hierarchy + (name,))


def _block_sid(element: ElementTree.Element) -> str:
    sid = element.get("SID")
    if not sid:
        raise ModelLoaderError("A block is missing SID.")
    return sid


def _block_type(element: ElementTree.Element) -> str:
    block_type = element.get("BlockType")
    if not block_type:
        raise ModelLoaderError("A block is missing BlockType.")
    return block_type


def _block_name(element: ElementTree.Element) -> str:
    name = element.get("Name")
    if name is None:
        raise ModelLoaderError("A block is missing Name.")
    return name


def _parse_flat_block(
    element: ElementTree.Element,
    block_id: str,
    name: str,
    parameters: dict[str, float],
    chart_sources: dict[str, str],
    matlab_function_sources: dict[str, str],
    flat_block_type: str | None = None,
    start_time: float = 0.0,
    step: float | None = None,
):
    block_type = flat_block_type or _block_type(element)
    if block_type == "Integrator":
        for parameter, default in (("ExternalReset", "none"), ("WrapState", "off")):
            value = _parameter(element, parameter) or default
            if value != default:
                raise ModelLoaderError(f"Integrator option {parameter}={value!r} is not implemented; no approximate state behavior is exported.")
    dtype = _parameter(element, "OutDataTypeStr")
    # logic blocks are boolean in Simulink; the runtime is double (1 / 0) like Simulink's signal
    # values. Data Type Conversion handles its own target type.
    if dtype and dtype not in {"double", "boolean"} and not dtype.startswith("Inherit:") and block_type != "DataTypeConversion":
        raise ModelLoaderError(f"Output data type {dtype!r} is not supported by the real scalar runtime.")
    if block_type == "Inport":
        width = _inport_width(element, name, parameters)
        inherited = _parameter(element, "PortDimensions") in (None, "", "-1")
        return Inport(block_id, name, width=width, inherited=inherited)
    if block_type == "Outport":
        return Outport(block_id, name)
    if block_type == "StateflowChart":
        try:
            return StateflowChart(block_id, name, chart_sources[name])
        except KeyError as error:
            raise ModelLoaderError(f"Missing Stateflow chart XML for '{name}'.") from error
        except ValueError as error:
            raise ModelLoaderError(f"Unsupported Stateflow chart '{name}': {error}") from error
    if block_type == "MatlabFunction":
        try:
            script = matlab_function_sources[name]
        except KeyError as error:
            raise ModelLoaderError(f"Missing Stateflow chart for MATLAB Function '{name}'.") from error
        try:
            return MatlabFunction(block_id, name, script)
        except ValueError as error:
            raise ModelLoaderError(f"Unsupported MATLAB Function '{name}': {error}") from error
    return _parse_block(element, block_id, name, parameters, start_time, step)


def _parse_block(element: ElementTree.Element, block_id: str, name: str, parameters: dict[str, float] | None = None,
                 start_time: float = 0.0, step: float | None = None):
    parameters = parameters if parameters is not None else {}
    block_type = _block_type(element)
    source_type = " ".join((_parameter(element, "SourceType") or "").split()) if block_type == "Reference" else None
    semantics_key = f"Reference:{source_type}" if block_type == "Reference" else block_type
    params = block_semantics.Params(
        text=lambda key: _block_parameter(element, key),
        number=lambda key, default: _scalar_parameter(element, key, default=default, parameters=parameters),
        matrix=lambda key, default: _matrix_parameter(element, key, default, parameters),
        start_time=start_time,
    )
    try:
        described = block_semantics.build(semantics_key, params)
    except block_semantics.SemanticsError as error:
        raise ModelLoaderError(f"{error} (block '{name}')") from error
    if described is not None:
        kind, input_count, outputs, *reduce = described
        return MathBlock(block_id, name, kind, input_count, outputs, reduce=reduce[0] if reduce else None)
    if block_type == "Reference":
        if semantics_key == "Reference:PID 1dof":
            return _pid_controller(block_id, name, params)
        source = " ".join((_parameter(element, "SourceBlock") or "").split())
        raise ModelLoaderError(f"Library block '{source}' (SourceType {source_type!r}) is not supported; its "
                               "contents are not stored in the .slx file.")
    if block_type == "DiscretePulseGenerator":
        if (_parameter(element, "PulseType") or "Time based") != "Time based" or (_parameter(element, "TimeSource") or "Use simulation time") != "Use simulation time":
            raise ModelLoaderError("Only scalar time-based Pulse Generator with simulation time is supported; sample-based/external-clock pulses require a different scheduler.")
        return PulseGenerator(block_id, name, **{key: _scalar_parameter(element, parameter, default=default, parameters=parameters)
                                                for key, parameter, default in (("amplitude", "Amplitude", 1.), ("period", "Period", 10.),
                                                                                ("width", "PulseWidth", 5.), ("delay", "PhaseDelay", 0.))})
    if block_type == "Step":
        return Step(
            block_id,
            name,
            step_time=_scalar_parameter(element, "Time", default=1.0, parameters=parameters),
            initial_value=_scalar_parameter(element, "Before", default=0.0, parameters=parameters),
            final_value=_scalar_parameter(element, "After", default=1.0, parameters=parameters),
        )
    if block_type == "Constant":
        return Constant(block_id, name, _vector(element, "Value", 1.0, parameters, name))
    if block_type == "Gain":
        multiplication = _parameter(element, "Multiplication") or "Element-wise(K.*u)"
        gain = np.real(np.asarray(_matrix_parameter(element, "Gain", 1.0, parameters)))
        if multiplication == "Element-wise(K.*u)":
            gain = _vector(element, "Gain", 1.0, parameters, name)
        try:
            return Gain(block_id, name, gain, multiplication)
        except ValueError as error:
            raise ModelLoaderError(f"{error} (block '{name}')") from error
    if block_type == "Demux":
        outputs = np.real(np.asarray(_matrix_parameter(element, "Outputs", 2.0, parameters))).ravel()
        return Demux(block_id, name, [int(v) for v in outputs] if outputs.size > 1 else int(outputs[0]))
    if block_type == "Selector":
        return _selector(element, block_id, name, parameters)
    if block_type == "Concatenate":
        if (_parameter(element, "Mode") or "Vector") not in {"Vector", "Multidimensional array"}:
            raise ModelLoaderError(f"Concatenate mode of '{name}' is not supported.")
        return Concatenate(block_id, name, int(_scalar_parameter(element, "NumInputs", 2.0, parameters)))
    if block_type == "Sum":
        signs_text = (_parameter(element, "Inputs") or "++").replace("|", "")
        if signs_text.strip().isdigit():
            signs_text = "+" * int(signs_text)
        if any(sign not in "+-" for sign in signs_text):
            raise ModelLoaderError(f"Unsupported Sum inputs '{signs_text}' on block '{name}'.")
        return Sum(block_id, name, [1.0 if sign == "+" else -1.0 for sign in signs_text])
    if block_type == "Integrator":
        initial_condition_source = _parameter(element, "InitialConditionSource") or "internal"
        if initial_condition_source not in {"internal", "external"}:
            raise ModelLoaderError(
                f"Unsupported InitialConditionSource '{initial_condition_source}' on Integrator '{name}'."
            )
        limited = (_parameter(element, "LimitOutput") or "off") == "on"
        if limited and (_parameter(element, "ShowSaturationPort") or "off") == "on":
            raise ModelLoaderError(f"The saturation port of Integrator '{name}' is not supported.")
        initial = _vector(element, "InitialCondition", 0.0, parameters, name)
        return Integrator(
            block_id,
            name,
            initial,
            external_initial_condition=initial_condition_source == "external",
            lower_limit=_scalar_parameter(element, "LowerSaturationLimit", default=-np.inf, parameters=parameters) if limited else -np.inf,
            upper_limit=_scalar_parameter(element, "UpperSaturationLimit", default=np.inf, parameters=parameters) if limited else np.inf,
        )
    if block_type == "SecondOrderIntegrator":
        for parameter, default in (("ICSourceX", "internal"), ("ICSourceDXDT", "internal"), ("LimitX", "off"),
                                   ("LimitDXDT", "off"), ("WrapX", "off"), ("ExternalReset", "none"),
                                   ("ShowSaturationPorts", "off")):
            value = _parameter(element, parameter) or default
            if value != default:
                raise ModelLoaderError(f"Second-Order Integrator option {parameter}={value!r} is not implemented.")
        return SecondOrderIntegrator(block_id, name, _scalar_parameter(element, "ICX", 0.0, parameters),
                                     _scalar_parameter(element, "ICDXDT", 0.0, parameters),
                                     show_output=_parameter(element, "ShowOutput") or "both")
    if block_type == "ZeroPole":
        zeros = np.asarray(_matrix_parameter(element, "Zeros", [1.0], parameters)).ravel()
        poles = np.asarray(_matrix_parameter(element, "Poles", [0.0, -1.0], parameters)).ravel()
        gain = np.asarray(_matrix_parameter(element, "Gain", 1.0, parameters)).ravel()
        if gain.size != 1:
            raise ModelLoaderError(f"Zero-Pole '{name}' with several outputs is not supported.")
        numerator = np.real_if_close(gain[0] * np.poly(zeros)) if zeros.size else np.real_if_close(gain)
        denominator = np.real_if_close(np.poly(poles))
        if np.iscomplexobj(numerator) or np.iscomplexobj(denominator):
            raise ModelLoaderError(f"Zero-Pole '{name}': complex zeros / poles must come in conjugate pairs.")
        try:
            return TransferFcn(block_id, name, np.atleast_1d(numerator).astype(float),
                               np.atleast_1d(denominator).astype(float), 0.0)
        except ValueError as error:
            raise ModelLoaderError(f"Unsupported Zero-Pole '{name}': {error}") from error
    if block_type == "StateSpace":
        matrices = [np.real(np.asarray(_matrix_parameter(element, key, default, parameters)))
                    for key, default in (("A", 1.0), ("B", 1.0), ("C", 1.0), ("D", 1.0))]
        initial = np.real(np.asarray(_matrix_parameter(element, "InitialCondition", 0.0, parameters))).ravel()
        try:
            return StateSpace(block_id, name, *matrices, initial_condition=initial)
        except ValueError as error:
            raise ModelLoaderError(f"Unsupported State-Space '{name}': {error}") from error
    if block_type == "TransferFcn":
        def real_vector(key, default):
            value = np.asarray(_matrix_parameter(element, key, default, parameters))
            if np.iscomplexobj(value):
                raise ModelLoaderError(f"Parameter '{key}' of transfer function '{name}' is complex.")
            if value.ndim == 2 and min(value.shape) > 1:
                raise ModelLoaderError(f"Transfer function '{name}' with several outputs (matrix {key}) is not supported.")
            return value.ravel().astype(float)

        numerator = real_vector("Numerator", [1.0])
        # Simulink omits parameters that retain their library defaults. The
        # Transfer Fcn block's default denominator is [1 1], not [1].
        denominator = real_vector("Denominator", [1.0, 1.0])
        initial_condition = real_vector("InitialCondition", [0.0])
        try:
            return TransferFcn(block_id, name, numerator, denominator, initial_condition)
        except ValueError as error:
            raise ModelLoaderError(f"Unsupported transfer function '{name}': {error}") from error
    if block_type == "Product":
        operations = (_parameter(element, "Inputs") or "**").replace("|", "")
        if operations.strip().isdigit():
            operations = "*" * int(operations)
        if (_parameter(element, "Multiplication") or "Element-wise(.*)") != "Element-wise(.*)":
            raise ModelLoaderError(f"Matrix multiplication in Product '{name}' is not supported.")
        return Product(block_id, name, operations)
    if block_type == "Saturate":
        if (_parameter(element, "OutputSaturationStatus") or "off") != "off":
            raise ModelLoaderError("Saturation status output is not implemented.")
        return Saturate(
            block_id,
            name,
            _scalar_parameter(element, "LowerLimit", default=-0.5, parameters=parameters),
            _scalar_parameter(element, "UpperLimit", default=0.5, parameters=parameters),
            zero_crossing=_parameter(element, "ZeroCross") != "off",
        )
    if block_type == "Mux":
        return Mux(block_id, name, _port_count(element, "in"))
    if block_type == "ManualSwitch":
        return ManualSwitch(
            block_id,
            name,
            int(_scalar_parameter(element, "CurrentSetting", default=1.0, parameters=parameters)),
        )
    if block_type in _DISCRETE_TYPES:
        try:
            return _discrete_block(block_type, element, block_id, name, parameters, start_time, step)
        except discrete_semantics.DiscreteSemanticsError as error:
            raise ModelLoaderError(str(error)) from error
    if block_type == "Fcn":
        expression = _fcn_with_values(_required_parameter(element, "Expr"), name, parameters)
        try:
            return Fcn(block_id, name, expression)
        except ValueError as error:
            raise ModelLoaderError(str(error)) from error
    raise ModelLoaderError(f"Unsupported block type '{block_type}' on block '{name}'.")


_DISCRETE_TYPES = {"Memory", "Derivative", "UnitDelay", "ZeroOrderHold", "DiscreteIntegrator",
                   "DiscreteTransferFcn", "DiscreteStateSpace", "RateLimiter", "Relay", "Backlash"}


def _discrete_block(block_type: str, element, block_id: str, name: str, parameters, start_time: float,
                    step: float | None = None):
    """Blocks with discrete memory (discrete_semantics). The defaults are Simulink's (R2025b) for
    parameters the .slx omits; options that are not checked against Simulink are refused."""
    def text(key, default):
        return " ".join((_parameter(element, key) or default).split())

    def require(key, default, allowed=None):
        value = text(key, default)
        if value != (allowed or default):
            raise ModelLoaderError(f"{block_type} option {key}={value!r} is not supported (block '{name}').")

    def initial(key="InitialCondition"):
        return _vector(element, key, 0.0, parameters, name)

    def sampled(default, build):
        """build(sample) -> the block; an inherited sample time (-1) is resolved once the model is
        wired (sample_times.resolve, which calls block.inherit with the rate Simulink compiles)"""
        sample = discrete_semantics.sample_time(_matrix_parameter(element, "SampleTime", default, parameters), name)
        if sample is not None:
            return build(sample)
        block = build((step or 1.0, 0.0))
        block.inherit = build
        return block

    if block_type == "Memory":
        require("InheritSampleTime", "off")
        return discrete_semantics.memory(block_id, name, initial())
    if block_type == "Derivative":
        return discrete_semantics.derivative(block_id, name, start_time)
    if block_type == "RateLimiter":
        require("SampleTimeMode", "inherited")
        if step is None:
            raise ModelLoaderError(f"Rate Limiter '{name}' in a variable-step model is not supported.")
        return discrete_semantics.rate_limiter(block_id, name, _scalar_parameter(element, "RisingSlewLimit", 1.0, parameters),
                                               _scalar_parameter(element, "FallingSlewLimit", -1.0, parameters),
                                               initial(), start_time, step)
    if block_type == "Relay":
        eps = float(np.finfo(float).eps)
        return discrete_semantics.relay(block_id, name, _scalar_parameter(element, "OnSwitchValue", eps, parameters),
                                        _scalar_parameter(element, "OffSwitchValue", eps, parameters),
                                        _scalar_parameter(element, "OnOutputValue", 1.0, parameters),
                                        _scalar_parameter(element, "OffOutputValue", 0.0, parameters))
    if block_type == "Backlash":
        return discrete_semantics.backlash(block_id, name, _scalar_parameter(element, "BacklashWidth", 1.0, parameters),
                                           initial("InitialOutput"))
    if block_type == "UnitDelay":
        value = initial()
        return sampled(-1.0, lambda s: discrete_semantics.unit_delay(block_id, name, value, s))
    if block_type == "ZeroOrderHold":
        return sampled(-1.0, lambda s: discrete_semantics.zero_order_hold(block_id, name, s))
    if block_type == "DiscreteIntegrator":
        require("IntegratorMethod", "Integration: Forward Euler")
        for key, default in (("ExternalReset", "none"), ("InitialConditionSource", "internal"), ("LimitOutput", "off"),
                             ("ShowStatePort", "off"), ("ShowSaturationPort", "off")):
            require(key, default)
        gain, value = _scalar_parameter(element, "gainval", 1.0, parameters), initial()
        return sampled(1.0, lambda s: discrete_semantics.discrete_integrator(block_id, name, gain, value, s))
    if block_type == "DiscreteTransferFcn":
        for key, default in (("NumeratorSource", "Dialog"), ("DenominatorSource", "Dialog"), ("ExternalReset", "None"),
                             ("InitialStatesSource", "Dialog")):
            require(key, default)
        if np.any(np.asarray(_matrix_parameter(element, "InitialStates", 0.0, parameters)) != 0):
            raise ModelLoaderError(f"Discrete Transfer Fcn '{name}' with non-zero initial states is not supported.")
        numerator = np.real(np.asarray(_matrix_parameter(element, "Numerator", [1.0], parameters))).ravel()
        denominator = np.real(np.asarray(_matrix_parameter(element, "Denominator", [1.0, 0.5], parameters))).ravel()
        # R2025b omits SampleTime -1 of a Discrete Transfer Fcn: its default is inherited
        return sampled(-1.0, lambda s: discrete_semantics.discrete_transfer_fcn(block_id, name, numerator, denominator, s))
    if block_type == "DiscreteStateSpace":
        matrices = [np.real(np.asarray(_matrix_parameter(element, key, 1.0, parameters))) for key in "ABCD"]
        initial_state = np.real(np.asarray(_matrix_parameter(element, "InitialCondition", 0.0, parameters))).ravel()
        return sampled(1.0, lambda s: discrete_semantics.discrete_state_space(block_id, name, "Discrete State-Space",
                                                                             *matrices, initial_state, s))
    raise ModelLoaderError(f"Unsupported block type '{block_type}' on block '{name}'.")


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
        raise ModelLoaderError(f"Invalid {direction} endpoint '{value}'.") from error
    if port < 0:
        raise ModelLoaderError(f"Invalid {direction} endpoint '{value}'.")
    return Endpoint(block_id, port)


def _parameter(element: ElementTree.Element, name: str) -> str | None:
    for parameter in element.findall("P"):
        if parameter.get("Name") == name:
            return parameter.text or ""
    return None


def _required_parameter(element: ElementTree.Element, name: str) -> str:
    value = _parameter(element, name)
    if value is None:
        raise ModelLoaderError(f"Missing required parameter '{name}'.")
    return value


def _required_nested_parameter(element: ElementTree.Element, name: str) -> str:
    for parameter in element.iter("P"):
        if parameter.get("Name") == name:
            return parameter.text or ""
    raise ModelLoaderError(f"Missing required parameter '{name}'.")


def _solver_options(config_root, parameters):
    defaults = {"MaxStep": "auto", "InitialStep": "auto", "MinStep": "auto", "AbsTol": "auto",
                "AutoScaleAbsTol": "on", "ZeroCrossControl": "UseLocalSettings", "ZeroCrossAlgorithm": "Nonadaptive"}
    raw = {name: next((p.text or "" for p in config_root.iter("P") if p.get("Name") == name), default)
           for name, default in defaults.items()}
    result = {**raw, "raw": raw}
    for name in ("MaxStep", "InitialStep", "MinStep"):
        if raw[name] != "auto":
            result[name] = _numeric_expression(raw[name], name, parameters)
    return result


def _tolerance_parameter(element: ElementTree.Element, name: str, default: float) -> float:
    text = _required_nested_parameter(element, name)
    if text == "auto":
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise ModelLoaderError(f"Solver tolerance '{name}' must be numeric or 'auto'.") from error
    if value <= 0.0 or not np.isfinite(value):
        raise ModelLoaderError(f"Solver tolerance '{name}' must be positive and finite.")
    return value


def _scalar_parameter(element: ElementTree.Element, name: str, default: float, parameters: dict[str, float]) -> float:
    text = _block_parameter(element, name)
    if text is None or not text.strip():
        return default
    if not np.isfinite(default) and text.strip().lstrip("+-").lower() == "inf":
        return float(text.strip().lower().replace("inf", "1")) * np.inf     # a limit left open
    return _numeric_expression(text, name, parameters)


def _block_parameter(element: ElementTree.Element, name: str) -> str | None:
    """A parameter of the block: its own P, or for a library block a mask parameter (InstanceData)."""
    value = _parameter(element, name)
    if value is not None:
        return value
    instance = element.find("InstanceData")
    return _parameter(instance, name) if instance is not None else None


_MATLAB_CONSTANTS = {"pi": np.pi, "inf": np.inf, "Inf": np.inf, "eps": np.finfo(float).eps,
                     "true": 1.0, "false": 0.0}
_MATLAB_PARAMETER_FUNCTIONS = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10, "abs": np.abs,
                               "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan, "deg2rad": np.deg2rad}


def _matrix_parameter(element: ElementTree.Element, name: str, default, parameters: dict[str, float]) -> np.ndarray:
    """A scalar / vector / matrix parameter in MATLAB syntax ([1 2; 3 4], complex 1+2i, pi, inf,
    sqrt(2), workspace parameters given explicitly) -> 2-D array (complex only if needed)."""
    text = _block_parameter(element, name)
    if text is None or not text.strip():
        return np.atleast_2d(np.asarray(default, dtype=complex if np.iscomplexobj(default) else float))
    return _matrix_value(text, name, parameters)


def _note_used(text: str, parameters) -> None:
    """record the workspace variables a parameter uses (see workspace.UsedParameters)"""
    if isinstance(parameters, workspace.UsedParameters):
        parameters.note(text)


def _matrix_value(text: str, name: str, parameters: dict[str, float]) -> np.ndarray:
    _note_used(text, parameters)
    try:
        return _matrix_literal(text, name, parameters)
    except ModelLoaderError as error:
        value = _workspace_value(text, name, parameters, error)
        return value.real.copy() if np.iscomplexobj(value) and np.all(value.imag == 0) else value


def _workspace_value(text: str, name: str, parameters: dict, error: ModelLoaderError) -> np.ndarray:
    """A parameter the plain number syntax does not cover: MATLAB expressions with workspace
    variables (vectors / matrices too) and functions (eye, zeros, inv ...), evaluated by the
    MATLAB interpreter of ``workspace`` (no code is executed)."""
    try:
        return workspace.evaluate(text, parameters, getattr(parameters, "missing", None))
    except workspace.UnknownName as unknown:
        raise ModelLoaderError(
            f"Parameter '{name}' references workspace symbol '{unknown.name}'. "
            "Provide it explicitly with --parameter NAME=VALUE."
        ) from unknown
    except (workspace.MatlabError, ArithmeticError, TypeError, ValueError, IndexError, np.linalg.LinAlgError) as failure:
        raise ModelLoaderError(f"{error} ({failure})") from failure


def _matrix_literal(text: str, name: str, parameters: dict[str, float]) -> np.ndarray:
    body = text.strip()
    transpose = body.endswith("'")            # [1 2 3]' (or .'): a column
    if transpose:
        body = body[:-2] if body.endswith(".'") else body[:-1]
        body = body.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    rows = []
    for row in re.split(r"[;\n]", body):
        if row.strip():
            rows.append([_complex_expression(token, name, parameters) for token in _matrix_elements(row)])
    if not rows:
        return np.zeros((0, 0))
    if len({len(row) for row in rows}) != 1:
        raise ModelLoaderError(f"Parameter '{name}': the rows of the matrix have different lengths.")
    value = np.array(rows, dtype=complex)
    if transpose:
        value = value.T
    return value.real.copy() if np.all(value.imag == 0) else value


def _matrix_elements(row: str) -> list[str]:
    """MATLAB's element separation: commas or blanks; [1 -2] is two elements, [1 - 2] one."""
    tokens = [t for t in re.split(r"[,\s]+", row.strip()) if t]
    elements: list[str] = []
    for token in tokens:
        if elements and (token in {"+", "-", "*", "/", "^"} or elements[-1][-1] in "+-*/^(" or token[0] in "*/^)"
                         or elements[-1].count("(") > elements[-1].count(")")):
            elements[-1] += token
        else:
            elements.append(token)
    return elements


def _complex_expression(text: str, parameter_name: str, parameters: dict[str, float]) -> complex:
    source = re.sub(r"(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[ij]\b", r"\1j", text.strip())
    source = re.sub(r"(?<![\w.])[ij]\b", "1j", source).replace(".^", "**").replace("^", "**").replace(".*", "*").replace("./", "/")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ModelLoaderError(f"Parameter '{parameter_name}' is not a numeric expression: {text!r}.") from error
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed) or (isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in _MATLAB_PARAMETER_FUNCTIONS or node.keywords)):
            raise ModelLoaderError(f"Parameter '{parameter_name}' contains an unsupported expression: {text!r}.")
    try:
        with np.errstate(all="ignore"):
            return complex(eval(compile(tree, "<simulink-parameter>", "eval"), {"__builtins__": {}},
                                {**_MATLAB_CONSTANTS, **_MATLAB_PARAMETER_FUNCTIONS, **parameters}))
    except NameError as error:
        raise ModelLoaderError(
            f"Parameter '{parameter_name}' references workspace symbol '{error.name}'. "
            "Provide it explicitly with --parameter NAME=VALUE."
        ) from error
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ModelLoaderError(f"Parameter '{parameter_name}' is not a valid number: {text!r}.") from error


def _vector(element: ElementTree.Element, key: str, default, parameters: dict[str, float], name: str):
    """a scalar or a vector parameter (a float, or a 1-D array); matrices are not signals here"""
    value = np.asarray(_matrix_parameter(element, key, default, parameters))
    if np.iscomplexobj(value):
        raise ModelLoaderError(f"Parameter '{key}' of '{name}' is complex.")
    if value.ndim == 2 and min(value.shape) > 1:
        raise ModelLoaderError(f"Parameter '{key}' of '{name}' is a matrix; matrix signals are not supported (vectors are).")
    value = value.ravel()
    return float(value[0]) if value.size == 1 else value


def _inport_width(element: ElementTree.Element, name: str, parameters: dict[str, float]) -> int:
    """the width of a root Inport: PortDimensions, -1 (inherited) taken as a scalar"""
    dims = np.real(np.asarray(_matrix_parameter(element, "PortDimensions", -1.0, parameters))).ravel()
    if dims.size == 1:
        return 1 if dims[0] <= 0 else int(dims[0])
    if dims.size == 2 and min(dims) == 1:
        return int(max(dims))
    raise ModelLoaderError(f"Inport '{name}' has the dimensions {dims.astype(int).tolist()}; matrix signals are not supported.")


def _selector(element: ElementTree.Element, block_id: str, name: str, parameters: dict[str, float]) -> Selector:
    """Selector of a vector: an index vector, or a starting index and a count, in the dialog"""
    if int(_scalar_parameter(element, "NumberOfDimensions", 1.0, parameters)) != 1:
        raise ModelLoaderError(f"Selector '{name}' with several dimensions is not supported.")
    option = _parameter(element, "IndexOptions") or "Index vector (dialog)"
    base = 0 if (_parameter(element, "IndexMode") or "One-based") == "Zero-based" else 1
    indices = np.real(np.asarray(_matrix_parameter(element, "Indices", 1.0, parameters))).ravel().astype(int)
    if option == "Index vector (dialog)":
        selected = [int(i) - base for i in indices]
    elif option == "Starting index (dialog)":
        count = int(np.real(np.asarray(_matrix_parameter(element, "OutputSizes", 1.0, parameters))).ravel()[0])
        selected = list(range(int(indices[0]) - base, int(indices[0]) - base + count))
    elif option == "Select all":
        width = int(_scalar_parameter(element, "InputPortWidth", -1.0, parameters))
        if width < 1:
            raise ModelLoaderError(f"Selector '{name}': 'Select all' needs the input port width.")
        selected = list(range(width))
    else:
        raise ModelLoaderError(f"Selector '{name}': index option {option!r} (an index port) is not supported.")
    try:
        return Selector(block_id, name, selected)
    except ValueError as error:
        raise ModelLoaderError(f"Selector '{name}': {error}") from error


def _pid_controller(block_id: str, name: str, p: block_semantics.Params) -> StateSpace:
    """PID Controller (continuous, parallel or ideal form) as a state space. Simulink's parallel
    form: u = P e + I_state + N (D e - F_state), dI_state/dt = I e, dF_state/dt = N (D e - F_state)."""
    for parameter, default in (("TimeDomain", "Continuous-time"), ("ControllerParametersSource", "internal"),
                               ("InitialConditionSource", "internal"), ("ExternalReset", "none"),
                               ("LimitOutput", "off"), ("AntiWindupMode", "none"), ("TrackingMode", "off"),
                               ("UseExternalDerivativeSource", "off"), ("LimitIntegrator", "off")):
        if p.get(parameter, default) != default:
            raise ModelLoaderError(f"PID Controller option {parameter}={p.get(parameter, default)!r} is not "
                                   f"supported (block '{name}').")
    controller, form = p.get("Controller", "PID"), p.get("Form", "Parallel")
    if controller not in {"P", "I", "PI", "PD", "PID"} or form not in {"Parallel", "Ideal"}:
        raise ModelLoaderError(f"PID Controller {controller} ({form}) is not supported (block '{name}').")
    kp = p.number("P", 1.0) if "P" in controller else 0.0
    ki = p.number("I", 1.0) if "I" in controller else 0.0
    kd = p.number("D", 0.0) if "D" in controller else 0.0
    n = p.number("N", 100.0)
    if form == "Ideal":
        ki, kd = kp * ki, kp * kd
    if kd != 0 and p.get("UseFilter", "on") != "on":
        raise ModelLoaderError(f"PID Controller '{name}' without derivative filter is not supported.")
    a, b, c, x0 = [], [], [], []
    d = kp
    if ki != 0 or "I" in controller:
        a.append([0.0]); b.append([ki]); c.append(1.0); x0.append(p.number("InitialConditionForIntegrator", 0.0))
    if kd != 0:
        a.append([-n]); b.append([n * kd]); c.append(-n); x0.append(p.number("InitialConditionForFilter", 0.0))
        d += n * kd
    size = len(a)
    matrix_a = np.zeros((size, size))
    for i in range(size):
        matrix_a[i, i] = a[i][0]
    return StateSpace(block_id, name, matrix_a, np.array(b).reshape(size, 1), np.array(c).reshape(1, size),
                      np.array([[d]]), initial_condition=np.array(x0) if size else 0.0)


def _numeric_expression(text: str, parameter_name: str, parameters: dict[str, float]) -> float:
    """Evaluate Simulink's scalar arithmetic parameter syntax safely."""
    _note_used(text, parameters)
    try:
        return _numeric_literal(text, parameter_name, parameters)
    except ModelLoaderError as error:
        value = _workspace_value(text, parameter_name, parameters, error)
        if value.size != 1 or (np.iscomplexobj(value) and value.ravel()[0].imag != 0):
            raise ModelLoaderError(f"Parameter '{parameter_name}' must be a real scalar; {text!r} is not.") from error
        number = float(np.real(value.ravel()[0]))
        if not np.isfinite(number):
            raise ModelLoaderError(f"Parameter '{parameter_name}' is not finite.") from error
        return number


def _numeric_literal(text: str, parameter_name: str, parameters: dict[str, float]) -> float:
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ModelLoaderError(f"Parameter '{parameter_name}' must be a scalar numeric expression.") from error
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    )
    if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        raise ModelLoaderError(f"Parameter '{parameter_name}' contains an unsupported numeric expression.")
    try:
        value = float(eval(compile(tree, "<simulink-parameter>", "eval"), {"__builtins__": {}}, parameters))
    except NameError as error:
        raise ModelLoaderError(
            f"Parameter '{parameter_name}' references workspace symbol '{error.name}'. "
            "Provide it explicitly with --parameter NAME=VALUE."
        ) from error
    except (ArithmeticError, TypeError, ValueError) as error:
        raise ModelLoaderError(f"Parameter '{parameter_name}' is not a valid finite number.") from error
    if not np.isfinite(value):
        raise ModelLoaderError(f"Parameter '{parameter_name}' is not finite.")
    return value


_FCN_NAMES = {"u", "pi"}                    # the names a Fcn expression knows itself
_FCN_CONSTANTS = {"e", "inf", "Inf", "eps", "true", "false"}


def _fcn_with_values(expression: str, name: str, parameters: dict) -> str:
    """A Fcn expression with its workspace variables (scalars) written as numbers. A name that is
    neither u, a function (followed by '(') nor a known variable is reported like a block parameter."""
    _note_used(expression, parameters)
    def replace(match: re.Match) -> str:
        word = match.group(0)
        if word in _FCN_NAMES or re.match(r"\s*\(", expression[match.end():]):
            return word
        if word not in parameters and word in _FCN_CONSTANTS:
            return word
        if word not in parameters and getattr(parameters, "missing", None) is not None:
            parameters.missing.add(word)
            return "(1.0)"
        if word not in parameters:
            raise ModelLoaderError(f"Parameter 'Expr' of Fcn '{name}' references workspace symbol '{word}'. "
                                   "Provide it explicitly with --parameter NAME=VALUE.")
        value = np.asarray(parameters[word])
        if value.size != 1 or np.iscomplexobj(value):
            raise ModelLoaderError(f"Fcn '{name}' uses the workspace variable '{word}', which is not a real scalar.")
        return f"({float(value.ravel()[0])!r})"
    return re.sub(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", replace, expression)


def _port_count(element: ElementTree.Element, direction: str) -> int:
    port_counts = element.find("PortCounts")
    if port_counts is not None:
        return int(port_counts.get(direction, "0"))
    if _block_type(element) == "Mux":
        inputs = (_parameter(element, "Inputs") or "1").strip()
        # a number of inputs, or the widths of the inputs: [2 1]
        return len(inputs.strip("[]").replace(",", " ").split()) if inputs.startswith("[") else int(inputs)
    raise ModelLoaderError(f"Missing {direction} port count on block '{_block_name(element)}'.")

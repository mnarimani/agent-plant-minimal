"""A model without root Inports / Outports: its inputs and outputs chosen on its block diagram
(backend_core/simulink/direct/diagram.py, topology.SignalChoice).

tst5 without its Outports, and its Inports turned into Constant blocks, is what a user sends who
did not put Inport / Outport blocks in the model. Choosing on its diagram the signals the original
ports had must give exactly the conversion of the original model.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_core.simulink import intake  # noqa: E402
from backend_core.simulink.direct import diagram, slx_direct  # noqa: E402
from backend_core.simulink.direct.slx2py.model_converter.topology import SignalChoice  # noqa: E402

EXAMPLES = ROOT / "tests" / "fixtures"
# tst5: In1 (18), In2 (19), In3 (33) -> ... -> Out1 = Saturation (35), Out2 = Integrator (3), Out3 = Gain (27)
ORIGINAL_INPUTS = ("18#1", "19#1", "33#1")
ORIGINAL_OUTPUTS = ("35#1", "3#1", "27#1")


def _slx(name: str) -> bytes:
    return (EXAMPLES / ("%s.slx" % name)).read_bytes()


def without_ports(data: bytes) -> bytes:
    """the .slx without its root Outports (and their lines), its root Inports turned into Constants"""
    source, buffer = zipfile.ZipFile(io.BytesIO(data)), io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for item in source.infolist():
            content = source.read(item.filename)
            if item.filename.endswith("system_root.xml"):
                root = ElementTree.fromstring(content)
                outports = {e.get("SID") for e in root.findall("Block") if e.get("BlockType") == "Outport"}
                for element in root.findall("Block"):
                    if element.get("BlockType") == "Outport":
                        root.remove(element)
                    elif element.get("BlockType") == "Inport":
                        element.set("BlockType", "Constant")
                        for p in element.findall("P"):
                            if p.get("Name") == "Port":
                                element.remove(p)
                        ElementTree.SubElement(element, "P", {"Name": "Value"}).text = "0"

                def prune(parent):
                    for child in list(parent):
                        if child.tag in ("Line", "Branch"):
                            dst = next((p.text for p in child.findall("P") if p.get("Name") == "Dst"), None)
                            if dst and dst.split("#")[0] in outports:
                                parent.remove(child)
                            else:
                                prune(child)
                prune(root)
                content = ElementTree.tostring(root, encoding="utf-8")
            out.writestr(item, content)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    return slx_direct.convert_slx("tst5.slx", _slx("tst5"), tmp_path_factory.mktemp("runs"), log=lambda m: None)


@pytest.fixture(scope="module")
def bare():
    return without_ports(_slx("tst5"))


def test_without_ports_the_whole_model_converts_and_says_it_needs_inputs_and_outputs(bare, original, tmp_path):
    """the whole model is converted - every state - even without an Outport; the app then waits for the
    choice on the diagram (DirectResult.no_input)"""
    r = slx_direct.convert_slx("tst5.slx", bare, tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert r.draft["metadata"]["states"] == original.draft["metadata"]["states"]
    assert r.draft["metadata"]["inputs"] == [] and r.draft["metadata"]["outputs"] == []
    assert r.no_input == slx_direct.NO_INPORT + " " + slx_direct.NO_OUTPUT
    assert "def output" not in r.draft["python_code"]
    assert slx_direct.chosen_on_diagram(slx_direct.NO_OUTPORT)
    assert not slx_direct.chosen_on_diagram(None) and not slx_direct.chosen_on_diagram("something else")


def test_choosing_the_signals_of_the_original_ports_gives_the_original_conversion(original, bare, tmp_path):
    choice = SignalChoice(ORIGINAL_INPUTS, ORIGINAL_OUTPUTS,
                          tuple(zip(ORIGINAL_INPUTS + ORIGINAL_OUTPUTS, ("In1", "In2", "In3", "Out1", "Out2", "Out3"))))
    r = slx_direct.convert_slx("tst5.slx", bare, tmp_path, log=lambda m: None, signals=choice)
    assert r.ok, r.reason
    assert r.draft["python_code"] == original.draft["python_code"]
    assert r.draft["metadata"]["inputs"] == original.draft["metadata"]["inputs"]
    assert r.draft["metadata"]["outputs"] == original.draft["metadata"]["outputs"]
    assert all(s.ok for s in r.steps)


def test_an_input_cuts_the_signal_and_any_signal_even_in_a_subsystem_can_be_an_output(bare, tmp_path):
    # Sum (6) as the input: In1 / In3 (now Constants) are cut away; the Integrator inside the Subsystem as an output
    r = slx_direct.convert_slx("tst5.slx", bare, tmp_path, log=lambda m: None,
                               signals=SignalChoice(("6#1",), ("10::16#1", "35#1")))
    assert r.ok, r.reason
    md = r.draft["metadata"]
    assert md["inputs"] == ["u1: root Inport Sum"]
    assert [o.split(" = ")[0] for o in md["outputs"]] == ["y1 (Integrator)", "y2 (Saturation)"]
    assert len(md["states"]) == 5


@pytest.mark.parametrize("choice, words", [
    (SignalChoice((), ("nothing#1",)), "not a signal of this model"),
    (SignalChoice((), ("3#2",)), "has no output 2"),
])
def test_a_choice_that_cannot_work_is_said_plainly(bare, tmp_path, choice, words):
    r = slx_direct.convert_slx("tst5.slx", bare, tmp_path, log=lambda m: None, signals=choice)
    assert not r.ok and words in r.reason


def test_the_diagram_has_every_system_block_line_and_signal(bare):
    d = diagram.read_diagram(_slx("tst5"), "tst5.slx")
    assert d.model == "tst5"
    assert {p: s.title for p, s in d.systems.items()} == {
        (): "tst5", ("10",): "tst5/Subsystem", ("10", "13"): "tst5/Subsystem/Subsystem"}
    root = d.systems[()]
    assert {b.name for b in root.blocks} >= {"Integrator", "Transfer Fcn", "Sum", "Saturation", "Out1", "In1"}
    assert next(b for b in root.blocks if b.sid == "10").child == ("10",)
    assert all(line.key for line in root.lines)              # every line of tst5 carries a computed signal
    assert sorted(d.model_inputs) == sorted(ORIGINAL_INPUTS)
    assert sorted(d.model_outputs) == sorted(ORIGINAL_OUTPUTS)
    assert d.label("27#1") == "Gain (Gain)" and d.label("10::16#1") == "Subsystem/Integrator (Integrator)"
    # a line starts at the output port of its block and its branches start where it ends
    trunk = next(line for line in root.lines if line.key == "27#1").polylines
    assert trunk[0][0] == (885.0, 220.0) and trunk[1][0] == trunk[0][-1] == trunk[2][0]
    # the diagram of the model without ports has the same signals, and nothing chosen by the model
    bare_diagram = diagram.read_diagram(bare, "tst5.slx")
    assert set(bare_diagram.signals) == set(d.signals)
    assert bare_diagram.model_inputs == [] and bare_diagram.model_outputs == []


def test_a_choice_on_the_diagram_names_the_new_ports_after_their_blocks(bare):
    d = diagram.read_diagram(bare, "tst5.slx")
    c = diagram.choice(d, ["6#1"], ["35#1", "10::16#1"])
    assert c == SignalChoice(("6#1",), ("35#1", "10::16#1"),
                             (("6#1", "Sum"), ("35#1", "Saturation"), ("10::16#1", "Integrator")))
    assert intake.chosen_names(c) == {"inputs": "Sum", "outputs": "Saturation, Integrator"}


def test_the_figure_marks_each_signal_once_and_a_click_gives_its_key(bare):
    pytest.importorskip("plotly")
    d = diagram.read_diagram(bare, "tst5.slx")
    keys = diagram.marker_keys(d, ())
    fig = diagram.figure(d, (), ["6#1"], ["35#1"])
    markers = fig.data[-1]
    assert list(markers.customdata) == keys and len(keys) == len(set(keys))
    colors = dict(zip(keys, markers.marker.color))
    assert colors["6#1"] == diagram.INPUT_COLOR and colors["35#1"] == diagram.OUTPUT_COLOR
    assert colors["27#1"] == diagram.PLAIN_COLOR
    assert len(fig.layout.shapes) == len(d.systems[()].blocks)
    event = {"selection": {"points": [{"curve_number": len(fig.data) - 1, "point_index": 2, "customdata": keys[2]}]}}
    assert diagram.selected_keys(event, keys) == [keys[2]]
    no_data = {"selection": {"points": [{"curve_number": len(fig.data) - 1, "point_index": 1}]}}
    assert diagram.selected_keys(no_data, keys, curve=len(fig.data) - 1) == [keys[1]]
    assert diagram.selected_keys({"selection": {"points": []}}, keys) == []


def test_the_agent_is_told_about_the_diagram_and_the_choice():
    info = intake.ModelInfo(name="tst5", file="tst5.slx", fmt="slx", model_fix=slx_direct.NO_OUTPORT, diagram=True)
    note = intake.just_uploaded_note(info)
    assert "Convert with these signals" in note and "No root Outport" in note
    context = intake.agent_context(info, None)
    assert "Convert with these signals" in context
    info.signals = SignalChoice(("6#1",), ("6#1",), (("6#1", "Sum"),))
    info.direct_error = "The model has no state."
    assert "inputs: Sum; outputs: Sum" in intake.agent_context(info, None)
    failed = intake.signals_failed_note(info)
    assert "inputs: Sum" in failed and "The model has no state." in failed


def _segments(d):
    for system in d.systems.values():
        for line in system.lines:
            for polyline in line.polylines:
                yield from zip(polyline, polyline[1:])


def test_lines_are_drawn_at_right_angles_as_in_simulink(bare):
    """a line that ends a little off the port computed from the block size, or a branch without points
    of its own, still enters its port straight (mimo_test8: lines into a MATLAB Function block)"""
    for data in (_slx("tst5"), bare):
        d = diagram.read_diagram(data, "tst5.slx")
        assert all(abs(a[0] - b[0]) < 0.5 or abs(a[1] - b[1]) < 0.5 for a, b in _segments(d))
    assert diagram._into_port((150.0, 175.0), (170.0, 268.3), True) == [(150.0, 268.3), (170.0, 268.3)]
    assert diagram._into_port((150.0, 175.0), (170.0, 173.3), True) == [(170.0, 175.0)]


def test_root_inports_that_reach_no_output_are_still_inputs_of_the_model(tmp_path):
    """the whole model, as in Simulink: a root Inport is an input u, wherever its line goes (here: nowhere)"""
    def cut(data):
        source, buffer = zipfile.ZipFile(io.BytesIO(data)), io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as out:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename.endswith("system_root.xml"):
                    root = ElementTree.fromstring(content)
                    for line in root.findall("Line"):
                        if any(p.get("Name") == "Src" and p.text in ("18#out:1", "19#out:1", "33#out:1")
                               for p in line.findall("P")):
                            root.remove(line)
                    content = ElementTree.tostring(root, encoding="utf-8")
                out.writestr(item, content)
        return buffer.getvalue()
    data = cut(_slx("tst5"))
    d = diagram.read_diagram(data, "tst5.slx")
    assert sorted(d.model_inputs) == sorted(ORIGINAL_INPUTS) and sorted(d.model_outputs) == sorted(ORIGINAL_OUTPUTS)
    r = slx_direct.convert_slx("tst5.slx", data, tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert r.no_input is None and len(r.draft["metadata"]["inputs"]) == 3


def _without_line(data: bytes, source: str) -> bytes:
    """the .slx without the root line that leaves ``source`` ('32#out:1'): what it fed is unconnected"""
    archive, buffer = zipfile.ZipFile(io.BytesIO(data)), io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for item in archive.infolist():
            content = archive.read(item.filename)
            if item.filename.endswith("system_root.xml"):
                root = ElementTree.fromstring(content)
                for line in root.findall("Line"):
                    if any(p.get("Name") == "Src" and p.text == source for p in line.findall("P")):
                        root.remove(line)
                content = ElementTree.tostring(root, encoding="utf-8")
            out.writestr(item, content)
    return buffer.getvalue()


def test_an_input_port_nothing_is_connected_to_can_be_chosen_as_an_input(tmp_path):
    """the user's mimo_test8: nothing enters the Transfer Fcn. Its free input chosen as u gives what
    cutting the signal that used to enter it gives"""
    data = _without_line(_slx("tst5"), "32#out:1")         # MATLAB Function -> Transfer Fcn removed
    d = diagram.read_diagram(data, "tst5.slx")
    assert diagram.is_input_port(d, "2#in1") and not diagram.is_input_port(d, "2#1")
    assert d.label("2#in1") == "Transfer Fcn (TransferFcn), input 1 - not connected"
    stub = next(line for line in d.systems[()].lines if line.key == "2#in1")
    assert stub.polylines[0][-1] == (625.0, 220.0)          # it ends at the input port of the block
    assert "2#in1" in diagram.marker_keys(d, ())
    names = (("2#in1", "u_tf"), ("32#1", "u_tf"))
    free = slx_direct.convert_slx("tst5.slx", data, tmp_path / "a", log=lambda m: None,
                                  signals=SignalChoice(("2#in1",), (), names))
    cut = slx_direct.convert_slx("tst5.slx", _slx("tst5"), tmp_path / "b", log=lambda m: None,
                                 signals=SignalChoice(("32#1",), (), names))
    assert free.ok and cut.ok, (free.reason, cut.reason)
    assert free.draft["python_code"] == cut.draft["python_code"]
    assert free.draft["metadata"]["inputs"] == cut.draft["metadata"]["inputs"] == [
        "u1: root Inport In1", "u2: root Inport In2", "u3: root Inport In3", "u4: root Inport u_tf"]


def test_a_choice_of_an_input_port_is_checked():
    data = _without_line(_slx("tst5"), "32#out:1")
    with pytest.raises(ValueError, match="only a signal can be an output"):
        _topology(data, SignalChoice((), ("2#in1",)))
    with pytest.raises(ValueError, match="is connected already"):
        _topology(data, SignalChoice(("27#in1",), ()))


def _topology(data, choice):
    import tempfile
    from backend_core.simulink.direct.slx2py.model_converter import pipeline, topology
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "m.slx").write_bytes(data)
        return topology.build_topology(pipeline.extract_slx(Path(tmp) / "m.slx", Path(tmp) / "m"), choice)


def test_an_unconnected_subsystem_input_reads_0_and_can_be_chosen(tmp_path):
    """as in Simulink, an unconnected port of a subsystem reads 0 (it used to stop the conversion)"""
    data = _without_line(_slx("tst5"), "36#out:1")         # From -> Subsystem input 1 removed
    r = slx_direct.convert_slx("tst5.slx", data, tmp_path / "a", log=lambda m: None)
    assert r.ok, r.reason
    d = diagram.read_diagram(data, "tst5.slx")
    assert diagram.is_input_port(d, "10#in1") and d.label("10#in1").startswith("Subsystem (SubSystem), input 1")
    r = slx_direct.convert_slx("tst5.slx", data, tmp_path / "b", log=lambda m: None,
                               signals=diagram.choice(d, ["10#in1"], []))
    assert r.ok, r.reason
    assert r.draft["metadata"]["inputs"][-1] == "u4: root Inport Subsystem"
    assert all(s.ok for s in r.steps)


def test_a_model_without_inputs_waits_for_the_choice_before_it_is_converted():
    info = intake.ModelInfo(name="m8", file="m8.slx", fmt="slx", diagram=True, no_inputs=True,
                            direct_error=slx_direct.NO_INPORT)
    note = intake.just_uploaded_note(info)
    assert "has not converted it yet" in note and "Convert as it is" in note and "diamond" in note
    assert "has NOT converted it yet" in intake.agent_context(info, None)
    assert "has not converted it yet" in intake.values_received_note(info, ["the chat"])


def test_a_signal_the_model_already_shows_is_not_a_second_output(original, tmp_path):
    """the user's mimo_test8: Selector was chosen as an output, but Out1 already shows it"""
    r = slx_direct.convert_slx("tst5.slx", _slx("tst5"), tmp_path / "a", log=lambda m: None,
                               signals=SignalChoice((), ("35#1", "10::16#1", "27#1")))   # Out1, new, Out3
    assert r.ok, r.reason
    outputs = r.draft["metadata"]["outputs"]
    assert outputs[:3] == original.draft["metadata"]["outputs"] and len(outputs) == 4
    assert outputs[3].startswith("y4 (Integrator)")
    dynamics = lambda code: code.split("def output")[0]                      # noqa: E731
    assert dynamics(r.draft["python_code"]) == dynamics(original.draft["python_code"])


def _with_blocks(data: bytes, blocks: str, lines: str) -> bytes:
    """the .slx with more blocks and lines at its top level (XML snippets)"""
    archive, buffer = zipfile.ZipFile(io.BytesIO(data)), io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for item in archive.infolist():
            content = archive.read(item.filename)
            if item.filename.endswith("system_root.xml"):
                text = content.decode("utf-8")
                at = text.rindex("</System>")
                content = (text[:at] + blocks + lines + text[at:]).encode("utf-8")
            out.writestr(item, content)
    return buffer.getvalue()


_DEAD_TF = ('<Block BlockType="TransferFcn" Name="Dead TF" SID="901"><P Name="Position">[900, 500, 960, 540]</P>'
            '<P Name="Numerator">[1]</P><P Name="Denominator">[1 3]</P></Block>')


def test_the_whole_model_is_converted_also_what_reaches_no_output(original, tmp_path):
    """a Transfer Fcn whose output goes nowhere still has its state in dx (Simulink simulates it too)"""
    line = '<Line><P Name="Src">3#out:1</P><P Name="Dst">901#in:1</P></Line>'
    r = slx_direct.convert_slx("tst5.slx", _with_blocks(_slx("tst5"), _DEAD_TF, line), tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    states = r.draft["metadata"]["states"]
    assert len(states) == len(original.draft["metadata"]["states"]) + 1
    assert "dx[5] = ((-3.0 * x6) + x1)" in r.draft["python_code"]
    assert r.draft["metadata"]["outputs"] == original.draft["metadata"]["outputs"]


def test_a_block_that_cannot_be_converted_and_reaches_no_output_is_left_out_with_a_note(original, tmp_path):
    blocks = ('<Block BlockType="NoSuchBlock" Name="Strange" SID="900"><PortCounts in="1" out="1"/>'
              '<P Name="Position">[800, 500, 840, 540]</P></Block>' + _DEAD_TF)
    lines = ('<Line><P Name="Src">3#out:1</P><P Name="Dst">900#in:1</P></Line>'
             '<Line><P Name="Src">900#out:1</P><P Name="Dst">901#in:1</P></Line>')
    r = slx_direct.convert_slx("tst5.slx", _with_blocks(_slx("tst5"), blocks, lines), tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert r.draft["python_code"] == original.draft["python_code"]       # 'Dead TF' after it goes too
    assert "Not converted - NoSuchBlock 'Strange'" in r.message and "'Dead TF'" in r.message
    # on a path to an output it still stops the conversion
    lines = ('<Line><P Name="Src">3#out:1</P><P Name="Dst">900#in:1</P></Line>'
             '<Line><P Name="Src">900#out:1</P><P Name="Dst">901#in:1</P></Line>')
    r = slx_direct.convert_slx("tst5.slx", _with_blocks(_slx("tst5"), blocks, lines), tmp_path / "b",
                               log=lambda m: None, signals=SignalChoice((), ("901#1",)))
    assert not r.ok and "NoSuchBlock" in r.reason


def test_the_code_has_output_y_checked_against_the_block_model(original):
    from backend_core import sandbox
    code = original.draft["python_code"]
    output = sandbox.compile_output(code)
    x, u = [1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.1, -1.0]
    assert list(output(0.0, x, u)) == pytest.approx([0.5, 1.0, 0.1 * (0.2 * 3 + 0.1 * 4 + 0.5 * 5)])
    assert any(s.label.startswith("Equations and outputs y vs the block model: PASS") for s in original.steps)


def test_the_simulation_shows_the_outputs_y(original):
    pytest.importorskip("plotly")
    from backend_core import sandbox
    from backend_core.simulink.c2py import convert
    sim = original.draft["metadata"]["simulink"]["simulation"]
    r = convert.simulate(original.draft["python_code"], tableau=original.tableau, fixed_step=sim["fixed_step"],
                         x0=sim["initial_state"], n_inputs=sim["n_inputs"], t_final=2.0,
                         output_names=["y1 (Out1)", "y2 (Out2)", "y3 (Out3)"])
    assert r.success and r.y.shape == (len(r.t), 3)
    assert list(r.y[:, 1]) == pytest.approx(list(r.x[:, 0]))             # y2 = x1
    fig = sandbox.build_figure(r)
    assert fig.layout.annotations[0].text == "Outputs y" and [d.name for d in fig.data[:3]] == ["y1 (Out1)", "y2 (Out2)", "y3 (Out3)"]

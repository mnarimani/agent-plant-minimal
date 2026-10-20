"""Scalar hierarchical Stateflow interpreter with explicit supported syntax.

No chart/data/event names have special meaning. Temporal operators, history,
messages, matrix data and arbitrary embedded code are rejected.
"""
from __future__ import annotations

import ast
import re
from xml.etree import ElementTree

from .stateflow import StateflowError, _xml_parameter, _evaluate_expression


class StateMachine:
    def __init__(self, source):
        self.source = source
        chart = ElementTree.fromstring(source)
        children = chart.find("Children")
        if children is None:
            raise StateflowError("Chart has no Children definition.")
        self.input_names = self._data(children, "INPUT_DATA")
        self.output_names = self._data(children, "OUTPUT_DATA")
        self.local_names = self._data(children, "LOCAL_DATA")
        events = children.findall("event")
        if any(_xml_parameter(e, "scope") != "INPUT_EVENT" for e in events):
            raise StateflowError("Only externally triggered input events are supported.")
        self.event_names = tuple(_xml_parameter(e, "name") for e in events)
        self.event_triggers = tuple(_xml_parameter(e, "trigger") or "EITHER_EDGE_EVENT" for e in events)
        if any(t not in {"EITHER_EDGE_EVENT", "RISING_EDGE_EVENT", "FALLING_EDGE_EVENT"} for t in self.event_triggers):
            raise StateflowError("Unsupported event trigger type.")
        if not self.input_names or not self.output_names:
            raise StateflowError("Chart must declare scalar input/output data.")
        self.states, self.functions, self.transitions, self.junctions = {}, {}, {}, set()
        self.child_states = {"root": []}
        self.decomposition = {"root": _xml_parameter(chart, "decomposition") or "CLUSTER_STATE"}
        self.state_outputs = {d.get("name"): _xml_parameter(d, "outputState") for d in children.findall("data")
                              if _xml_parameter(d, "outputState")}
        self.initial_values = {}
        for data in children.findall("data"):
            initial = data.findtext("props/initialValue") or "0"
            try:
                self.initial_values[data.get("name")] = float(initial)
            except ValueError as error:
                raise StateflowError("Only numeric scalar chart initial values are supported.") from error
        self.enter_at_initialization = _xml_parameter(chart, "executeAtInitialization") == "1"
        self._collect(children, "root")
        self._expressions = {}
        for values in self.child_states.values():
            values.sort(key=lambda sid: self.states[sid]["order"])
        for values in self.transitions.values():
            values.sort(key=lambda t: t["order"])
        self._validate()
        self.reset()

    @staticmethod
    def _data(children, scope):
        result = []
        for data in children.findall("data"):
            if _xml_parameter(data, "scope") != scope:
                continue
            name = data.get("name", "")
            size = data.findtext("props/array/P[@Name='size']")
            if not name.isidentifier() or size not in {None, "", "1", "[1 1]"}:
                raise StateflowError("Only named scalar chart data is supported.")
            result.append(name)
        return tuple(result)

    def _collect(self, children, parent):
        for state in children.findall("state"):
            sid = state.get("SSID")
            label = _xml_parameter(state, "labelString") or ""
            kind = _xml_parameter(state, "type") or "OR_STATE"
            nested = state.find("Children")
            if kind == "FUNC_STATE":
                signature = re.fullmatch(r"\s*(?:(\w+)\s*=\s*)?(\w+)\(([^)]*)\)\s*", label)
                if not signature or nested is None:
                    raise StateflowError(f"Unsupported graphical function {label!r}.")
                output, name, arguments = signature.groups()
                self.functions[name] = (sid, tuple(v.strip() for v in arguments.split(",") if v.strip()), output)
                self._collect(nested, sid)
                continue
            if kind not in {"OR_STATE", "AND_STATE"}:
                raise StateflowError(f"Unsupported state type {kind!r}.")
            name = label.splitlines()[0].strip()
            if not name.isidentifier():
                raise StateflowError(f"Unsupported state label {name!r}.")
            self.states[sid] = {"name": name, "parent": parent,
                                "order": int(_xml_parameter(state, "executionOrder") or "1"),
                                "kind": kind, "actions": self._state_actions(label)}
            self.child_states.setdefault(parent, []).append(sid)
            self.child_states[sid] = []
            self.decomposition[sid] = _xml_parameter(state, "decomposition") or "CLUSTER_STATE"
            if nested is not None:
                self._collect(nested, sid)
        for junction in children.findall("junction"):
            if _xml_parameter(junction, "type") not in {None, "CONNECTIVE_JUNCTION"}:
                raise StateflowError("History junctions are unsupported.")
            self.junctions.add(junction.get("SSID"))
        for transition in children.findall("transition"):
            src = _xml_parameter(transition.find("src"), "SSID")
            dst = _xml_parameter(transition.find("dst"), "SSID")
            label = _xml_parameter(transition, "labelString") or ""
            match = re.fullmatch(r"\s*(?P<event>(?:en\([\w.]+\)|\w+))?\s*(?:\[(?P<guard>[^\]]+)\])?\s*(?:\{(?P<action>[^}]*)\})?\s*(?:/\s*(?P<after>.*))?", label, re.S)
            if not match:
                raise StateflowError(f"Unsupported transition label {label!r}.")
            record = {**match.groupdict(), "src": src, "dst": dst, "parent": parent,
                      "order": int(_xml_parameter(transition, "executionOrder") or "1")}
            self.transitions.setdefault(src or ("default", parent), []).append(record)

    @staticmethod
    def _state_actions(label):
        lines = label.splitlines()[1:]
        actions, section = {"entry": "", "during": "", "exit": ""}, None
        for line in lines:
            match = re.match(r"\s*(entry|en|during|du|exit|ex)\s*:\s*(.*)", line)
            if match:
                section = {"en": "entry", "du": "during", "ex": "exit"}.get(match[1], match[1])
                line = match[2]
            elif section is None and line.strip():
                raise StateflowError(f"Unsupported state action {line!r}.")
            if section:
                actions[section] += line + "\n"
        return actions

    def _expression(self, expression, context, validate=False):
        if expression in self._expressions:
            return None if validate else self._evaluate_compiled(self._expressions[expression], context)
        text = re.sub(r"\.\.\.\s*\n", " ", expression).strip()
        text = text.replace("&&", " and ").replace("||", " or ").replace("~=", "!=")
        text = re.sub(r"~|!(?!=)", " not ", text).replace("^", "**")
        text = re.sub(r"\btrue\b", "True", text)
        text = re.sub(r"\bfalse\b", "False", text)
        text = re.sub(r"\bin\(([^)]+)\)", lambda m: "is_active(" + repr(m[1].strip()) + ")", text)
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as error:
            raise StateflowError(f"Unsupported chart expression {expression!r}.") from error
        allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
                   ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise StateflowError(f"Unsupported chart expression {expression!r}.")
        builtin = {"abs", "min", "max", "sin", "cos", "tan", "exp", "sqrt", "log", "is_active"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in builtin | self.functions.keys()):
                raise StateflowError(f"Unsupported chart function in {expression!r}.")
        code = compile(tree, "<stateflow>", "eval")
        self._expressions[expression] = code
        if validate:
            return None
        return self._evaluate_compiled(code, context)

    def _evaluate_compiled(self, code, context):
        functions = {name: (lambda *args, name=name: self._function(name, args, context)) for name in self.functions}
        functions["is_active"] = lambda name: self._state_by_name(name) in self.active
        return _evaluate_expression(code, {**context, **functions})

    def _actions(self, text, context, validate=False):
        text = re.sub(r"\.\.\.\s*\n", " ", text or "")
        for statement in text.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            assignment = re.fullmatch(r"(\w+)\s*=\s*(.+)", statement, re.S)
            expression = assignment[2] if assignment else statement
            value = self._expression(expression, context, validate)
            if not validate and assignment:
                context[assignment[1]] = float(value)

    def _validate(self):
        for sid, names, output in self.functions.values():
            if ("default", sid) not in self.transitions or any(not name.isidentifier() for name in names):
                raise StateflowError("Graphical function requires valid arguments and a default flow path.")
        for state in self.states.values():
            for text in state["actions"].values():
                self._actions(text, {}, True)
        for transitions in self.transitions.values():
            for t in transitions:
                if t["dst"] not in self.states and t["dst"] not in self.junctions:
                    raise StateflowError("Transition destination is missing or unsupported.")
                if t["src"] in self.states and t["dst"] in self.junctions:
                    raise StateflowError("State transitions through junctions are not yet supported.")
                if t["event"] and t["event"] not in self.event_names and not t["event"].startswith("en("):
                    raise StateflowError(f"Unsupported transition event {t['event']!r}.")
                if t["guard"]:
                    self._expression(t["guard"], {}, True)
                self._actions(t["action"], {}, True)
                self._actions(t["after"], {}, True)

    def reset(self):
        self.active, self.context = set(), dict(self.initial_values)
        self.initialized = False
        self.entered = set()
        self.depth = 0

    def initial_outputs(self):
        return {name: self.context.get(name, 0.0) for name in self.output_names}

    def _state_by_name(self, name):
        matches = []
        for sid, state in self.states.items():
            path, parent = state["name"], state["parent"]
            while parent != "root":
                path = self.states[parent]["name"] + "." + path
                parent = self.states[parent]["parent"]
            if name == state["name"] or name == path or path.endswith("." + name):
                matches.append(sid)
        if len(matches) != 1:
            raise StateflowError(f"Missing or ambiguous state name {name!r}.")
        return matches[0]

    def _eligible(self, transition, event, context):
        trigger = transition["event"]
        if trigger:
            if trigger.startswith("en("):
                if self._state_by_name(trigger[3:-1]) not in self.entered:
                    return False
            elif trigger != event:
                return False
        return not transition["guard"] or bool(self._expression(transition["guard"], context))

    def _flow(self, source, event, context):
        for t in self.transitions.get(source, ()):
            if not self._eligible(t, event, context):
                continue
            self._actions(t["action"], context)
            destination = t["dst"]
            if destination in self.junctions:
                if self.transitions.get(destination):
                    self.depth += 1
                    if self.depth > 128:
                        raise StateflowError("Graphical function/transition recursion limit exceeded.")
                    found, destination = self._flow(destination, event, context)
                    self.depth -= 1
                    if not found:
                        continue
                else:
                    destination = None
            self._actions(t["after"], context)
            return True, destination
        return False, None

    def _function(self, name, arguments, context):
        self.depth += 1
        if self.depth > 128:
            raise StateflowError("Graphical function recursion limit exceeded.")
        sid, names, output = self.functions[name]
        if len(names) != len(arguments):
            raise StateflowError(f"Incorrect argument count for {name}.")
        local = {**context, **dict(zip(names, arguments))}
        found, _ = self._flow(("default", sid), None, local)
        if not found:
            raise StateflowError(f"Graphical function {name!r} has no valid terminating path.")
        for key in context:
            if key not in names and key != output:
                context[key] = local[key]
        self.depth -= 1
        if output and output not in local:
            raise StateflowError(f"Graphical function {name!r} did not assign its output.")
        return local[output] if output else 0.0

    def _children_enter(self, parent, event):
        children = self.child_states.get(parent, ())
        if not children:
            return
        if self.decomposition[parent] == "SET_CHART":
            for sid in children:
                self._enter(sid, event)
        else:
            found, destination = self._flow(("default", parent), event, self.context)
            if destination not in self.states:
                raise StateflowError(f"Exclusive state {parent!r} has no valid default transition.")
            self._enter(destination, event)

    def _enter(self, sid, event, descend=True):
        self.active.add(sid)
        self._state_data()
        self.entered.add(sid)
        self._actions(self.states[sid]["actions"]["entry"], self.context)
        if descend:
            self._children_enter(sid, event)

    def _exit(self, sid):
        for child in self.child_states.get(sid, ()):
            if child in self.active:
                self._exit(child)
        self._actions(self.states[sid]["actions"]["exit"], self.context)
        self.active.discard(sid)
        self._state_data()

    def _state_data(self):
        for name, sid in self.state_outputs.items():
            self.context[name] = float(sid in self.active)

    def _ancestors(self, sid):
        path = []
        while sid != "root":
            path.append(sid)
            sid = self.states[sid]["parent"]
        return [*path, "root"]

    def _take(self, source, target, event, action):
        source_path, target_path = self._ancestors(source), self._ancestors(target)
        common = next(sid for sid in source_path if sid in target_path)
        if source == target:
            common = self.states[source]["parent"]
        if common == source:
            for child in self.child_states[source]:
                if child in self.active:
                    self._exit(child)
        else:
            branch = source_path[source_path.index(common) - 1]
            self._exit(branch)
        self._actions(action, self.context)
        entry_path = target_path[:target_path.index(common)]
        for sid in reversed(entry_path):
            self._enter(sid, event, descend=sid == target)

    def _tick(self, sid, event):
        for inner in (False, True):
            if inner:
                self._actions(self.states[sid]["actions"]["during"], self.context)
            for t in self.transitions.get(sid, ()):
                target = t["dst"]
                if target not in self.states:
                    raise StateflowError("State transitions through junctions are not yet supported.")
                is_inner = sid in self._ancestors(target) and target != sid
                if is_inner != inner or not self._eligible(t, event, self.context):
                    continue
                self._actions(t["action"], self.context)
                self._take(sid, target, event, t["after"])
                return
        for child in self.child_states.get(sid, ()):
            if child in self.active:
                self._tick(child, event)

    def execute(self, values, time, events=()):
        self.context.update({name: float(values[name]) for name in self.input_names})
        self.context["time"] = float(time)
        occurrences = events or ((None,) if not self.event_names or self.enter_at_initialization and not self.initialized else ())
        for event in occurrences:
            self.entered = set()
            if not self.initialized:
                self._children_enter("root", event)
                self.initialized = True
            else:
                for sid in self.child_states["root"]:
                    if sid in self.active:
                        self._tick(sid, event)
        for name, sid in self.state_outputs.items():
            self.context[name] = float(sid in self.active)
        return self.initial_outputs()


def compile_state_machine(source):
    return StateMachine(source)

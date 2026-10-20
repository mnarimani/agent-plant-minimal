"""MATLAB workspace variables of a model, without MATLAB.

Simulink parameters often name MATLAB variables (Gain ``m``, A ``[0 1; -k/m -c/m]``) whose values
are not stored with the blocks. They are set by the model's callbacks (PreLoadFcn / PostLoadFcn /
InitFcn, stored in the .slx), by the model workspace, or by a script the user runs in MATLAB.

This module reads such scripts with a small MATLAB interpreter of its own (nothing is executed
by Python's eval): assignments of numbers, vectors, matrices and struct fields; MATLAB's
operators (matrix and element-wise), ranges, indexing, ``end``; the common math, matrix and
control functions (``eye``, ``inv``, ``lqr``, ``care`` ...); ``if`` / ``for`` blocks; comments and
``...``. Whatever it cannot evaluate is not guessed: the variable is listed in
``Workspace.unresolved`` with the reason, and so is every variable computed from it.

    ws = model_workspace(extracted_dir)          # the model's own callbacks
    ws.run("k = 2; A = [0 1; -k -0.5];", "chat") # more values, e.g. from the user
    parameters = ws.parameters()                 # name -> float | ndarray, for slx_reader
    evaluate("[1 2]*k'", parameters)             # one block parameter
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

# Simulink runs these model callbacks, in this order, before a simulation
MODEL_CALLBACKS = ("PreLoadFcn", "PostLoadFcn", "InitFcn")
MAX_LOOP = 100_000                         # iterations of all for / while loops of one script
MAX_ELEMENTS = 1_000_000                   # elements of one value


class MatlabError(ValueError):
    """A statement or expression that cannot be evaluated here (the message says why)."""


class UnknownName(MatlabError):
    """A name that is neither a variable nor a function known here."""

    def __init__(self, name: str, reason: str | None = None) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"{name!r} is not defined" + (f" ({reason})" if reason else ""))


# ---------------------------------------------------------------------------------------------
# the workspace
# ---------------------------------------------------------------------------------------------
class UsedParameters(dict):
    """The parameters of the block reader (name -> value) that also record which names the block
    parameters used (``slx_reader`` adds them to ``used``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[str] = set()
        self.missing: set[str] | None = None     # while set: unknown names are collected, not errors

    def note(self, text: str) -> None:
        for name in re.findall(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", text or ""):
            if name in self:
                self.used.add(name)


@dataclass
class Workspace:
    values: dict[str, object] = field(default_factory=dict)       # name -> 2-D ndarray | str
    unresolved: dict[str, str] = field(default_factory=dict)      # name -> why it has no value
    sources: dict[str, str] = field(default_factory=dict)         # name -> where its value comes from
    notes: list[str] = field(default_factory=list)                # statements skipped, files not read

    def run(self, script: str, source: str) -> None:
        """Run a MATLAB script; never raises (see ``unresolved`` / ``notes``)."""
        try:
            statements = _Parser(_tokens(script), script).statements()
        except Exception as error:                # noqa: BLE001 - a script that cannot be read is a note
            self.notes.append(f"{source}: could not be read ({error}).")
            return
        _Interpreter(self, source).block(statements)

    def set(self, name: str, value, source: str) -> None:
        self.values[name] = value
        self.sources[name] = source
        self.unresolved.pop(name, None)

    def parameters(self) -> dict[str, object]:
        """name -> float (scalars) or a 2-D ndarray, for the block parameters (strings are left out)"""
        result: dict[str, object] = {}
        for name, value in self.values.items():
            if isinstance(value, np.ndarray):
                result[name] = _python_value(value)
        return result

    def report(self) -> dict:
        """for model.json: every variable with its value and source, what has no value and why"""
        def item(name, value):
            entry = {"value": format_value(value, 200), "source": self.sources.get(name, "")}
            if isinstance(value, np.ndarray) and value.size == 1 and not np.iscomplexobj(value)                     and np.isfinite(value.ravel()[0]):
                entry["number"] = float(value.ravel()[0])      # exact, for the metadata
            return entry
        return {"variables": {name: item(name, value) for name, value in self.values.items()},
                "unresolved": dict(self.unresolved), "notes": list(self.notes)}

    def summary(self, names=None) -> list[str]:
        """'name = value  (source)' lines, short values only"""
        lines = []
        for name in names if names is not None else self.values:
            if name in self.values:
                lines.append(f"{name} = {format_value(self.values[name])}  ({self.sources.get(name, '?')})")
        return lines


def model_workspace(directory: str | Path, base: Workspace | None = None) -> Workspace:
    """The variables an extracted .slx sets itself: its model workspace (when defined by MATLAB
    code) and its model callbacks (PreLoadFcn, PostLoadFcn, InitFcn, in Simulink's order).
    ``base``: the MATLAB workspace before (e.g. the user's init script); the model's callbacks
    run on top of it, as in Simulink."""
    workspace = base if base is not None else Workspace()
    diagram = Path(directory) / "simulink" / "blockdiagram.xml"
    if not diagram.is_file():
        return workspace
    try:
        root = ElementTree.parse(diagram).getroot()
    except ElementTree.ParseError:
        workspace.notes.append("blockdiagram.xml could not be read.")
        return workspace
    callbacks = {}
    for parameter in root.iter("P"):              # blockdiagram.xml holds no blocks: these are the model's
        if parameter.get("Name") in MODEL_CALLBACKS and parameter.text:
            callbacks.setdefault(parameter.get("Name"), parameter.text)
    model_ws = root.find(".//ModelWorkspace")
    if model_ws is not None:
        code = next((p.text for p in model_ws.iter("P") if p.get("Name") in {"MATLABCode", "MatlabCode"} and p.text), None)
        source = next((p.text for p in model_ws.iter("P") if p.get("Name") == "DataSource"), None)
        if code:
            workspace.run(code, "the model workspace (MATLAB code)")
        elif source and source not in {"Model File", "MDL-File"}:
            workspace.notes.append(f"The model workspace is loaded from {source!r}, which is not part of the .slx file.")
    if list((Path(directory) / "simulink").glob("modelWorkspace*")):
        workspace.notes.append("The model workspace holds data stored in MATLAB's binary format; it is not read here.")
    for name in MODEL_CALLBACKS:
        if callbacks.get(name):
            workspace.run(callbacks[name], f"the model's {name}")
    return workspace


def evaluate(text: str, values: dict[str, object] | None = None, missing: set[str] | None = None) -> np.ndarray:
    """One MATLAB expression (e.g. a block parameter) -> 2-D ndarray. Raises UnknownName / MatlabError.
    ``missing``: collect the unknown names instead (each counts as 1), to name them all at once."""
    parser = _Parser(_tokens(text), text)
    node = parser.expression()
    parser.expect_end()
    workspace = Workspace(values={k: _matlab_value(v) for k, v in (values or {}).items()})
    interpreter = _Interpreter(workspace, "")
    interpreter.missing = missing
    value = interpreter.value(node)
    if isinstance(value, str):
        raise MatlabError("a text, not a number")
    return value


# Persian / Arabic digits and separators, as people type them in the chat
_CHAT_CHARACTERS = str.maketrans({**{chr(0x06F0 + k): str(k) for k in range(10)},
                                  **{chr(0x0660 + k): str(k) for k in range(10)},
                                  "\u066b": ".", "\u066c": ",", "\u060c": ",", "\u061b": ";", "\u00a0": " "})
# an assignment anywhere in a message: name = value (also name: value), not ==
_CHAT_ASSIGNMENT = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*(?:=(?!=)|:(?![=:]))\s*")


def assignments_in_text(text: str) -> str:
    """The MATLAB assignments in a chat message, as one script (``name = value;`` per line).

    People write values the way they talk: ``m = 1200, k = 50``, ``m=1200 k=50``, inside a Persian
    sentence (``جرم m = 1200 و k = 50``), with Persian digits (``۱۲۰۰``), with units
    (``g = 9.81 m/s^2``), as ``m: 1200``, matrices over several lines. Every ``name =`` (or
    ``name:``) in the message starts a value; the value is the longest part of what follows that is
    a MATLAB expression - so a unit or a word after it is left out, never read as a value."""
    text = (text or "").translate(_CHAT_CHARACTERS).replace("\r", "").replace("`", " ")
    starts = list(_CHAT_ASSIGNMENT.finditer(text))
    lines = []
    for k, match in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(text)
        value = _chat_value(text[match.end():end])
        if value is not None:
            lines.append("%s = %s;" % (match.group(1), value))
    return "\n".join(lines)


def _chat_value(rest: str) -> str | None:
    """the longest leading part of ``rest`` that is one MATLAB expression (None: there is none)"""
    # a matrix may span lines; otherwise the value ends with its line
    depth, cut = 0, len(rest)
    for i, c in enumerate(rest):
        if c == "[":
            depth += 1
        elif c == "]":
            depth = max(0, depth - 1)
        elif c == "\n" and depth == 0:
            cut = i
            break
        elif not c.isascii():                 # a word of the sentence (Persian ...) ends the value
            cut = i
            break
    candidate = re.sub(r"%.*", "", rest[:cut]).strip().rstrip(",;")
    words = candidate.split()
    for n in range(len(words), 0, -1):        # drop trailing words (units, prose) until it parses
        expression = " ".join(words[:n]).strip().rstrip(",;.")
        if not expression:
            continue
        try:
            parser = _Parser(_tokens(expression), expression)
            parser.expression()
            parser.expect_end()
        except Exception:                     # noqa: BLE001 - not an expression yet
            continue
        return expression
    return None


def format_value(value, limit: int = 60) -> str:
    if isinstance(value, str):
        return repr(value)
    value = np.asarray(value)
    if value.size == 1:
        text = _number(value.ravel()[0])
    else:
        rows = ["  ".join(_number(v) for v in row) for row in value]
        text = "[" + "; ".join(rows) + "]"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _number(v) -> str:
    if np.iscomplexobj(v) and v.imag != 0:
        return f"{v.real:.6g}{v.imag:+.6g}i"
    return f"{float(np.real(v)):.6g}"


def _python_value(value: np.ndarray):
    value = value.real.copy() if np.iscomplexobj(value) and np.all(value.imag == 0) else value
    if value.size == 1 and not np.iscomplexobj(value):
        return float(value.ravel()[0])
    return value


def _matlab_value(value) -> np.ndarray | str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    array = array.astype(complex if np.iscomplexobj(array) else float)
    return array.reshape(1, 1) if array.ndim == 0 else (array.reshape(1, -1) if array.ndim == 1 else array)


# ---------------------------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------------------------
@dataclass
class _Token:
    kind: str           # num | id | str | op | kw | sep (statement end) | eof
    text: str
    pos: int
    space: bool = False  # whitespace before the token
    value: object = None


_KEYWORDS = {"if", "elseif", "else", "end", "for", "while", "switch", "case", "otherwise", "try", "catch",
             "function", "return", "break", "continue", "global", "persistent", "parfor", "do", "until"}
_OPERATORS = sorted(["==", "~=", "!=", "<=", ">=", "&&", "||", ".*", "./", ".\\", ".^", ".'", "+", "-", "*", "/",
                     "\\", "^", "<", ">", "&", "|", "~", "!", "=", "(", ")", "[", "]", "{", "}", ",", ";", ":",
                     ".", "@", "'"], key=len, reverse=True)
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?([ij](?![\w]))?")


def _value_end(token: _Token | None) -> bool:
    return token is not None and (token.kind in {"num", "id", "str"} or token.text in {")", "]", "}", "'", ".'"}
                                  or (token.kind == "kw" and token.text == "end"))


def _tokens(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    stack: list[str] = []
    i, n, space = 0, len(text), False
    line_start = True

    def previous() -> _Token | None:
        return tokens[-1] if tokens else None

    def separator(kind_text: str) -> None:
        if tokens and tokens[-1].kind != "sep":
            tokens.append(_Token("sep", kind_text, i))

    while i < n:
        c = text[i]
        # block comments: %{ ... %} on lines of their own
        if line_start and re.match(r"[ \t]*%\{[ \t]*(\r?\n|$)", text[i:]):
            end = re.compile(r"^[ \t]*%\}[ \t]*$", re.M).search(text, i)
            i = end.end() if end else n
            continue
        if c in " \t":
            i, space = i + 1, True
            continue
        if c in "%#" or (c == "." and text.startswith("...", i)):
            continuation = c == "."
            while i < n and text[i] != "\n":
                i += 1
            if continuation:                 # ... joins the next line
                i, space = i + 1, True
            continue
        if c in "\r\n":
            if stack and stack[-1] == "[":
                if previous() is not None and previous().text not in {"[", ";"}:
                    tokens.append(_Token("op", ";", i))
            elif not stack:
                separator("\n")
            i, space, line_start = i + 1, True, True
            continue
        line_start = False
        start = i
        # inside [ ]: a blank between two values separates elements ([1 -2] is two, [1 - 2] one)
        in_matrix = bool(stack) and stack[-1] in "[{"
        if in_matrix and space and _value_end(previous()):
            elementwise = text.startswith((".*", "./", ".^", ".'", ".\\"), i)
            starts_value = (c.isalnum() or c in "_([{\"@" or (c == "." and not elementwise)
                            or c == "'")         # a blank and a quote: a text, not a transpose
            if c in "+-~!" and i + 1 < n and text[i + 1] not in " 	=":
                starts_value = True              # [1 -2]: two elements; [1 - 2]: one
            if starts_value:
                tokens.append(_Token("op", ",", i))
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            match = _NUMBER.match(text, i)
            literal = match.group(0)
            imaginary = bool(match.group(1))
            number = float(re.sub(r"[dD]", "e", literal[:-1] if imaginary else literal))
            tokens.append(_Token("num", literal, start, space, complex(0, number) if imaginary else number))
            i = match.end()
        elif (c.isascii() and c.isalpha()) or c == "_":
            match = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(text, i)
            word = match.group(0)
            kind = "kw" if word in _KEYWORDS and not (stack and word != "end") else "id"
            if word == "end" and stack:
                kind = "kw"
            tokens.append(_Token(kind, word, start, space))
            i = match.end()
        elif c == '"' or (c == "'" and not (_value_end(previous()) and not space)):
            quote, j, chars = c, i + 1, []
            while j < n and text[j] != "\n":
                if text[j] == quote:
                    if j + 1 < n and text[j + 1] == quote:
                        chars.append(quote)
                        j += 2
                        continue
                    break
                chars.append(text[j])
                j += 1
            if j >= n or text[j] != quote:
                raise MatlabError("a text without its closing quote")
            tokens.append(_Token("str", text[i:j + 1], start, space, "".join(chars)))
            i = j + 1
        else:
            op = next((o for o in _OPERATORS if text.startswith(o, i)), None)
            if op is None:
                raise MatlabError(f"the character {c!r}")
            if op in {"(", "[", "{"}:
                stack.append(op)
            elif op in {")", "]", "}"}:
                if stack:
                    stack.pop()
            if op in {";", ","} and not stack:
                separator(op)
            else:
                tokens.append(_Token("op", op, start, space))
            i += len(op)
        space = False
    separator("\n")
    tokens.append(_Token("eof", "", n))
    return tokens


# ---------------------------------------------------------------------------------------------
# syntax: statements and expressions (nodes are tuples)
# ---------------------------------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens: list[_Token], source: str) -> None:
        self.tokens, self.i, self.source = tokens, 0, source

    # -- helpers
    def peek(self, offset: int = 0) -> _Token:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def next(self) -> _Token:
        token = self.peek()
        self.i += 1
        return token

    def accept(self, text: str) -> bool:
        if self.peek().text == text and self.peek().kind in {"op", "kw"}:
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> None:
        if not self.accept(text):
            raise MatlabError(f"expected {text!r} near {self.peek().text or 'the end'!r}")

    def expect_end(self) -> None:
        while self.peek().kind == "sep":
            self.i += 1
        if self.peek().kind != "eof":
            raise MatlabError(f"unexpected {self.peek().text!r}")

    def skip_separators(self) -> None:
        while self.peek().kind == "sep":
            self.i += 1

    # -- statements
    def statements(self, until: tuple[str, ...] = ()) -> list:
        result = []
        while True:
            self.skip_separators()
            token = self.peek()
            if token.kind == "eof":
                if until:
                    raise MatlabError(f"a block without its 'end'")
                return result
            if token.kind == "kw" and token.text in until:
                return result
            start = self.i
            try:
                result.append(self.statement())
            except MatlabError as error:
                result.append(self._unreadable(start, error))

    def _unreadable(self, start: int, error: MatlabError):
        """a statement that cannot be read: skipped up to its end, with the names it would set"""
        self.i = start
        first, second = self.peek(), self.peek(1)
        names = [first.text] if first.kind == "id" and second.text in {"=", "(", "."} else []
        text_start = first.pos
        while self.peek().kind not in {"sep", "eof"}:
            self.i += 1
        if self.i == start:
            self.i += 1
        return ("unreadable", names, str(error), self._text(text_start))

    def statement(self):
        start = self.i
        token = self.peek()
        text_start = token.pos
        if token.kind == "kw":
            if token.text == "if":
                return self._if()
            if token.text in {"for", "parfor"}:
                self.next()
                parenthesized = self.accept("(")
                name = self.next()
                if name.kind != "id":
                    raise MatlabError("a for loop without a variable")
                self.expect("=")
                values = self.expression()
                if parenthesized:
                    self.expect(")")
                body = self.statements(("end",))
                self.expect("end")
                return ("for", name.text, values, body, self._text(text_start))
            if token.text == "while":
                self.next()
                condition = self.expression()
                body = self.statements(("end",))
                self.expect("end")
                return ("while", condition, body, self._text(text_start))
            # switch, try, function ...: skipped as a whole, with what they assign
            return self._skip_block(token.text, text_start)
        # command syntax: clear a b, clc, close all, format long, disp ...
        if token.kind == "id" and self.peek(1).kind in {"id", "sep", "eof", "num"} and \
                (self.peek(1).kind != "id" or self.peek(1).space):
            if self.peek(1).kind in {"sep", "eof"} or token.text in _COMMANDS:
                words = []
                self.next()
                while self.peek().kind not in {"sep", "eof"}:
                    words.append(self.next().text)
                return ("command", token.text, words, self._text(text_start))
        # [a, b] = f(...)
        if token.text == "[" and self._is_multi_assignment():
            self.next()
            targets = []
            while not self.accept("]"):
                if self.accept(","):
                    continue
                item = self.next()
                if item.kind == "id" or item.text in {"~", "!"}:
                    targets.append(item.text if item.kind == "id" else None)
                else:
                    raise MatlabError("unexpected target in [ ] = ...")
            self.expect("=")
            value = self.expression()
            self._statement_end()
            return ("massign", targets, value, self._text(text_start))
        if token.kind == "id" and self._is_assignment():
            target = self.target()
            self.expect("=")
            value = self.expression()
            self._statement_end()
            return ("assign", target, value, self._text(text_start))
        self.i = start
        value = self.expression()
        self._statement_end()
        return ("expr", value, self._text(text_start))

    def _statement_end(self) -> None:
        if self.peek().kind not in {"sep", "eof"}:
            raise MatlabError(f"unexpected {self.peek().text!r}")

    def _text(self, start: int) -> str:
        end = self.tokens[self.i - 1].pos + len(self.tokens[self.i - 1].text) if self.i else start
        return " ".join(self.source[start:end].split())

    def _is_assignment(self) -> bool:
        depth, j = 0, self.i
        while True:
            token = self.tokens[j]
            if token.kind in {"sep", "eof"}:
                return False
            if token.text in {"(", "[", "{"}:
                depth += 1
            elif token.text in {")", "]", "}"}:
                depth -= 1
            elif token.text == "=" and depth == 0:
                return True
            j += 1

    def _is_multi_assignment(self) -> bool:
        depth, j = 0, self.i
        while self.tokens[j].kind not in {"sep", "eof"}:
            if self.tokens[j].text == "[":
                depth += 1
            elif self.tokens[j].text == "]":
                depth -= 1
                if depth == 0:
                    return self.tokens[j + 1].text == "=" and self.tokens[j + 1].kind == "op"
            j += 1
        return False

    def _if(self):
        text_start = self.peek().pos
        self.expect("if")
        branches = []
        condition = self.expression()
        body = self.statements(("elseif", "else", "end"))
        branches.append((condition, body))
        while True:
            if self.accept("elseif"):
                condition = self.expression()
                branches.append((condition, self.statements(("elseif", "else", "end"))))
            elif self.accept("else"):
                branches.append((None, self.statements(("end",))))
            else:
                self.expect("end")
                return ("if", branches, self._text(text_start))

    def _skip_block(self, keyword: str, text_start: int):
        if keyword in {"end", "else", "elseif", "case", "otherwise", "catch"}:
            raise MatlabError(f"{keyword!r} without its block")
        depth, assigned = 0, []
        while True:
            token = self.next()
            if token.kind == "eof":
                if keyword == "function":     # a function file / local functions: up to the end
                    break
                raise MatlabError(f"{keyword!r} without its 'end'")
            if token.kind == "kw" and token.text in {"if", "for", "parfor", "while", "switch", "try", "function"}:
                depth += 1
            elif token.kind == "kw" and token.text == "end":
                depth -= 1
                if depth == 0:
                    break
            elif token.kind == "id" and self.peek().text == "=" and self.peek().kind == "op" and \
                    self.tokens[self.i - 2].kind == "sep":
                assigned.append(token.text)
        return ("skip", keyword, assigned, self._text(text_start))

    def target(self):
        node = ("name", self.next().text)
        while True:
            if self.peek().text == "." and self.peek(1).kind == "id":
                self.next()
                node = ("field", node, self.next().text)
            elif self.peek().text == "(" and not self.peek().space:
                self.next()
                node = ("index", node, self._arguments(")"))
            else:
                return node

    # -- expressions, by MATLAB's precedence (low to high)
    def expression(self):
        return self._binary(0)

    _LEVELS = (("||",), ("&&",), ("|",), ("&",), ("<", "<=", ">", ">=", "==", "~=", "!="))

    def _binary(self, level: int):
        if level == len(self._LEVELS):
            return self._range()
        node = self._binary(level + 1)
        while self.peek().kind == "op" and self.peek().text in self._LEVELS[level]:
            op = self.next().text
            node = ("bin", "~=" if op == "!=" else op, node, self._binary(level + 1))
        return node

    def _range(self):
        first = self._additive()
        if self.peek().text == ":" and self.peek().kind == "op":
            self.next()
            second = self._additive()
            if self.peek().text == ":" and self.peek().kind == "op":
                self.next()
                return ("range", first, second, self._additive())
            return ("range", first, ("num", 1.0), second)
        return first

    def _additive(self):
        node = self._multiplicative()
        while self.peek().kind == "op" and self.peek().text in {"+", "-"}:
            op = self.next().text
            node = ("bin", op, node, self._multiplicative())
        return node

    def _multiplicative(self):
        node = self._unary()
        while self.peek().kind == "op" and self.peek().text in {"*", "/", "\\", ".*", "./", ".\\"}:
            op = self.next().text
            node = ("bin", op, node, self._unary())
        return node

    def _unary(self):
        if self.peek().kind == "op" and self.peek().text in {"-", "+", "~", "!"}:
            op = self.next().text
            return ("unary", "~" if op == "!" else op, self._unary())
        return self._power()

    def _power(self):
        node = self._postfix()
        while self.peek().kind == "op" and self.peek().text in {"^", ".^"}:
            op = self.next().text
            if self.peek().kind == "op" and self.peek().text in {"-", "+", "~", "!"}:
                sign = self.next().text
                operand = ("unary", "~" if sign == "!" else sign, self._postfix())
            else:
                operand = self._postfix()
            node = ("bin", op, node, operand)
        return node

    def _postfix(self):
        node = self._primary()
        while True:
            token = self.peek()
            if token.kind == "op" and token.text in {"'", ".'"} and not token.space:
                self.next()
                node = ("transpose", token.text, node)
            elif token.kind == "op" and token.text == "(":
                self.next()
                node = ("call", node, self._arguments(")"))
            elif token.kind == "op" and token.text == "." and self.peek(1).kind == "id":
                self.next()
                node = ("field", node, self.next().text)
            elif token.kind == "op" and token.text == "{":
                raise MatlabError("cell arrays are not supported")
            else:
                return node

    def _arguments(self, closing: str) -> list:
        arguments = []
        if self.accept(closing):
            return arguments
        while True:
            if self.peek().text == ":" and self.peek(1).text in {",", closing}:
                self.next()
                arguments.append(("colon",))
            else:
                arguments.append(self.expression())
            if self.accept(closing):
                return arguments
            self.expect(",")

    def _primary(self):
        token = self.next()
        if token.kind == "num":
            return ("num", token.value)
        if token.kind == "str":
            return ("str", token.value)
        if token.kind == "id":
            return ("name", token.text)
        if token.kind == "kw" and token.text == "end":
            return ("end",)
        if token.text == "(":
            node = self.expression()
            self.expect(")")
            return ("paren", node)
        if token.text == "[":
            rows, row = [], []
            while True:
                if self.accept("]"):
                    if row:
                        rows.append(row)
                    return ("matrix", rows)
                if self.accept(";"):
                    rows.append(row)
                    row = []
                    continue
                if self.accept(","):
                    continue
                row.append(self.expression())
        if token.text == "@":
            raise MatlabError("function handles are not supported")
        raise MatlabError(f"unexpected {token.text or 'end of the text'!r}")


_COMMANDS = {"clear", "clearvars", "clc", "close", "format", "disp", "warning", "figure", "hold", "grid", "load",
             "run", "addpath", "rmpath", "cd", "more", "beep", "echo", "diary", "open_system", "load_system",
             "set_param", "sim", "syms", "global", "pack"}
_IGNORED_FUNCTIONS = {"clc", "close", "format", "disp", "fprintf", "display", "warning", "figure", "hold", "grid",
                      "plot", "subplot", "xlabel", "ylabel", "title", "legend", "axis", "drawnow", "pause", "tic",
                      "toc", "addpath", "rmpath", "more", "beep", "diary", "echo", "open_system", "load_system",
                      "set_param", "Simulink.BlockDiagram.buildRapidAcceleratorTarget"}


# ---------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------
class _Interpreter:
    def __init__(self, workspace: Workspace, source: str) -> None:
        self.ws, self.source = workspace, source
        self.loops = 0
        self.end_stack: list[int] = []
        self.missing: set[str] | None = None     # see evaluate()

    # -- statements
    def block(self, statements) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, node) -> None:
        kind = node[0]
        try:
            if kind == "assign":
                _, target, value_node, _text = node
                if target[0] == "name":
                    self.ws.set(target[1], self.value(value_node), self.source)
                else:
                    self.assign_into(target, self.value(value_node))
            elif kind == "massign":
                _, targets, value_node, _text = node
                if value_node[0] != "call" or value_node[1][0] != "name":
                    raise MatlabError("[a, b] = ... needs a function call")
                outputs = self.call(value_node[1][1], value_node[2], nargout=len(targets))
                if len(outputs) < len(targets):
                    raise MatlabError(f"{value_node[1][1]} gives {len(outputs)} outputs")
                for name, value in zip(targets, outputs):
                    if name is not None:
                        self.ws.set(name, value, self.source)
            elif kind == "command":
                self.command(node[1], node[2], node[3])
            elif kind == "expr":
                value_node = node[1]
                name = value_node[1][1] if value_node[0] == "call" and value_node[1][0] == "name" else \
                    value_node[1] if value_node[0] == "name" else None
                if name in _IGNORED_FUNCTIONS:
                    return
                if name in {"clear", "clearvars"}:
                    self.command(name, [a[1] for a in value_node[2] if a[0] == "str"] if value_node[0] == "call" else [], node[2])
                    return
                self.value(value_node)       # ans = ...: nothing to keep, but a failure is a note
            elif kind == "if":
                for condition, body in node[1]:
                    if condition is None or _truth(self.value(condition)):
                        self.block(body)
                        break
            elif kind == "for":
                _, name, values_node, body, _text = node
                values = self.value(values_node)
                if isinstance(values, str):
                    raise MatlabError("a for loop over a text")
                for column in range(values.shape[1]):
                    self._count_loop()
                    self.ws.set(name, values[:, column:column + 1].copy(), self.source)
                    self.block(body)
            elif kind == "while":
                while _truth(self.value(node[1])):
                    self._count_loop()
                    self.block(node[2])
            elif kind == "unreadable":
                _, names, message, text = node
                for name in names:
                    self._unresolve(name, f"{text} - could not be read ({message})")
                if not names:
                    self.ws.notes.append(f"{self.source}: skipped `{text}` (could not be read: {message}).")
            elif kind == "skip":
                for name in node[2]:
                    self._unresolve(name, f"set inside a '{node[1]}' block, which is not evaluated here")
                self.ws.notes.append(f"{self.source}: a '{node[1]}' block was not evaluated.")
        except _LoopLimit:
            raise
        except MatlabError as error:
            self._failed(node, error)
        except (ArithmeticError, ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            self._failed(node, MatlabError(str(error) or type(error).__name__))

    def _count_loop(self) -> None:
        self.loops += 1
        if self.loops > MAX_LOOP:
            raise _LoopLimit()

    def _failed(self, node, error: MatlabError) -> None:
        names = _assigned_names(node)
        reason = error.reason if isinstance(error, UnknownName) and error.reason else \
            (f"uses {error.name}, which is not defined there" if isinstance(error, UnknownName) else str(error))
        text = node[-1] if isinstance(node[-1], str) else ""
        for name in names:
            self._unresolve(name, f"{text} - {reason}" if text else reason)
        if not names:
            self.ws.notes.append(f"{self.source}: skipped `{text}` ({reason}).")

    def _unresolve(self, name: str, reason: str) -> None:
        self.ws.values.pop(name, None)
        self.ws.sources.pop(name, None)
        self.ws.unresolved[name] = f"{reason} (in {self.source})" if self.source else reason

    def command(self, name: str, words: list[str], text: str) -> None:
        if name in {"clear", "clearvars"}:
            targets = [w for w in words if not w.startswith("-")]
            if not targets or targets[0] in {"all", "variables", "-all"}:
                self.ws.values.clear()
                self.ws.sources.clear()
            else:
                for target in targets:
                    self.ws.values.pop(target, None)
                    self.ws.sources.pop(target, None)
        elif name in {"load", "run"}:
            self.ws.notes.append(f"{self.source}: `{text}` reads another file, which is not part of the .slx file; "
                                 "its variables are not known here.")
        elif name in _IGNORED_FUNCTIONS or name in {"format", "more"}:
            return
        elif name in self.ws.values:
            return                           # a variable shown in the command window
        elif name in _FUNCTIONS or name in _CONSTANTS:
            return
        else:
            self.ws.notes.append(f"{self.source}: `{text}` runs {name}, which is not known here; "
                                 "variables it sets are not known.")

    def assign_into(self, target, value) -> None:
        """s.f = v, a(i, j) = v, s.f(i) = v"""
        name = _dotted(target)
        if name is not None:
            self.ws.set(name, value, self.source)
            return
        if target[0] != "index" or _dotted(target[1]) is None:
            raise MatlabError("this indexed assignment is not supported")
        name = _dotted(target[1])
        current = self.ws.values.get(name)
        if isinstance(current, str) or isinstance(value, str):
            raise MatlabError("indexing with texts")
        if current is None and name in self.ws.unresolved:
            raise UnknownName(name, f"uses {name}, which has no value here")
        current = np.zeros((0, 0)) if current is None else current
        arguments = target[2]
        indices = [self.index_value(a, current, k, len(arguments)) for k, a in enumerate(arguments)]
        self.ws.set(name, _assign_index(current, indices, value), self.source)

    # -- expressions
    def value(self, node):
        kind = node[0]
        if kind == "num":
            return np.array([[node[1]]], dtype=complex if isinstance(node[1], complex) else float)
        if kind == "str":
            return node[1]
        if kind == "paren":
            return self.value(node[1])
        if kind == "name":
            return self.name(node[1])
        if kind == "field":
            name = _dotted(node)
            if name is not None:
                if name in self.ws.values:
                    return self.ws.values[name]
                if name in self.ws.unresolved:
                    raise UnknownName(name, f"uses {name}, which has no value here")
                if name.split(".")[0] in self.ws.values or name in _QUALIFIED_FUNCTIONS:
                    return self.call(name, [])[0] if name in _QUALIFIED_FUNCTIONS else _no_field(name)
                raise UnknownName(name)
            raise MatlabError("this field access is not supported")
        if kind == "matrix":
            return self.matrix(node[1])
        if kind == "unary":
            operand = _numeric(self.value(node[2]))
            if node[1] == "-":
                return -operand
            if node[1] == "+":
                return operand
            return (operand == 0).astype(float)
        if kind == "transpose":
            operand = _numeric(self.value(node[2]))
            return operand.T.copy() if node[1] == ".'" else operand.conj().T.copy()
        if kind == "bin":
            if node[1] in {"&&", "||"}:
                left = _truth(self.value(node[2]))
                if node[1] == "&&" and not left or node[1] == "||" and left:
                    return np.array([[float(left)]])
                return np.array([[float(_truth(self.value(node[3])))]])
            return _binary(node[1], _numeric(self.value(node[2])), _numeric(self.value(node[3])))
        if kind == "range":
            start, step, stop = (_scalar(self.value(n), "a range bound") for n in node[1:])
            if step == 0 or (stop - start) / step < 0:
                return np.zeros((1, 0))
            count = int(math.floor((stop - start) / step + 1e-10)) + 1
            if count > MAX_ELEMENTS:
                raise MatlabError("a range too long")
            return (start + step * np.arange(count)).reshape(1, -1)
        if kind == "end":
            if not self.end_stack:
                raise MatlabError("'end' outside an index")
            return np.array([[float(self.end_stack[-1])]])
        if kind == "call":
            function = node[1]
            name = function[1] if function[0] == "name" else _dotted(function)
            if name is not None and name in self.ws.values:
                base = self.ws.values[name]
                if isinstance(base, str):
                    raise MatlabError("indexing into a text")
                indices = [self.index_value(a, base, k, len(node[2])) for k, a in enumerate(node[2])]
                return _index(base, indices)
            if name is not None and name in self.ws.unresolved:
                raise UnknownName(name, f"uses {name}, which has no value here")
            if name is None:
                raise MatlabError("this call is not supported")
            return self.call(name, node[2], nargout=1)[0]
        if kind == "colon":
            raise MatlabError("':' outside an index")
        raise MatlabError(f"unsupported expression {kind}")

    def name(self, name: str):
        if name in self.ws.values:
            return self.ws.values[name]
        if name in self.ws.unresolved:
            raise UnknownName(name, f"uses {name}, which has no value here")
        if name in _CONSTANTS:
            return np.array([[_CONSTANTS[name]]])
        if name in _FUNCTIONS:
            return self.call(name, [], nargout=1)[0]
        if self.missing is not None:
            self.missing.add(name)
            return np.ones((1, 1))
        raise UnknownName(name)

    def index_value(self, node, base: np.ndarray, position: int, count: int):
        if node[0] == "colon":
            return slice(None)
        size = base.size if count == 1 else (base.shape[position] if position < 2 else 1)
        self.end_stack.append(size)
        try:
            value = self.value(node)
        finally:
            self.end_stack.pop()
        if isinstance(value, str):
            raise MatlabError("a text as an index")
        value = np.real(value)
        if value.size and set(np.unique(value)) <= {0.0, 1.0} and _is_logical_hint(node):
            return value.astype(bool)
        if np.any(value != np.round(value)) or np.any(value < 1):
            raise MatlabError("indices must be positive integers")
        return value.astype(int) - 1

    def matrix(self, rows):
        built = []
        for row in rows:
            items = [self.value(item) for item in row]
            if any(isinstance(item, str) for item in items):
                if all(isinstance(item, str) for item in items) and len(rows) == 1:
                    return "".join(items)
                raise MatlabError("a text in a matrix")
            items = [_numeric(item) for item in items if _numeric(item).size]
            if items:
                if len({item.shape[0] for item in items}) != 1:
                    raise MatlabError("dimensions of the matrix are not consistent")
                built.append(np.hstack(items))
        if not built:
            return np.zeros((0, 0))
        if len({item.shape[1] for item in built}) != 1:
            raise MatlabError("dimensions of the matrix are not consistent")
        return np.vstack(built)

    def call(self, name: str, argument_nodes, nargout: int = 1):
        if name in _IGNORED_FUNCTIONS:
            raise MatlabError(f"{name} gives no value")
        function = _FUNCTIONS.get(name) or _QUALIFIED_FUNCTIONS.get(name)
        if function is None:
            if name in _CONSTANTS:
                arguments = [self.value(a) for a in argument_nodes]
                return [_fill(_CONSTANTS[name], arguments)]
            if self.missing is not None:
                self.missing.add(name)
                return [np.ones((1, 1))]
            raise UnknownName(name, f"uses {name}, a function or variable not known here")
        arguments = [self.value(a) for a in argument_nodes]
        if name == "size" and nargout > 1 and len(arguments) == 1:
            whole = _size(arguments[0]).ravel()
            return [np.array([[whole[0]]]), np.array([[whole[1]]])]
        result = function(*arguments)
        outputs = list(result) if isinstance(result, tuple) else [result]
        outputs = [o if isinstance(o, str) else _numeric(o) for o in outputs]
        for o in outputs:
            if not isinstance(o, str) and o.size > MAX_ELEMENTS:
                raise MatlabError("a value too large")
        return outputs


class _LoopLimit(Exception):
    pass


def _dotted(node) -> str | None:
    if node[0] == "name":
        return node[1]
    if node[0] == "field":
        base = _dotted(node[1])
        return f"{base}.{node[2]}" if base is not None else None
    return None


def _no_field(name: str):
    raise UnknownName(name, f"uses {name}, a field that is not set")


def _assigned_names(node) -> list[str]:
    kind = node[0]
    if kind == "assign":
        target = node[1]
        name = _dotted(target)
        while name is None and target[0] in {"index", "field"}:
            target = target[1]
            name = _dotted(target)
        return [name] if name else []
    if kind == "massign":
        return [n for n in node[1] if n]
    if kind == "if":
        return [n for _, body in node[1] for s in body for n in _assigned_names(s)]
    if kind in {"for", "while"}:
        body = node[3] if kind == "for" else node[2]
        return ([node[1]] if kind == "for" else []) + [n for s in body for n in _assigned_names(s)]
    if kind in {"skip", "unreadable"}:
        return list(node[2] if kind == "skip" else node[1])
    return []


def _is_logical_hint(node) -> bool:
    return node[0] in {"bin", "unary"} and node[1] in {"<", "<=", ">", ">=", "==", "~=", "&", "|", "~"}


# ---------------------------------------------------------------------------------------------
# values: 2-D arrays, as in MATLAB
# ---------------------------------------------------------------------------------------------
def _numeric(value) -> np.ndarray:
    if isinstance(value, str):
        raise MatlabError("a text where a number is needed")
    array = np.asarray(value)
    if array.dtype == bool:
        array = array.astype(float)
    if not np.iscomplexobj(array):
        array = array.astype(float)
    elif np.all(array.imag == 0):
        array = array.real.astype(float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim > 2:
        raise MatlabError("arrays of more than two dimensions are not supported")
    return array


def _scalar(value, what: str) -> float:
    value = _numeric(value)
    if value.size != 1:
        raise MatlabError(f"{what} must be a scalar")
    return float(np.real(value.ravel()[0]))


def _truth(value) -> bool:
    value = _numeric(value)
    return bool(value.size) and bool(np.all(value != 0))


def _broadcast(a: np.ndarray, b: np.ndarray) -> None:
    for x, y in zip(a.shape, b.shape):
        if x != y and x != 1 and y != 1:
            raise MatlabError(f"sizes {a.shape[0]}x{a.shape[1]} and {b.shape[0]}x{b.shape[1]} do not match")


def _power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _broadcast(a, b)
    if not np.iscomplexobj(a) and not np.iscomplexobj(b) and np.any((a < 0) & (b != np.round(b))):
        a = a.astype(complex)
    with np.errstate(all="ignore"):
        return np.power(a, b)


def _binary(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        if op in {"+", "-", ".*", "./", ".\\", "<", "<=", ">", ">=", "==", "~=", "&", "|"}:
            _broadcast(a, b)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == ".*":
            return a * b
        if op == "./":
            return a / b
        if op == ".\\":
            return b / a
        if op == ".^":
            return _power(a, b)
        if op == "*":
            if a.size == 1 or b.size == 1:
                return a * b
            if a.shape[1] != b.shape[0]:
                raise MatlabError(f"inner dimensions {a.shape[1]} and {b.shape[0]} of * do not match")
            return a @ b
        if op == "/":
            if b.size == 1:
                return a / b
            if a.shape[1] != b.shape[1]:
                raise MatlabError("dimensions of / do not match")
            return _solve(b.T, a.T).T
        if op == "\\":
            if a.size == 1:
                return b / a
            if a.shape[0] != b.shape[0]:
                raise MatlabError("dimensions of \\ do not match")
            return _solve(a, b)
        if op == "^":
            if a.size == 1 and b.size == 1:
                return _power(a, b)
            if b.size == 1 and a.shape[0] == a.shape[1] and float(np.real(b[0, 0])).is_integer():
                k = int(np.real(b[0, 0]))
                return np.linalg.matrix_power(a, k) if k >= 0 else np.linalg.matrix_power(np.linalg.inv(a), -k)
            raise MatlabError("this matrix power is not supported")
        comparisons = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
                       "==": np.equal, "~=": np.not_equal}
        if op in comparisons:
            return comparisons[op](np.real(a), np.real(b)).astype(float) if op not in {"==", "~="} else \
                comparisons[op](a, b).astype(float)
        if op == "&":
            return ((a != 0) & (b != 0)).astype(float)
        if op == "|":
            return ((a != 0) | (b != 0)).astype(float)
    raise MatlabError(f"operator {op}")


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == a.shape[1]:
        return np.linalg.solve(a, b)
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _index(base: np.ndarray, indices: list) -> np.ndarray:
    if len(indices) == 1:
        flat = base.ravel(order="F")
        index = indices[0]
        if isinstance(index, slice):
            return flat.reshape(-1, 1).copy()
        vector = base.shape[0] == 1 or base.shape[1] == 1
        if index.dtype == bool:
            positions = np.flatnonzero(index.ravel(order="F"))
            shape = (1, positions.size) if base.shape[0] == 1 else (positions.size, 1)
        else:
            positions = index.ravel(order="F")
            shape = index.shape
            if vector and (index.shape[0] == 1 or index.shape[1] == 1):
                shape = (1, positions.size) if base.shape[0] == 1 else (positions.size, 1)
        if np.any(positions >= flat.size):
            raise MatlabError("index exceeds the number of elements")
        return flat[positions].reshape(shape, order="F")
    if len(indices) == 2:
        rows, columns = (_axis(i, base.shape[k]) for k, i in enumerate(indices))
        return base[np.ix_(rows, columns)].copy()
    raise MatlabError("indexing with more than two indices is not supported")


def _axis(index, size: int) -> np.ndarray:
    if isinstance(index, slice):
        return np.arange(size)
    index = np.flatnonzero(index.ravel()) if index.dtype == bool else index.ravel()
    if np.any(index >= size):
        raise MatlabError("index exceeds the matrix dimensions")
    return index


def _assign_index(current: np.ndarray, indices: list, value: np.ndarray) -> np.ndarray:
    value = _numeric(value)
    result = current.astype(complex if np.iscomplexobj(value) or np.iscomplexobj(current) else float)
    if len(indices) == 1:
        index = indices[0]
        if isinstance(index, slice):
            index = np.arange(result.size)
        index = np.flatnonzero(index.ravel(order="F")) if index.dtype == bool else index.ravel()
        needed = int(index.max()) + 1 if index.size else 0
        if needed > result.size:
            if result.size == 0:
                result = np.zeros((1, needed), dtype=result.dtype)
            elif result.shape[0] == 1:
                result = np.hstack([result, np.zeros((1, needed - result.size), dtype=result.dtype)])
            elif result.shape[1] == 1:
                result = np.vstack([result, np.zeros((needed - result.size, 1), dtype=result.dtype)])
            else:
                raise MatlabError("a linear index beyond a matrix")
        flat = result.ravel(order="F")
        flat[index] = value.ravel(order="F") if value.size != 1 else value.ravel()[0]
        return flat.reshape(result.shape, order="F")
    if len(indices) == 2:
        sizes = [result.shape[0], result.shape[1]]
        axes = []
        for k, index in enumerate(indices):
            if isinstance(index, slice):
                axes.append(np.arange(sizes[k] if sizes[k] else (value.shape[k] if value.size != 1 else 1)))
            else:
                axes.append(np.flatnonzero(index.ravel()) if index.dtype == bool else index.ravel())
        grown = [max(sizes[k], int(axes[k].max()) + 1 if axes[k].size else 0) for k in range(2)]
        if grown != sizes:
            bigger = np.zeros(grown, dtype=result.dtype)
            bigger[: sizes[0], : sizes[1]] = result
            result = bigger
        block = value if value.size != 1 else np.full((axes[0].size, axes[1].size), value.ravel()[0])
        if block.shape != (axes[0].size, axes[1].size):
            raise MatlabError("the sizes of the assignment do not match")
        result[np.ix_(axes[0], axes[1])] = block
        return result
    raise MatlabError("indexing with more than two indices is not supported")


# ---------------------------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------------------------
_CONSTANTS = {"pi": math.pi, "inf": math.inf, "Inf": math.inf, "nan": math.nan, "NaN": math.nan,
              "eps": float(np.finfo(float).eps), "true": 1.0, "false": 0.0, "i": 1j, "j": 1j,
              "realmax": float(np.finfo(float).max), "realmin": float(np.finfo(float).tiny)}


def _fill(constant, arguments) -> np.ndarray:
    """inf(2, 3), nan(1, 4), true(2) ..."""
    rows, columns = _dimensions(arguments)
    return np.full((rows, columns), constant)


def _dimensions(arguments) -> tuple[int, int]:
    numbers = [_numeric(a).ravel() for a in arguments if not isinstance(a, str)]
    if not numbers:
        return 1, 1
    if len(numbers) == 1:
        if numbers[0].size == 1:
            return int(numbers[0][0]), int(numbers[0][0])
        return int(numbers[0][0]), int(numbers[0][1])
    return int(numbers[0][0]), int(numbers[1][0])


def _elementwise(function):
    def apply(x, *rest):
        if rest:
            raise MatlabError("too many arguments")
        with np.errstate(all="ignore"):
            return function(_numeric(x))
    return apply


def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5) if not np.iscomplexobj(x) else \
        _matlab_round(x.real) + 1j * _matlab_round(x.imag)


def _mod(x, y):
    x, y = _numeric(x), _numeric(y)
    _broadcast(x, y)
    with np.errstate(all="ignore"):
        return np.where(y == 0, x, x - np.floor(x / np.where(y == 0, 1, y)) * y)


def _rem(x, y):
    x, y = _numeric(x), _numeric(y)
    _broadcast(x, y)
    with np.errstate(all="ignore"):
        return np.where(y == 0, x, x - np.trunc(x / np.where(y == 0, 1, y)) * y)


def _first_dimension(x: np.ndarray) -> int:
    return 1 if x.shape[0] == 1 else 0


def _reduce(function):
    def apply(x, dim=None):
        x = _numeric(x)
        axis = _first_dimension(x) if dim is None else int(_scalar(dim, "a dimension")) - 1
        if x.size == 0:
            return np.array([[function(np.zeros(0))]])
        return np.expand_dims(function(x, axis=axis), axis)
    return apply


def _extreme(function, arg_function):
    def apply(x, y=None, dim=None):
        x = _numeric(x)
        if y is not None and _numeric(y).size:
            y = _numeric(y)
            _broadcast(x, y)
            return function(x, y)
        axis = _first_dimension(x) if dim is None else int(_scalar(dim, "a dimension")) - 1
        values = np.expand_dims((np.max if function is np.maximum else np.min)(x, axis=axis), axis)
        indices = np.expand_dims(arg_function(x, axis=axis), axis) + 1.0
        return values, indices
    return apply


def _zeros(*arguments):
    return np.zeros(_dimensions(arguments))


def _ones(*arguments):
    return np.ones(_dimensions(arguments))


def _eye(*arguments):
    rows, columns = _dimensions(arguments)
    return np.eye(rows, columns)


def _diag(x, k=None):
    x = _numeric(x)
    offset = int(_scalar(k, "a diagonal")) if k is not None else 0
    if x.shape[0] == 1 or x.shape[1] == 1:
        return np.diag(x.ravel(), offset)
    return np.diag(x, offset).reshape(-1, 1)


def _size(x, dim=None):
    shape = (1, len(x)) if isinstance(x, str) else _numeric(x).shape
    if dim is not None:
        k = int(_scalar(dim, "a dimension"))
        return np.array([[float(shape[k - 1] if k <= 2 else 1)]])
    return np.array([[float(shape[0]), float(shape[1])]])


def _length(x):
    shape = (1, len(x)) if isinstance(x, str) else _numeric(x).shape
    return np.array([[float(0 if 0 in shape else max(shape))]])


def _numel(x):
    return np.array([[float(len(x) if isinstance(x, str) else _numeric(x).size)]])


def _linspace(a, b, n=None):
    count = 100 if n is None else int(_scalar(n, "the number of points"))
    return np.linspace(_scalar(a, "a bound"), _scalar(b, "a bound"), count).reshape(1, -1)


def _repmat(x, m, n=None):
    rows, columns = _dimensions([m] if n is None else [m, n])
    return np.tile(_numeric(x), (rows, columns))


def _reshape(x, *dims):
    rows, columns = _dimensions(dims)
    x = _numeric(x)
    if rows * columns != x.size:
        raise MatlabError("reshape does not keep the number of elements")
    return x.ravel(order="F").reshape((rows, columns), order="F")


def _norm(x, p=None):
    x = _numeric(x)
    order = None if p is None else (np.inf if isinstance(p, str) and p.lower() == "inf" else
                                    "fro" if isinstance(p, str) else _scalar(p, "a norm"))
    if x.shape[0] == 1 or x.shape[1] == 1:
        return np.array([[np.linalg.norm(x.ravel(), order if order != "fro" else None)]])
    return np.array([[np.linalg.norm(x, 2 if order is None else order)]])


def _square(x, what: str) -> np.ndarray:
    x = _numeric(x)
    if x.shape[0] != x.shape[1]:
        raise MatlabError(f"{what} needs a square matrix")
    return x


def _expm(x):
    from scipy.linalg import expm
    return expm(_square(x, "expm"))


def _sqrt_psd(x: np.ndarray) -> np.ndarray:
    """a factor L with L L' = x (x symmetric, positive semidefinite)"""
    symmetric = (x + x.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    if np.any(values < -1e-10 * max(1.0, float(np.max(np.abs(values))))):
        raise MatlabError("the matrix is not positive semidefinite")
    return vectors * np.sqrt(np.clip(values, 0, None))


def _care(a, b, q, r=None, s=None, e=None):
    """[X, K, L] = care(A, B, Q, R, S): A'X + XA - (XB + S) R^-1 (B'X + S') + Q = 0, K = R^-1 (B'X + S')"""
    from scipy.linalg import solve_continuous_are
    if e is not None:
        raise MatlabError("care with E is not supported")
    a, b, q = _numeric(a), _numeric(b), _numeric(q)
    r = np.eye(b.shape[1]) if r is None or _numeric(r).size == 0 else _numeric(r)
    s = None if s is None or _numeric(s).size == 0 else _numeric(s)
    x = solve_continuous_are(a, b, q, r, s=s)
    gain = np.linalg.solve(r, b.T @ x + (s.T if s is not None else 0))
    closed = np.linalg.eigvals(a - b @ gain).reshape(-1, 1)
    return x, gain, closed


def _lqr(a, b, q, r, n=None):
    """[K, S, P] = lqr(A, B, Q, R, N)"""
    x, gain, closed = _care(a, b, q, r, n)
    return gain, x, closed


def _are(a, b, c):
    """X = are(A, B, C): A'X + XA - XBX + C = 0, B symmetric positive semidefinite"""
    from scipy.linalg import solve_continuous_are
    a, b, c = _numeric(a), _numeric(b), _numeric(c)
    factor = _sqrt_psd(b)
    return solve_continuous_are(a, factor, c, np.eye(factor.shape[1]))


def _lyap(a, q, c=None):
    from scipy.linalg import solve_continuous_lyapunov, solve_sylvester
    if c is not None:
        return solve_sylvester(_numeric(a), _numeric(q), -_numeric(c))
    return solve_continuous_lyapunov(_numeric(a), -_numeric(q))


def _ctrb(a, b):
    a, b = _numeric(a), _numeric(b)
    blocks, current = [], b
    for _ in range(a.shape[0]):
        blocks.append(current)
        current = a @ current
    return np.hstack(blocks)


def _obsv(a, c):
    return _ctrb(_numeric(a).T, _numeric(c).T).T


def _acker(a, b, poles):
    """K = acker(A, B, p): Ackermann's formula, one input"""
    a, b = _square(a, "acker"), _numeric(b)
    if b.shape[1] != 1:
        raise MatlabError("acker needs one input")
    coefficients = np.real(np.poly(_numeric(poles).ravel()))
    phi = sum(c * np.linalg.matrix_power(a, len(coefficients) - 1 - k) for k, c in enumerate(coefficients))
    unit = np.zeros((1, a.shape[0]))
    unit[0, -1] = 1.0
    return unit @ np.linalg.solve(_ctrb(a, b), phi)


def _cross(x, y):
    x, y = _numeric(x), _numeric(y)
    return np.cross(x.ravel(), y.ravel()).reshape(x.shape)


def _dot(x, y):
    return np.array([[np.sum(np.conj(_numeric(x).ravel()) * _numeric(y).ravel())]])


def _str2num(text):
    if not isinstance(text, str):
        raise MatlabError("str2num needs a text")
    return evaluate(text)


def _double(x):
    if isinstance(x, str):
        return np.array([[float(ord(c)) for c in x]])
    return _numeric(x)


def _deg2rad(x):
    return _numeric(x) * (math.pi / 180.0)


def _rad2deg(x):
    return _numeric(x) * (180.0 / math.pi)


def _atan2(y, x):
    y, x = _numeric(y), _numeric(x)
    _broadcast(y, x)
    return np.arctan2(y, x)


def _hypot(x, y):
    x, y = _numeric(x), _numeric(y)
    _broadcast(x, y)
    return np.hypot(x, y)


def _power_function(x, y):
    return _power(_numeric(x), _numeric(y))


def _sqrt(x):
    x = _numeric(x)
    return np.sqrt(x.astype(complex)) if not np.iscomplexobj(x) and np.any(x < 0) else np.sqrt(x)


def _log(function):
    def apply(x):
        x = _numeric(x)
        with np.errstate(all="ignore"):
            return function(x.astype(complex)) if not np.iscomplexobj(x) and np.any(x < 0) else function(x)
    return apply


def _cumulative(function):
    def apply(x, dim=None):
        x = _numeric(x)
        axis = _first_dimension(x) if dim is None else int(_scalar(dim, "a dimension")) - 1
        return function(x, axis=axis)
    return apply


def _transpose(x):
    return _numeric(x).T.copy()


_FUNCTIONS = {
    "sqrt": _sqrt, "exp": _elementwise(np.exp), "log": _log(np.log), "log10": _log(np.log10), "log2": _log(np.log2),
    "abs": _elementwise(np.abs), "sin": _elementwise(np.sin), "cos": _elementwise(np.cos), "tan": _elementwise(np.tan),
    "asin": _elementwise(np.arcsin), "acos": _elementwise(np.arccos), "atan": _elementwise(np.arctan),
    "sinh": _elementwise(np.sinh), "cosh": _elementwise(np.cosh), "tanh": _elementwise(np.tanh),
    "sind": _elementwise(lambda x: np.sin(np.deg2rad(x))), "cosd": _elementwise(lambda x: np.cos(np.deg2rad(x))),
    "tand": _elementwise(lambda x: np.tan(np.deg2rad(x))),
    "floor": _elementwise(np.floor), "ceil": _elementwise(np.ceil), "fix": _elementwise(np.trunc),
    "round": _elementwise(_matlab_round), "sign": _elementwise(np.sign), "real": _elementwise(np.real),
    "imag": _elementwise(np.imag), "conj": _elementwise(np.conj), "angle": _elementwise(np.angle),
    "atan2": _atan2, "hypot": _hypot, "mod": _mod, "rem": _rem, "power": _power_function,
    "deg2rad": _deg2rad, "rad2deg": _rad2deg,
    "max": _extreme(np.maximum, np.argmax), "min": _extreme(np.minimum, np.argmin),
    "sum": _reduce(np.sum), "prod": _reduce(np.prod), "mean": _reduce(np.mean),
    "cumsum": _cumulative(np.cumsum), "cumprod": _cumulative(np.cumprod),
    "zeros": _zeros, "ones": _ones, "eye": _eye, "diag": _diag, "linspace": _linspace, "repmat": _repmat,
    "reshape": _reshape, "size": _size, "length": _length, "numel": _numel, "transpose": _transpose,
    "inv": lambda x: np.linalg.inv(_square(x, "inv")), "pinv": lambda x: np.linalg.pinv(_numeric(x)),
    "det": lambda x: np.array([[np.linalg.det(_square(x, "det"))]]),
    "trace": lambda x: np.array([[np.trace(_square(x, "trace"))]]),
    "eig": lambda x: np.linalg.eigvals(_square(x, "eig")).reshape(-1, 1),
    "rank": lambda x: np.array([[float(np.linalg.matrix_rank(_numeric(x)))]]),
    "norm": _norm, "expm": _expm, "kron": lambda x, y: np.kron(_numeric(x), _numeric(y)),
    "cross": _cross, "dot": _dot,
    "care": _care, "lqr": _lqr, "are": _are, "lyap": _lyap, "ctrb": _ctrb, "obsv": _obsv, "acker": _acker,
    "double": _double, "str2num": _str2num, "isempty": lambda x: np.array([[float(_numel(x)[0, 0] == 0)]]),
    "logical": lambda x: (_numeric(x) != 0).astype(float),
}
_QUALIFIED_FUNCTIONS: dict[str, object] = {}

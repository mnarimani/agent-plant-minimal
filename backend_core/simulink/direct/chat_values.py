"""Values of missing MATLAB variables, read from a chat message.

A Simulink model may wait for the values of MATLAB workspace variables (slx_direct: they are not
stored in the .slx file). The user answers in the chat. Two readers, in this order:

1. the MATLAB parser (``workspace.assignments_in_text``): every ``name = value`` in the message,
   however it is written (in a Persian sentence, Persian digits, units after the value ...).
   Deterministic, free, and it takes the numbers exactly as written.
2. for the missing variables the parser did not find in the message: an LLM, for values given by
   description instead of ``name = value`` ("the mass of the car is 1200 kg", "m is two"). It sees
   those names and the blocks that use them, and returns name -> value in MATLAB form with the
   words it read.

Nothing an LLM returns is used unchecked - a wrong value would give a model that the checks call
correct, since they compare the equations with a block model built from the same values. A value
is taken only if
  - its name is one of the missing variables,
  - it is a plain MATLAB value (a number, a vector or matrix of numbers, pi ...), no names, and
  - every number in it appears in the user's message, in digits or in words ("two", "twenty-five",
    "two point five", «دو», «بیست و پنج» - read by ``word_numbers``, not by the LLM); no invented numbers,
    no unit conversion (a value in a unit that is not the model's goes to ``unclear`` and the user is asked).
Every value taken is listed with the words it came from (``ChatValues.read``) for the user to see.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...prompts import fill, text as prompt_text

PARSER_SOURCE = "the chat"
LLM_SOURCE = "the chat (read by the LLM)"
MAX_MESSAGE = 4000                 # characters of the message the LLM sees

_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+")

# the prompt: backend_core/prompts/simulink/chat_values.yaml
PROMPTS = "simulink/chat_values"


@dataclass
class ChatValues:
    script: str = ""                                  # MATLAB assignments ("name = value;" per line)
    source: str = PARSER_SOURCE                       # who read them: the parser or the LLM
    read: List[str] = field(default_factory=list)     # "m = 1200  (from «...»)" - shown to the user
    unclear: List[str] = field(default_factory=list)  # what the LLM could not take, for the agent to ask
    rejected: List[str] = field(default_factory=list)  # LLM answers the checks refused (and why)


def read_values(text: str, missing: List[str], uses: Optional[Dict[str, List[str]]] = None,
                llm: Optional[Callable[[List[Dict[str, str]]], str]] = None) -> ChatValues:
    """The values in a chat message for a model that waits for ``missing`` (see the module doc).
    Never raises: an LLM that fails or answers nonsense gives no values."""
    from .slx2py.model_converter import workspace
    script = workspace.assignments_in_text(text)
    parsed = ChatValues(script=script, source=PARSER_SOURCE, read=[line.rstrip(";") for line in script.splitlines()])
    named = {line.split("=", 1)[0].strip().split(".")[0].split("(")[0] for line in script.splitlines()}
    rest = [name for name in missing if name not in named]       # what the parser did not read
    if not rest or llm is None or not (text or "").strip():
        return parsed
    try:
        answer = _ask(llm, text, rest, uses or {})
    except Exception as exc:                          # noqa: BLE001 - the LLM is an extra, never a failure
        parsed.rejected.append("the LLM could not be asked (%s)" % type(exc).__name__)
        return parsed
    found = _checked(answer, text, rest)
    if not script:
        return found
    if not found.script:                              # the LLM added nothing: the parser's values
        parsed.unclear, parsed.rejected = found.unclear, found.rejected
        return parsed
    # both: the parser's values, then the LLM's for the rest
    found.script = "\n".join(line for line in (script, found.script) if line)
    found.read = parsed.read + found.read
    return found


def _ask(llm, text: str, missing: List[str], uses: Dict[str, List[str]]) -> dict:
    variables = "\n".join(fill(prompt_text(PROMPTS, "variable"), name=name,
                                uses="; ".join(uses.get(name) or [prompt_text(PROMPTS, "no_uses")]))
                           for name in missing)
    messages = [{"role": "system", "content": fill(prompt_text(PROMPTS, "system"), variables=variables)},
                {"role": "user", "content": (text or "")[:MAX_MESSAGE]}]
    raw = llm(messages)
    data = json.loads(raw) if isinstance(raw, str) else raw
    return data if isinstance(data, dict) else {}


def _numbers(text: str) -> List[float]:
    return [float(n) for n in _NUMBER.findall(text)]


# numbers written in words, English and Persian: word -> (value, kind)
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_FA_UNITS = ("صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده"
             ).split()
_FA_TENS = "بیست سی چهل پنجاه شصت هفتاد هشتاد نود".split()
_FA_HUNDREDS = "صد دویست سیصد چهارصد پانصد ششصد هفتصد هشتصد نهصد".split()
_WORDS: Dict[str, tuple] = {
    **{w: (float(i), "unit") for i, w in enumerate(_UNITS)},
    **{w: (float(i), "unit") for i, w in enumerate(_FA_UNITS)},
    "هیجده": (18.0, "unit"),
    **{w: (20.0 + 10 * i, "tens") for i, w in enumerate(_TENS)},
    **{w: (20.0 + 10 * i, "tens") for i, w in enumerate(_FA_TENS)},
    **{w: (100.0 * (i + 1), "hundreds") for i, w in enumerate(_FA_HUNDREDS[1:], start=1)},
    "hundred": (100.0, "scale100"), "صد": (100.0, "hundreds"),
    "thousand": (1e3, "scale"), "هزار": (1e3, "scale"), "million": (1e6, "scale"), "میلیون": (1e6, "scale"),
    "half": (0.5, "alone"), "نیم": (0.5, "alone"),
}
_POINT = {"point", "ممیز"}
_JOIN = {"and", "و", "-"}
_WORD = re.compile(r"[A-Za-z]+|[\u0600-\u06FF]+|-")


def word_numbers(text: str) -> set:
    """the numbers written in words in ``text``: "two" -> 2, "twenty-five" -> 25, "two point five"
    -> 2.5, «بیست و پنج» -> 25, «دو ممیز پنج» -> 2.5, "one two three" -> 1, 2, 3"""
    found: set = set()
    total = current = 0.0
    last = None                                   # the kind of the last number word, None: no number open
    decimals = None                               # the digits after "point"

    def flush():
        nonlocal total, current, last, decimals
        if last is not None:
            value = total + current
            if decimals:
                value += float("0." + "".join(decimals))
            found.add(value)
        total = current = 0.0
        last, decimals = None, None

    for word in _WORD.findall((text or "").lower()):
        if word in _POINT and last is not None and decimals is None:
            decimals = []
            continue
        if word in _JOIN and last is not None:
            continue
        item = _WORDS.get(word)
        if item is None:
            flush()
            continue
        value, kind = item
        if decimals is not None:
            if kind == "unit" and value < 10:
                decimals.append(str(int(value)))
                continue
            flush()
        if kind == "alone":
            flush()
            found.add(value)
            continue
        if kind == "unit":
            if last in ("unit",) or (last == "tens" and value >= 10):
                flush()
            current += value
        elif kind == "tens":
            if last in ("unit", "tens"):
                flush()
            current += value
        elif kind == "hundreds":
            if last in ("unit", "tens", "hundreds"):
                flush()
            current += value
        elif kind == "scale100":
            current = (current or 1.0) * value
        else:                                     # thousand, million
            total += (current or 1.0) * value
            current = 0.0
        last = kind
    flush()
    return found


def _checked(answer: dict, text: str, missing: List[str]) -> ChatValues:
    """the LLM's values that pass the checks of the module doc"""
    from .slx2py.model_converter import workspace
    normalized = (text or "").translate(workspace._CHAT_CHARACTERS)
    in_message = set(_numbers(normalized)) | word_numbers(normalized)
    flat_message = " ".join(normalized.split())
    result = ChatValues(source=LLM_SOURCE)
    lines = []
    for item in answer.get("values") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip().translate(workspace._CHAT_CHARACTERS)
        quote = " ".join(str(item.get("quote") or "").split())
        if name not in missing:
            result.rejected.append("%s: not one of the missing variables" % (name or "?"))
            continue
        try:
            workspace.evaluate(value, {})                 # a plain value: numbers, no names
        except Exception:                                 # noqa: BLE001
            result.rejected.append("%s = %s: not a plain MATLAB value" % (name, value))
            continue
        numbers = _numbers(value)
        if not numbers or any(n not in in_message for n in numbers):
            result.rejected.append("%s = %s: a number that is not in the message" % (name, value))
            continue
        if any(line.startswith(name + " = ") for line in lines):
            continue
        lines.append("%s = %s;" % (name, value))
        shown = quote if quote and quote.translate(workspace._CHAT_CHARACTERS) in flat_message else ""
        result.read.append("%s = %s%s" % (name, value, "  (from «%s»)" % shown if shown else ""))
    result.unclear = [" ".join(str(u).split()) for u in answer.get("unclear") or [] if str(u).strip()]
    result.script = "\n".join(lines)
    return result

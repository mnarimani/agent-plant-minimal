"""Every prompt of the app, as YAML files in this folder - the code holds none (README.md says
which file is used where, and when).

    chat/       the plant-model chat agent: its system prompt and metadata prompt, the notes the
                app adds to a turn
    search/     document search (RAG) and web search
    simulink/   the Simulink part: what the agent is told about an upload (upload_guidance), the
                LLM that reads values from the chat (chat_values), the c2py agent (c2py_agent)

    load("simulink/chat_values")              -> the whole file (a dict)
    text("simulink/chat_values", "system")    -> one prompt (str); a dotted key reaches into a
                                                 mapping: text(name, "rule7.free")
    fill(template, file="x.slx", ...)         -> {{file}} ... replaced by the values

Placeholders are {{name}}; the prompts of the c2py agent (from the C_to_Python project) keep their
Python {name} placeholders, filled with str.format.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> Dict[str, Any]:
    """``name``: the path of the file in this folder, without .yaml (e.g. "chat/plant_model_agent")"""
    with (PROMPTS_DIR / ("%s.yaml" % name)).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def text(name: str, key: str) -> str:
    value: Any = load(name)
    for part in key.split("."):
        value = value[part]
    if not isinstance(value, str):
        raise KeyError("%s: %s is not a text" % (name, key))
    return value


_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def fill(template: str, **values: Any) -> str:
    """{{name}} -> str(value), in one pass (a value is never searched for placeholders itself).
    A placeholder of the template without a value is an error, never sent to an LLM."""
    missing = sorted(set(_PLACEHOLDER.findall(template)) - set(values))
    if missing:
        raise KeyError("prompt placeholder(s) without a value: %s" % ", ".join(missing))
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)

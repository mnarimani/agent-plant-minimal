"""Every prompt is in backend_core/prompts/*.yaml (see its README.md), loaded by backend_core.prompts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_core import prompts  # noqa: E402

FILES = sorted(p.relative_to(prompts.PROMPTS_DIR).with_suffix("").as_posix()
               for p in prompts.PROMPTS_DIR.rglob("*.yaml"))


def test_the_prompt_files_are_the_ones_the_readme_lists():
    readme = (prompts.PROMPTS_DIR / "README.md").read_text(encoding="utf-8")
    assert FILES == sorted(["chat/agent_notes", "chat/plant_model_agent", "chat/plant_model_metadata",
                            "search/rag", "search/websearch", "simulink/c2py_agent", "simulink/chat_values",
                            "simulink/upload_guidance"])
    for name in FILES:
        assert name.split("/")[-1] + ".yaml" in readme and "`%s`" % name in readme, name


@pytest.mark.parametrize("name", FILES)
def test_every_prompt_file_loads(name):
    data = prompts.load(name)
    assert isinstance(data, dict) and data


def test_fill_needs_every_placeholder_and_fills_in_one_pass():
    assert prompts.fill("a {{x}} b", x="{{y}}") == "a {{y}} b"
    with pytest.raises(KeyError):
        prompts.fill("{{missing}}")


def test_no_long_prompt_text_is_left_in_the_code_that_sends_prompts():
    """the modules that talk to an LLM keep no long text of their own"""
    for path in ("backend_core/agent.py", "backend_core/rag.py", "backend_core/websearch.py",
                 "backend_core/simulink/intake.py", "backend_core/simulink/direct/chat_values.py",
                 "backend_core/simulink/c2py/agent.py"):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert len(node.value) < 400 or "\n\n" not in node.value, (path, node.value[:80])

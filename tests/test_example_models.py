"""Every model of examples/simulink_models/ converts directly from its .slx (route 1), with its checks.

    python -m pytest tests/test_example_models.py -q        (about 40 s)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_core.simulink.direct import slx_direct  # noqa: E402

MODELS = sorted((ROOT / "examples" / "simulink_models").glob("*.slx"))


def test_the_examples_are_there():
    assert len(MODELS) >= 19


@pytest.mark.parametrize("path", MODELS, ids=[p.stem for p in MODELS])
def test_the_example_converts_and_passes_its_checks(path, tmp_path):
    r = slx_direct.convert_slx(path.name, path.read_bytes(), tmp_path, log=lambda m: None)
    assert r.ok, r.reason
    assert all(step.ok for step in r.steps), [s.label for s in r.steps if not s.ok]
    md = r.draft["metadata"]
    assert md["states"]
    if path.stem.endswith("_no_ports"):              # the app first asks for the choice on the diagram
        assert r.no_input and not md["inputs"] and not md["outputs"]
    else:
        assert r.no_input is None and md["inputs"] and md["outputs"]
        assert "def output(t, x, u)" in r.draft["python_code"]

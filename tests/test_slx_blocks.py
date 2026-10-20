"""Every block of the slx2py engine against Simulink itself (see slx_conformance.py).

    python -m pytest tests/test_slx_blocks.py          # group A: must match Simulink
    python tests/slx_conformance.py                    # the table of every case, groups B and C too

Groups A (scalar blocks) and B (vector signals) are supported and must match, block model and
extracted equations. Group C (discrete memory): the blocks in SUPPORTED_C must match too, the
others must be refused with a reason, never converted into something that differs from Simulink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slx_conformance as conformance  # noqa: E402

pytestmark = pytest.mark.skipif(not conformance.REFERENCE.is_file(), reason="no Simulink reference data")


def _names(group):
    return [c["name"] for c in conformance.cases(group)] if conformance.REFERENCE.is_file() else []


@pytest.mark.parametrize("name", _names("A") + _names("B"))
def test_supported_block_matches_simulink(name, tmp_path):
    result = conformance.run_case(name, tmp_path)
    assert result["status"] == "pass", result
    assert result["dynamics"] is not None, "the equations were not extracted"


SUPPORTED_C = {"b_memory", "b_derivative", "b_unit_delay", "b_zero_order_hold", "b_discrete_integrator",
               "b_discrete_transfer_fcn", "b_discrete_state_space", "b_rate_limiter", "b_relay", "b_backlash"}


def _supported_c(name):
    return name in SUPPORTED_C or name.startswith("b_inh_")     # inherited sample times (-1), all checked


@pytest.mark.parametrize("name", [n for n in _names("C") if _supported_c(n)])
def test_discrete_block_matches_simulink(name, tmp_path):
    result = conformance.run_case(name, tmp_path)
    assert result["status"] == "pass", result
    assert result["dynamics"] is not None, "the equations (with their update) were not extracted"


@pytest.mark.parametrize("name", [n for n in _names("C") if not _supported_c(n)])
def test_unsupported_block_is_refused(name, tmp_path):
    result = conformance.run_case(name, tmp_path)
    assert result["status"] == "unsupported", result


def _inherited_cases():
    if not conformance.REFERENCE.is_file():
        return []
    return [c for c in conformance.manifest()["cases"] if c["name"].startswith("b_inh_") and c.get("sample_times", {}).get("B")]


@pytest.mark.parametrize("case", _inherited_cases(), ids=lambda case: case["name"])
def test_inherited_sample_time_is_the_one_simulink_compiles(case, tmp_path):
    """the rate the engine gives a discrete block with sample time -1 = Simulink's CompiledSampleTime
    (also for the cases whose outputs Simulink logged only at their hits)"""
    import json
    import re
    import zipfile
    with zipfile.ZipFile(conformance.REFERENCE) as z:
        (tmp_path / "m.slx").write_bytes(z.read(conformance.FOLDER + case["name"] + ".slx"))
    conformance.pipeline.convert_slx(tmp_path / "m.slx", tmp_path / "x" / "m", tmp_path / "out")
    notes = json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))["warnings"]
    ours = [float(re.search(r"\): ([0-9.e+-]+) s", n).group(1)) for n in notes if "'B' inherits" in n]
    simulink = float(case["sample_times"]["B"].strip("[]").split()[0])
    assert ours and ours[0] == pytest.approx(simulink, abs=1e-12), (ours, simulink)


# ---- whole models (slx_models_check.py): every model the engine converts must match Simulink
import slx_models_check as models_check  # noqa: E402


def _model_cases():
    return models_check.load(models_check.DEFAULT)[0] if models_check.DEFAULT.is_file() else []


@pytest.mark.parametrize("case", _model_cases(), ids=lambda case: case["name"])
def test_converted_model_matches_simulink(case, tmp_path):
    _, files = models_check.load(models_check.DEFAULT)
    result = models_check.run_case(case, files, tmp_path)
    assert not result.startswith("FAIL"), result

"""Values of missing MATLAB variables read from a chat message (backend_core/simulink/direct/chat_values.py).

The parser reads every ``name = value``; an LLM only reads values given in words, and nothing it
returns is taken unchecked: the name must be a missing one, the value a plain MATLAB value, and
every number in it must be in the message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_core.simulink.direct import chat_values  # noqa: E402

MISSING = ["m", "ks", "cs"]
USES = {"m": ["Gain '1/m', parameter Gain"], "ks": ["Gain 'Spring', parameter Gain"], "cs": ["Gain 'Damper'"]}


class FakeLLM:
    """answers with a fixed JSON object and records what it was asked"""

    def __init__(self, answer) -> None:
        self.answer, self.calls = answer, []

    def __call__(self, messages):
        self.calls.append(messages)
        if isinstance(self.answer, Exception):
            raise self.answer
        return json.dumps(self.answer) if isinstance(self.answer, dict) else self.answer


def test_name_equals_value_is_read_by_the_parser_without_the_llm():
    llm = FakeLLM({"values": []})
    r = chat_values.read_values("m = 1200 و ks = ۳۵۰۰۰ و cs = 5", MISSING, USES, llm)
    assert r.source == chat_values.PARSER_SOURCE and not llm.calls       # every missing one is there
    assert r.script.splitlines() == ["m = 1200;", "ks = 35000;", "cs = 5;"]
    r = chat_values.read_values("m = 1200 و ks = ۳۵۰۰۰", MISSING, USES, llm)
    assert r.source == chat_values.PARSER_SOURCE and len(llm.calls) == 1  # asked for cs only, found none
    assert "- cs:" in llm.calls[0][0]["content"] and "- m:" not in llm.calls[0][0]["content"]
    assert r.script.splitlines() == ["m = 1200;", "ks = 35000;"]


def test_values_in_words_are_read_by_the_llm_with_where_they_come_from():
    text = "جرم ماشین ۱۲۰۰ کیلوگرم است و سختی فنر 35000"
    llm = FakeLLM({"values": [{"name": "m", "value": "1200", "quote": "جرم ماشین ۱۲۰۰ کیلوگرم"},
                              {"name": "ks", "value": "35000", "quote": "سختی فنر 35000"}], "unclear": []})
    r = chat_values.read_values(text, MISSING, USES, llm)
    assert r.source == chat_values.LLM_SOURCE and r.script.splitlines() == ["m = 1200;", "ks = 35000;"]
    assert r.read[0] == "m = 1200  (from «جرم ماشین ۱۲۰۰ کیلوگرم»)"
    prompt = llm.calls[0][0]["content"]
    assert "json" in prompt and "- ks: used in Gain 'Spring', parameter Gain" in prompt


def test_a_number_that_is_not_in_the_message_is_never_taken():
    llm = FakeLLM({"values": [{"name": "m", "value": "1.2", "quote": "۱٫۲ تن"},           # in the message
                              {"name": "ks", "value": "1200", "quote": "۱٫۲ تن"}]})       # converted: refused
    r = chat_values.read_values("جرم ۱٫۲ تن است", MISSING, USES, llm)
    assert r.script == "m = 1.2;"
    assert any("ks = 1200" in x and "not in the message" in x for x in r.rejected)


def test_only_missing_names_and_plain_values_are_taken():
    llm = FakeLLM({"values": [{"name": "g", "value": "9.81"},                            # not missing
                              {"name": "m", "value": "mass_of_car"},                     # not a value
                              {"name": "cs", "value": "[1 2]"}]})
    r = chat_values.read_values("g is 9.81, damping 1 and 2", MISSING, USES, llm)
    assert r.script == "cs = [1 2];"
    assert len(r.rejected) == 2


def test_unclear_values_go_back_to_the_agent():
    llm = FakeLLM({"values": [], "unclear": ["m: given in tons (1.2 t)"]})
    r = chat_values.read_values("جرم ۱٫۲ تن", MISSING, USES, llm)
    assert not r.script and r.unclear == ["m: given in tons (1.2 t)"]


def test_an_llm_that_fails_or_answers_nonsense_gives_no_values():
    for answer in (RuntimeError("no network"), "not json", {"values": "nope"}):
        r = chat_values.read_values("the mass is 1200", MISSING, USES, FakeLLM(answer))
        assert not r.script


def test_no_llm_is_asked_when_nothing_waits_for_values():
    llm = FakeLLM({"values": [{"name": "m", "value": "1200"}]})
    assert not chat_values.read_values("the mass is 1200", [], USES, llm).script and not llm.calls


def test_the_conversion_says_where_each_missing_variable_is_used(tmp_path):
    import test_slx_workspace as T
    from backend_core.simulink.direct import slx_direct
    r = slx_direct.convert_slx("tst5.slx", T._with_variables(), tmp_path, log=lambda m: None)
    assert r.variable_uses["k"] == ["Gain 'Gain', parameter Gain"]
    assert r.variable_uses["num"] == ["TransferFcn 'Transfer Fcn', parameter Numerator"]
    assert r.variable_uses["den"] == ["TransferFcn 'Transfer Fcn'"]


def test_numbers_written_in_words_are_numbers_of_the_message():
    """'m is two': the LLM reads m = 2, and 2 is in the message - in words (English or Persian)"""
    llm = FakeLLM({"values": [{"name": "m", "value": "2", "quote": "m is two"}], "unclear": []})
    r = chat_values.read_values("m is two", MISSING, USES, llm)
    assert r.script == "m = 2;" and not r.rejected
    llm = FakeLLM({"values": [{"name": "m", "value": "1200", "quote": "هزار و دویست"},
                              {"name": "ks", "value": "2.5", "quote": "دو ممیز پنج"}]})
    r = chat_values.read_values("جرم هزار و دویست و سختی دو ممیز پنج", MISSING, USES, llm)
    assert r.script.splitlines() == ["m = 1200;", "ks = 2.5;"]
    llm = FakeLLM({"values": [{"name": "m", "value": "3", "quote": "m is two"}]})       # invented: refused
    assert chat_values.read_values("m is two", MISSING, USES, llm).script == ""


def test_word_numbers():
    w = chat_values.word_numbers
    assert w("twenty-five") == {25.0} and w("two point five") == {2.5} and w("one two three") == {1.0, 2.0, 3.0}
    assert w("twelve thousand five hundred") == {12500.0} and w("half") == {0.5}
    assert w("بیست و پنج") == {25.0} and w("دوازده هزار و پانصد") == {12500.0}
    assert w("جرم دو کیلوگرم و سختی سیصد و پنجاه") == {2.0, 350.0}
    assert w("جرم ۱٫۲ تن است") == set()


def test_the_llm_reads_what_the_parser_did_not():
    """'m = 1200 and the spring is thirty five thousand': the parser takes m, the LLM is asked for ks and cs only"""
    llm = FakeLLM({"values": [{"name": "ks", "value": "35000", "quote": "the spring is thirty five thousand"}]})
    r = chat_values.read_values("m = 1200 and the spring is thirty five thousand", MISSING, USES, llm)
    assert r.script.splitlines() == ["m = 1200;", "ks = 35000;"]
    assert r.source == chat_values.LLM_SOURCE and r.read[0] == "m = 1200"
    prompt = llm.calls[0][0]["content"]
    assert "- ks:" in prompt and "- cs:" in prompt and "- m:" not in prompt

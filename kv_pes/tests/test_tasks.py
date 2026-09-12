"""Unit tests for task construction and grading (Phase 2 correctness gate)."""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kv_pes.tasks import (build_case, grade_answer, grade_multi,
                          needle_in_haystack, multi_fact_qa)


def test_needle_deterministic_per_seed():
    a = needle_in_haystack(random.Random(7), depth=0.5)
    b = needle_in_haystack(random.Random(7), depth=0.5)
    assert a.prompt == b.prompt and a.answer == b.answer


def test_needle_answer_present_and_gradable():
    for depth in (0.1, 0.5, 0.9):
        c = needle_in_haystack(random.Random(1), depth=depth)
        assert f"magic number for {c.key} is {c.answer}" in c.prompt
        assert grade_answer(f"The magic number is {c.answer}.", c.answer)
        assert not grade_answer("I do not know.", c.answer)
        wrong = str(int(c.answer) + 1)
        assert not grade_answer(f"It is {wrong}.", c.answer)


def test_needle_depth_control():
    c0 = needle_in_haystack(random.Random(3), depth=0.02)
    c1 = needle_in_haystack(random.Random(3), depth=0.98)
    assert c0.prompt.find("magic number") < len(c0.prompt) * 0.15
    assert c1.prompt.find("magic number") > len(c1.prompt) * 0.85


def test_distractors_same_surface_form():
    c = needle_in_haystack(random.Random(5), depth=0.1, distractors=6)
    n_needles = len(re.findall(r"magic number for", c.prompt))
    assert n_needles == 7  # 1 real + 6 distractors
    assert c.meta["distractors"] == 6
    # real needle appears before every distractor (delayed-relevance probe)
    first_fakes = re.findall(r"magic number for (\w+)", c.prompt)
    assert first_fakes[0] == c.key


def test_multifact_structure():
    c = multi_fact_qa(random.Random(11), k=8, ask_all=False)
    assert c.meta["k"] == 8
    assert c.answer.isdigit()
    assert len(re.findall(r"magic number for", c.prompt)) == 8
    c_all = multi_fact_qa(random.Random(11), k=8, ask_all=True)
    assert ":" in c_all.answer


def test_grade_multi():
    assert grade_multi("magpie: 123, lantern: 456", "magpie: 123, lantern: 456") == 1.0
    assert grade_multi("magpie: 123", "magpie: 123, lantern: 456") == 0.5
    assert grade_multi("The number is 777.", "777") == 1.0


def test_build_case_dispatch():
    for task in ["needle_early", "needle_mid", "needle_late", "delayed_needle", "multifact"]:
        c = build_case(task, seed=0, difficulty=0)
        assert c.prompt and c.answer
    c = build_case("delayed_needle", seed=0, difficulty=8)
    assert c.meta["distractors"] >= 4

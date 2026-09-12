"""Long-context evaluation tasks.

Task 1 -- needle_in_haystack: long filler text with a single key-value needle
inserted at a controlled depth; question at the end; graded by exact match.

Task 2 -- multi_fact_qa: k key-value facts sprinkled at varied depths in long
filler; one (or all) asked at the end; graded per fact. This task stresses
*nonlinear credit assignment*: many facts must survive, and the attention value
of any single fact is low most of the time -- exactly the regime the paper's
Sec. 7.4 predicts value-based pricing should degrade in.

Task 3 -- delayed_needle (boundary probe): the needle appears early, followed
by a long stretch of attention-grabbing distractor "fake needles" (same surface
form, wrong keys). An EMA/recency value signal down-weights the early needle;
a cumulative signal (H2O) keeps it. This is the deliberate attempt to break PES.

All text is generated deterministically from a seed (no external downloads).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

FILLER_SENTENCES = [
    "The old bridge crossed the river just north of the mill.",
    "Every autumn the town repainted the benches in the square.",
    "She kept a ledger of rainfall going back thirty years.",
    "The bakery on Elm Street opened before sunrise each day.",
    "Trains stopped running on that line shortly after the war.",
    "A colony of herons nested in the tallest poplar.",
    "The library's reading room kept lamps lit until midnight.",
    "Local farmers argued about drainage for an entire decade.",
    "The clock above the gate ran four minutes fast in winter.",
    "Children raced paper boats along the flooded gutter.",
    "The orchard yielded more plums than anyone could sell.",
    "A fiddler played on the pier every Sunday evening.",
    "The quarry was flooded and stocked with perch in 1962.",
    "Fog from the bay softened the outline of the hills.",
    "The postmistress knew every recipient by nickname.",
    "Bicycles outnumbered cars on the market square for years.",
    "The school's brass band rehearsed in the grain hall.",
    "Letters were left on the ledge when the tide was high.",
]

FAKE_KEYS = ["magpie", "lantern", "harbor", "satchel", "meadow", "whistle",
             "cobblestone", "ember", "thistle", "weathervane"]

ANSWER_RE = re.compile(r"-?\d+")


@dataclass
class NeedleCase:
    prompt: str
    question_len: int          # tokens at the end that must never be evicted
    key: str
    answer: str
    needle_char_pos: float     # 0..1 relative depth of the needle in the prompt
    n_facts: int = 1
    case_id: str = ""
    meta: dict = field(default_factory=dict)


def _filler(rng: random.Random, n_words: int) -> str:
    out = []
    while sum(len(s.split()) for s in out) < n_words:
        out.append(rng.choice(FILLER_SENTENCES))
    return " ".join(out)




def _fmt_answer(value: int) -> str:
    return str(value)


def grade_answer(text: str, answer: str) -> bool:
    """Exact-match grading: the answer must appear as the (only) number."""
    nums = ANSWER_RE.findall(text.strip())
    return len(nums) >= 1 and nums[0] == answer


def needle_in_haystack(rng: random.Random, depth: float, n_filler_words: int = 600,
                       distractors: int = 0,
                       question_first: str = "What is the magic number for {key}?",
                       value_range=(100, 999)) -> NeedleCase:
    """depth in [0,1]: 0 = needle at the very start, 1 = just before the question.
    `distractors` adds fake needles with other keys (same surface form) after
    the needle to raise difficulty (boundary-mapping knob)."""
    key = rng.choice(FAKE_KEYS)
    value = rng.randint(*value_range)
    needle = f" The magic number for {key} is {value}."
    body = _filler(rng, n_filler_words)
    insert_at = int(len(body) * depth)
    body = body[:insert_at] + needle + body[insert_at:]
    # distractors spread over the region AFTER the needle
    step = max(1, (len(body) - insert_at) // (distractors + 1))
    last = insert_at
    for _ in range(distractors):
        fk = rng.choice([k for k in FAKE_KEYS if k != key])
        fv = rng.randint(*value_range)
        f = f" The magic number for {fk} is {fv}."
        at = min(len(body), last + step)
        body = body[:at] + f + body[at:]
        last = at + len(f)
    q = f" Question: {question_first.format(key=key)} Answer with just the number. Answer:"
    prompt_text = body + q
    return NeedleCase(prompt=prompt_text, question_len=max(8, len(q) // 4),
                      key=key, answer=_fmt_answer(value), needle_char_pos=depth,
                      meta={"distractors": distractors, "n_filler_words": n_filler_words})


def multi_fact_qa(rng: random.Random, k: int = 8, n_filler_words: int = 600,
                  ask_all: bool = False) -> NeedleCase:
    """k facts at random depths; question about a random (or all) fact(s)."""
    keys = rng.sample(FAKE_KEYS, min(k, len(FAKE_KEYS)))
    facts = [(key, rng.randint(100, 999)) for key in keys]
    body = _filler(rng, n_filler_words)
    positions = sorted(rng.sample(range(len(body)), k))
    for (key, val), at in zip(facts, reversed(positions)):
        body = body[:at] + f" The magic number for {key} is {val}." + body[at:]
    if ask_all:
        q = (" Question: For every object mentioned above, what is its magic number? "
             "Answer as a list of 'key: number' pairs. Answer:")
        answers = ", ".join(f"{key}: {val}" for key, val in facts)
    else:
        key, val = facts[rng.randrange(len(facts))]
        q = f" Question: What is the magic number for {key}? Answer with just the number. Answer:"
        answers = _fmt_answer(val)
    prompt_text = body + q
    return NeedleCase(prompt=prompt_text, question_len=max(8, len(q) // 4),
                      key=key if not ask_all else "ALL", answer=answers,
                      needle_char_pos=-1.0, n_facts=k,
                      meta={"k": k, "ask_all": ask_all})


def grade_multi(text: str, answer: str) -> float:
    """Fraction of 'key: value' pairs recovered (used when ask_all=True)."""
    if ":" in answer:
        want = [tuple(p.strip().split(":")) for p in answer.split(",")]
        got = dict(re.findall(r"(\w+)\s*:\s*(\d+)", text))
        correct = sum(1 for k, v in want if got.get(k) == v.strip())
        return correct / len(want)
    return float(grade_answer(text, answer))


def tokenize_case(case: NeedleCase, tokenizer) -> NeedleCase:
    """Tokenize the prompt and set the exact question token count (tokens at the
    end of the prompt that the cache must protect from eviction)."""
    ids = tokenizer(case.prompt)["input_ids"]
    q_ids = tokenizer(_question_suffix(case), add_special_tokens=False)["input_ids"]
    case.question_len = max(1, len(q_ids))
    case.meta["n_prompt_tokens"] = len(ids)
    case.meta["prompt_token_ids"] = ids
    return case


def _question_suffix(case: NeedleCase) -> str:
    i = case.prompt.find(" Question:")
    return case.prompt[i:] if i >= 0 else case.prompt[-60:]


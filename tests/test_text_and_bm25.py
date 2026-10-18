import json
import math

import numpy as np
import pytest

from council_rag.bm25 import BM25Index
from council_rag.text import contains_normalized, s_stem, tokenize
from tests.conftest import FIXTURES

CASES = json.loads((FIXTURES / "tokenizer_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:24] for c in CASES])
def test_tokenizer_shared_cases(case: dict) -> None:
    # the same cases are checked against docs/assets/search.js by scripts/check_search_parity.mjs
    assert tokenize(case["text"]) == case["tokens"]


def test_s_stemmer_rules() -> None:
    assert s_stem("policies") == "policy"
    assert s_stem("cases") == "case"
    assert s_stem("bus") == "bus"
    assert s_stem("grass") == "grass"
    assert s_stem("toes") == "toe"  # "oes" is exempt from the "es" rule
    assert s_stem("agrees") == "agree"


def test_contains_normalized_ignores_case_quotes_and_spacing() -> None:
    assert contains_normalized("St. John\u2019s  Council", "st. john's council")
    assert contains_normalized("Notices Published \u2013 12 Example", "published - 12 example")
    assert not contains_normalized("Council approved", "council rejected")


def test_bm25_matches_hand_computed_score() -> None:
    docs = ["snow clearing bylaw", "parking bylaw bylaw", "community garden"]
    index = BM25Index(k1=1.2, b=0.75).fit(docs)
    scores = index.scores("bylaw")
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avgdl = (3 + 3 + 2) / 3
    expected0 = idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
    expected1 = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl))
    assert scores[0] == pytest.approx(expected0, rel=1e-5)
    assert scores[1] == pytest.approx(expected1, rel=1e-5)
    assert scores[2] == 0
    assert [i for i, _ in index.top_k("bylaw", 5)] == [1, 0]


def test_bm25_mask_and_unknown_terms() -> None:
    index = BM25Index().fit(["alpha beta", "alpha gamma", "delta"])
    mask = np.array([False, True, True])
    assert [i for i, _ in index.top_k("alpha", 5, mask)] == [1]
    assert index.top_k("zzz", 5) == []
    assert BM25Index().fit([]).top_k("alpha", 3) == []

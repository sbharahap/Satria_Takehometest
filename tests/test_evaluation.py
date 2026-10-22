import pytest

from scripts.evaluate_rag import summarize, validate_judgment


def test_judge_scores_must_match_all_retrieved_contexts():
    body = {
        "answer_relevance": 3,
        "groundedness": 4,
        "context_relevance": [{"id": "S1", "score": 2}],
    }
    assert validate_judgment(body, ["S1"]) is body
    with pytest.raises(ValueError):
        validate_judgment(body, ["S1", "S2"])
    body["answer_relevance"] = True
    with pytest.raises(ValueError):
        validate_judgment(body, ["S1"])


def test_abstention_and_judge_failures_are_not_counted_as_perfect_grounding():
    records = [
        {
            "agent": "hr",
            "expected_abstention": False,
            "abstained": False,
            "metrics": {"answer_relevance": 0.5, "groundedness": 0.75, "context_relevance": 0.25},
        },
        {
            "agent": "hr",
            "expected_abstention": True,
            "abstained": True,
            "citations": [],
            "metrics": {"answer_relevance": 1, "groundedness": None, "context_relevance": 0},
        },
        {"agent": "legal", "expected_abstention": False, "judge_error": "ValueError"},
    ]
    overall = summarize(records)["all"]
    assert overall["cases"] == 3 and overall["judged"] == 2
    assert overall["metrics"]["groundedness"] == {"mean": 0.75, "n": 1}
    assert overall["answerable_metrics"]["answer_relevance"]["mean"] == 0.5
    assert overall["expected_abstention"] == {"correct": 1, "n": 1}


def test_keyed_judge_contexts_are_normalized_without_inventing_scores():
    body = {
        "answer_relevance": 2,
        "groundedness": None,
        "context_relevance": {"S1": {"score": 1, "reason": "Hanya satu topik."}},
    }
    result = validate_judgment(body, ["S1"])
    assert result["context_relevance"] == [{"id": "S1", "score": 1, "reason": "Hanya satu topik."}]

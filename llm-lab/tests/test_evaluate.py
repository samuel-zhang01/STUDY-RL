import pytest

from llm_lab import evaluate


def test_counts_complete_correct_answers_not_just_format(monkeypatch):
    outputs = iter([
        {"text": "7", "terminated": True, "generated_tokens": 2},
        {"text": "8", "terminated": True, "generated_tokens": 2},
        {"text": "7", "terminated": False, "generated_tokens": 12},
        {"text": "The answer is 7", "terminated": True, "generated_tokens": 6},
    ])
    monkeypatch.setattr(evaluate, "generate_completion_details", lambda *args: next(outputs))
    report = evaluate.evaluate_rows(None, None, [{"prompt": "3+4", "completion": "7"}] * 4)
    assert report["exact_matches"] == 1
    assert report["exact_match_rate"] == 0.25
    assert report["format_rate"] == 0.5
    assert report["length_cutoffs"] == 1


def test_empty_evaluation_fails():
    with pytest.raises(ValueError, match="at least one"):
        evaluate.evaluate_rows(None, None, [])

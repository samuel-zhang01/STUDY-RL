import hashlib
import json

import pytest

from llm_lab.data import build_splits, score_completion, write_splits


def test_splits_are_reproducible_disjoint_and_correct():
    splits = build_splits()
    assert splits == build_splits()
    assert splits != build_splits(17)
    seen = set()
    for split, count in (("train", 192), ("validation", 32), ("test", 32)):
        rows = splits[f"sft_{split}"]
        groups = {tuple(sorted(row["operands"])) for row in rows}
        assert len(rows) == len(groups) == count
        assert not seen.intersection(groups)
        seen.update(groups)
        for row in rows:
            assert str(sum(row["operands"])) == row["completion"]
    for split in ("train", "validation"):
        assert [r["id"] for r in splits[f"sft_{split}"]] == [r["id"] for r in splits[f"preferences_{split}"]]
        for row in splits[f"preferences_{split}"]:
            assert score_completion(row["chosen"], row["chosen"])["exact_match"]
            assert not score_completion(row["rejected"], row["chosen"])["exact_match"]
            assert score_completion(row["rejected"], row["chosen"])["valid_format"]


@pytest.mark.parametrize("text,valid,correct", [
    ("7", True, True), (" 7\n", True, True), ("8", True, False),
    ("The answer is 7", False, False), ("7 or 8", False, False),
    ("7.0", False, False), ("07", False, False), ("７", False, False),
    ("", False, False), ("__import__('os').system('bad')", False, False),
])
def test_verifier_rejects_reward_shortcuts(text, valid, correct):
    assert score_completion(text, "7") == {"valid_format": valid, "exact_match": correct}


def test_manifest_hashes_exact_bytes(tmp_path):
    manifest = write_splits(tmp_path)
    for name, info in manifest["files"].items():
        contents = (tmp_path / name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == info["sha256"]
        assert len([json.loads(line) for line in contents.splitlines()]) == info["rows"]

"""Invalid environment controls fail before expensive work starts."""

import pytest

from solution.hybrid_project.config import Settings


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOP_K", "0"),
        ("TOP_K", "-1"),
        ("MAX_SEQUENCE_LENGTH", "0"),
        ("MAX_NEW_TOKENS", "0"),
        ("TRAIN_EPOCHS", "0"),
        ("TRAIN_EPOCHS", "-1"),
        ("LEARNING_RATE", "0"),
        ("LEARNING_RATE", "-0.1"),
        ("LEARNING_RATE", "nan"),
        ("LEARNING_RATE", "inf"),
        ("MIN_SCORE", "nan"),
        ("MIN_SCORE", "inf"),
        ("MIN_SCORE", "1.1"),
        ("MIN_SCORE", "-1.1"),
        ("RANDOM_SEED", "-1"),
        ("RANDOM_SEED", str(2**32)),
    ],
)
def test_invalid_setting_has_actionable_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


@pytest.mark.parametrize("score", ["-1", "0", "1"])
def test_valid_cosine_thresholds(monkeypatch, score):
    monkeypatch.setenv("MIN_SCORE", score)
    assert Settings.from_env().min_score == float(score)

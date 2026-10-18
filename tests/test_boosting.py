import numpy as np
import pytest

from honest_model.boosting import GradientBoostingScratch, find_best_stump


def test_known_answer_stump():
    # Tiny dataset where the right cut-off and values are known
    X = np.array([[1], [2], [3], [4]])
    residuals = np.array([-3, -1, 0, 4])
    stump = find_best_stump(X, residuals)

    assert stump["feature_index"] == 0
    # Cut 3.5 leaves squared error 4.67; cut 2.5 would leave 10, so 3.5 must win
    assert stump["threshold"] == pytest.approx(3.5)
    assert stump["left_value"] == pytest.approx(-4 / 3)  # mean of [-3, -1, 0]
    assert stump["right_value"] == pytest.approx(4.0)  # mean of [4]


def test_refit_resets_stumps():
    # Fit twice and check stump count is not doubled
    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])
    model = GradientBoostingScratch(n_estimators=5, learning_rate=0.1)

    model.fit(X, y)
    assert len(model.stumps) == 5

    model.fit(X, y)
    assert len(model.stumps) == 5  # mistake #19 guarded forever

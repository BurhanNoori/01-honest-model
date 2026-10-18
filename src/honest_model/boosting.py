"""
A module where we are learning the implementation of gradient boosting from scratch.
Here we will take a small dataset from scikit-learn load_diabetes
and implement gradient boosting on it.
We will also compare the results with sklearn's implementation of gradient boosting.

Note: This is a toy implementation and is not optimized for performance.
We are doing GradientBoostingRegressor because analysing the loss function
for regression is easier than classification.
We will implement gradient boosting for classification on Day 11.
"""

import numpy as np


def find_best_stump(X, residuals):
    """
    Fit a decision stump (one-level tree) to residuals.
    Each side predicts the mean residual, minimizing squared error.

    Parameters:
    X (numpy.ndarray): Input features
    residuals (numpy.ndarray): Current residuals

    Returns:
    dict: {
        "feature_index": int,
        "threshold": float,
        "left_value": float,
        "right_value": float
    }
    """
    best_stump = {}
    min_error = float("inf")

    # Try every feature, because a stump may split on any one of them
    for feature_index in range(X.shape[1]):
        # Midpoints between sorted unique values, so neither side is ever empty
        values = np.unique(X[:, feature_index])
        thresholds = (values[:-1] + values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask

            # Each side predicts its mean residual, the value that minimises squared error
            left_mean = residuals[left_mask].mean()
            right_mean = residuals[right_mask].mean()

            predictions = np.where(left_mask, left_mean, right_mean)
            error = np.sum((residuals - predictions) ** 2)

            if error < min_error:
                min_error = error
                best_stump = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "left_value": left_mean,
                    "right_value": right_mean,
                }

    return best_stump


def predict_stump(stump, X):
    """
    Predict new values using a fitted stump.

    Parameters:
    stump (dict): feature_index, threshold, left_value, right_value
    X (numpy.ndarray): Input features

    Returns:
    numpy.ndarray: Predictions for each row
    """
    mask = X[:, stump["feature_index"]] <= stump["threshold"]
    return np.where(mask, stump["left_value"], stump["right_value"])


class GradientBoostingScratch:
    """
    Simple Gradient Boosting Regressor built from scratch.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.init_value = None

    def fit(self, X, y):
        # Reset state so refits don't pile up stumps
        self.stumps = []

        # Start from mean(y)
        self.init_value = np.mean(y)
        predictions = np.full_like(y, self.init_value, dtype=float)

        for _ in range(self.n_estimators):
            residuals = y - predictions
            stump = find_best_stump(X, residuals)
            if not stump:  # guard for degenerate case (no cut-offs)
                break
            self.stumps.append(stump)

            # Each stump corrects the current guess by a fraction (learning_rate)
            predictions += self.learning_rate * predict_stump(stump, X)

        return self  # sklearn convention

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_value, dtype=float)
        for stump in self.stumps:
            preds += self.learning_rate * predict_stump(stump, X)
        return preds

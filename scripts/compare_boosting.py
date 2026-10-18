"""
Day 10: compare the from-scratch gradient booster (src/honest_model/boosting.py)
with scikit-learn's GradientBoostingRegressor on the same data and settings.

Usage:
    uv run python scripts/compare_boosting.py
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from honest_model.boosting import GradientBoostingScratch

# Step 1: Load dataset
X, y = load_diabetes(return_X_y=True)

# Step 2: Train/test split (20% holdout)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train both models with matched settings
scratch_model = GradientBoostingScratch(n_estimators=50, learning_rate=0.1).fit(X_train, y_train)
# Each stump is a small correction step; max_depth=1 ensures gentle adjustments
sk_model = GradientBoostingRegressor(
    max_depth=1,
    criterion="squared_error",
    loss="squared_error",  # explicit for clarity
    n_estimators=50,
    learning_rate=0.1,
).fit(X_train, y_train)

# Step 4: Predictions
scratch_preds_train = scratch_model.predict(X_train)
scratch_preds_test = scratch_model.predict(X_test)

sk_preds_train = sk_model.predict(X_train)
sk_preds_test = sk_model.predict(X_test)

# Step 5: Compare errors
print("Scratch MSE (train):", mean_squared_error(y_train, scratch_preds_train))
print("Sklearn MSE (train):", mean_squared_error(y_train, sk_preds_train))
print("Scratch MSE (test):", mean_squared_error(y_test, scratch_preds_test))
print("Sklearn MSE (test):", mean_squared_error(y_test, sk_preds_test))

# Step 6: Largest difference between predictions
diff_train = np.max(np.abs(scratch_preds_train - sk_preds_train))
diff_test = np.max(np.abs(scratch_preds_test - sk_preds_test))
print("Max prediction difference (train):", diff_train)
print("Max prediction difference (test):", diff_test)

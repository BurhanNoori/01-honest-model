"""
This module will do the comparison between the single CV score and nested CV score
to report the optimism gap
The following steps will be performed:
  1. Single CV Score (GridSearchCV / RandomizedSearchCV):
      • Run a single 5-fold CV hyperparameter search across X_numeric.
      • Record the best mean score reported by the search (this is the optimistic score).
  2. Nested CV Score (cross_val_score + GridSearchCV / RandomizedSearchCV):
      • Run an outer 5-fold CV where each outer fold runs an inner hyperparameter search on
        the outer training set.
      • Record the mean score across the 5 outer test folds (this is the unbiased, honest score).
  3. Report the Optimism Gap:
      • Calculate Single CV Score - Nested CV Score and print the delta.
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from honest_model.memory import reduce_memory_usage

# Step 0: Load application_train
df = pd.read_csv("data/raw/application_train.csv")

# Step 1: Reduce memory usage of the DataFrame because
# we have to train the model multiple times during
# cross-validation and we want to avoid memory issues
df = reduce_memory_usage(df)

y = df["TARGET"]

# Select only numeric columns (integers and floats)
X_raw = df.drop(columns=["TARGET", "SK_ID_CURR"])
X_numeric = X_raw.select_dtypes(include=["number"])

# Step 2: Build the pipeline — imputer, scaler, and model as a single estimator
# median chosen over mean because mean is pulled by extreme outliers in financial data
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(random_state=42)),
    ]
)


# ------------------------------------------------------------------
#   Step 3: Single CV Score (GridSearchCV / RandomizedSearchCV)
# -------------------------------------------------------------------

param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [3, 5, 10],
    "model__learning_rate": [0.01, 0.1],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
)

# Fit grid_search on the entire dataset
grid_search.fit(X_numeric, y)

# Get the optimistic score reported by Single CV search
single_cv_optimistic_score = grid_search.best_score_


# -------------------------------------------------------------------------------
#   Step 4: Nested CV Score (cross_val_score + GridSearchCV / RandomizedSearchCV)
# -------------------------------------------------------------------------------

# Inner loop search (tunes hyperparameters on whatever subset of data it is given)
inner_cv = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
)

# Outer loop (splits data into 5 outer folds, passes outer train to inner_cv, tests on outer test)
nested_scores = cross_val_score(
    inner_cv,
    X_numeric,
    y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
)


# -------------------------------------------------------------------------------
#   Step 5: Report the Optimism Gap
# -------------------------------------------------------------------------------
nested_unbiased_score = nested_scores.mean()
optimism_gap = single_cv_optimistic_score - nested_unbiased_score

print("=" * 55)
print(f"Single CV (Optimistic) ROC-AUC:  {single_cv_optimistic_score:.5f}")
print(f"Nested CV (Unbiased) ROC-AUC:    {nested_unbiased_score:.5f}")
print(f"Optimism Gap (Single - Nested):  {optimism_gap:+.5f}")
print("=" * 55)

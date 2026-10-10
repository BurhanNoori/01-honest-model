"""
This module contains the validation logic for the model training process
Here we are using the various cross-validation techniques to validate the model and its performance
And check which cross-validation technique is best suited for our model and data
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    TimeSeriesSplit,
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

# Step 3: Fit on training data only — pipeline internally applies fit+transform in order
scores = cross_val_score(
    pipeline,
    X_numeric,
    y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
)

# Step 4: Print the cross-validation scores and their mean
print("=" * 55)
print(f"Cross-Validation KFold scores: {scores}")
print(f"Mean KFold ROC-AUC: {scores.mean():.5f} and Standard Deviation: {scores.std():.5f}")


# Repeat step 3 to 4 for different cross-validation techniques and compare the results
scores = cross_val_score(
    pipeline,
    X_numeric,
    y,
    cv=GroupKFold(n_splits=5, shuffle=True, random_state=42),
    groups=df["SK_ID_CURR"],
    scoring="roc_auc",
)

print("=" * 55)
print(f"Cross-Validation GroupKFold scores: {scores}")
print(f"Mean GroupKFold ROC-AUC: {scores.mean():.5f} and Standard Deviation: {scores.std():.5f}")

# TimeSeriesSplit
scores = cross_val_score(
    pipeline,
    X_numeric,
    y,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="roc_auc",
)

print("=" * 55)
print(f"Cross-Validation TimeSeriesSplit scores: {scores}")
print(
    f"Mean TimeSeriesSplit ROC-AUC: {scores.mean():.5f} and Standard Deviation: {scores.std():.5f}"
)

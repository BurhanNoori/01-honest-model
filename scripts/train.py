"""
In production, human error happens when people manually call .fit() and .transform()
across multiple files. To solve this permanently, scikit-learn provides the Pipeline object.

A Pipeline bundles a sequence of data transformers and a final model into a single,
unified estimator.
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 0: Load application_train
df = pd.read_csv("data/raw/application_train.csv")
y = df["TARGET"]

# Select only numeric columns (integers and floats)
X_raw = df.drop(columns=["TARGET", "SK_ID_CURR"])
X_numeric = X_raw.select_dtypes(include=["number"])

# Step 1: Split FIRST — the pipeline sees only training data when it fits
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

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
pipeline.fit(X_train, y_train)

# Step 4: Predict probability of default (column 1 = positive class)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Step 5: Evaluate and compare against the leaked naive baseline
NAIVE_LEAKED_AUC = 0.75061
honest_auc = roc_auc_score(y_test, y_pred_proba)
delta = honest_auc - NAIVE_LEAKED_AUC

print("=" * 55)
print(f"Honest Pipeline ROC-AUC:  {honest_auc:.5f}")
print(f"Naive (Leaked) Baseline:  {NAIVE_LEAKED_AUC:.5f}")
print(f"Delta (honest - leaked):  {delta:+.5f}")
print("=" * 55)

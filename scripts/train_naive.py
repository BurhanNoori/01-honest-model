"""
Deliberately Naive Baseline Training Pipeline (Day 6 - The Inflated Model).

Purpose:
    Demonstrates the standard "flawed" pipeline prevalent in online Kaggle notebooks:
    1. Pre-Split Preprocessing Leak: Imputer and Scaler are fit on the full dataset,
       allowing summary statistics from the test set to leak into training features.
    2. Random Split: Standard random train_test_split ignoring temporal/entity ordering.

Model Selection:
    LightGBM (LGBMClassifier) is used over linear scikit-learn models (LogisticRegression)
    because tabular credit risk data exhibits non-linear relationships (e.g. debt thresholds)
    and histogram binning allows training on 250k+ rows in <2 seconds.
"""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 0: Load application_train
df = pd.read_csv("data/raw/application_train.csv")
y = df["TARGET"]

# Select only numeric columns (integers and floats)
X_raw = df.drop(columns=["TARGET", "SK_ID_CURR"])
X_numeric = X_raw.select_dtypes(include=["number"])

# Step 1: Impute missing values on numeric columns
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_numeric)


# Step 2: Standardize the features
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

# scikit-learn returns: X_train, X_test, THEN y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(
    standardized_data, y, test_size=0.2, random_state=42
)

# Step 3: Train a LightGBM model
model = LGBMClassifier()
model.fit(X_train, y_train)

# Predict probability of default (column 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate and print the leaked AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("\n=======================================================")
print(f"Naive (Leaked) Baseline ROC-AUC: {auc_score:.5f}")
print("=======================================================\n")

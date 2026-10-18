"""
Version 2: tests whether the single-CV vs nested-CV optimism gap is real or just split luck.

WHAT VERSION 1 FOUND
    hyperparameter_tuning.py (v1) reported single CV 0.74970, nested CV 0.74931, and an
    optimism gap of +0.00039, which was read as "single CV overestimates performance".

WHY THAT CONCLUSION DID NOT HOLD
    1. The gap was smaller than the noise. The Day 8 fold-to-fold standard deviation was
       0.00144, about four times the gap. A gap that size can appear by luck alone.
    2. There was only one run. A single seed gives one random draw of the gap. A different
       seed could have produced a different size, or even the opposite sign. One number
       cannot reveal its own noise.
    3. The two procedures differed in more than one way. v1 passed cv=3 to the inner
       search. For a classifier, scikit-learn silently turns an integer into
       StratifiedKFold(3, shuffle=False). So the inner loop was stratified, unshuffled,
       and 3-fold, while the other two loops were plain, shuffled, and 5-fold. Any of those
       differences, not only "nested vs not nested", could have produced the gap.
    4. A small gap was expected anyway. Six candidates, several of them near-duplicates
       (in LightGBM, max_depth 5 and 10 are both capped by num_leaves=31), leave little
       room for selection bias. This is a hypothesis; this version is the test of it.
    In short: +0.00039 is an observation, not a finding.

HOW VERSION 2 FIXES IT
    1. One splitter for all three loops: KFold(5, shuffle=True, random_state=seed). The
       two procedures now differ only in being nested or not.
    2. The model seed is held fixed (MODEL_SEED = 42), so the variation between runs
       measures split luck only, not randomness inside training.
    3. The whole experiment is repeated over several splitter seeds, giving one gap per
       seed instead of one gap in total.
    4. The gaps are summarised by their mean, their sample standard deviation (ddof=1),
       and the standard error of the mean, SE = std / sqrt(R).
    5. Decision rule: if |mean gap| > 2 SE, single CV is reported as optimistic (or
       pessimistic). Otherwise the result is "no gap was detected at this precision",
       which is not the same as "there is no gap".
    Scope: every rerun uses the same rows, so this tests split luck on this dataset, not
    how a new sample of applicants would behave (see notes.md, Remark in section 8.7).

The following steps are performed once per splitter seed:
  1. Single CV Score (GridSearchCV):
      • Run a single 5-fold CV hyperparameter search across X_numeric.
      • Record the best mean score reported by the search (the optimistic score).
  2. Nested CV Score (cross_val_score wrapping GridSearchCV):
      • Run an outer 5-fold CV where each outer fold runs an inner 5-fold search on
        the outer training set.
      • Record the mean score across the 5 outer test folds (the honest score).
  3. Record the gap: Single CV Score - Nested CV Score.

Then, across all seeds:
  4. Report the mean gap, its standard deviation, and its standard error (SE),
     and say whether the mean gap is more than 2 SE away from zero.

Usage:
    python scripts/hyperparameter_tuning_v2.py --n-seeds 10
"""

import argparse
import math

import numpy as np
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

# Model seed is held fixed across every run, so the spread of the gaps measures
# split luck only (which rows land in which fold), not randomness inside training
MODEL_SEED = 42

N_FOLDS = 5

PARAM_GRID = {
    "model__n_estimators": [100],
    "model__max_depth": [3, 5, 10],
    "model__learning_rate": [0.01, 0.1],
}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load application_train once and return numeric features and target."""
    # Step 1: Load and shrink the table once, because every seed reuses the same rows
    df = pd.read_csv("data/raw/application_train.csv")
    df = reduce_memory_usage(df)

    # Step 2: Keep numeric columns only; drop the target and the row identifier
    y = df["TARGET"]
    X_raw = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X_numeric = X_raw.select_dtypes(include=["number"])
    return X_numeric, y


def build_pipeline() -> Pipeline:
    """Imputer, scaler, and model as a single estimator, so every fit stays inside its fold."""
    # median chosen over mean because mean is pulled by extreme outliers in financial data
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(random_state=MODEL_SEED, verbose=-1)),
        ]
    )


def run_one_seed(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[float, float]:
    """Run single CV and nested CV under one splitter seed; return both scores."""

    # Step 1: One splitter definition for all three loops, so the two procedures
    # differ only in being nested or not (same type, same k, shuffled, same seed)
    def splitter() -> KFold:
        return KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # Step 2: Single CV — search and score on the same folds (the optimistic number).
    # refit=False because only best_score_ is needed; a final refit on all rows is wasted work
    single_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        cv=splitter(),
        scoring="roc_auc",
        refit=False,
    )
    single_search.fit(X, y)
    single_score = single_search.best_score_

    # Step 3: Nested CV — the inner search only ever sees the outer-train part, and
    # refit=True (default) retrains each round's winner on the full outer-train before scoring
    inner_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        cv=splitter(),
        scoring="roc_auc",
    )
    nested_scores = cross_val_score(
        inner_search,
        X,
        y,
        cv=splitter(),
        scoring="roc_auc",
    )
    return single_score, nested_scores.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="number of splitter seeds to rerun the experiment with (at least 2)",
    )
    args = parser.parse_args()
    if args.n_seeds < 2:
        parser.error("--n-seeds must be at least 2 to estimate a spread")

    # Step 1: Load data once
    X, y = load_features()

    # Step 2: Rerun the whole experiment once per splitter seed; each run is one
    # random draw of the gap
    gaps = []
    for seed in range(args.n_seeds):
        single_score, nested_score = run_one_seed(X, y, seed)
        gap = single_score - nested_score
        gaps.append(gap)
        print(
            f"seed {seed:>3}:  single {single_score:.5f}  "
            f"nested {nested_score:.5f}  gap {gap:+.5f}",
            flush=True,
        )

    # Step 3: Summarise the gaps. ddof=1 because the spread is estimated from a
    # small sample of runs (np.std defaults to ddof=0, which understates it)
    gaps = np.array(gaps)
    n_runs = len(gaps)
    mean_gap = gaps.mean()
    std_gap = gaps.std(ddof=1)
    se_gap = std_gap / math.sqrt(n_runs)

    # Step 4: Decide — a mean gap more than 2 SE from zero is unlikely to be split luck
    print("=" * 55)
    print(f"Runs (splitter seeds):           {n_runs}")
    print(f"Mean gap (Single - Nested):      {mean_gap:+.5f}")
    print(f"Std of gaps (one run's wobble):  {std_gap:.5f}")
    print(f"Standard error of mean gap:      {se_gap:.5f}")
    print(f"Mean gap / SE:                   {mean_gap / se_gap:+.2f}")
    if abs(mean_gap) > 2 * se_gap:
        direction = "optimistic" if mean_gap > 0 else "pessimistic"
        print(f"Verdict: single CV is {direction} on this dataset (|mean| > 2 SE)")
    else:
        print("Verdict: no gap was detected at this precision (|mean| <= 2 SE)")
    print("=" * 55)


if __name__ == "__main__":
    main()

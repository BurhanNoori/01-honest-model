"""
The idea is to aggregate values of child tables before apply an merge or join operation.
The question is why?
So following are the reasons:
Because our application_train has SK_ID_CURR represents ID of the current applicant
But other child tables such as bureau has multiple entries of same applicant.
1. It would do the row explosion in our training data application_train
2. Some entries in this children tables are post date of loan application
    So we need to filter predate (< application date) data otherwise it would
    be a leak for our model and could impact its decision making
3. We also have to consider data bureau for DAYS_CREDIT < 0 instead DAYS_CREDIT <= 0.
    The Day = 0 is ambiguos. It means the current loan is also approved and it is reflecting
    in the Bureau table which means before our ML model has find any target the loan is approved.
    This pattern will be learned by our model and can impact its accuracy.
    We need to avoid any data to be taken from child tables that are made post loan approval
    decision or events occured after loan approval for each customer.

    This module will helps us to filtering the bad features and preparing
    good features before we start training.

    1. Implement aggregate_bureau(bureau_df: pd.DataFrame) -> pd.DataFrame.
      • Step 1: Filter DAYS_CREDIT < 0.
      • Step 2: Aggregate numerical columns (AMT_CREDIT_SUM, DAYS_CREDIT, AMT_CREDIT_OVERDUE etc.).
      • Step 3: (Optional/Bonus) Aggregate categorical columns (CREDIT_ACTIVE).
      • Step 4: Flatten column names with a BUREAU_ prefix.
  2. Write a unit test in tests/test_features.py to verify:
      • Step 5: Downcast memory of the final collapsed table with reduce_memory_usage().
      • Output DataFrame has 1 row per applicant.
      • No multi-index column names remain.
      • Rows with DAYS_CREDIT >= 0 were properly excluded.
"""

import pandas as pd

from honest_model.memory import reduce_memory_usage


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:

    # Collapse bureau_balance to one row per credit line FIRST
    bb = bureau_balance.copy()
    bb["IS_OVERDUE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bb_agg = (
        bb.groupby("SK_BUREAU_ID")
        .agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_OVERDUE_MONTHS=("IS_OVERDUE", "sum"),
        )
        .reset_index()
    )

    # Step 1 + 3: drop anything at or after the application date
    bureau = bureau[bureau["DAYS_CREDIT"] < 0].copy()

    # 1-to-1 now, so this cannot explode rows
    bureau = bureau.merge(bb_agg, on="SK_BUREAU_ID", how="left")
    bureau[["BB_MONTHS_COUNT", "BB_OVERDUE_MONTHS"]] = bureau[
        ["BB_MONTHS_COUNT", "BB_OVERDUE_MONTHS"]
    ].fillna(0)

    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
    # NOTE: sum() already skips NaN, so fillna(0) here is a no-op for the total.
    # The information that the debt was *unknown* would be lost, so it is kept
    # as its own count below.
    bureau["ACTIVE_DEBT"] = (
        bureau["AMT_CREDIT_SUM_DEBT"].fillna(0).where(bureau["IS_ACTIVE"] == 1, 0.0)
    )
    bureau["ACTIVE_DEBT_MISSING"] = (
        (bureau["IS_ACTIVE"] == 1) & bureau["AMT_CREDIT_SUM_DEBT"].isna()
    ).astype(int)

    enddate_known = bureau["DAYS_CREDIT_ENDDATE"].notna()
    bureau["MONTHS_REMAINING"] = (bureau["DAYS_CREDIT_ENDDATE"] / 30).clip(lower=0)
    is_ongoing = enddate_known & (bureau["MONTHS_REMAINING"] > 0)
    bureau["IS_ONGOING"] = is_ongoing.astype(int)
    bureau["ENDDATE_MISSING"] = (~enddate_known).astype(int)
    bureau["ANNUITY_IF_ONGOING"] = bureau["AMT_ANNUITY"].where(is_ongoing)

    bureau_agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            BUREAU_LOAN_COUNT=("DAYS_CREDIT", "count"),
            BUREAU_ACTIVE_LOAN_COUNT=("IS_ACTIVE", "sum"),
            BUREAU_CLOSED_LOAN_COUNT=("IS_CLOSED", "sum"),
            BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
            BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
            BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
            BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
            BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
            BUREAU_ACTIVE_DEBT_TOTAL=("ACTIVE_DEBT", "sum"),
            BUREAU_ACTIVE_DEBT_MISSING_COUNT=("ACTIVE_DEBT_MISSING", "sum"),
            BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_MONTHS_REMAINING_MAX=("MONTHS_REMAINING", "max"),
            BUREAU_ONGOING_LOAN_COUNT=("IS_ONGOING", "sum"),
            BUREAU_ENDDATE_MISSING_COUNT=("ENDDATE_MISSING", "sum"),
            BUREAU_MAX_ANNUITY_IF_ONGOING=("ANNUITY_IF_ONGOING", "max"),
            BUREAU_MONTHS_TOTAL=("BB_MONTHS_COUNT", "sum"),
            BUREAU_OVERDUE_MONTHS_TOTAL=("BB_OVERDUE_MONTHS", "sum"),
        )
        .reset_index()
    )

    bureau_agg["BUREAU_DUES_FAILURE_RATE"] = (
        bureau_agg["BUREAU_OVERDUE_MONTHS_TOTAL"].div(
            bureau_agg["BUREAU_MONTHS_TOTAL"].where(lambda s: s > 0)
        )
    ) * 100

    return reduce_memory_usage(bureau_agg)


def join_features(application: pd.DataFrame, child: pd.DataFrame) -> pd.DataFrame:

    return application.merge(child, on="SK_ID_CURR", how="left")

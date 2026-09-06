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


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:

    # Step 1: Filter the bureau data based on the DAYS_CREDIT < 0 + Step 3
    bureau = bureau[bureau["DAYS_CREDIT"] < 0].copy()

    # Step 2: Aggregate the numerical columns (AMT_CREDIT_SUM, DAYS_CREDIT etc)
    # bureau = bureau.groupby("SK_ID_CURR").agg({
    # "AMT_CREDIT_SUM": ["mean", "max", "sum"]
    # })

    # shortcut it will flatten column names #Step 2 + Step 4

    # Create active-loan debt helper
    bureau["IS_ACTIVE"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    bureau["IS_CLOSED"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype(int)
    bureau["ACTIVE_DEBT"] = bureau["AMT_CREDIT_SUM_DEBT"].where(bureau["IS_ACTIVE"] == 1, 0.0)

    # Aggregate both lifetime and active signals
    bureau_agg = (
        bureau.groupby("SK_ID_CURR")
        .agg(
            # How many bureau loans this applicant has in total.
            BUREAU_LOAN_COUNT=("DAYS_CREDIT", "count"),
            # Number of active loans (since IS_ACTIVE is 1 for active, 0 otherwise).
            BUREAU_ACTIVE_LOAN_COUNT=("IS_ACTIVE", "sum"),
            # The earliest loan (most negative days = longest time ago).
            BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
            # The most recent loan (closest to today).
            BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
            # Average loan size across all bureau loans.
            BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
            # Largest single loan amount.
            BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
            # Total credit amount across all bureau loans.
            BUREAU_AMT_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
            # Total debt from active loans only.
            BUREAU_ACTIVE_DEBT_TOTAL=("ACTIVE_DEBT", "sum"),
            # Worst overdue among all the loans.
            BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
            # Loans successfully closed
            BUREAU_CLOSED_LOAN_COUNT=("IS_CLOSED", "sum"),
        )
        .reset_index()
    )

    return reduce_memory_usage(bureau_agg)


def join_features(application: pd.DataFrame, child: pd.DataFrame) -> pd.DataFrame:

    return application.merge(child, on="SK_ID_CURR", how="left")

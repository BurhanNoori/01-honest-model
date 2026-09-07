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


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:

    # Collapse bureau_balance to one row per credit line FIRST
    bb = bureau_balance.copy()
    bb["IS_OVERDUE"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bb_agg = (
        bb.groupby("SK_ID_BUREAU")
        .agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_OVERDUE_MONTHS=("IS_OVERDUE", "sum"),
        )
        .reset_index()
    )

    # Step 1 + 3: drop anything at or after the application date
    bureau = bureau[bureau["DAYS_CREDIT"] < 0].copy()

    # 1-to-1 now, so this cannot explode rows
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Filling NaN or missing values with 0
    bureau[["BB_MONTHS_COUNT", "BB_OVERDUE_MONTHS"]] = (
        bureau[  # [[]] is for dataframe whereas [] is used for series
            ["BB_MONTHS_COUNT", "BB_OVERDUE_MONTHS"]
        ].fillna(0)
    )

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

    # Checking NaN on the loan pending enddate (-ve means loan settled +ve means still left rel.
    # to the date of application)
    enddate_known = bureau[
        "DAYS_CREDIT_ENDDATE"
    ].notna()  # series of boolean False for NaN otherwise True

    bureau["MONTHS_REMAINING"] = (bureau["DAYS_CREDIT_ENDDATE"] / 30).clip(lower=0)
    is_ongoing = enddate_known & (bureau["MONTHS_REMAINING"] > 0)
    bureau["IS_ONGOING"] = is_ongoing.astype(int)

    # enddate's not is endate_missing same as endate_missing = !enddate_known
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

    return bureau_agg


def aggregate_pos_cash(pos_cash) -> pd.DataFrame:
    pos_cash = pos_cash.copy()  # work on a copy
    # Check all passed failed payment
    pos_cash["IS_PAYMENT_FAILED"] = (pos_cash["MONTHS_BALANCE"] < 0) & (
        pos_cash["SK_DPD"] > 0
    ).astype(int)
    pos_cash_agg = (
        pos_cash.groupby("SK_ID_CURR")
        .agg(
            POS_PAST_INSTALMENT_FAILED_ON_TIME=("IS_PAYMENT_FAILED", "sum"),
            POS_PAST_TOTAL_INSTALLMENTS=("IS_PAYMENT_FAILED", "count"),
        )
        .reset_index()
    )

    pos_cash_agg["POS_PAYMENT_FAILURE_RATE"] = (
        pos_cash_agg["POS_PAST_INSTALMENT_FAILED_ON_TIME"].div(
            pos_cash_agg["POS_PAST_INSTALMENT_FAILED_ON_TIME"].where(lambda s: s > 0)
        )
        * 100
    )

    return pos_cash_agg


def aggregate_installments(installments) -> pd.DataFrame:
    installments = installments.copy()  # work on a copy
    installments["FAILED_TO_PAY"] = (
        installments["AMT_PAYMENT"] < installments["AMT_INSTALMENT"]
    ).astype(int)
    installments_agg = (
        installments.groupby("SK_ID_CURR")
        .agg(
            INSTALMENTS_TOTAL_FAILURES=("FAILED_TO_PAY", "sum"),
            INSTALMENTS_TOTAL=("FAILED_TO_PAY", "count"),
        )
        .reset_index()
    )

    return installments_agg


def aggregate_credit_card(cc) -> pd.DataFrame:
    cc = cc.copy()  # work on a copy
    # What we want
    # 1. Failure rate to pay the bills on time
    cc["BILL_FAILURE"] = (
        (cc["AMT_BALANCE"] > 0) & (cc["AMT_PAYMENT_TOTAL_CURRENT"] < cc["AMT_BALANCE"])
    ).astype(int)

    # 2. What are his mean monthly expenses
    cc_agg = (
        cc.groupby("SK_ID_CURR")
        .agg(
            CREDIT_CARD_TOTAL_MONTHS=("MONTHS_BALANCE", "count"),
            CREDIT_CARD_MEAN_MONTHLY_EXPENSE=("AMT_TOTAL_RECEIVABLE", "mean"),
            CREDIT_CARD_MEAN_MONTHLY_PAID=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
            CREDIT_CARD_BILL_FAILURE=("BILL_FAILURE", "sum"),
        )
        .reset_index()
    )

    cc_agg["CREDIT_CARD_BILL_FAILURE_RATE"] = (
        cc_agg["CREDIT_CARD_BILL_FAILURE"].div(
            cc_agg["CREDIT_CARD_TOTAL_MONTHS"].where(lambda s: s > 0)
        )
        * 100
    )
    return cc_agg


def aggregate_previous_applications(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()  # work on a copy
    # Step 1. Temporal filter, only past decisions
    prev = prev[prev["DAYS_DECISION"] < 0].copy()

    # Step 2. Indicators for approval and refusal
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["APP_CREDIT_RATIO"] = prev["AMT_CREDIT"].div(prev["AMT_APPLICATION"]) * 100

    # Step 3. Group by applicant
    prev_agg = (
        prev.groupby("SK_ID_CURR")
        .agg(
            PREV_APP_COUNT=("DAYS_DECISION", "count"),
            PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
            PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
            PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
            PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
            PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
            PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
            PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        )
        .reset_index()
    )

    prev_agg["PREV_REFUSAL_RATE"] = (
        prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_APP_COUNT"]
    ) * 100

    return prev_agg

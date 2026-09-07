"""Tests for aggregate_bureau.

Fixture design notes (these are load-bearing, don't "simplify" them away):

* Every applicant has exactly one loan with DAYS_CREDIT == 0, and those rows
  carry CREDIT_DAY_OVERDUE == 999 -- larger than any legitimate value. If the
  leak filter ever breaks, the overdue assertions fail loudly instead of
  passing by luck.
* Balance history is UNEVEN across loans (3, 12, 1, 5, 2 months). A 1-to-many
  merge multiplies each loan by its own month count, so uneven counts expose
  the explosion; equal counts would scale everything uniformly and could hide
  inside a ratio.
* Some loans have no balance rows at all, exercising the left-join fill.
* Loan 14 is Active with an unknown debt figure, so a "missing" count can be
  distinguished from a genuine zero.
* Applicant 3 has no balance history whatsoever -> zero total months, which is
  the divide-by-zero path for the failure rate.
* Applicant 4's only loan is at DAYS_CREDIT == 0, so they vanish from the
  output entirely. That is intended behaviour and is asserted below.
"""

import numpy as np
import pandas as pd
import pytest

from honest_model.features import (
    aggregate_bureau,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_applications,
)

# SK_ID_BUREAU -> monthly STATUS codes, most recent first.
# STATUS "1".."5" counts as an overdue month; "0" and "C" do not.
_BALANCE_HISTORY = {
    11: ["0", "0", "1"],
    12: ["5"] * 8,  # excluded loan (DAYS_CREDIT == 0) -- must not be counted
    13: ["0", "0", "0", "0", "1", "2", "0", "0", "0", "0", "0", "0"],
    14: ["0"],
    16: ["0", "0", "0", "0", "0"],
    17: ["4"] * 6,  # excluded loan -- must not be counted
    18: ["0", "3"],
    # 15, 19, 20 deliberately have no balance rows
}


@pytest.fixture
def bureau_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4],
            "SK_ID_BUREAU": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "CREDIT_ACTIVE": [
                "Active",
                "Active",
                "Closed",
                "Active",
                "Closed",
                "Closed",
                "Active",
                "Closed",
                "Closed",
                "Active",
                "Active",
                "Active",
            ],
            "DAYS_CREDIT": [-300, 0, -150, -100, -50, -400, 0, -200, -100, -30, -60, 0],
            "CREDIT_DAY_OVERDUE": [0, 999, 0, 10, 0, 0, 999, 20, 0, 15, 0, 999],
            "AMT_CREDIT_SUM": [
                50000,
                30000,
                20000,
                40000,
                10000,
                90000,
                10000,
                15000,
                20000,
                25000,
                5000,
                7000,
            ],
            # loan 14 is Active with a missing debt figure -> exercises the
            # fillna-before-where path in ACTIVE_DEBT
            "AMT_CREDIT_SUM_DEBT": [
                20000,
                10000,
                0.0,
                np.nan,
                0.0,
                40000,
                2000,
                5000,
                10000,
                12000,
                1000,
                3000,
            ],
            "DAYS_CREDIT_ENDDATE": [
                -200,
                500,
                np.nan,
                300,
                -10,
                -350,
                100,
                400,
                np.nan,
                200,
                90,
                120,
            ],
            "AMT_ANNUITY": [
                1000,
                2000,
                3000,
                4000,
                5000,
                500,
                600,
                700,
                800,
                900,
                300,
                400,
            ],
        }
    )


@pytest.fixture
def bureau_balance_df():
    rows = [
        {"SK_ID_BUREAU": sk_id, "MONTHS_BALANCE": -i, "STATUS": status}
        for sk_id, statuses in _BALANCE_HISTORY.items()
        for i, status in enumerate(statuses)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def agg(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame):
    return aggregate_bureau(bureau_df, bureau_balance_df).set_index("SK_ID_CURR")


@pytest.fixture
def credit_card_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 3],
            "SK_ID_PREV": [101, 102, 201, 202, 301],
            "MONTHS_BALANCE": [-1, -2, -1, 0, -3],
            "AMT_BALANCE": [1000, 2000, 1500, 2500, np.nan],
            "AMT_TOTAL_RECEIVABLE": [900, 2100, 1400, 2600, np.nan],
            "AMT_PAYMENT_TOTAL_CURRENT": [1200, 2200, 1600, 2700, np.nan],
        }
    )


@pytest.fixture
def installments_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 3],
            "SK_ID_PREV": [101, 102, 201, 202, 301],
            "AMT_INSTALMENT": [1000, 2000, 1500, 2500, 3000],
            "AMT_PAYMENT": [900, 2100, 1400, 2600, 2900],
        }
    )


@pytest.fixture
def pos_cash_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 3],
            "SK_ID_PREV": [101, 102, 201, 202, 301],
            "MONTHS_BALANCE": [-1, -2, -1, -2, 0],
            "SK_DPD": [0, 5, 0, 10, 0],
        }
    )


@pytest.fixture
def previous_application_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2, 3],
            "SK_ID_PREV": [101, 102, 201, 202, 301],
            "AMT_APPLICATION": [10000, np.nan, 15000, 25000, 30000],
            "AMT_CREDIT": [9000, np.nan, 14000, 26000, 29000],
            "AMT_DOWN_PAYMENT": [1000, np.nan, 1500, 2500, 3000],
            "DAYS_DECISION": [-10, -20, -15, -25, -30],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Refused", "Approved"],
        }
    )


# --------------------------------------------------------------------------
# Shape and schema
# --------------------------------------------------------------------------


def test_one_row_per_applicant(agg: pd.DataFrame):
    assert agg.index.is_unique


def test_no_multiindex_columns(agg: pd.DataFrame):
    assert not isinstance(agg.columns, pd.MultiIndex)
    assert all(isinstance(c, str) for c in agg.columns)


def test_expected_columns_present(agg: pd.DataFrame):
    expected = {
        "BUREAU_LOAN_COUNT",
        "BUREAU_ACTIVE_LOAN_COUNT",
        "BUREAU_CLOSED_LOAN_COUNT",
        "BUREAU_DAYS_CREDIT_MIN",
        "BUREAU_DAYS_CREDIT_MAX",
        "BUREAU_AMT_CREDIT_SUM_MEAN",
        "BUREAU_AMT_CREDIT_SUM_MAX",
        "BUREAU_AMT_CREDIT_SUM_TOTAL",
        "BUREAU_ACTIVE_DEBT_TOTAL",
        "BUREAU_ACTIVE_DEBT_MISSING_COUNT",
        "BUREAU_CREDIT_DAY_OVERDUE_MAX",
        "BUREAU_ONGOING_LOAN_COUNT",
        "BUREAU_ENDDATE_MISSING_COUNT",
        "BUREAU_MAX_ANNUITY_IF_ONGOING",
        "BUREAU_MONTHS_TOTAL",
        "BUREAU_OVERDUE_MONTHS_TOTAL",
        "BUREAU_DUES_FAILURE_RATE",
    }
    assert expected <= set(agg.columns)


def test_input_frames_are_not_mutated(
    bureau_df: pd.DataFrame,
    bureau_balance_df: pd.DataFrame,
    credit_card_df: pd.DataFrame,
    installments_df: pd.DataFrame,
    pos_cash_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
):
    before_bureau = bureau_df.copy()
    before_balance = bureau_balance_df.copy()
    before_credit_card = credit_card_df.copy()
    before_installments = installments_df.copy()
    before_pos_cash = pos_cash_df.copy()
    before_previous_application = previous_application_df.copy()
    aggregate_bureau(bureau_df, bureau_balance_df)
    aggregate_credit_card(credit_card_df)
    aggregate_installments(installments_df)
    aggregate_pos_cash(pos_cash_df)
    aggregate_previous_applications(previous_application_df)
    pd.testing.assert_frame_equal(bureau_df, before_bureau)
    pd.testing.assert_frame_equal(bureau_balance_df, before_balance)
    pd.testing.assert_frame_equal(credit_card_df, before_credit_card)
    pd.testing.assert_frame_equal(installments_df, before_installments)
    pd.testing.assert_frame_equal(pos_cash_df, before_pos_cash)
    pd.testing.assert_frame_equal(previous_application_df, before_previous_application)


def test_expected_columns_present_in_credit_card(credit_card_df: pd.DataFrame):
    expected = {
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_TOTAL_RECEIVABLE",
        "AMT_PAYMENT_TOTAL_CURRENT",
    }
    assert expected <= set(credit_card_df.columns)


def test_expected_columns_present_in_installments(installments_df: pd.DataFrame):
    expected = {
        "SK_ID_CURR",
        "SK_ID_PREV",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    }
    assert expected <= set(installments_df.columns)


def test_expected_columns_present_in_pos_cash(pos_cash_df: pd.DataFrame):
    expected = {
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "SK_DPD",
    }
    assert expected <= set(pos_cash_df.columns)


def test_expected_columns_present_in_previous_application(previous_application_df: pd.DataFrame):
    expected = {
        "SK_ID_CURR",
        "SK_ID_PREV",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
    }
    assert expected <= set(previous_application_df.columns)


# --------------------------------------------------------------------------
# Leak filter: DAYS_CREDIT must be strictly negative
# --------------------------------------------------------------------------


def test_days_credit_zero_rows_are_excluded(agg: pd.DataFrame):
    assert (agg["BUREAU_DAYS_CREDIT_MIN"] < 0).all()
    assert (agg["BUREAU_DAYS_CREDIT_MAX"] < 0).all()


def test_excluded_rows_do_not_leak_into_aggregates(agg: pd.DataFrame):
    # The 999s live only on DAYS_CREDIT == 0 rows.
    assert agg["BUREAU_CREDIT_DAY_OVERDUE_MAX"].max() < 999
    assert agg.loc[1, "BUREAU_CREDIT_DAY_OVERDUE_MAX"] == 10
    assert agg.loc[2, "BUREAU_CREDIT_DAY_OVERDUE_MAX"] == 20


def test_applicant_with_only_day_zero_loans_is_dropped(agg: pd.DataFrame):
    assert 4 not in agg.index


# --------------------------------------------------------------------------
# The row-explosion regression
# --------------------------------------------------------------------------


def test_balance_merge_does_not_explode_loan_count(agg: pd.DataFrame):
    assert agg.loc[1, "BUREAU_LOAN_COUNT"] == 4
    assert agg.loc[2, "BUREAU_LOAN_COUNT"] == 4
    assert agg.loc[3, "BUREAU_LOAN_COUNT"] == 1


def test_balance_merge_does_not_inflate_sums(agg: pd.DataFrame):
    assert agg.loc[1, "BUREAU_AMT_CREDIT_SUM_TOTAL"] == 120000
    assert agg.loc[2, "BUREAU_AMT_CREDIT_SUM_TOTAL"] == 150000
    assert agg.loc[1, "BUREAU_AMT_CREDIT_SUM_MEAN"] == 30000
    assert agg.loc[1, "BUREAU_AMT_CREDIT_SUM_MAX"] == 50000


def test_month_totals_count_each_loan_once(agg: pd.DataFrame):
    # applicant 1: 3 (loan 11) + 12 (13) + 1 (14) + 0 (15)
    assert agg.loc[1, "BUREAU_MONTHS_TOTAL"] == 16
    # applicant 2: 5 (loan 16) + 2 (18) + 0 + 0
    assert agg.loc[2, "BUREAU_MONTHS_TOTAL"] == 7


def test_loans_without_balance_history_count_as_zero_months(agg: pd.DataFrame):
    # applicant 3's only loan has no balance rows at all
    assert agg.loc[3, "BUREAU_MONTHS_TOTAL"] == 0


# --------------------------------------------------------------------------
# Credit status counts
# --------------------------------------------------------------------------


def test_active_and_closed_counts(agg: pd.DataFrame):
    assert agg.loc[1, "BUREAU_ACTIVE_LOAN_COUNT"] == 2
    assert agg.loc[1, "BUREAU_CLOSED_LOAN_COUNT"] == 2
    assert agg.loc[2, "BUREAU_ACTIVE_LOAN_COUNT"] == 1
    assert agg.loc[2, "BUREAU_CLOSED_LOAN_COUNT"] == 3


def test_status_counts_never_exceed_loan_count(agg: pd.DataFrame):
    total = agg["BUREAU_ACTIVE_LOAN_COUNT"] + agg["BUREAU_CLOSED_LOAN_COUNT"]
    assert (total <= agg["BUREAU_LOAN_COUNT"]).all()


def test_active_debt_excludes_closed_loans(agg: pd.DataFrame):
    # loan 11 (Active, 20000) + loan 14 (Active, debt unknown -> contributes 0)
    assert agg.loc[1, "BUREAU_ACTIVE_DEBT_TOTAL"] == 20000
    # only loan 20 is Active
    assert agg.loc[2, "BUREAU_ACTIVE_DEBT_TOTAL"] == 12000


def test_unknown_active_debt_is_recorded_not_just_zeroed(agg: pd.DataFrame):
    # A zero total and a total built from unknown figures must be
    # distinguishable. Loan 14 is Active with a missing debt amount.
    assert agg.loc[1, "BUREAU_ACTIVE_DEBT_MISSING_COUNT"] == 1
    assert agg.loc[2, "BUREAU_ACTIVE_DEBT_MISSING_COUNT"] == 0
    assert agg.loc[3, "BUREAU_ACTIVE_DEBT_MISSING_COUNT"] == 0


# --------------------------------------------------------------------------
# Ongoing loans / missing end dates
# --------------------------------------------------------------------------


def test_ongoing_and_missing_enddate_counts(agg: pd.DataFrame):
    # applicant 1: loan 14 ongoing; loan 13 has a missing end date
    assert agg.loc[1, "BUREAU_ONGOING_LOAN_COUNT"] == 1
    assert agg.loc[1, "BUREAU_ENDDATE_MISSING_COUNT"] == 1
    # applicant 2: loans 18 and 20 ongoing; loan 19 missing
    assert agg.loc[2, "BUREAU_ONGOING_LOAN_COUNT"] == 2
    assert agg.loc[2, "BUREAU_ENDDATE_MISSING_COUNT"] == 1


def test_annuity_only_counted_for_ongoing_loans(agg: pd.DataFrame):
    # loan 14 is the only ongoing loan for applicant 1
    assert agg.loc[1, "BUREAU_MAX_ANNUITY_IF_ONGOING"] == 4000
    # max of loans 18 (700) and 20 (900)
    assert agg.loc[2, "BUREAU_MAX_ANNUITY_IF_ONGOING"] == 900


def test_missing_enddate_is_not_treated_as_ongoing(agg: pd.DataFrame):
    # loans 13 and 19 have NaN end dates and annuities of 3000 / 800.
    # If NaN were treated as ongoing, applicant 1's max would be 5000+.
    assert agg.loc[1, "BUREAU_MAX_ANNUITY_IF_ONGOING"] < 5000


# --------------------------------------------------------------------------
# Failure rate
# --------------------------------------------------------------------------


def test_failure_rate_is_a_percentage_of_months(agg: pd.DataFrame):
    # applicant 1: 3 overdue months out of 16
    assert agg.loc[1, "BUREAU_OVERDUE_MONTHS_TOTAL"] == 3
    assert agg.loc[1, "BUREAU_DUES_FAILURE_RATE"] == pytest.approx(18.75, rel=1e-2)
    # applicant 2: 1 overdue month out of 7
    assert agg.loc[2, "BUREAU_OVERDUE_MONTHS_TOTAL"] == 1
    assert agg.loc[2, "BUREAU_DUES_FAILURE_RATE"] == pytest.approx(100 / 7, rel=1e-2)


def test_failure_rate_is_nan_not_inf_when_no_history(agg: pd.DataFrame):
    # applicant 3 has zero months -> unknown rate, not 0 and not inf
    assert pd.isna(agg.loc[3, "BUREAU_DUES_FAILURE_RATE"])


def test_failure_rate_is_within_bounds(agg: pd.DataFrame):
    rate = agg["BUREAU_DUES_FAILURE_RATE"].dropna()
    assert ((rate >= 0) & (rate <= 100)).all()


# --------------------------------------------------------------------------
# Downcasting must not corrupt values
# --------------------------------------------------------------------------


def test_counts_have_no_nan(agg: pd.DataFrame):
    # NaN is correct for MEAN/MAX/RATE columns when there is nothing to
    # aggregate, so only counts and totals are checked here.
    count_cols = [c for c in agg.columns if c.endswith(("_COUNT", "_TOTAL"))]
    assert agg[count_cols].notna().all().all()


def test_no_inf_after_downcast(agg: pd.DataFrame):
    numeric = agg.select_dtypes(include="number")
    assert not np.isinf(numeric.to_numpy(dtype="float64")).any()


def test_large_totals_survive_downcast(agg: pd.DataFrame):
    # float16 tops out at 65504; applicant 2's total is 150000.
    assert agg.loc[2, "BUREAU_AMT_CREDIT_SUM_TOTAL"] == pytest.approx(150000, rel=1e-3)

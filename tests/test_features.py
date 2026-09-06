import pandas as pd
import pytest

from honest_model.features import aggregate_bureau


@pytest.fixture
def _load_bureau():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "SK_ID_BUREAU": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
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
            ],
            "DAYS_CREDIT": [-300, 0, -150, -100, -50, -400, 0, -200, -100, -30],
            "CREDIT_DAY_OVERDUE": [0, 5, 0, 10, 0, 0, 0, 20, 0, 15],
            "AMT_CREDIT_MAX_OVERDUE": [0, 1000, 0, 2000, 0, 0, 0, 500, 0, 1500],
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
            ],
            "AMT_CREDIT_SUM_DEBT": [20000, 10000, 0, 15000, 0, 40000, 2000, 5000, 10000, 12000],
        }
    )


def test_aggregate_should_not_have_days_credit_equals_to_zero(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    # Check that min and max DAYS_CREDIT are strictly negative
    assert (bureau["BUREAU_DAYS_CREDIT_MIN"] < 0).all()
    assert (bureau["BUREAU_DAYS_CREDIT_MAX"] < 0).all()


def test_aggregate_should_have_agg_cols(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    assert "BUREAU_LOAN_COUNT" in bureau.columns
    assert "BUREAU_ACTIVE_LOAN_COUNT" in bureau.columns
    assert "BUREAU_DAYS_CREDIT_MIN" in bureau.columns
    assert "BUREAU_DAYS_CREDIT_MAX" in bureau.columns
    assert "BUREAU_AMT_CREDIT_SUM_MEAN" in bureau.columns
    assert "BUREAU_AMT_CREDIT_SUM_MAX" in bureau.columns
    assert "BUREAU_AMT_CREDIT_SUM_TOTAL" in bureau.columns
    assert "BUREAU_ACTIVE_DEBT_TOTAL" in bureau.columns
    assert "BUREAU_CREDIT_DAY_OVERDUE_MAX" in bureau.columns
    assert "BUREAU_CLOSED_LOAN_COUNT" in bureau.columns


def test_aggregate_should_have_one_row_each_id(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    assert len(bureau) == 2


def test_no_nan_in_aggregated_features(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    assert bureau.notna().all().all()


def test_loan_count_matches_fixture(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    assert bureau.loc[bureau["SK_ID_CURR"] == 1, "BUREAU_LOAN_COUNT"].iloc[0] == 4
    assert bureau.loc[bureau["SK_ID_CURR"] == 2, "BUREAU_LOAN_COUNT"].iloc[0] == 4


def test_max_overdue_days(_load_bureau):
    bureau = aggregate_bureau(_load_bureau)
    assert bureau.loc[bureau["SK_ID_CURR"] == 1, "BUREAU_CREDIT_DAY_OVERDUE_MAX"].iloc[0] == 10
    assert bureau.loc[bureau["SK_ID_CURR"] == 2, "BUREAU_CREDIT_DAY_OVERDUE_MAX"].iloc[0] == 20

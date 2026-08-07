import pandas as pd

from ppa_analysis.helper_functions import quarterly_indexation, yearly_indexation


def test_yearly_indexation_float_rate(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-06-01 00:00,   0
        2027-06-01 00:00,   0
        2028-06-01 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = yearly_indexation(df, strike_price=100.0, indexation=5.0)

    # 2026: 100 -> 2027: 100 + 5% = 105 -> 2028: 105 + 5% = 110.25
    expected = pd.Series(
        [100.0, 105.0, 110.25],
        index=pd.DatetimeIndex(
            ["2026-06-01", "2027-06-01", "2028-06-01"], name="DateTime"
        ),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)


def test_yearly_indexation_list_shorter_than_years_repeats_last_value(csv_str_to_df):
    # Regression case for the bug fixed in 20ded8e, where the last year's price picked
    # up an extra compounding step. 3 years, only 2 indexation rates given -> the second
    # rate (20%) should repeat for the third year, not compound an extra time.
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-06-01 00:00,   0
        2027-06-01 00:00,   0
        2028-06-01 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = yearly_indexation(df, strike_price=100.0, indexation=[10.0, 20.0])

    # 2026: 100 -> 2027: 100 + 10% = 110 -> 2028: 110 + 20% (repeated rate) = 132
    expected = pd.Series(
        [100.0, 110.0, 132.0],
        index=pd.DatetimeIndex(
            ["2026-06-01", "2027-06-01", "2028-06-01"], name="DateTime"
        ),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)


def test_yearly_indexation_list_longer_than_years_ignores_extras(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-06-01 00:00,   0
        2027-06-01 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = yearly_indexation(
        df, strike_price=100.0, indexation=[10.0, 20.0, 999.0, 999.0]
    )

    # Only the first 2 rates (matching the 2 years present) are used -> 999.0 never applied.
    expected = pd.Series(
        [100.0, 110.0],
        index=pd.DatetimeIndex(["2026-06-01", "2027-06-01"], name="DateTime"),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)


def test_quarterly_indexation_float_rate(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-15 00:00,   0
        2026-04-15 00:00,   0
        2026-07-15 00:00,   0
        2026-10-15 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = quarterly_indexation(df, strike_price=100.0, indexation=5.0)

    # Q1: 100 -> Q2: 105 -> Q3: 110.25 -> Q4: 115.7625
    expected = pd.Series(
        [100.0, 105.0, 110.25, 115.7625],
        index=pd.DatetimeIndex(
            ["2026-01-15", "2026-04-15", "2026-07-15", "2026-10-15"], name="DateTime"
        ),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)


def test_quarterly_indexation_list_shorter_than_quarters_repeats_last_value(
    csv_str_to_df,
):
    # Same regression case as the yearly test above, but for the quarterly helper.
    # One year -> 4 quarters, only 2 rates given -> the second rate (20%) repeats for
    # Q3 and Q4.
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-15 00:00,   0
        2026-04-15 00:00,   0
        2026-07-15 00:00,   0
        2026-10-15 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = quarterly_indexation(df, strike_price=100.0, indexation=[10.0, 20.0])

    # Q1: 100 -> Q2: 110 -> Q3: 132 -> Q4: 158.4
    expected = pd.Series(
        [100.0, 110.0, 132.0, 158.4],
        index=pd.DatetimeIndex(
            ["2026-01-15", "2026-04-15", "2026-07-15", "2026-10-15"], name="DateTime"
        ),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)


def test_quarterly_indexation_list_longer_than_quarters_ignores_extras(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-15 00:00,   0
        2026-04-15 00:00,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = quarterly_indexation(
        df, strike_price=100.0, indexation=[10.0, 20.0, 999.0, 999.0]
    )

    # Only the first 2 rates (matching Q1/Q2, the quarters actually present) are used.
    expected = pd.Series(
        [100.0, 110.0],
        index=pd.DatetimeIndex(["2026-01-15", "2026-04-15"], name="DateTime"),
        name="Strike Price (Indexed)",
    )
    pd.testing.assert_series_equal(result, expected)

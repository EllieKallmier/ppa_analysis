import pandas as pd
import pytest

from ppa_analysis.advanced_settings import DISCOUNT_RATE
from ppa_analysis.helper_functions import (
    _check_interval_consistency,
    _check_missing_data,
    calculate_lcoe,
    check_leap_year,
    concat_shaped_profiles,
    get_all_lcoes,
    get_interval_length,
    get_load_data_chunk,
    get_percentile_profile,
    get_seasons,
    get_weekends,
    quarterly_indexation,
    yearly_indexation,
)


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


def test_calculate_lcoe_isolates_capital_term():
    # Fixed and Variable O&M zeroed out so only the capital-recovery term contributes.
    # Economic Life = 1 year makes the denominator (1+r)^1 - 1 = r, which cancels the
    # discount_rate factor in the numerator -- leaving a hand-checkable expression:
    # capital * (1+r)^(construction_years + economic_life) / (8760 * capacity_factor)
    # = 1,000,000 * 1.07 / 8760
    generator_info = {
        "Capital ($/kW)": 1000.0,
        "Construction Time (years)": 0,
        "Economic Life (years)": 1,
        "Fixed O&M ($/kW)": 0.0,
        "Variable O&M ($/kWh)": 0.0,
        "Capacity Factor": 1.0,
    }

    result = calculate_lcoe(generator_info)

    assert result == pytest.approx(1_000_000 * (1 + DISCOUNT_RATE) / 8760)


def test_calculate_lcoe_isolates_om_term():
    # Capital ($/kW) = 0 zeroes out the capital-recovery term, leaving only:
    # variable_om * (fixed_om * 1000) / (8760 * capacity_factor) = 0.02 * 100,000 / 4380
    generator_info = {
        "Capital ($/kW)": 0.0,
        "Construction Time (years)": 5,
        "Economic Life (years)": 25,
        "Fixed O&M ($/kW)": 100.0,
        "Variable O&M ($/kWh)": 0.02,
        "Capacity Factor": 0.5,
    }

    result = calculate_lcoe(generator_info)

    assert result == pytest.approx(0.02 * 100_000 / 4380)


def test_get_all_lcoes_wraps_each_generator_and_skips_out_key():
    # This is needed because the current setup is designed for use with Jupyter
    # notebook widgets, which needs an 'out' entry that these calcs should ignore.
    capital_only = {
        "Capital ($/kW)": 1000.0,
        "Construction Time (years)": 0,
        "Economic Life (years)": 1,
        "Fixed O&M ($/kW)": 0.0,
        "Variable O&M ($/kWh)": 0.0,
        "Capacity Factor": 1.0,
    }
    om_only = {
        "Capital ($/kW)": 0.0,
        "Construction Time (years)": 5,
        "Economic Life (years)": 25,
        "Fixed O&M ($/kW)": 100.0,
        "Variable O&M ($/kWh)": 0.02,
        "Capacity Factor": 0.5,
    }
    generator_data_dict = {
        "Wind": capital_only,
        "Solar": om_only,
        "out": {"anything": "here should be skipped"},
    }

    result = get_all_lcoes(generator_data_dict)

    assert result == {
        "Wind": calculate_lcoe(capital_only),
        "Solar": calculate_lcoe(om_only),
    }


def test_get_percentile_profile_yearly_groups_by_hour_only(csv_str_to_df):
    # Two days, each contributing one value per hour -> the median (percentile=0.5) at
    # each hour is just the average of the two days' values for that hour.
    data = csv_str_to_df(
        """
        DateTime,           Generation
        2026-01-01 00:00,   10.0
        2026-01-02 00:00,   20.0
        2026-01-01 01:00,   30.0
        2026-01-02 01:00,   40.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = get_percentile_profile("Y", data, 0.5)

    expected = pd.DataFrame(
        {"Generation": [15.0, 35.0]}, index=pd.Index([0, 1], name="Hour")
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_percentile_profile_monthly_groups_by_month_and_hour(csv_str_to_df):
    # Two months, two days each, so each (Month, Hour) group has two values to median.
    data = csv_str_to_df(
        """
        DateTime,           Generation
        2026-01-01 00:00,   10.0
        2026-01-15 00:00,   20.0
        2026-01-01 01:00,   30.0
        2026-01-15 01:00,   40.0
        2026-02-01 00:00,   50.0
        2026-02-15 00:00,   60.0
        2026-02-01 01:00,   70.0
        2026-02-15 01:00,   80.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = get_percentile_profile("M", data, 0.5)

    expected = pd.DataFrame(
        {"Generation": [15.0, 35.0, 55.0, 75.0]},
        index=pd.MultiIndex.from_tuples(
            [(1, 0), (1, 1), (2, 0), (2, 1)], names=["Month", "Hour"]
        ),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_percentile_profile_quarterly_groups_by_quarter_and_hour(csv_str_to_df):
    # Same shape as the monthly case, but January/April so the two days land in
    # different quarters (Q1/Q2) rather than different months within the same quarter.
    data = csv_str_to_df(
        """
        DateTime,           Generation
        2026-01-01 00:00,   10.0
        2026-01-15 00:00,   20.0
        2026-01-01 01:00,   30.0
        2026-01-15 01:00,   40.0
        2026-04-01 00:00,   50.0
        2026-04-15 00:00,   60.0
        2026-04-01 01:00,   70.0
        2026-04-15 01:00,   80.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = get_percentile_profile("Q", data, 0.5)

    expected = pd.DataFrame(
        {"Generation": [15.0, 35.0, 55.0, 75.0]},
        index=pd.MultiIndex.from_tuples(
            [(1, 0), (1, 1), (2, 0), (2, 1)], names=["Quarter", "Hour"]
        ),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_concat_shaped_profiles_yearly_broadcasts_by_hour(csv_str_to_df):
    # Same shaped values as the get_percentile_profile 'Y' test above -- here checking
    # that concat_shaped_profiles broadcasts them onto a full timeseries by hour alone,
    # regardless of which day/month each row falls on.
    shaped_data = pd.DataFrame(
        {"Generation": [15.0, 35.0]}, index=pd.Index([0, 1], name="Hour")
    )
    long_data = csv_str_to_df(
        """
        DateTime
        2026-03-01 00:00
        2026-03-01 01:00
        2026-03-02 00:00
        """,
        parse_dates=["DateTime"],
    )

    result = concat_shaped_profiles("Y", shaped_data, long_data)

    expected = csv_str_to_df(
        """
        DateTime,           Generation
        2026-03-01 00:00,   15.0
        2026-03-01 01:00,   35.0
        2026-03-02 00:00,   15.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_concat_shaped_profiles_monthly_broadcasts_by_month_and_hour(csv_str_to_df):
    shaped_data = pd.DataFrame(
        {"Generation": [15.0, 35.0, 55.0, 75.0]},
        index=pd.MultiIndex.from_tuples(
            [(1, 0), (1, 1), (2, 0), (2, 1)], names=["Month", "Hour"]
        ),
    )
    long_data = csv_str_to_df(
        """
        DateTime
        2026-01-05 00:00
        2026-01-05 01:00
        2026-02-10 00:00
        2026-02-10 01:00
        """,
        parse_dates=["DateTime"],
    )

    result = concat_shaped_profiles("M", shaped_data, long_data)

    expected = csv_str_to_df(
        """
        DateTime,           Generation
        2026-01-05 00:00,   15.0
        2026-01-05 01:00,   35.0
        2026-02-10 00:00,   55.0
        2026-02-10 01:00,   75.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_check_leap_year_true_for_leap_year(csv_str_to_df):
    # 2024 has 366 days, so 365 days after Jan 1 lands on Dec 31 (day 31 != day 1).
    df = csv_str_to_df(
        """
        DateTime,    Load
        2024-01-01,  1.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    assert check_leap_year(df) is True


def test_check_leap_year_false_for_non_leap_year(csv_str_to_df):
    # 2023 has 365 days, so 365 days after Jan 1 lands back on Jan 1 the following year.
    df = csv_str_to_df(
        """
        DateTime,    Load
        2023-01-01,  1.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    assert check_leap_year(df) is False


def test_get_seasons_maps_each_month_and_the_dec_jan_boundary(csv_str_to_df):
    # One row per season, plus a December row alongside the January row -- both map to
    # Summer despite falling on either side of a calendar year change.
    df = csv_str_to_df(
        """
        DateTime,    Load
        2026-01-15,  1.0
        2026-04-15,  2.0
        2026-07-15,  3.0
        2026-10-15,  4.0
        2026-12-15,  5.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = get_seasons(df)

    expected = csv_str_to_df(
        """
        DateTime,    Load,  Season
        2026-01-15,  1.0,   Summer
        2026-04-15,  2.0,   Autumn
        2026-07-15,  3.0,   Winter
        2026-10-15,  4.0,   Spring
        2026-12-15,  5.0,   Summer
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_get_weekends_flags_saturday_sunday_and_public_holiday(csv_str_to_df):
    # region follows the AEMO-style code (e.g. 'NSW1') -- get_weekends strips the
    # trailing digit itself (region[:-1]) to get the state subdivision for the
    # holidays package, so 'NSW1' here becomes subdiv 'NSW'.
    # 2026-01-01 (New Year's Day, a Thursday) isolates the public-holiday branch from
    # the day-of-week branch; 2026-01-05 (Monday) is an ordinary weekday.
    df = csv_str_to_df(
        """
        DateTime,    Load
        2026-01-01,  1.0
        2026-01-03,  2.0
        2026-01-04,  3.0
        2026-01-05,  4.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = get_weekends(df, region="NSW1")

    expected = csv_str_to_df(
        """
        DateTime,    Load,  Weekend
        2026-01-01,  1.0,   1
        2026-01-03,  2.0,   1
        2026-01-04,  3.0,   1
        2026-01-05,  4.0,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_concat_shaped_profiles_quarterly_broadcasts_by_quarter_and_hour(csv_str_to_df):
    shaped_data = pd.DataFrame(
        {"Generation": [15.0, 35.0, 55.0, 75.0]},
        index=pd.MultiIndex.from_tuples(
            [(1, 0), (1, 1), (2, 0), (2, 1)], names=["Quarter", "Hour"]
        ),
    )
    long_data = csv_str_to_df(
        """
        DateTime
        2026-02-05 00:00
        2026-02-05 01:00
        2026-05-10 00:00
        2026-05-10 01:00
        """,
        parse_dates=["DateTime"],
    )

    result = concat_shaped_profiles("Q", shaped_data, long_data)

    expected = csv_str_to_df(
        """
        DateTime,           Generation
        2026-02-05 00:00,   15.0
        2026-02-05 01:00,   35.0
        2026-05-10 00:00,   55.0
        2026-05-10 01:00,   75.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_check_missing_data_fills_nan_rows_with_zero_and_logs(csv_str_to_df, caplog):
    df = csv_str_to_df(
        """
        DateTime,    Load
        2026-01-01,  1.0
        2026-01-02,
        2026-01-03,  3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    with caplog.at_level("WARNING"):
        result = _check_missing_data(df)

    assert "Some missing data found. Filled with zeros." in caplog.text
    expected = csv_str_to_df(
        """
        DateTime,    Load
        2026-01-01,  1.0
        2026-01-02,  0.0
        2026-01-03,  3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_check_missing_data_passes_through_unchanged_when_no_nans(
    csv_str_to_df, caplog
):
    df = csv_str_to_df(
        """
        DateTime,    Load
        2026-01-01,  1.0
        2026-01-02,  2.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    with caplog.at_level("WARNING"):
        result = _check_missing_data(df)

    assert caplog.text == ""
    pd.testing.assert_frame_equal(result, df)


def test_check_missing_data_logs_when_df_is_empty(caplog):
    df = pd.DataFrame(columns=["Load"])

    with caplog.at_level("WARNING"):
        _check_missing_data(df)

    assert "DataFrame is empty." in caplog.text


def test_get_interval_length_consistent_intervals_no_log(csv_str_to_df, caplog):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:00,   1.0
        2026-01-01 01:00,   2.0
        2026-01-01 02:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    with caplog.at_level("WARNING"):
        result = get_interval_length(df)

    assert result == 60
    assert caplog.text == ""


def test_get_interval_length_inconsistent_intervals_logs_and_uses_first_gap(
    csv_str_to_df, caplog
):
    # First gap (00:00 -> 01:00) is 60 minutes, last gap (01:00 -> 03:00) is 120 minutes --
    # inconsistent, so the function logs a warning and falls back to the first gap.
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:00,   1.0
        2026-01-01 01:00,   2.0
        2026-01-01 03:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    with caplog.at_level("WARNING"):
        result = get_interval_length(df)

    assert result == 60
    assert "Interval lengths are different throughout dataset." in caplog.text


def test_check_interval_consistency_true_for_consistent_intervals(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:00,   1.0
        2026-01-01 01:00,   2.0
        2026-01-01 02:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    assert _check_interval_consistency(df, mins=60)


def test_check_interval_consistency_false_for_inconsistent_intervals(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:00,   1.0
        2026-01-01 01:00,   2.0
        2026-01-01 03:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    assert not _check_interval_consistency(df, mins=60)


def test_get_load_data_chunk_boundary_exactly_on_a_row(csv_str_to_df):
    # end_date is 2026-01-31, so the true cutoff (end_date + 1 day) is 2026-02-01 00:00 --
    # a row exactly on that timestamp should still land in chunk (inclusive boundary).
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-31 12:00,   1.0
        2026-02-01 00:00,   2.0
        2026-02-01 01:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    chunk, remainder = get_load_data_chunk(df, pd.Timestamp("2026-01-31"))

    expected_chunk = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-31 12:00,   1.0
        2026-02-01 00:00,   2.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    expected_remainder = csv_str_to_df(
        """
        DateTime,           Load
        2026-02-01 01:00,   3.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(chunk, expected_chunk)
    pd.testing.assert_frame_equal(remainder, expected_remainder)


def test_get_load_data_chunk_boundary_falls_between_rows(csv_str_to_df):
    # No row sits exactly on the 2026-02-01 00:00 cutoff here -- the split still falls
    # cleanly either side of it.
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-31 12:00,   1.0
        2026-02-01 12:00,   2.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    chunk, remainder = get_load_data_chunk(df, pd.Timestamp("2026-01-31"))

    expected_chunk = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-31 12:00,   1.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    expected_remainder = csv_str_to_df(
        """
        DateTime,           Load
        2026-02-01 12:00,   2.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(chunk, expected_chunk)
    pd.testing.assert_frame_equal(remainder, expected_remainder)

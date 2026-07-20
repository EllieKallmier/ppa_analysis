import pandas as pd

from ppa_analysis.matching import (
    calc_bulk_match,
    calc_hourly_match,
    calc_unmatched_emissions,
)


def test_calc_hourly_match_caps_at_100(csv_str_to_df):
    # col_to_match_to below, at, and above Load in three separate hours — the third row
    # would be 150% uncapped, so this pins down the np.minimum(..., 100) cap.
    df = csv_str_to_df(
        """
        DateTime,          Load,  Contracted Energy
        2026-01-01 00:00,  100,   50
        2026-01-01 01:00,  100,   100
        2026-01-01 02:00,  100,   150
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calc_hourly_match(
        df,
        col_to_match_to="Contracted Energy",
        resample_period="h",
        load_region="NSW1",
    )

    expected = pd.Series(
        [50.0, 100.0, 100.0],
        index=pd.DatetimeIndex(
            ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"], freq="h"
        ),
        name="Hourly Match %",
    )
    expected.index.name = "DateTime"
    pd.testing.assert_series_equal(result, expected)


def test_calc_hourly_match_load_zero_gives_100_percent(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          Load,  Contracted Energy
        2026-01-01 00:00,  0,     20
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calc_hourly_match(
        df,
        col_to_match_to="Contracted Energy",
        resample_period="h",
        load_region="NSW1",
    )

    expected = pd.Series(
        [100.0],
        index=pd.DatetimeIndex(["2026-01-01 00:00"], freq="h"),
        name="Hourly Match %",
    )
    expected.index.name = "DateTime"
    pd.testing.assert_series_equal(result, expected)


def test_calc_hourly_match_resamples_as_mean_of_capped_ratios(csv_str_to_df):
    # Two hours in the same day: 50% and a capped 100% (150% uncapped). Resampling to 'D'
    # averages the already-capped per-row percentages, giving 75% — not a sum-then-divide
    # over the day's totals. See test_calc_bulk_match_resamples_before_dividing for the
    # contrasting behaviour of calc_bulk_match on the same input.
    df = csv_str_to_df(
        """
        DateTime,          Load,  Contracted Energy
        2026-01-01 00:00,  100,   50
        2026-01-01 12:00,  100,   150
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calc_hourly_match(
        df,
        col_to_match_to="Contracted Energy",
        resample_period="D",
        load_region="NSW1",
    )

    expected = pd.Series(
        [75.0],
        index=pd.DatetimeIndex(["2026-01-01"], freq="D"),
        name="Hourly Match %",
    )
    expected.index.name = "DateTime"
    pd.testing.assert_series_equal(result, expected)


def test_calc_bulk_match_resamples_before_dividing(csv_str_to_df):
    # calc_bulk_match sums Load and Contracted Energy across the day first (200 and 200),
    # then divides — giving 100%, not the 75% that calc_hourly_match gives on identical
    # input. This is the behavioural distinction between the two functions: resample-then-
    # ratio (here) vs ratio-then-resample (calc_hourly_match).
    df = csv_str_to_df(
        """
        DateTime,          Load,  Contracted Energy
        2026-01-01 00:00,  100,   50
        2026-01-01 12:00,  100,   150
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calc_bulk_match(
        df,
        col_to_match_to="Contracted Energy",
        resample_period="D",
        load_region="NSW1",
    )

    expected = pd.Series(
        [100.0],
        index=pd.DatetimeIndex(["2026-01-01"], freq="D"),
        name="Bulk Match %",
    )
    expected.index.name = "DateTime"
    pd.testing.assert_series_equal(result, expected)


def test_calc_unmatched_emissions_clips_negative_to_zero_and_sums_over_period(
    csv_str_to_df,
):
    # Row 1: Load > col_to_match_to -> positive emissions (40 * 0.5 = 20).
    # Row 2: Load <= col_to_match_to -> shortfall is negative, clipped to 0 emissions.
    # Both rows fall in the same day, so the resample sum (20 + 0 = 20) also exercises
    # the resample step in the same test.
    df = csv_str_to_df(
        """
        DateTime,          Load,  Contracted Energy,  AEI: NSW1
        2026-01-01 00:00,  100,   60,                  0.5
        2026-01-01 12:00,  100,   120,                 0.5
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calc_unmatched_emissions(
        df,
        col_to_match_to="Contracted Energy",
        resample_period="D",
        load_region="NSW1",
    )

    expected = pd.Series(
        [20.0],
        index=pd.DatetimeIndex(["2026-01-01"], freq="D"),
        name="Emissions",
    )
    expected.index.name = "DateTime"
    pd.testing.assert_series_equal(result, expected)

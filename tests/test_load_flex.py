import pandas as pd

from ppa_analysis.load_flex import (
    _create_base_days,
    _get_daily_load_sums,
    daily_load_shifting,
)


def test_get_daily_load_sums_resamples_by_day(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:00,   1.0
        2026-01-01 01:00,   2.0
        2026-01-01 02:00,   3.0
        2026-01-02 00:00,   4.0
        2026-01-02 01:00,   5.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = _get_daily_load_sums(df)

    expected = pd.Series(
        [6.0, 9.0],
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], name="DateTime", freq="D"),
        name="Load",
    )
    pd.testing.assert_series_equal(result, expected)


def test_create_base_days_splits_weekday_weekend_and_groups_by_month_hour(
    csv_str_to_df,
):
    df = csv_str_to_df(
        """
        DateTime,           Load,   Weekend
        2026-01-05 00:00,   10.0,   0
        2026-01-12 00:00,   30.0,   0
        2026-01-03 00:00,   100.0,  1
        2026-01-04 00:00,   300.0,  1
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    base_weekday, base_weekend = _create_base_days(df, base_load_quantile=0.5)

    expected_weekday = pd.DataFrame({"Month": [1], "Hour": [0], "Load": [20.0]})
    pd.testing.assert_frame_equal(base_weekday, expected_weekday)

    expected_weekend = pd.DataFrame({"Month": [1], "Hour": [0], "Load": [200.0]})
    pd.testing.assert_frame_equal(base_weekend, expected_weekend)


def test_daily_load_shifting_moves_flexible_load_to_the_cheaper_hour(csv_str_to_df):
    # Two full 24-hour weekdays, hour-ending timestamps (2026-01-05 01:00 is
    # the first interval of day 1, 2026-01-06 00:00 is its last).
    # base_load_quantile=0.0 makes each hour's "base load" the minimum of
    # that hour across the two days, so day 2 exists only to shape day 1's
    # base profile - it has no flexibility of its own.
    #
    # Only the interesting hours are shown below; every other hour on both
    # days is flat (Load=10, Contracted Energy=10, RRP=100) and passes
    # straight through unchanged (Load dispatch=0, Firming=0, Load with
    # flex=10):
    #
    #             Load  Contracted  Base    Load      Load
    #                   Energy      load    dispatch  with flex
    #   Day1 04:00  10      4        4         0         4    <- stays dipped: dispatching
    #                                                           here would cost (CE only 4)
    #   Day1 06:00   4     16        4         6        10    <- filled instead: free (CE=16)
    #
    #   Day2 04:00   4     10        4         0         4    <- pulls day1's 04:00 base down
    #   Day2 06:00   4     10        4         0         4    <- matches day1's own 06:00 dip,
    #                                                           so day1's dip there stays "own load"
    #
    # The day's total load is fixed and dispatch is otherwise capped at each
    # day's own peak (10), so the 6 MWh of flexible dispatch freed up by the
    # 04:00 dip has to land somewhere - the optimiser puts it at the cheaper
    # hour (06:00) rather than restoring 04:00.
    day1_index = pd.date_range("2026-01-05 01:00", periods=24, freq="h")
    day2_index = pd.date_range("2026-01-06 01:00", periods=24, freq="h")

    load1 = [10.0] * 24
    load1[5] = 4.0  # 06:00
    contracted1 = [10.0] * 24
    contracted1[3] = 4.0  # 04:00
    contracted1[5] = 16.0  # 06:00
    rrp1 = [100.0] * 24

    load2 = [10.0] * 24
    load2[3] = 4.0  # 04:00
    load2[5] = 4.0  # 06:00
    contracted2 = [10.0] * 24
    rrp2 = [100.0] * 24

    df = pd.DataFrame(
        {
            "Load": load1 + load2,
            "Contracted Energy": contracted1 + contracted2,
            "RRP": rrp1 + rrp2,
        },
        index=day1_index.append(day2_index),
    )
    df.index.name = "DateTime"

    timeseries_out, results_df = daily_load_shifting(
        df, base_load_quantile=0.0, raise_price=0.001
    )

    dispatch1 = [0.0] * 24
    dispatch1[5] = 6.0
    contracted_out1 = contracted1
    original1 = load1
    base1 = [10.0] * 24
    base1[3] = 4.0  # pulled down by day 2's 04:00 dip
    base1[5] = 4.0  # equals day 1's own 06:00 dip, so not flagged flexible
    firming1 = [0.0] * 24
    flex1 = [10.0] * 24
    flex1[3] = 4.0  # 04:00 stays dipped - the flexible load moved away from here

    expected_day1 = pd.DataFrame(
        {
            "Load dispatch": dispatch1,
            "Contracted Energy": contracted_out1,
            "Original load": original1,
            "Base load": base1,
            "Firming": firming1,
            "Load with flex": flex1,
        },
        index=day1_index,
    )

    dispatch2 = [0.0] * 24
    contracted_out2 = contracted2
    original2 = load2
    base2 = load2  # day 2 has no flexibility of its own, base equals its own load
    flex2 = load2
    expected_day2 = pd.DataFrame(
        {
            "Load dispatch": dispatch2,
            "Contracted Energy": contracted_out2,
            "Original load": original2,
            "Base load": base2,
            "Firming": [float("nan")] * 24,
            "Load with flex": flex2,
        },
        index=day2_index,
    )

    expected_results = pd.concat([expected_day1, expected_day2])
    expected_results.index.name = "DateTime"
    expected_results.index.freq = None
    pd.testing.assert_frame_equal(results_df, expected_results, check_dtype=False)

    expected_timeseries = pd.DataFrame(
        {
            "Load": load1 + load2,
            "Contracted Energy": contracted1 + contracted2,
            "RRP": rrp1 + rrp2,
            "Load with flex": flex1 + flex2,
        },
        index=day1_index.append(day2_index),
    )
    expected_timeseries.index.name = "DateTime"
    pd.testing.assert_frame_equal(
        timeseries_out, expected_timeseries, check_dtype=False
    )


def test_daily_load_shifting_passes_partial_day_through_unchanged(csv_str_to_df):
    # Fewer than 24 rows for the day falls outside the len == 24 branch, so
    # no optimisation runs - dispatch is fixed at 0 and Base load is left NaN.
    df = csv_str_to_df(
        """
        DateTime,           Load,   Contracted Energy,  RRP
        2026-01-05 01:00,   10.0,   10.0,                50.0
        2026-01-05 02:00,   12.0,   10.0,                50.0
        2026-01-05 03:00,   8.0,    10.0,                50.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    timeseries_out, results_df = daily_load_shifting(df)

    expected_results = pd.DataFrame(
        {
            "Load dispatch": [0.0, 0.0, 0.0],
            "Contracted Energy": [10.0, 10.0, 10.0],
            "Original load": [10.0, 12.0, 8.0],
            "Base load": [float("nan")] * 3,
            "Firming": [float("nan")] * 3,
            "Load with flex": [float("nan")] * 3,
        },
        index=df.index,
    )
    pd.testing.assert_frame_equal(results_df, expected_results, check_dtype=False)

    expected_timeseries = csv_str_to_df(
        """
        DateTime,           Load,   Contracted Energy,  RRP,    Load with flex
        2026-01-05 01:00,   10.0,   10.0,                50.0,
        2026-01-05 02:00,   12.0,   10.0,                50.0,
        2026-01-05 03:00,   8.0,    10.0,                50.0,
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(
        timeseries_out, expected_timeseries, check_dtype=False
    )

import pandas as pd
import pytest

from ppa_analysis.firming_contracts import (
    choose_firming_type,
    part_wholesale_exposure,
    retail_tariff_contract,
    tariff_firming_col_fill,
    total_wholesale_exposure,
)


def test_total_wholesale_exposure_copies_rrp(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-01 00:00,  50
        2026-01-01 01:00,  120
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = total_wholesale_exposure(df)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-01 00:00,  50,   50
        2026-01-01 01:00,  120,  120
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_part_wholesale_exposure_clips_to_bounds(csv_str_to_df):
    # RRP below lower_bound, inside the bounds, and above upper_bound.
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-01 00:00,  10
        2026-01-01 01:00,  50
        2026-01-01 02:00,  150
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = part_wholesale_exposure(df, upper_bound=100, lower_bound=20)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-01 00:00,  10,   20
        2026-01-01 01:00,  50,   50
        2026-01-01 02:00,  150,  100
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def _tou_component(time_interval, month=(1,), weekday=True, weekend=True, value=0.10):
    return {
        "Month": list(month),
        "TimeIntervals": {"T1": list(time_interval)},
        "Unit": "$/kWh",
        "Value": value,
        "Weekday": weekday,
        "Weekend": weekend,
    }


def test_tariff_firming_col_fill_normal_time_window(csv_str_to_df):
    # 15:00 falls inside the 14:00-20:00 window, 10:00 doesn't. Weekday/Weekend/Month all
    # left permissive so only the time-window logic is under test here.
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 15:00,  0
        2026-01-05 10:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(["14:00", "20:00"], value=0.10)

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 15:00,  100
        2026-01-05 10:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_tariff_firming_col_fill_normalises_24_00_to_00_00(csv_str_to_df):
    # TimeIntervals end of "24:00" should behave as midnight, wrapping the window across
    # the day boundary (22:00 exclusive -> 00:00 inclusive).
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 23:00,  0
        2026-01-06 00:00,  0
        2026-01-05 12:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(["22:00", "24:00"], value=0.08)

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 23:00,  80
        2026-01-06 00:00,  80
        2026-01-05 12:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    # test the start interval set to "24:00" also gets set to "00:00":\
    # (00:00 exclusive -> 12:00 inclusive).
    start_component = _tou_component(["24:00", "12:00"], value=0.02)
    result_start = tariff_firming_col_fill(df, start_component)
    expected_start = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 23:00,  0
        2026-01-06 00:00,  0
        2026-01-05 12:00,  20
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result_start, expected_start, check_dtype=False)


def test_tariff_firming_col_fill_applies_to_whole_day_when_times_equal(csv_str_to_df):
    # start_time == end_time (e.g. an anytime/flat component)
    # skips between_time filtering entirely, so every row in the eligible days gets it.
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 00:00,  0
        2026-01-05 08:00,  0
        2026-01-05 23:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(["14:00", "14:00"], value=0.02)

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 00:00,  20
        2026-01-05 08:00,  20
        2026-01-05 23:00,  20
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_tariff_firming_col_fill_weekday_only(csv_str_to_df):
    # Weekday=True, Weekend=False -> only the Monday row should pick up the charge.
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  0
        2026-01-03 08:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(
        ["00:00", "00:00"], weekday=True, weekend=False, value=0.02
    )

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  20
        2026-01-03 08:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_tariff_firming_col_fill_weekend_only(csv_str_to_df):
    # Weekday=False, Weekend=True -> only the Saturday row should pick up the charge.
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  0
        2026-01-03 08:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(
        ["00:00", "00:00"], weekday=False, weekend=True, value=0.02
    )

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  0
        2026-01-03 08:00,  20
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_tariff_firming_col_fill_month_filtering(csv_str_to_df):
    # Month=[1] -> only the January row should pick up the charge.
    df = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  0
        2026-02-05 08:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    component = _tou_component(["00:00", "00:00"], month=(1,), value=0.02)

    result = tariff_firming_col_fill(df, component)

    expected = csv_str_to_df(
        """
        DateTime,          Firming price
        2026-01-05 08:00,  20
        2026-02-05 08:00,  0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_retail_tariff_contract_flatrate_only(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-05 08:00,  40
        2026-01-05 20:00,  60
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {"Parameters": {"NUOS": {"FlatRate": {"Value": 0.05}}}}

    result = retail_tariff_contract(df, tariff_details)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-05 08:00,  40,   50
        2026-01-05 20:00,  60,   50
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_retail_tariff_contract_tou_only(csv_str_to_df):
    # 15:00 falls inside the TOU window, 10:00 doesn't -> mixed result, no fallback.
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-05 15:00,  40
        2026-01-05 10:00,  60
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {
        "Parameters": {
            "NUOS": {"TOU1": {"T1": _tou_component(["14:00", "20:00"], value=0.10)}}
        }
    }

    result = retail_tariff_contract(df, tariff_details)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-05 15:00,  40,   100
        2026-01-05 10:00,  60,   0
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_retail_tariff_contract_falls_back_to_rrp_when_all_zero(csv_str_to_df):
    # TOU component only applies in June; all rows are January, so Firming price stays 0
    # for every row after the loop -> falls back to RRP instead of returning all zeroes.
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-05 15:00,  45.5
        2026-01-05 10:00,  60.2
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {
        "Parameters": {
            "NUOS": {
                "TOU1": {
                    "T1": _tou_component(["14:00", "20:00"], month=(6,), value=0.10)
                }
            }
        }
    }

    result = retail_tariff_contract(df, tariff_details)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,   Firming price
        2026-01-05 15:00,  45.5,  45.5
        2026-01-05 10:00,  60.2,  60.2
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_choose_firming_type_dispatches_wholesale_exposed(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-01 00:00,  50
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = choose_firming_type("Wholesale exposed", df)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-01 00:00,  50,   50
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_choose_firming_type_dispatches_partially_wholesale_exposed(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-01 00:00,  150
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = choose_firming_type(
        "Partially wholesale exposed", df, upper_bound=100, lower_bound=20
    )

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-01 00:00,  150,  100
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_choose_firming_type_dispatches_retail(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-05 08:00,  40
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {"Parameters": {"NUOS": {"FlatRate": {"Value": 0.05}}}}

    result = choose_firming_type("Retail", df, tariff_details=tariff_details)

    expected = csv_str_to_df(
        """
        DateTime,          RRP,  Firming price
        2026-01-05 08:00,  40,   50
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_choose_firming_type_raises_on_invalid_type(csv_str_to_df):
    df = csv_str_to_df(
        """
        DateTime,          RRP
        2026-01-01 00:00,  50
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    with pytest.raises(ValueError, match="firming_type must be one of"):
        choose_firming_type("Not a real firming type", df)

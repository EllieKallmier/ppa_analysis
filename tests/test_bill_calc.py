import pandas as pd

from ppa_analysis.bill_calc import (
    calculate_firming,
    calculate_ppa,
    calculate_tariff_bill,
)


def _minimal_flatrate_tariff(rate: float) -> dict:
    """A minimal tariff dict shaped the way tariff_bill_calculator expects to receive it.

    In the real pipeline, tariffs arrive here already passed through
    helper_functions.get_selected_tariff -> tariffs.convert_network_tariff_to_retail_tariff,
    which flattens 'Parameters' out from under 'NUOS' and sets ProviderType to 'Retailer'
    (done for both Network and Retail tariffs -- see that function's docstring/source).
    tariffs.py is out of scope for this test pass, so this fixture exists purely to satisfy
    calculate_tariff_bill's dependency on that shape -- it's not testing tariff correctness,
    just enough of a FlatRate component to drive a bill through.
    """
    return {
        "ProviderType": "Retailer",
        "Parameters": {
            "FlatRate": {"Unit": "$/kWh", "Value": rate},
        },
    }


def test_calculate_tariff_bill_network_dispatch(csv_str_to_df):
    # Two months, one row each, deliberately at 00:30 (not on the month boundary itself) --
    # see calculate_firming_retail below for why that timing matters for chunking.
    load = csv_str_to_df(
        """
        DateTime,           Load
        2026-01-01 00:30,   0.1
        2026-02-01 00:30,   0.2
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {"Network": _minimal_flatrate_tariff(0.10)}

    result = calculate_tariff_bill(load, "M", tariff_details, "Network")

    # Load is converted MWh -> kWh inside calculate_tariff_bill, so 0.1 MWh = 100 kWh.
    expected = pd.DataFrame(
        {"Network Bill ($)": [10.0, 20.0]},
        index=pd.DatetimeIndex(["2026-01-31", "2026-02-28"], name="DateTime", freq="M"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_tariff_bill_retail_dispatch(csv_str_to_df):
    unmatched = csv_str_to_df(
        """
        DateTime,           Unmatched Energy
        2026-01-01 00:30,   0.05
        2026-02-01 00:30,   0.1
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {"Retail": _minimal_flatrate_tariff(0.30)}

    result = calculate_tariff_bill(unmatched, "M", tariff_details, "Retail")

    expected = pd.DataFrame(
        {"Retail Bill ($)": [15.0, 30.0]},
        index=pd.DatetimeIndex(["2026-01-31", "2026-02-28"], name="DateTime", freq="M"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_firming_non_retail_uses_firming_price_column(csv_str_to_df):
    # calculate_firming only branches on firming_type == "Retail" vs anything else -- both
    # 'Wholesale exposed' and 'Partially wholesale exposed' take this same arithmetic path
    # (they differ upstream, in what choose_firming_type already put in 'Firming price';
    # that's covered in test_firming_contracts.py). One case here is enough to cover the
    # branch.
    df = csv_str_to_df(
        """
        DateTime,          Load,   Contracted Energy,  Firming price
        2026-01-01 00:00,  100.0,  60.0,                80.0
        2026-01-01 12:00,  100.0,  120.0,                80.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calculate_firming(df, "Wholesale exposed", {}, "D")

    # Row 1: Unmatched = max(100-60, 0) = 40 -> Firming Costs = 40 * 80 = 3200
    # Row 2: Unmatched = max(100-120, 0) = 0 -> Firming Costs = 0
    # Resampled (summed) over the single day these both fall in.
    expected = pd.DataFrame(
        {
            "Load": [200.0],
            "Contracted Energy": [180.0],
            "Firming price": [160.0],
            "Unmatched Energy": [40.0],
            "Firming Costs": [3200.0],
        },
        index=pd.DatetimeIndex(["2026-01-01"], name="DateTime", freq="D"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_firming_retail_routes_through_tariff_bill(csv_str_to_df):
    # Unlike the non-retail case, 'Unmatched Energy' isn't supplied directly here -- it's
    # computed inside calculate_firming itself before being handed to calculate_tariff_bill.
    df = csv_str_to_df(
        """
        DateTime,           Load,   Contracted Energy
        2026-01-01 00:30,   0.15,   0.10
        2026-02-01 00:30,   0.30,   0.20
        """,
        index_col="DateTime",
        parse_dates=True,
    )
    tariff_details = {"Retail": _minimal_flatrate_tariff(0.30)}

    result = calculate_firming(df, "Retail", tariff_details, "M")

    # Unmatched = max(Load - Contracted Energy, 0): Jan 0.05 MWh (50 kWh), Feb 0.1 MWh (100 kWh)
    # Bill = kWh * 0.30 -> Jan $15.00, Feb $30.00. calculate_tariff_bill's output columns
    # entirely replace the input's, so only 'Firming Costs' survives (renamed from
    # 'Retail Bill ($)').
    expected = pd.DataFrame(
        {"Firming Costs": [15.0, 30.0]},
        index=pd.DatetimeIndex(["2026-01-31", "2026-02-28"], name="DateTime", freq="M"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_ppa_no_indexation_floor_not_binding(csv_str_to_df):
    # Two rows on the same day, resampled with settlement_period='D', so the
    # settlement period sum has to combine both rows' worth of results.
    df = csv_str_to_df(
        """
        DateTime,           RRP,   Contracted Energy
        2026-01-01 00:00,   50.0,  100.0
        2026-01-01 12:00,   60.0,  80.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calculate_ppa(
        df, strike_price=75.0, settlement_period="D", indexation=0, floor_price=-1000.0
    )

    # Row 1: Price = 75 - max(50, -1000) = 25 -> Settlement = 100*25 = 2500, Wholesale = 100*50 = 5000
    # Row 2: Price = 75 - max(60, -1000) = 15 -> Settlement = 80*15 = 1200, Wholesale = 80*60 = 4800
    # Floor never binds here, so Wholesale Cost + PPA Settlement collapses to PPA Value exactly
    # (Final Cost == Value), which is why both columns match below.
    expected = pd.DataFrame(
        {
            "RRP": [110.0],
            "Contracted Energy": [180.0],
            "Strike Price (Indexed)": [150.0],
            "Price": [40.0],
            "Wholesale Cost": [9800.0],
            "PPA Settlement": [3700.0],
            "PPA Value": [13500.0],
            "PPA Final Cost": [13500.0],
        },
        index=pd.DatetimeIndex(["2026-01-01"], name="DateTime", freq="D"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_ppa_floor_price_binds(csv_str_to_df):
    # Two separate days (not resampled together) so each row's result can be checked in
    # isolation: day 1's RRP sits below the floor, day 2's sits above it.
    df = csv_str_to_df(
        """
        DateTime,           RRP,      Contracted Energy
        2026-01-01 00:00,   -20.0,    10.0
        2026-01-02 00:00,   50.0,     10.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calculate_ppa(
        df, strike_price=50.0, settlement_period="D", indexation=0, floor_price=0.0
    )

    # Day 1: floor binds -> Price = 50 - max(-20, 0) = 50 -> Settlement = 10*50 = 500
    #   but Wholesale Cost still uses the raw RRP: 10 * -20 = -200, so Final Cost (300)
    #   diverges from PPA Value (500) -- unlike the no-floor case above.
    # Day 2: floor doesn't bind -> Price = 50 - max(50, 0) = 0 -> Settlement = 0
    expected = pd.DataFrame(
        {
            "RRP": [-20.0, 50.0],
            "Contracted Energy": [10.0, 10.0],
            "Strike Price (Indexed)": [50.0, 50.0],
            "Price": [50.0, 0.0],
            "Wholesale Cost": [-200.0, 500.0],
            "PPA Settlement": [500.0, 0.0],
            "PPA Value": [500.0, 500.0],
            "PPA Final Cost": [300.0, 500.0],
        },
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], name="DateTime", freq="D"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_ppa_yearly_indexation_dispatch(csv_str_to_df):
    # One row per calendar year, RRP=0 so Price/Settlement/Value all read straight off the
    # indexed strike price. This only checks that index_period='Y' wires through to
    # yearly_indexation
    df = csv_str_to_df(
        """
        DateTime,           RRP,   Contracted Energy
        2026-06-01 00:00,   0.0,   1.0
        2027-06-01 00:00,   0.0,   1.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calculate_ppa(
        df,
        strike_price=100.0,
        settlement_period="Y",
        indexation=10.0,
        index_period="Y",
        floor_price=-1000.0,
    )

    # yearly_indexation(strike=100, rate=10%) over 2026/2027 -> [100.0, 110.0]
    expected = pd.DataFrame(
        {
            "RRP": [0.0, 0.0],
            "Contracted Energy": [1.0, 1.0],
            "Strike Price (Indexed)": [100.0, 110.0],
            "Price": [100.0, 110.0],
            "Wholesale Cost": [0.0, 0.0],
            "PPA Settlement": [100.0, 110.0],
            "PPA Value": [100.0, 110.0],
            "PPA Final Cost": [100.0, 110.0],
        },
        index=pd.DatetimeIndex(
            ["2026-12-31", "2027-12-31"], name="DateTime", freq="A-DEC"
        ),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_calculate_ppa_quarterly_indexation_dispatch(csv_str_to_df):
    # Same idea as the yearly dispatch test, but for index_period='Q' -> quarterly_indexation.
    df = csv_str_to_df(
        """
        DateTime,           RRP,   Contracted Energy
        2026-01-01 00:00,   0.0,   1.0
        2026-04-01 00:00,   0.0,   1.0
        """,
        index_col="DateTime",
        parse_dates=True,
    )

    result = calculate_ppa(
        df,
        strike_price=100.0,
        settlement_period="Q",
        indexation=10.0,
        index_period="Q",
        floor_price=-1000.0,
    )

    # quarterly_indexation(strike=100, rate=10%) over 2026 Q1/Q2 -> [100.0, 110.0]
    expected = pd.DataFrame(
        {
            "RRP": [0.0, 0.0],
            "Contracted Energy": [1.0, 1.0],
            "Strike Price (Indexed)": [100.0, 110.0],
            "Price": [100.0, 110.0],
            "Wholesale Cost": [0.0, 0.0],
            "PPA Settlement": [100.0, 110.0],
            "PPA Value": [100.0, 110.0],
            "PPA Final Cost": [100.0, 110.0],
        },
        index=pd.DatetimeIndex(
            ["2026-03-31", "2026-06-30"], name="DateTime", freq="Q-DEC"
        ),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

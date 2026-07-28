import pandas as pd

from ppa_analysis.bill_calc import calculate_firming, calculate_tariff_bill


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

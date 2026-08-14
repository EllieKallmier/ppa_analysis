import pandas as pd
import pytest

from ppa_analysis.hybrid import (
    create_hybrid_generation,
    hybrid_247,
    hybrid_baseload,
    hybrid_pac,
    hybrid_pap,
    hybrid_shaped,
)

GENERATOR_INFO = {"Gen1": 50.0}
N_HOURS = 48


def _load_matches_generator_df():
    # A single generator whose output exactly equals load at every interval,
    # so the "obviously correct" optimum is 100% contracted regardless of
    # contract type. Only a couple of days are needed - each hybrid_* function
    # slices to the "first year" internally, but that slice is just
    # iloc[:24 * days_in_year] and silently takes whatever rows exist if
    # fewer are provided.
    idx = pd.date_range("2026-01-01 01:00", periods=N_HOURS, freq="h")
    return pd.DataFrame(
        {
            "Load": [10.0] * N_HOURS,
            "Gen1": [10.0] * N_HOURS,
            "RRP": [100.0] * N_HOURS,
        },
        index=idx,
    )


def _expected_percentages():
    return {
        "Gen1": {
            "Percent of generator output": pytest.approx(100.0),
            "Percent of hybrid trace": pytest.approx(100.0),
        }
    }


def test_hybrid_pap_fully_contracts_single_matching_generator():
    df, percentages = hybrid_pap(100.0, _load_matches_generator_df(), GENERATOR_INFO, None, None)

    expected = _load_matches_generator_df()
    expected["Hybrid"] = 10.0
    expected["Contracted Energy"] = 10.0
    pd.testing.assert_frame_equal(df, expected, check_exact=False)
    assert percentages == _expected_percentages()


def test_hybrid_pac_fully_contracts_single_matching_generator():
    df, percentages = hybrid_pac(100.0, _load_matches_generator_df(), GENERATOR_INFO, None, None)

    expected = _load_matches_generator_df()
    expected["Hybrid"] = 10.0
    expected["Contracted Energy"] = 10.0
    pd.testing.assert_frame_equal(df, expected, check_exact=False)
    assert percentages == _expected_percentages()


def test_hybrid_247_fully_contracts_single_matching_generator():
    df, percentages = hybrid_247(100.0, _load_matches_generator_df(), GENERATOR_INFO, None, None)

    expected = _load_matches_generator_df()
    expected["Hybrid"] = 10.0
    expected["Contracted Energy"] = 10.0
    pd.testing.assert_frame_equal(df, expected, check_exact=False)
    assert percentages == _expected_percentages()


def test_hybrid_baseload_fully_contracts_single_matching_generator():
    df, percentages = hybrid_baseload(
        100.0, _load_matches_generator_df(), GENERATOR_INFO, "Y", None
    )

    expected = _load_matches_generator_df()
    expected["Contracted Energy"] = 10.0
    expected["Hybrid"] = 10.0
    pd.testing.assert_frame_equal(df, expected, check_exact=False)
    assert percentages == _expected_percentages()


def test_hybrid_shaped_fully_contracts_single_matching_generator():
    df, percentages = hybrid_shaped(
        100.0, _load_matches_generator_df(), GENERATOR_INFO, "Y", 50.0
    )

    expected = _load_matches_generator_df()
    expected["Contracted Energy"] = 10.0
    expected["Hybrid"] = 10.0
    pd.testing.assert_frame_equal(df, expected, check_exact=False)
    assert percentages == _expected_percentages()


def test_create_hybrid_generation_dispatches_to_the_matching_contract_function():
    # Wiring only - hybrid_pap itself is covered above.
    df, percentages = create_hybrid_generation(
        "Pay as Produced", 100.0, _load_matches_generator_df(), GENERATOR_INFO
    )
    assert list(df.columns) == ["Load", "Gen1", "RRP", "Hybrid", "Contracted Energy"]
    assert len(df) == N_HOURS


def test_create_hybrid_generation_raises_for_invalid_contract_type():
    with pytest.raises(ValueError, match="contract_type must be one of"):
        create_hybrid_generation(
            "Not A Contract", 100.0, _load_matches_generator_df(), GENERATOR_INFO
        )


def test_create_hybrid_generation_raises_for_invalid_redef_period():
    with pytest.raises(ValueError, match="redef_period must be one of"):
        create_hybrid_generation(
            "Shaped",
            100.0,
            _load_matches_generator_df(),
            GENERATOR_INFO,
            redef_period="not-a-period",
            percentile_val=50.0,
        )

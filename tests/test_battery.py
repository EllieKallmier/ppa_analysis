import pandas as pd

from ppa_analysis.battery import run_battery_optimisation


def test_run_battery_optimisation_charges_on_excess_gen_and_discharges_on_excess_load(
    caplog,
):
    # Hour 1 has 4 MWh excess generation, hour 2 has 10 MWh excess load at a
    # higher price. The battery starts half-charged (SoE 10/20 MWh) and only
    # needs to draw down to the 4 MWh floor (20% of 20 MWh) to fully cover
    # hour 2's excess load, so charge/discharge amounts are pinned exactly.
    df = pd.DataFrame(
        {
            "Load": [10.0, 6.0, 15.0, 10.0],
            "Contracted Energy": [10.0, 10.0, 5.0, 10.0],
            "RRP": [50.0, 50.0, 100.0, 50.0],
        }
    )

    with caplog.at_level("WARNING"):
        result = run_battery_optimisation(
            df,
            rated_power_capacity=10,
            size_in_mwh=20,
            charging_efficiency=1.0,
            discharging_efficiency=1.0,
        )

    expected = pd.DataFrame(
        {
            "Load": [10.0, 6.0, 15.0, 10.0],
            "Contracted Energy": [10.0, 10.0, 5.0, 10.0],
            "RRP": [50.0, 50.0, 100.0, 50.0],
            "Load with battery": [10.0, 10.0, 5.0, 10.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert caplog.text == ""


def test_run_battery_optimisation_returns_input_unchanged_when_infeasible(caplog):
    # Negative size_in_mwh flips MIN_SOC/MAX_SOC into an infeasible SoE bound
    # (lower bound above upper bound), forcing the solver to fail regardless
    # of the load/generation data.
    # TODO: this is a solid test but probably should be validating against negative
    # size inputs to avoid this case actually happening...
    df = pd.DataFrame(
        {
            "Load": [10.0, 10.0],
            "Contracted Energy": [10.0, 10.0],
            "RRP": [50.0, 50.0],
        }
    )

    with caplog.at_level("WARNING"):
        result = run_battery_optimisation(df, rated_power_capacity=5, size_in_mwh=-10)

    pd.testing.assert_frame_equal(result, df)
    assert "This battery optimisation was infeasible." in caplog.text

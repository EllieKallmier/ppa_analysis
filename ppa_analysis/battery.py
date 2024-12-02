import numpy as np
import pandas as pd
from mip import (
    BINARY,
    CBC,
    CONTINUOUS,
    GUROBI,
    Model,
    OptimizationStatus,
    minimize,
    xsum,
)

from ppa_analysis import advanced_settings


def run_battery_optimisation(
    timeseries_data: pd.DataFrame,
    rated_power_capacity: float,
    size_in_mwh: float,
    charging_efficiency: float = 0.9,
    discharging_efficiency: float = 0.88,
) -> pd.DataFrame:
    """
    Optimises battery dispatch to minimise cost of purchasing energy not covered by a PPA at the wholesale spot price.

    Optimises battery dispatch to store excess energy purchased through a PPA for use at times when the PPA energy is
    not sufficient to cover the load. Charging and discharging decisions are made to minimise the cost of purchasing
    additional energy to meet the load at the wholesale spot price. Optimisation is performed using mixed integer linear
    programing.

    :param timeseries_data: pd.DataFrame containing the load, generation, and price timeseries data. Needs to contain
        columns 'Load' (MWh), 'Contracted Energy' (MWh, the energy being purchased from renewable energy generators
        through the PPA), and a wholesale price column named 'RRP' ($/MWh). The data is assumed to be sorted in
        sequential order and be hourly interval data.
    :param rated_power_capacity: float, the maximum discharging and charging rate of the battery in MW.
    :param size_in_mwh: float, the volume of energy that can be stored in the battery, in MWh. Note, however that the
        optimisation will not fully charge or discharge the battery to minimise degradation, the default limits on the
        state of charge are 20% to 80%, but can be configured in advanced_settings.py, see MIN_SOC and MAX_SOC.
    :param charging_efficiency: float, between 0.0 and 1.0, the fraction of energy which the battery draws from the grid
        that is stored.
    :param discharging_efficiency: float, between 0.0 and 1.0, the fraction of energy which when drawn from the battery
        is delivered to the grid.
    :return: pd.DataFrame, the timeseries data supplied with an additional column 'Load with battery' specifying the
        load after adding the battery charging and discharging.
    """

    # Get the useful traces first - it depends a bit on the order of operations
    # but the load to be used here will either just be 'Load' or 'Shifted load' ??
    wholesale_prices = timeseries_data["RRP"].clip(lower=1.0).values
    excess_load = np.maximum(
        timeseries_data["Load"] - timeseries_data["Contracted Energy"], 0
    ).values
    excess_gen = np.maximum(
        timeseries_data["Contracted Energy"] - timeseries_data["Load"], 0
    ).values

    min_soe = size_in_mwh * advanced_settings.MIN_SOC
    max_soe = size_in_mwh * advanced_settings.MAX_SOC

    if advanced_settings.solver == "GUROBI":
        solver = GUROBI
    elif advanced_settings.solver == "CBC":
        solver = CBC
    else:
        raise ValueError(f"Solver name {advanced_settings.solver} not recognised.")

    m = Model(solver_name=solver)

    len_timeseries = range(len(excess_load))

    battery_discharge = [
        m.add_var(var_type=CONTINUOUS, lb=0.0, ub=rated_power_capacity)
        for i in len_timeseries
    ]
    battery_charge = [
        m.add_var(var_type=CONTINUOUS, lb=0.0, ub=rated_power_capacity)
        for i in len_timeseries
    ]
    soe = [
        m.add_var(var_type=CONTINUOUS, lb=min_soe, ub=max_soe) for i in len_timeseries
    ]

    # Binary coefficients to disallow charging/discharging simultaneously
    charge_coef = [m.add_var(var_type=BINARY) for i in len_timeseries]
    discharge_coef = [m.add_var(var_type=BINARY) for i in len_timeseries]

    # Initial simple objective: minimise cost of firming - firming = excess load - battery discharge
    m.objective = minimize(
        xsum(
            (excess_load[i] - battery_discharge[i]) * (wholesale_prices[i])
            for i in len_timeseries
        )
    )

    # TODO: if needed, update to take into account the time interval (to convert from energy to power or vice versa)

    # Define soe as previous soe + battery charge (*efficiency) - battery discharge (*efficiency)
    for i in range(1, len(excess_load)):
        m += (
            soe[i]
            <= soe[i - 1]
            + charging_efficiency * battery_charge[i - 1]
            - (1 / discharging_efficiency) * battery_discharge[i - 1]
        )

        m += (
            soe[i]
            >= soe[i - 1]
            + charging_efficiency * battery_charge[i - 1]
            - (1 / discharging_efficiency) * battery_discharge[i - 1]
        )

    # Set the initial state of energy to half battery size:
    m += soe[0] <= size_in_mwh * 0.5
    m += soe[0] >= size_in_mwh * 0.5

    # s.t. constraints:
    # 1. Can't charge and discharge at the same time
    for i in len_timeseries:
        m += battery_charge[i] <= excess_gen[i]
        m += battery_discharge[i] <= excess_load[i]
        m += charge_coef[i] + discharge_coef[i] <= 1
        m += battery_charge[i] <= charge_coef[i] * rated_power_capacity
        m += battery_discharge[i] <= discharge_coef[i] * rated_power_capacity

    m.verbose = 0
    status = m.optimize()

    if status == OptimizationStatus.INFEASIBLE:
        print("This battery optimisation was infeasible.")
        m.clear()
        return timeseries_data

    if status == OptimizationStatus.FEASIBLE or status == OptimizationStatus.OPTIMAL:
        # get results:
        battery_discharge_result = [battery_discharge[i].x for i in len_timeseries]
        battery_charge_result = [battery_charge[i].x for i in len_timeseries]
        soe_result = [soe[i].x for i in len_timeseries]
        charge_coef_result = [charge_coef[i].x for i in len_timeseries]
        discharge_coef_result = [discharge_coef[i].x for i in len_timeseries]

        battery_data = timeseries_data.copy()
        battery_data["Discharge"] = battery_discharge_result
        battery_data["Charge"] = battery_charge_result
        battery_data["SoE"] = soe_result
        battery_data["P_c"] = charge_coef_result
        battery_data["P_d"] = discharge_coef_result

        battery_data["Load with battery"] = (
            battery_data["Load"] + battery_data["Charge"] - battery_data["Discharge"]
        )

        timeseries_data["Load with battery"] = battery_data["Load with battery"].copy()

        m.clear()

    return timeseries_data

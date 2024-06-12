import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, CONTINUOUS, BINARY, OptimizationStatus

# TODO: add documentation!!
def run_battery_optimisation(
        df:pd.DataFrame,
        load_col_to_use:str,
        region:str,
        rated_power_capacity:float,
        size_in_mwh:float,
        charging_efficiency:float=0.9,
        discharging_efficiency:float=0.88
) -> pd.DataFrame:
    
    # Get the useful traces first - it depends a bit on the order of operations
    # but the load to be used here will either just be 'Load' or 'Shifted load' ??
    wholesale_prices = df[f'RRP: {region}'].clip(lower=1.0).values
    excess_load = np.maximum(df[load_col_to_use] - df['Contracted Energy'], 0).values
    excess_gen = np.maximum(df['Contracted Energy'] - df[load_col_to_use], 0).values

    min_soe = size_in_mwh * 0.2
    max_soe = size_in_mwh * 0.8

    m = Model()
    I = range(len(excess_load))

    battery_discharge = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=rated_power_capacity) for i in I]
    battery_charge = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=rated_power_capacity) for i in I]
    soe = [m.add_var(var_type=CONTINUOUS, lb=min_soe, ub=max_soe) for i in I]

    # Binary coefficients to disallow charging/discharging simultaneously
    charge_coef = [m.add_var(var_type=BINARY) for i in I]
    discharge_coef = [m.add_var(var_type=BINARY) for i in I]

    # Initial simple objective: minimise cost of firming - firming = excess load - battery discharge
    m.objective = minimize(xsum((excess_load[i] - battery_discharge[i]) * (wholesale_prices[i]) for i in I))

    # TODO: if needed, update to take into account the time interval (to convert from energy to power or vice versa)

    # Define soe as previous soe + battery charge (*efficiency) - battery discharge (*efficiency)
    for i in range(1, len(excess_load)):
        m += soe[i] <= soe[i-1] + charging_efficiency*battery_charge[i-1] - \
            (1/discharging_efficiency)*battery_discharge[i-1]
        
        m += soe[i] >= soe[i-1] + charging_efficiency*battery_charge[i-1] - \
            (1/discharging_efficiency)*battery_discharge[i-1]

    # Set the initial state of energy to half battery size:
    m += soe[0] <= size_in_mwh * 0.5
    m += soe[0] >= size_in_mwh * 0.5

    # s.t. constraints:
    # 1. Can't charge and discharge at the same time
    for i in I:
        m += battery_charge[i] <= excess_gen[i]
        m += battery_discharge[i] <= excess_load[i]
        m += charge_coef[i] + discharge_coef[i] <= 1 
        m += battery_charge[i] <= charge_coef[i] * rated_power_capacity 
        m += battery_discharge[i] <= discharge_coef[i] * rated_power_capacity

    m.verbose = 0
    status = m.optimize()

    if status == OptimizationStatus.INFEASIBLE:
        print('This battery optimisation was infeasible.')
        m.clear()
        return df

    if status == OptimizationStatus.FEASIBLE or status == OptimizationStatus.OPTIMAL:
        # get results:
        battery_discharge_result = [battery_discharge[i].x for i in I]
        battery_charge_result = [battery_charge[i].x for i in I]
        soe_result = [soe[i].x for i in I]
        charge_coef_result = [charge_coef[i].x for i in I]
        discharge_coef_result = [discharge_coef[i].x for i in I]

        battery_data = df.copy()
        battery_data['Discharge'] = battery_discharge_result
        battery_data['Charge'] = battery_charge_result
        battery_data['SoE'] = soe_result
        battery_data['P_c'] = charge_coef_result
        battery_data['P_d'] = discharge_coef_result

        battery_data['Load with battery'] = battery_data[load_col_to_use] + \
            battery_data['Charge'] - battery_data['Discharge']
        
        # TODO: add a validation test for this optimisation
        df['Load with battery'] = battery_data['Load with battery'].copy()

        m.clear()

    return df
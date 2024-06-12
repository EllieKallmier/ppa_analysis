# CONTEXT NOTE:
# This file should help supply the CONTRACTED generation profile to the rest
# of the functions throughout the tool. 


import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, OptimizationStatus, CONTINUOUS, CBC
from ppa_analysis.helper_functions import *

# ---------------------------- HYBRID OPTIMISATION -----------------------------

# def run_hybrid_optimisation
def run_hybrid_optimisation(
        contracted_energy:pd.Series,
        wholesale_prices:pd.Series,
        generation_data:pd.DataFrame,
        gen_costs:dict,
        total_sum:float,
        contract_type:str,
        cfe_score_min:float=0.0
) -> tuple[pd.Series, dict[str:dict[str:float]]]:

    # TODO: consider if this return structure is actually best/fit for purpose here
    gen_names = {}
    gen_data_series = {}
    lcoe = {}
    wholesale_prices_vals = np.array(wholesale_prices.clip(lower=1.0).values)   # clipped to avoid over-valuing energy in negative pricing intervals, and to enforce the physical constraints (need 'some' price at all times otherwise 'unmatched' energy can get huge)

    market_floor = 1000.0  # market price floor value to use as oversupply penalty - this is max. "bad outcome" if buyer is left with excess to sell on, wholesale exposed.

    if contract_type in ['24/7']:#, 'Baseload']:#== '24/7':
        unmatched_penalty = 16600.0  # market price cap to use as penalty for an unmet CFE score - meeting this score is a priority for sellers to mitigate risk
        
    else:
        unmatched_penalty = 0.0

    for _, gen in enumerate(generation_data):
        gen_data_series[str(_)] = generation_data[gen].copy()
        gen_names[str(_)] = gen
        lcoe[str(_)] = gen_costs[gen]

    # Create the optimisation model and set up constants/variables:
    R = range(len(contracted_energy))       # how many time intervals in total
    G = range(len(generation_data.columns))         # how many columns of generators

    m = Model(solver_name=CBC)
    percent_of_generation = {}
    # Add a 'percentage' variable for each generator
    for g in G:
        percent_of_generation[str(g)] = m.add_var(var_type=CONTINUOUS, lb=0.0, ub=1.0)

    # Define each of the key variables, all continuous, including lb==0.0 for all.
    # excess = [m.add_var(var_type=CONTINUOUS, lb=0.0) for r in R]
    unmatched = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub = contracted_energy.max()) for r in R]
    hybrid_gen_sum = [m.add_var(var_type=CONTINUOUS, lb=0.0) for r in R]

    # Penalties added to avoid inflexible constraints:
    oversupply_flip_var = m.add_var(var_type=CONTINUOUS, lb=0.0)
    unmet_cfe_score = m.add_var(var_type=CONTINUOUS, lb=0.0)

    # add the objective: to minimise firming (unmatched)
    m.objective = minimize(
        xsum((unmatched[r]*wholesale_prices_vals[r] + xsum(gen_data_series[str(g)][r]*percent_of_generation[str(g)]*lcoe[str(g)] for g in G)) for r in R) \
        + oversupply_flip_var * market_floor \
        + unmatched_penalty * unmet_cfe_score
    )

    # Add to hybrid_gen_sum variable by adding together each generation trace by the percentage variable
    for r in R:
        m += hybrid_gen_sum[r] <= sum([gen_data_series[str(g)][r] * percent_of_generation[str(g)] for g in G])
        m += hybrid_gen_sum[r] >= sum([gen_data_series[str(g)][r] * percent_of_generation[str(g)] for g in G])

    for r in R:
        m += unmatched[r] >= contracted_energy[r] - hybrid_gen_sum[r]
        m += unmatched[r] <= contracted_energy[r]

    # Add constraint to make sure the hybrid total is greater than or equal to
    # the "total_sum" value - keeps assumption of 100% load met (not matched)
    m += xsum(hybrid_gen_sum[r] for r in R) >= total_sum

    # Set the oversupply variable: multiplied by market floor this disincentivises
    # overcontracting unless specified directly.
    m += oversupply_flip_var >= xsum(hybrid_gen_sum[r] for r in R) - total_sum
    m += unmet_cfe_score >= xsum(unmatched[r] for r in R) - (1 - cfe_score_min) * total_sum
    
    m.verbose = 0
    status = m.optimize()

    hybrid_trace = pd.DataFrame(generation_data)
    hybrid_trace['Hybrid'] = 0
    
    # If the optimisation is infeasible: try again with different constraints based
    # on the contract type. 
    # TODO: get rid of this recursion!!
    if status == OptimizationStatus.INFEASIBLE:
        print('Infeasible problem under current constraints.')
        m.clear()
        return #    maybe need to raise an error here instead/as well? For user benefit?
        
        
    # If the optimisation is solvable (and solved): create the new hybrid trace
    # by multiplying out each gen trace by the percentage variable
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for g in G:         
            hybrid_trace['Hybrid'] += gen_data_series[str(g)] * percent_of_generation[str(g)].x

        results = {}
        for g in G:
            name = gen_names[str(g)]
            details = {
                'Percent of generator output' : round(percent_of_generation[str(g)].x*100, 1),
                'Percent of hybrid trace' : round(
                    sum(percent_of_generation[str(g)].x * gen_data_series[str(g)]) / sum(hybrid_trace['Hybrid']) * 100, 1)
            }

            results[name] = details

        
        # Add some checks to make sure optimisation is running correctly.
        # Check here to see that the variables for 'unmatched' and 'excess' match
        # the actual values (to see that the contraints are working as expected).
        check_df = pd.DataFrame()
        check_df['Contracted'] = contracted_energy.copy()
        check_df['Hybrid Gen'] = hybrid_trace['Hybrid'].copy()

        check_df['Unmatched'] = [unmatched[r].x for r in R]
        check_df['Real Unmatched'] = (check_df['Contracted'] - check_df['Hybrid Gen']).clip(lower=0.0)
        check_df['Check unmatched'] = (check_df['Real Unmatched'].round(2) == check_df['Unmatched'].round(2))
        check_df = check_df[(check_df['Check unmatched'] == False)].copy()

        assert check_df.empty == True, "Unmatched and/or excess variables are not being calculated correctly. Check constraints."

        # clear the model at end of run so memory isn't overworked.
        m.clear()

        return hybrid_trace['Hybrid'], results
    

# 
def hybrid_shaped(
        redef_period:str,
        contracted_amount:float, 
        df:pd.DataFrame,
        region:str,
        generator_info:dict[str:float],
        interval:str,
        percentile_val:float
) -> pd.DataFrame:
    
    if contracted_amount < 0 or contracted_amount > 100:
        raise ValueError('contracted_amount must be a float between 0 - 100')
    
    if percentile_val < 0 or percentile_val > 1.0:
        raise ValueError('percentile_val must be a float between 0 - 1.0.')
    
    percentile_val = 1 - percentile_val

    # also need to find out if it's a leap year:
    leap_year = check_leap_year(df)
    first_year = df.iloc[:24 * (365 + leap_year)].copy()

    # Get the load and gen:
    first_year_load = first_year['Load'].copy()
    first_year_gen = first_year[generator_info.keys()].copy()

    # sum of total load in first year:
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    # Create a new df to hold the shaped (percentile) profiles, make sure timestamps
    # all line up.
    shaped_first_year = pd.DataFrame()
    shaped_first_year['DateTime'] = pd.date_range(
        first_year_load.index[0], 
        first_year_load.index[-1], 
        freq='H'
    )

    # TODO: add commenting detail here to explain what's going on!!
    resampled_gen_data = get_percentile_profile(redef_period, first_year_gen, percentile_val)
    shaped_first_year = concat_shaped_profiles(redef_period, resampled_gen_data, shaped_first_year)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=shaped_first_year.copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
        contract_type='Shaped'
    )

    hybrid_trace_whole_length = pd.DataFrame(columns=['DateTime'])
    hybrid_trace_whole_length['DateTime'] = df.index.copy()

    # Now add the hybrid P[x] profile to df as contracted energy
    resampled_gen_data['Contracted Energy'] = 0
    for name, det_dict in percentages.items():
        if name != 'Hybrid':
            contracted_percent_gen = det_dict['Percent of generator output'] / 100
            resampled_gen_data['Contracted Energy'] += resampled_gen_data[name] * (contracted_percent_gen)
    
    contracted_gen_full_length = concat_shaped_profiles(redef_period, resampled_gen_data, hybrid_trace_whole_length)

    df = pd.concat([df, contracted_gen_full_length['Contracted Energy']], axis='columns')

    # Now add the 'actual' hybrid profile (each gen * allocated output %)
    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

    return df, percentages


def hybrid_baseload(
        redef_period:str,
        contracted_amount:float, 
        df:pd.DataFrame,
        region:str,
        generator_info:dict[str:float],
        interval:str,
        percentile_val:float
) -> pd.DataFrame:

    if contracted_amount < 0:
        raise ValueError('contracted_amount must be greater than 0.')
    
    # also need to find out if it's a leap year:
    leap_year = check_leap_year(df)
    first_year = df.iloc[:24 * (365 + leap_year)].copy()

    # Resample to hourly load, then take the hourly average per chosen period
    first_year_load = first_year['Load'].copy()

    # Use a map to allocate hourly values across all years of load data:
    if redef_period == 'Y':
        avg_hourly_load = first_year_load.mean(numeric_only=True)

        # the contracted energy needs to be updated by the contracted_amount percentage:
        df['Contracted Energy'] = round(avg_hourly_load) * (contracted_amount / 100)
    
    else:
        # the contracted energy needs to be updated by the contracted_amount percentage:
        avg_hourly_load = pd.DataFrame(first_year_load.resample(redef_period).mean(numeric_only=True) * (contracted_amount / 100))

        avg_hourly_load['Load'] = avg_hourly_load['Load']
        avg_hourly_load['M'] = avg_hourly_load.index.month
        avg_hourly_load['Q'] = avg_hourly_load.index.quarter

        map_dict = dict(zip(avg_hourly_load[redef_period], avg_hourly_load['Load']))

        df['M'] = df.index.month
        df['Q'] = df.index.quarter

        df['Contracted Energy'] = df[redef_period].copy()
        df['Contracted Energy'] = df['Contracted Energy'].map(map_dict)

        df = df.drop(columns=['M', 'Q'])

    first_year = df.iloc[:24 * (365 + leap_year)].copy()
    
    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contract_type='Baseload',
        contracted_energy=first_year['Contracted Energy'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load.sum(numeric_only=True),
        # cfe_score_min=0.9
        # consider adding here a cfe_score_min to enfore a penalty on meeting the contracted trace...
    )

    first_year = pd.concat([first_year, hybrid_trace_series], axis='columns')

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

    return df, percentages

## The contracted amount for a 24/7 PPA gives the minimum cfe score
def hybrid_247(
        redef_period:str,
        contracted_amount:float, 
        df:pd.DataFrame,
        region:str,
        generator_info:dict[str:float],
        interval:str,
        percentile_val:float
) -> pd.DataFrame:

    if contracted_amount < 0 or contracted_amount > 100:
        raise ValueError('contracted_amount must be a float between 0-100')

    # also need to find out if it's a leap year:
    leap_year = check_leap_year(df)
    first_year = df.iloc[:24 * (365 + leap_year)].copy()

    # Get first year load (and total sum):
    first_year_load = first_year['Load'].copy()
    first_year_load_sum = first_year_load.sum(numeric_only=True)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year['Load'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
        contract_type='24/7',
        cfe_score_min=contracted_amount/100
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

    df['Contracted Energy'] = df['Hybrid'].copy()

    return df, percentages

def hybrid_pap(
        redef_period:str,
        contracted_amount:float, 
        df:pd.DataFrame,
        region:str,
        generator_info:dict[str:float],
        interval:str,
        percentile_val:float
) -> pd.DataFrame:

    if contracted_amount < 0:
        raise ValueError('contracted_amount must be greater than 0.')

    # also need to find out if it's a leap year:
    leap_year = check_leap_year(df)
    first_year = df.iloc[:24 * (365 + leap_year)].copy()

    # Get first year load (and total sum):
    first_year_load = first_year['Load'].copy()
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
        contract_type='Pay as Produced'
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen
    
    df['Contracted Energy'] = df['Hybrid'].copy()

    return df, percentages

def hybrid_pac(
        redef_period:str,
        contracted_amount:float, 
        df:pd.DataFrame,
        region:str,
        generator_info:dict[str:float],
        interval:str,
        percentile_val:float
) -> pd.DataFrame:

    if contracted_amount < 0:
        raise ValueError('contracted_amount must be greater than 0.')

    # Use only the first year of data to create the hybrid/contracted energy trace
    # also need to find out if it's a leap year:
    leap_year = check_leap_year(df)
    first_year = df.iloc[:24 * (365 + leap_year)].copy()

    # Get first year load (and total sum):
    first_year_load = first_year['Load'].copy()
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year['Load'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
        contract_type='Pay as Consumed'
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen
    
    df['Contracted Energy'] = np.minimum(df['Load'], df['Hybrid'])

    return df, percentages

def create_hybrid_generation(
        contract_type:str, # describes contract delivery structure
        redef_period:str, # one of python's offset strings indicating when the contract gets "redefined"
        contracted_amount:float, # a number 0-100(+) indicating a percentage. Definition depends on contract type.
        df:pd.DataFrame, # df containing Load, all gen profiles, prices, emissions.
        region:str,
        generator_info:dict[str:float],
        interval:str, # time interval in minutes that data is currently in
        percentile_val:float # for Shaped contracts only: to define the percentile of generation profiles to match.
) -> pd.DataFrame:
    
    valid_contracts = {'Pay as Produced', 'Pay as Consumed', 'Shaped', 'Baseload', '24/7'}
    if contract_type not in valid_contracts:
        raise ValueError(f'contract_type must be one of {valid_contracts}')
    
    valid_periods = {'M', 'Q', 'Y'}
    if redef_period not in valid_periods:
        raise ValueError(f'redef_period must be one of {valid_periods}')

    opt_hybrid_funcs = {
        'Pay as Produced' : hybrid_pap, 
        'Pay as Consumed' : hybrid_pac,
        'Shaped' : hybrid_shaped, 
        'Baseload' : hybrid_baseload, 
        '24/7' : hybrid_247
    }

    df_with_hybrid = opt_hybrid_funcs[contract_type](redef_period, contracted_amount, df, region, generator_info, interval, percentile_val)

    return df_with_hybrid
# CONTEXT NOTE:
# This file should help supply the CONTRACTED generation profile to the rest
# of the functions throughout the tool. 


import pandas as pd
import numpy as np
import residuals
from mip import Model, xsum, minimize, OptimizationStatus, CONTINUOUS, CBC
from helper_functions import *

# TODO: add error checking

# -------------------------------- HYBRID CALC ---------------------------------
# Helper function to create hybrid generation profiles from a given set of gen 
# profiles.

"""
Parameters:
    - profiles
        -> pandas df containing DateTime column as half-hourly timestamp type 
           and all other columns with generator IDs as header.
    - gen_id_mix
        -> list of tuples with 2 elements: [('Wind', 0.5), ('SOlar', 0.5)]
            - [0] string, generator ID to hybridise together (must be found in 
                column names in above)
            - [1] float, percentage of the generation mix this ID will provide. 
                Given as a float between 0 - 1, e.g, 0.8 (= 80%).
    - mix_name
        -> string, the name you want to give your hybrid mix - this will be the
           column header for the mixed profile.

Returns:
    - gen_profiles
        -> same as input, with added column containing hybrid generation mix

"""
def create_hybrid(profiles, gen_id_mix, mix_name):
    # Return a new df rather than an edited version
    profiles = profiles.copy()
    profiles[mix_name] = 0
    
    # iterate over id_mix, then add each % * profile to mix_name column.
    for id in gen_id_mix:
        profiles[mix_name] += profiles[id[0]] * id[1]

    return profiles



# ------------------------------- SCALING CALC ---------------------------------
# Calculates a new generation profile scaled to a chosen proportion of load

"""
Parameters:
    - profiles
        -> pandas df containing DateTime column as half-hourly timestamp type, a
            load data column,  and all other columns with generator IDs as header.
    - gen_ids
        -> list of strings containing the generator IDs to be scaled.
    - scaling_period
        -> string, one of either "Yearly", "Quarterly", or "Monthly" that indicates
            the time period to re-sample the scaling over. Default = Yearly.
    - scaling_factor
        -> list of floats, can be specified if scaling to a fraction/percentage of the 
            load/gen rather than the total load/gen. Default = [1.0].
            Needs to be a list to accomodate contracts with variable purchasing percentages.
    - ppa_contract
        -> string, either 'Pay As Produced' or 'Pay As Consumed'. Determines whether 
            to scale to a percentage of the total load or the total generation based 
            on the PPA contract volume. As produced == scale to gen (scale to production)
            As consumed == scale to load (scale to consumption).
    - load_id
        -> string, name of the column in profiles that holds load data
    - scale_to_id 
        -> string, default None. Can be specified if contract is more complex and 
            you want to scale to a specific generation ID.

Returns:
    - profiles
        -> same df as input with generator profiles scaled to the load by period 
            and scaling factor. 
"""
def scale_gen_profile(profiles, gen_ids, ppa_contract, load_id, scaling_period="Yearly", scaling_factor=[1.0], scale_to_id=None):
    # return new df rather than editing existing one
    profiles = profiles.copy()

    if scale_to_id != None:
        scale_to_id = scale_to_id
    elif ppa_contract.lower() == 'pay as produced':
        scale_to_id = gen_ids
    elif ppa_contract.lower() == 'pay as consumed':
        scale_to_id = load_id
    else:
        # Keep the backup to scale to load for the moment
        scale_to_id = load_id

    if scaling_period == "Yearly":
        period = 'Y'       # Python offset string meaning year end frequency sampling
        profiles = scaling(profiles, scaling_factor, load_id, gen_ids, period, scale_to=scale_to_id)

    if scaling_period == "Quarterly":
        period = 'Q'        # Python offset string meaning quarter end frequency sampling
        profiles = scaling(profiles, scaling_factor, load_id, gen_ids, period, scale_to=scale_to_id)
            
    if scaling_period == "Monthly":
        period = 'M'        # Python offset string meaning month end frequency sampling
        profiles = scaling(profiles, scaling_factor, load_id, gen_ids, period, scale_to=scale_to_id)
        
    else:
        return profiles
        # Need to consider the scenario with no scaling period specified,
        # whether we throw an error or take the whole dataset with no re-sampling.

    return profiles



def scaling(profile_set, scaling_factor, load_id, gen_ids, period, scale_to):
    profiles = profile_set.copy()
    profiles['DateTime'] = pd.to_datetime(profiles['DateTime'])
    
    # If the scaling period is monthly or quarterly, the function needs to group 
    # by year as well to correctly apply the list of scaling factors to each period.
    scaling_df = profiles.groupby(pd.Grouper(key='DateTime', freq=period)).sum()
    
    # Check that the scaling_factor list is the correct length to match the index.
    # If too short, fill out by repeating the list values up to the correct length.
    # If too long, take the first list items up to the index length.
    if len(scaling_factor) < len(scaling_df.index):
        while len(scaling_factor) < len(scaling_df.index):
            scaling_factor.append(scaling_factor[0])
    elif len(scaling_factor) > len(scaling_df.index):
        scaling_factor = scaling_factor[0:len(scaling_df.index)]
    
    scaling_df['Scaling %'] = scaling_factor
    scale_by = 1    # set default value
    # If scaling to a percentage of total load OR to a specific gen profile:
    if scale_to != gen_ids:
        # First adjust the benchmark (sums of total load or gen) to the scaling % given
        scaling_df['Scale To'] = scaling_df[scale_to] * scaling_df['Scaling %']
        for date in scaling_df.index:
            for gen_id in gen_ids:
                # Check whether the generation is enough to meet demand from the load
                # If not, penalties can be applied later so we don't scale here.
                if scaling_df.loc[date, 'Scale To'] <= scaling_df.loc[date, gen_id]:
                    scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                        scaling_df.loc[date, 'Scale To']/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.to_period(period)==date.to_period(period)), gen_id] = profiles[gen_id] * scale_by

    # Or if scaling to the generation profiles (% or just total):
    else:
        for date in scaling_df.index:
            for gen_id in gen_ids:
                scale_by = scaling_df.loc[date, 'Scaling %']
                profiles.loc[(profiles.DateTime.dt.to_period(period)==date.to_period(period)), gen_id] = profiles[gen_id] * scale_by

    return profiles


# ---------------------------- HYBRID OPTIMISATION -----------------------------

# def run_hybrid_optimisation
def run_hybrid_optimisation(
        contracted_energy:pd.Series,
        wholesale_prices:pd.Series,
        generation_data:pd.DataFrame,
        gen_costs:dict,
        excess_penalty:float,
        total_sum:float,
        contract_type:str,
        cfe_score_min:float=0.0,
        upscale_factor:float=1.0
) -> tuple[pd.Series, dict[str:dict[str:float]]]:

    # TODO: consider if this return structure is actually best/fit for purpose here
    gen_names = {}
    gen_data_series = {}
    lcoe = {}
    wholesale_prices_vals = np.array(wholesale_prices.clip(lower=1.0).values)

    market_floor = 1000.0  # market price floor value to use as oversupply penalty - this is max. "bad outcome" if buyer is left with excess to sell on. 

    if contract_type == '24/7':
        penalty_247 = 16600.0  # market price cap to use as penalty for an unmet CFE score - to emphasise the result
    else:
        penalty_247 = 0.0

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
    excess = [m.add_var(var_type=CONTINUOUS, lb=0.0) for r in R]
    unmatched = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub = contracted_energy.max()) for r in R]
    hybrid_gen_sum = [m.add_var(var_type=CONTINUOUS, lb=0.0) for r in R]

    # Penalties added to avoid inflexible constraints:
    oversupply_flip_var = m.add_var(var_type=CONTINUOUS, lb=0.0)
    unmet_cfe_score = m.add_var(var_type=CONTINUOUS, lb=0.0)

    # add the objective: to minimise firming (unmatched)
    m.objective = minimize(
        xsum((unmatched[r]*wholesale_prices_vals[r] + excess[r]*excess_penalty + xsum(gen_data_series[str(g)][r]*percent_of_generation[str(g)]*lcoe[str(g)] for g in G)) for r in R) \
        + oversupply_flip_var * market_floor \
        + penalty_247 * unmet_cfe_score
    )

    # Add to hybrid_gen_sum variable by adding together each generation trace by the percentage variable
    for r in R:
        m += hybrid_gen_sum[r] <= sum([gen_data_series[str(g)][r] * percent_of_generation[str(g)] for g in G])
        m += hybrid_gen_sum[r] >= sum([gen_data_series[str(g)][r] * percent_of_generation[str(g)] for g in G])

    for r in R:
        m += unmatched[r] >= contracted_energy[r] - hybrid_gen_sum[r]
        m += excess[r] >= hybrid_gen_sum[r] - contracted_energy[r]

    # Add constraint to make sure the hybrid total is greater than or equal to
    # the "total_sum" value - keeps assumption of 100% load met (not matched)
    m += xsum(hybrid_gen_sum[r] for r in R) >= total_sum

    # Set the oversupply variable: multiplid by market cap this disincentivises
    # overcontracting unless specified directly.
    m += oversupply_flip_var >= xsum(hybrid_gen_sum[r] for r in R) - total_sum
    m += unmet_cfe_score >= xsum(unmatched[r] for r in R) - (1 - cfe_score_min) * total_sum

    # Add constraint around CFE matching percent:
    # if contract_type == '24/7':
    #     m += xsum(unmatched[r] for r in R) <= (1 - cfe_score_min) * total_sum
    
    m.verbose = 0
    status = m.optimize()

    hybrid_trace = pd.DataFrame(generation_data)
    hybrid_trace['Hybrid'] = 0
    
    # If the optimisation is infeasible: try again with different constraints based
    # on the contract type.
    if status == OptimizationStatus.INFEASIBLE:
        print('Infeasible problem under current constraints: running again with Pay as Produced structure')
        m.clear()
        return run_hybrid_optimisation(contracted_energy, wholesale_prices, generation_data, gen_costs, excess_penalty, total_sum, 'Pay as Produced')
        
        
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
        check_df['Excess'] = [excess[r].x for r in R]

        check_df['Real Unmatched'] = (check_df['Contracted'] - check_df['Hybrid Gen']).clip(lower=0.0)
        check_df['Real Excess'] = (check_df['Hybrid Gen'] - check_df['Contracted']).clip(lower=0.0)

        check_df['Check unmatched'] = (check_df['Real Unmatched'].round(2) == check_df['Unmatched'].round(2))
        check_df['Check excess'] = (check_df['Real Excess'].round(2) == check_df['Excess'].round(2))

        check_df = check_df[(check_df['Check unmatched'] == False) | (check_df['Check excess'] == False)].copy()

        assert check_df.empty == True, "Unmatched and/or excess variables are not being calculated correctly. Check constraints."

        # clear the model at end of run so memory isn't overworked.
        m.clear()

        return hybrid_trace['Hybrid'], results, upscale_factor
    

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

    hybrid_trace_series, percentages, upscale_factor = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=shaped_first_year.copy(),
        gen_costs=generator_info,
        excess_penalty=0.5,
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

    contracted_gen_full_length *= (upscale_factor + 0.1)

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
    
    hybrid_trace_series, percentages, upscale_factor = run_hybrid_optimisation(
        contract_type='Baseload',
        contracted_energy=first_year['Contracted Energy'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        excess_penalty=0.5,
        total_sum=first_year_load.sum(numeric_only=True)
    )

    first_year = pd.concat([first_year, hybrid_trace_series], axis='columns')

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

    return df, percentages

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
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    hybrid_trace_series, percentages, upscale_factor = run_hybrid_optimisation(
        contracted_energy=first_year['Load'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        excess_penalty=0.5,
        total_sum=first_year_load_sum,
        contract_type='24/7',
        cfe_score_min=contracted_amount/100
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

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

    hybrid_trace_series, percentages, upscale_factor = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        excess_penalty=0.5,     # note: need to add a small (even if negligable) penalty for excess - to enforce calculation of the 'excess' variable in optimisation.
        total_sum=first_year_load_sum,
        contract_type='Pay as Produced'
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

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

    hybrid_trace_series, percentages, upscale_factor = run_hybrid_optimisation(
        contracted_energy=first_year['Load'].copy(),
        wholesale_prices=first_year[f'RRP: {region}'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        excess_penalty=0.5,     # note: need to add a small (even if negligable) penalty for excess - to enforce calculation of the 'excess' variable in optimisation.
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
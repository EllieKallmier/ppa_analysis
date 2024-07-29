"""
This module is used for calculating the optimal contracting mix from a set of renewable energy generators,
given a load profile, a set of generation profiles, generator LCOEs, wholesale prices, a contract type, and other
settings as specified in the relevant function's interfaces. Broadly speaking, the optimisation attempts to match the
combined renewable energy generation profile to the load profile, by optimising the percentage of each generator's
capacity to contract. However, the specific methodology differs for each contract type and is best understood by
reading the documentation for both the contract specific functions (see names below), and the run_hybrid_optimisation
documentation.

Basic usage for the module is through the function hybrid.create_hybrid_generation, which calls one the functions
hybrid_pac, hybrid_shaped, hybrid_baseload, or hybrid_247 to implement the optimisation for the corresponding contract
type. Note these contract specific functions can also be used directly if desired. Lastly, run_hybrid_optimisation
performs the optimisation, and may be useful to the advanced user who want to implement a customised methodology.
"""


import numpy as np
from mip import Model, xsum, minimize, OptimizationStatus, CONTINUOUS, CBC, GUROBI
from ppa_analysis.helper_functions import *
from ppa_analysis import advanced_settings


def run_hybrid_optimisation(
        contracted_energy: pd.Series,
        wholesale_prices: pd.Series,
        generation_data: pd.DataFrame,
        gen_costs: dict[str:float],
        total_sum: float,
        cfe_score_min: float = None,
) -> tuple[pd.Series, dict[str:dict[str:float]]]:
    """
    Calculates an optimal mix of volume to contract from a set of renewable energy generators.

    The fraction of capacity to contract from a set of generators is optimised to lower the cost of procuring energy
    assuming energy is bought from the generators at the specified LCOEs and any energy not covered by the generators
    is bought at the wholesale spot market price. Additionally, the variables total_sum and cfe_score_min can be used
    to influence the optimisation outcomes.

    Renewable energy generation profiles are provided in absolute terms in MWh for each interval in the optimisation
    period. The optimisation decides what fraction of the output to buy from each generator, i.e. if 10% of a generators
    energy is contracted then whatever the instantaneous output of generator is, 10% of its volume will be added to
    the total contracted volume for that interval.

    :param contracted_energy: a pd.Series (i.e. a column from a pandas dataframe) that specifies the load on timeseries
        basis that the optimisation is attempting to match with the generators (in MWh).
    :param wholesale_prices: a pd.Series (i.e. a column from a pandas dataframe) that specifies wholesale spot prices
        in the same market region as the load (in $/MWh), on time series basis.
    :param generation_data: a pd.Dataframe where each column specifies the generation profile of a generator on a time
        series basis. The column names should be the generator names and match the names in the gen_costs dict.
    :param gen_costs: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param total_sum: float, used to place a constraint on the optimisation such that total energy from the generators
        must be greater than or equal to total_sum.
    :param cfe_score_min: a float specifying the threshold at which the contract is considered to be under
        supplied, as a percentage of total_sum (0.0-1.0), if combined generation is below this threshold the
        advanced_settings.UNDERSUPPLY_PENALTY applies in the optimisation objective function. Note this only applies
        if a float rather than the default None is provided.
    :return: A pd.Series specifying the combined contracted profile from all generators and a dict specifying the
        fraction of each generator contracted and fraction of the total energy contracted provided by each generator.
        Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }

    """

    gen_names = {}
    gen_data_series = {}
    lcoe = {}

    # clipped to avoid over-valuing energy in negative pricing intervals, and to enforce the physical constraints
    # (need 'some' price at all times otherwise 'unmatched' energy can get huge)
    wholesale_prices_vals = np.array(wholesale_prices.clip(lower=1.0).values)

    if cfe_score_min is not None:
        # market price cap to use as penalty for an unmet CFE score - meeting this score is a priority for sellers
        # to mitigate risk
        unmatched_penalty = advanced_settings.UNDERSUPPLY_PENALTY
    else:
        unmatched_penalty = 0.0

    for _, gen in enumerate(generation_data):
        gen_data_series[str(_)] = generation_data[gen].copy()
        gen_names[str(_)] = gen
        lcoe[str(_)] = gen_costs[gen]

    # Create the optimisation model and set up constants/variables:
    R = range(len(contracted_energy))       # how many time intervals in total
    G = range(len(generation_data.columns))         # how many columns of generators

    if advanced_settings.solver == 'GUROBI':
        solver = GUROBI
    elif advanced_settings.solver == 'CBC':
        solver = CBC
    else:
        raise ValueError(f'Solver name {advanced_settings.solver} not recognised.')

    m = Model(solver_name=solver)

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
        xsum((unmatched[r] * wholesale_prices_vals[r] +
              xsum(gen_data_series[str(g)][r] * percent_of_generation[str(g)] * lcoe[str(g)] for g in G))
             for r in R)
        + oversupply_flip_var * advanced_settings.OVERSUPPLY_PENALTY
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

    if cfe_score_min is not None:
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
                'Percent of generator output': percent_of_generation[str(g)].x*100,
                'Percent of hybrid trace':
                    sum(percent_of_generation[str(g)].x * gen_data_series[str(g)]) / sum(hybrid_trace['Hybrid']) * 100
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
        contracted_amount:float, 
        df:pd.DataFrame,
        generator_info:dict[str:float],
        redef_period:str,
        percentile_val:float
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a 'Shaped' contract
    type.

    Calls hybrid.run_hybrid_optimisation to perform the optimisation with specific settings for the 'Shaped'
    contract. When calling run_hybrid_optimisation:
        - the contracted_energy parameter is set to the load profile.
        - the generation_data parameter is set to artificial generation profiles created by using the percentile
          value (given in percentile_val) on a yearly, quarterly, or monthly basis according to the redef_period
          ('Y', 'Q', 'M'), e.g. in the quarterly case with percentile_val=0.5, an hourly generation profile would be
          created for each generator where the generation value for each hour was equal to the quarter's median
          generation for that hour of the day.
        - total_sum is set to the sum of load volume across the first year (not adjusted by contract_amount).
        - and wholesale price data and generation data is passed to run_hybrid_optimisation as provided in
          time_series_data

    Also note, only the first year of time series data provided in used in the optimisation.

    :param contracted_amount: float, percentage (fraction between 0.0-1.0) specifying the fraction of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, period over which to average the load to determine the base load, should be one of
        'Y', 'Q', or 'M'.
    :param percentile_val: float, the percentile value to take in each period (yearly, quarterly, monthly) when
        creating the artificial load profile for optimisation, as explained above.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' which is the load profile averaged on a yearly, quarterly, or monthly
          basis as specified by the redef_period parameter.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """
    
    if contracted_amount < 0 or contracted_amount > 100:
        raise ValueError('contracted_amount must be a float between 0 - 100')
    
    if percentile_val < 0 or percentile_val > 100:
        raise ValueError('percentile_val must be a float between 0 - 100.')
    
    percentile_val = 1 - (percentile_val/100)

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

    # Find the load profile percentile on the specified periodic basis (yearly, quarterly, or monthly).
    resampled_gen_data = get_percentile_profile(redef_period, first_year_gen, percentile_val)
    # Join the periodic percentile data with the full time series data such that each time interval has the
    # percentile value for it period.
    shaped_first_year = concat_shaped_profiles(redef_period, resampled_gen_data, shaped_first_year)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year['RRP'].copy(),
        generation_data=shaped_first_year.copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
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
        contracted_amount:float, 
        df:pd.DataFrame,
        generator_info:dict[str:float],
        redef_period:str,
        percentile_val:float
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a 'Baseload' contract
    type.

    Calls hybrid.run_hybrid_optimisation to perform the optimisation with specific settings for the 'Baseload'
    contract. When calling run_hybrid_optimisation:
        - the contracted_energy parameter is set to an artificial load profile created by averaging the provided
          load profile on a yearly, quarterly, or monthly basis according to the redef_period ('Y', 'Q', 'M'). After
          averaging the profile is then adjusted by the contracted_amount parameter.
        - total_sum is set to the sum of load volume across the first year (not adjusted by contract_amount).
        - and wholesale price data and generation data is passed to run_hybrid_optimisation as provided in
          time_series_data

    Also note, only the first year of time series data provided in used in the optimisation.

    :param contracted_amount: float, percentage (fraction between 0.0-1.0) specifying the fraction of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, period over which to average the load to determine the base load, should be one of
        'Y', 'Q', or 'M'.
    :param percentile_val: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' which is the load profile averaged on a yearly, quarterly, or monthly
          basis as specified by the redef_period parameter.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """

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
        contracted_energy=first_year['Contracted Energy'].copy(),
        wholesale_prices=first_year['RRP'].copy(),
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


def hybrid_247(
        contracted_amount:float, 
        df:pd.DataFrame,
        generator_info:dict[str:float],
        redef_period:str,
        percentile_val:float
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a '24/7' contract
    type.

    Calls hybrid.run_hybrid_optimisation to perform the optimisation with specific settings for the '24/7'
    contract. When calling run_hybrid_optimisation:
        - the contracted_energy parameter is set to the load profile (not adjusted by the contracted_amount parameter)
        - total_sum is set to the sum of load volume across the first year (not adjusted by contract_amount).
        - cfe_score_min is set to the contracted amount.
        - and wholesale price data and generation data is passed to run_hybrid_optimisation as provided in
          time_series_data

    Also note, only the first year of time series data provided in used in the optimisation.

    :param contracted_amount: float, percentage (fraction between 0.0-1.0) specifying the fraction of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :param percentile_val: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' which in this case is the same as 'Hybrid' because of the '24/7' contract.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """

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
        wholesale_prices=first_year['RRP'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
        cfe_score_min=contracted_amount/100
    )

    df['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        df['Hybrid'] += df[name] * hybrid_percent_gen

    df['Contracted Energy'] = df['Hybrid'].copy()

    return df, percentages


def hybrid_pap(
        contracted_amount:float,
        time_series_data:pd.DataFrame,
        generator_info:dict[str:float],
        redef_period:str,
        percentile_val:float
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a 'Pay as
    Produced' contract type.

    Calls hybrid.run_hybrid_optimisation to perform the optimisation with specific settings for the 'Pay as Produced'
    contract. When calling run_hybrid_optimisation:
        - the contracted_energy parameter is set to the load profile (not adjusted by the contracted_amount parameter)
        - total_sum is set to the sum of load volume across the first year of load data multiplied by the
          contracted_amount percentage.
        - and wholesale price data and generation data is passed to run_hybrid_optimisation as provided in
          time_series_data

    Also note, only the first year of time series data provided in used in the optimisation.

    :param contracted_amount: float, percentage (fraction between 0.0-1.0) specifying the fraction of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :param percentile_val: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' which in this case is the same as 'Hybrid' because of the 'Pay as
          Produced' contract.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """

    if contracted_amount < 0:
        raise ValueError('contracted_amount must be greater than 0.')

    # also need to find out if it's a leap year:
    leap_year = check_leap_year(time_series_data)
    first_year = time_series_data.iloc[:24 * (365 + leap_year)].copy()

    # Get first year load (and total sum):
    first_year_load = first_year['Load'].copy()
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year_load,
        wholesale_prices=first_year['RRP'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
    )

    time_series_data['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        time_series_data['Hybrid'] += time_series_data[name] * hybrid_percent_gen
  
    time_series_data['Contracted Energy'] = time_series_data['Hybrid'].copy()
    
    return time_series_data, percentages



def hybrid_pac(
        contracted_amount: float,
        time_series_data: pd.DataFrame,
        generator_info: dict[str:float],
        redef_period: str,
        percentile_val: float
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a 'Pay as 
    Consumed' contract type.
    
    Calls hybrid.run_hybrid_optimisation to perform the optimisation with specific settings for the 'Pay as Consumed'
    contract. When calling run_hybrid_optimisation:
        - the contracted_energy parameter is set to the load profile (not adjusted by the contracted_amount parameter)
        - total_sum is set to the sum of load volume across the first year of load data multiplied by the
          contracted_amount percentage.
        - and wholesale price data and generation data is passed to run_hybrid_optimisation as provided in
          time_series_data

    Also note, only the first year of time series data provided in used in the optimisation.

    :param contracted_amount: float, percentage (fraction between 0.0-1.0) specifying the fraction of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :param percentile_val: str, not used, included to simplify control logic in hybrid.create_hybrid_generation.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' being the minimum of 'Hybrid' and 'Load' for each time interval.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """

    if contracted_amount < 0:
        raise ValueError('contracted_amount must be greater than 0.')

    # Use only the first year of data to create the hybrid/contracted energy trace
    # also need to find out if it's a leap year:
    leap_year = check_leap_year(time_series_data)
    first_year = time_series_data.iloc[:24 * (365 + leap_year)].copy()

    # Get first year load (and total sum):
    first_year_load = first_year['Load'].copy()
    first_year_load_sum = first_year_load.sum(numeric_only=True) * (contracted_amount/100)

    hybrid_trace_series, percentages = run_hybrid_optimisation(
        contracted_energy=first_year['Load'].copy(),
        wholesale_prices=first_year['RRP'].copy(),
        generation_data=first_year[generator_info.keys()].copy(),
        gen_costs=generator_info,
        total_sum=first_year_load_sum,
    )

    time_series_data['Hybrid'] = 0

    for name, det_dict in percentages.items():
        hybrid_percent_gen = det_dict['Percent of generator output'] / 100
        time_series_data['Hybrid'] += time_series_data[name] * hybrid_percent_gen
    
    time_series_data['Contracted Energy'] = np.minimum(time_series_data['Load'], time_series_data['Hybrid'])

    return time_series_data, percentages


def create_hybrid_generation(
        contract_type:str,
        contracted_amount:float,
        time_series_data:pd.DataFrame,
        generator_info:dict[str:float],
        redef_period:str = None,
        percentile_val:float = None
) -> pd.DataFrame:
    """
    Calculates the optimal mix of volume to contract from a set of renewable energy generators for a given contract
    type.

    This is a high-level control function that performs some input validation and passes off to different functions
    depending on the contract_type, for specific methodology details see the functions corresponding to the contract
    type:
    - 'Pay as Produced': hybrid.hybrid_pap
    - 'Pay as Consumed': hybrid.hybrid_pac
    - 'Shaped': hybrid.hybrid_shaped
    - 'Baseload': hybrid.hybrid_baseload
    - '24/7': hybrid.hybrid_247

    :param contract_type: str, used to specify the function used for the optimisation, and also changes some
        behaviour within hybrid.run_hybrid_optimisation.
    :param contracted_amount: float, percentage (fraction between 0.0-100) specifying the percentage of load volume to
        aim to provide with contracted renewable generation.
    :param time_series_data: pd.DataFrame with columns specifying load volume, generation volume, and regional
        wholesale prices on a timeseries basis. The load columns should be named 'Load', generation volume columns
        should be named with a generator name that matches the names provided in generator_info, and a column
        named 'RRP' contain wholesale spot price data.
    :param generator_info: A dictionary specifying the LCOE of each generator. Keys should be generator names as
        strings and values should be floats specifying the generator LCOE in $/MWh.
    :param redef_period: str, only used for 'Shaped' and 'Baseload' see corresponding functions.
    :param percentile_val: str, the percentile value to take in each period (yearly, quarterly, monthly) when
        creating the artificial load profile for optimisation, as explained above.
    :return:
        - pd.DataFrame, the time_series_data dataframe, with two extra columns. 'Hybrid' specifying the combined
          profile of generation from the renewable energy generators based on the percentage contracted from each
          generator, and 'Contracted Energy' which differs depending on the contract type, see the corresponding
          function.
        - dict, specifying the fraction of each generator contracted and fraction of the total energy contracted
          provided by each generator. Dict structure looks like:

        {
            'GEN1': {
                'Percent of generator output': 10.0,
                'Percent of hybrid trace': 45.0
            },
            'ANOTHERGEN': {
                'Percent of generator output': 40.0
                'Percent of hybrid trace': 55.0
            }
        }
    """
    
    valid_contracts = {'Pay as Produced', 'Pay as Consumed', 'Shaped', 'Baseload', '24/7'}
    if contract_type not in valid_contracts:
        raise ValueError(f'contract_type must be one of {valid_contracts}')
    
    valid_periods = {'M', 'Q', 'Y'}
    if redef_period not in valid_periods and contract_type in ['Shaped', 'Baseload']:
        raise ValueError(f'redef_period must be one of {valid_periods}')

    opt_hybrid_funcs = {
        'Pay as Produced' : hybrid_pap, 
        'Pay as Consumed' : hybrid_pac,
        'Shaped' : hybrid_shaped, 
        'Baseload' : hybrid_baseload, 
        '24/7' : hybrid_247
    }

    df_with_hybrid = opt_hybrid_funcs[contract_type](
        contracted_amount, time_series_data, generator_info, redef_period, percentile_val)

    return df_with_hybrid
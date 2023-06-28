import pandas as pd
import numpy as np
import residuals

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
        -> list of tuples with 2 elements:
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
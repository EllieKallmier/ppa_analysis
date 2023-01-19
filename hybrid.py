import pandas as pd
import numpy as np
import residuals


# -------------- HYBRID CALC ---------------
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



# -------------- SCALING CALC ---------------
# Calculates a new generation profile scaled to a chosen proportion of load
"""
Parameters:
    - profiles
        -> pandas df containing DateTime column as half-hourly timestamp type, a
            "total_load" column,  and all other columns with generator IDs as header.
    - gen_ids
        -> list of strings containing the generator IDs to be scaled.
    - scaling_period
        -> string, one of either "Yearly", "Quarterly", or "Monthly" that indicates
            the time period to re-sample the scaling over. Default = Yearly.
    - scaling_factor
        -> float, can be specified if scaling to a fraction/percentage of the 
            load rather than the total load. Default = 1.0

Returns:
    - profiles
        -> same df as input with generator profiles scaled to the load by period 
            and scaling factor. 
"""
def scale_gen_profile(profiles, gen_ids, scaling_period="Yearly", scaling_factor=1.0):
    # return new df rather than editing existing one
    profiles = profiles.copy()
    
    if scaling_period == "Yearly":
        scaling_df = profiles.groupby(profiles['DateTime'].dt.year).sum() 
        scaling_df["total_load"] *= scaling_factor
        for date in scaling_df.index:
            for gen_id in gen_ids:
                scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                    scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.year==date), gen_id] = profiles[gen_id] * scale_by

    elif scaling_period == "Quarterly":
        scaling_df = profiles.groupby(profiles["DateTime"].dt.to_period('Q')).sum()
        scaling_df["total_load"] *= scaling_factor
        for date in scaling_df.index:
            for gen_id in gen_ids:
                scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                    scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.to_period('Q')==date), gen_id] = profiles[gen_id] * scale_by
        

    elif scaling_period == "Monthly":
        scaling_df = profiles.groupby(profiles["DateTime"].dt.to_period('m')).sum()
        scaling_df["total_load"] *= scaling_factor
        for date in scaling_df.index:
            for gen_id in gen_ids:
                scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                    scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.to_period('m')==date), gen_id] = profiles[gen_id] * scale_by
        
        
    else:
        # Need to consider the scenario with no scaling period specified,
        # whether we throw an error or take the whole dataset with no re-sampling.
        return 0
    
    return profiles
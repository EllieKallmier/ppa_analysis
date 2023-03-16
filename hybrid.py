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

# TODO: add capacity to intake an array of scaling factors if the period is set (i.e for UNSW using different scaling for each month based on forecasting.)
# TODO: add the option to flag whether the function is scaling based on LOAD or GEN
# -> when user selects/specifies the PPA contract type (volume type) this should determine whether scaling is for load or gen. 


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
    - scale_to
        -> string, either 'total_load' or '<<GENERATOR_ID>>'. Determines whether to scale to a percentage 
            of the total load or the total generation. ('total_load' can also be
            any other load-reference name).

Returns:
    - profiles
        -> same df as input with generator profiles scaled to the load by period 
            and scaling factor. 
"""
def scale_gen_profile(profiles, gen_ids, scaling_period="Yearly", scaling_factor=1.0, scale_to='Load'):
    # return new df rather than editing existing one
    profiles = profiles.copy()
    
    if scaling_period == "Yearly":
        period = 'Y'       # Python offset string meaning year end frequency sampling

        scaling_df = profiles.groupby(profiles['DateTime'].dt.year).sum()
        scaling_df["total_load"] *= scaling_factor
        scale_by = 1
        
        for date in scaling_df.index:
            for gen_id in gen_ids:
                # using the matching time-stamped data, normalise/scale the generator
                # profile data by (total load) / (total generated) * (half hourly generated)
                # First, check whether total generated RE is >= total load. 
                # If not, don't scale (penalties can then be applied later if required)
                if scaling_df.loc[date, "total_load"] <= scaling_df.loc[date, gen_id]:
                    scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                        scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.year==date), gen_id] = profiles[gen_id] * scale_by

    elif scaling_period == "Quarterly":
        period = 'Q'        # Python offset string meaning quarter end frequency sampling
        scaling_df = profiles.groupby(profiles["DateTime"].dt.to_period('Q')).sum()
        scaling_df["total_load"] *= scaling_factor
        for date in scaling_df.index:
            for gen_id in gen_ids:
                if scaling_df.loc[date, "total_load"] <= scaling_df.loc[date, gen_id]:
                    scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                        scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])
 
                profiles.loc[(profiles.DateTime.dt.to_period('Q')==date), gen_id] = profiles[gen_id] * scale_by
            

    elif scaling_period == "Monthly":
        period = 'M'        # Python offset string meaning month end frequency sampling
        scaling_df = profiles.groupby(profiles["DateTime"].dt.to_period('m')).sum()
        scaling_df["total_load"] *= scaling_factor
        for date in scaling_df.index:
            for gen_id in gen_ids:
                if scaling_df.loc[date, "total_load"] <= scaling_df.loc[date, gen_id]:
                    scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                        scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

                profiles.loc[(profiles.DateTime.dt.to_period('m')==date), gen_id] = profiles[gen_id] * scale_by
        
        
    else:
        # Need to consider the scenario with no scaling period specified,
        # whether we throw an error or take the whole dataset with no re-sampling.
        return 0
    
    return profiles


def scaling(profile_set, scaling_factor, gen_ids, period, scale_to):
    profiles = profile_set.copy()
    scaling_df = profiles.groupby(profiles["DateTime"].dt.to_period(period)).sum()
    # Above line creates a new df with entries matching the sampling time:
        # DateTime      total_load      gen_id      ...
        # 2019          123456          112333      ...
        # 2020          234567          223344      ...
        # ...           ...             ...         ...
        # each column entry is the sum of all data for that year.   

    scaling_df[scale_to] *= scaling_factor
    # scale the total_load value by the scaling factor (if supplied)

    for date in scaling_df.index:
        for gen_id in gen_ids:
            if scaling_df.loc[date, "total_load"] <= scaling_df.loc[date, gen_id]:
                scale_by = np.where(scaling_df.loc[date, gen_id] == 0 ,0, \
                    scaling_df.loc[date, "total_load"]/scaling_df.loc[date, gen_id])

            profiles.loc[(profiles.DateTime.dt.to_period(period)==date), gen_id] = profiles[gen_id] * scale_by

    return profiles
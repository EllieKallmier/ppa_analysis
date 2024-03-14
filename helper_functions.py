# File to hold functions that will assist with testing and validation. 
import pandas as pd
import numpy as np
import logging


# Helper function to check whether a float sits within bounds (inclusive)
def check_float_between(x, lb=None, ub=None):
    if lb != None and ub != None:
        return (x >= lb) & (x <= ub)
    elif lb != None:
        return x >= ub
    elif ub != None:
        return x <= lb
    else:
        return True

# Define columns and dtypes to check against
# TODO: implement check_float_between (somehow?) in necessary 'value' items:
def get_scenario_dtypes():
    dtype_dict = {
        'Scenario_ID' : {'type' : int, 'value' : None}, 
        'PPA' : {'type' : str, 'value' : ['Off-site - Contract for Difference', 'Off-site - Tariff Pass Through',
                        'Off-site - Physical Hedge', 'On-site RE Generator', 'No PPA']},
        'PPA_Volume' : {'type' : str, 'value' : ['Pay As Consumed', 'Pay As Produced', 'Shaped']},
        'Wholesale_Exposure_Volume' : {'type' : str, 'value' : ['All RE', 'RE Uptill Load', 'All Load', 'None']},
        'PPA_Price' : {'type' : float, 'value' : None},  
        'Floor_Price' : {'type' : float, 'value' : None}, 
        'Excess_RE_Purchase_Price' : {'type' : float, 'value' : None}, 
        'Excess_RE_Sale_Price' : {'type' : float, 'value' : None},  
        'LGC_Volume_Type' : {'type' : str, 'value' : None},  
        'LGC_Purhcase_Volume' : {'type' : str, 'value' : None}, 
        'LGC_Purchase_Price' : {'type' : float, 'value' : None},  
        'Load_MLF' : {'type' : float, 'value' : None},  
        'Load_DLF' : {'type' : float, 'value' : None},  
        'Generator_MLF' : {'type' : float, 'value' : None}, 
        'Generator_DLF' : {'type' : float, 'value' : None},  
        'Target_Period' : {'type' : str, 'value' : None},  
        'Yearly_Target_MWh' : {'type' : float, 'value' : None},  
        'Yearly_Short_Fall_Penalty_MWh' : {'type' : float, 'value' : None}, 
        'Yearly_LGC_target_LGC' : {'type' : float, 'value' : None},  
        'Yearly_LGC_short_fall_penalty_LGC' : {'type' : float, 'value' : None},  
        'Average_Wholesale_Price' : {'type' : float, 'value' : None}, 
        'Wholesale_Price_ID' : {'type' : str, 'value' : None}, 
        'Load_ID' : {'type' : str, 'value' : None},  
        'Generator_ID' : {'type' : str, 'value' : None},           # check against any cols with Generator_ID in the name
        'Emissions_Region_ID' : {'type' : str, 'value' : None},  
        'Scaling_Period' : {'type' : str, 'value' : None},         # should be one of a few choices - TODO where to check this?
        'Scaling_Factor' : {'type' : float, 'value' : None}, 
        'Scale_to_ID' : {'type' : str, 'value' : None}, 
        'Hybrid_Percent' : {'type' : float, 'value' : None},     # check against any cols with Hybrid_Percent in the name
        'Hybrid_Mix_Name' : {'type' : str, 'value' : None}
    }
    return dtype_dict

# Check scenarios df to make sure all columns are present and dtypes are correct:
def _check_scenarios(scenario_row, dtypes):
    dtypes = get_scenario_dtypes()

    for col, dtype in dtypes.items():
        data_type = dtype['type']
        values = dtype['value']
        pass

    return


# Check load/gen profiles: check datetime column w/ datetime type, check for NaN/
# empty columns or rows. 
def _check_load_gen_profiles():
    return

# Check the dates from each set of profiles: emissions, load/gen etc
def _check_dates():
    return

# Check that the same/constant timestamps are used across all profiles (if not, 
# resample and throw a message?)
def _check_timestamps():
    return


# In hybrid.py - check that the percentages add to either 1 or 100 (may need to 
# dynamically switch between 0-1 and 0-100 depending on inputs?)
def _check_hybrid_percents():
    return


# Check that regions called in msat_replicator exist in all necessary dfs:
def check_regions():
    return
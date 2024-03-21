# File to hold functions that will assist with testing and validation. 
import pandas as pd
import numpy as np
import logging
from datetime import timedelta, datetime

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

# Test help functions:
def _check_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print('DataFrame is empty.\n')
    
    nan_df = df[df.isna()]

    if not nan_df.empty:
        print('Some missing data found. Filled with zeros.\n')
        df = df.fillna(0.0)

    return df

# Returns an integer representing minutes in the interval
def get_interval_length(df:pd.DataFrame) -> int:
    # get the interval length for the first and last intervals - this will
    # be checked throughout the whole dataset next

    first_int = df['DateTime'].iloc[1] - df['DateTime'].iloc[0]
    last_int = df['DateTime'].iloc[-1] - df['DateTime'].iloc[-2]

    if first_int == last_int:
        return int(first_int.total_seconds() / 60)
    else:
        print('Interval lengths are different throughout dataset.\n')
        return int(first_int.total_seconds() / 60)

def _check_interval_consistency(df:pd.DataFrame, mins:int) -> bool:
    return (df['DateTime'].diff() == timedelta(minutes=mins)).iloc[1:].all()


# Function to check whether a years' worth of data contain a leap year
# If the first day + 365 days != same day (number in month) - it's a leap year
def check_leap_year(
        df:pd.DataFrame
) -> bool:
    day_one = df.index[0]
    day_365 = day_one + timedelta(days=365)

    return day_one.day != day_365.day
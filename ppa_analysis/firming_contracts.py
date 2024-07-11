import numpy as np
import pandas as pd
import copy
from datetime import datetime, time, timedelta
from ppa_analysis import helper_functions, advanced_settings
from sunspot_bill_calculator import bill_calculator, convert_network_tariff_to_retail_tariff, add_other_charges_to_tariff

# Define options for different firming contracts: 

# 1. Wholesale exposure: fully risk exposed, no retail contract
# 2. Partial wholesale exposure (cap, swap or collar)
# 3. Retail contract

# There could be a mix of all three, but for the moment sticking to simpler structures


# Total wholesale exposure:
def total_wholesale_exposure(
        df:pd.DataFrame,
        regions:list[str], 
        upper_bound:float,
        lower_bound:float,
        tariff_details:dict[str:float]
) -> pd.DataFrame:

    for region in regions:
        df[f'Firming price: {region}'] = df[f'RRP: {region}'].copy()

    return df


# Partial wholesale exposure:
def part_wholesale_exposure(
        df:pd.DataFrame,
        regions:list[str], 
        upper_bound:float,
        lower_bound:float,
        tariff_details:dict[str:float]
) -> pd.DataFrame:

    for region in regions:
        df[f'Firming price: {region}'] = df[f'RRP: {region}'].copy()\
            .clip(upper=upper_bound, lower=lower_bound)

    return df


# straight from sunspot bill calculator function...
# but renamed variables for clarity, changed return
# and slightly altered functionality (now fills in a column 'Firming' at the 
# chosen/specified times with the tariff rate corresponding).
def tariff_firming_col_fill(
        load_and_gen_data:pd.DataFrame, 
        tariff_component_details:dict
) -> pd.DatetimeIndex:
    """ 
    For a component of a TOU tariff, adds that component's value (or rate) in $/MWh
    to the rows corresponding to times in which that component applies.

    :param load_and_gen_data: Dataframe with datetime index, and at least a column named 'Firming' that is either filled with zeroes or partially filled with tariff rates.
    :param tariff_component_details: a dictionary with the following structure:
            tariff_component_details = {
                "Month": [1, 2, 12],        # list of integers representing months
                "TimeIntervals": {
                    "T1": [
                    start_time (string),
                    end_time (string)
                    ]
                },
                "Unit": "$/kWh",
                "Value": float,
                "Weekday": bool,
                "Weekend": bool
            }
    """
    load_and_gen_data = load_and_gen_data.copy()
    data_at_tariff_selected_intervals = pd.DataFrame()
    for label, time_value, in tariff_component_details['TimeIntervals'].items():
        if time_value[0][0:2] == '24':
            time_value[0] = time_value[1].replace("24", "00")
        if time_value[1][0:2] == '24':
            time_value[1] = time_value[1].replace("24", "00")
        if time_value[0] != time_value[1]:
            data_between_times = load_and_gen_data.between_time(start_time=time_value[0], end_time=time_value[1],
                                                            include_start=False, include_end=True)
        else:
            data_between_times = load_and_gen_data.copy()

        if not tariff_component_details['Weekday']:
            data_between_times = data_between_times.loc[data_between_times.index.weekday >= 5].copy()

        if not tariff_component_details['Weekend']:
            data_between_times = data_between_times.loc[data_between_times.index.weekday < 5].copy()

        data_between_times = data_between_times.loc[data_between_times.index.month.isin(tariff_component_details['Month']), :].copy()

        data_at_tariff_selected_intervals = pd.concat([data_at_tariff_selected_intervals, data_between_times])
    
    index_for_selected_times = data_at_tariff_selected_intervals.index
    load_and_gen_data.loc[index_for_selected_times, 'Firming'] += tariff_component_details['Value'] * 1000

    return load_and_gen_data


# Retail tariff contract:
def retail_tariff_contract(
        df:pd.DataFrame,
        regions:list[str], 
        upper_bound:float,
        lower_bound:float,
        tariff_details:dict
) -> pd.DataFrame:
    
    df_with_firming = df.copy()
    df_with_firming['Firming'] = 0
    
    for component_name, info in tariff_details['Parameters']['NUOS'].items():
        if 'FlatRate' in component_name:
            df_with_firming['Firming'] += info['Value'] * 1000
        if 'TOU' in component_name:
            for tou_component, tou_info in info.items():
                df_with_firming = tariff_firming_col_fill(df_with_firming, tou_info)

    if len(df_with_firming[df_with_firming['Firming']==0]) == len(df_with_firming):
        df_with_firming['Firming'] = df_with_firming[f'RRP: {regions[0]}'].copy()

    return df_with_firming


# Function to choose which firming contract to apply:
def choose_firming_type(
        firming_type:str,
        df:pd.DataFrame,
        regions:list[str], 
        upper_bound:float,
        lower_bound:float,
        tariff_details:dict[str:float]
) -> pd.DataFrame:

    valid_options = {'Wholesale exposed', 'Partially wholesale exposed', 'Retail'}
    if firming_type not in valid_options:
        raise ValueError(f'firming_type must be one of {valid_options}')

    firming_price_traces = {
        'Wholesale exposed' : total_wholesale_exposure,
        'Partially wholesale exposed' : part_wholesale_exposure,
        'Retail' : retail_tariff_contract
    }

    df = firming_price_traces[firming_type](df, regions, upper_bound, lower_bound, tariff_details)

    return df.copy()
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from ppa_analysis import helper_functions

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


# Retail tariff contract:
def retail_tariff_contract(
        df:pd.DataFrame,
        regions:list[str], 
        upper_bound:float,
        lower_bound:float,
        tariff_details:dict[str:float]
) -> pd.DataFrame:
    # TODO: integrate tariffs

    # Tariff options for ToU (simple):
    # Peak, Shoulder, Off-Peak
    # For each of those: 

    # definition of tariff_details structure:
    # tariff_details = {
    #     'Fixed ($/day)': float,
    #     'Volume ($/MWh)': {
    #         'Type': str('Flat', 'ToU'),  # one of the two options
    #         'Rate(s)': {
    #             'Peak': {
    #                 'Months':[],
    #                 'Weekdays':[],
    #                 'Hours':[],
    #                 'Cost ($/MWh)':float
    #             },
    #             'Shoulder':{
    #                 'Months':[],
    #                 'Weekdays':[],
    #                 'Hours':[],
    #                 'Cost ($/MWh)':float
    #             },
    #             'Off Peak':{
    #                 'Months':[],
    #                 'Weekdays':[],
    #                 'Hours':[],
    #                 'Cost ($/MWh)':float
    #             },
    #             'Flat':float
    #         }
    #     }
    # }
    
    
    df['Tariff Rate ($/MWh)'] = 0.0

    # Average the fixed daily cost across each interval - then can sum the whole column to get the 
    # value of the fixed charge across a variable number of days/
    df['Fixed Daily'] = tariff_details['Fixed ($/day)'] / (1440 / helper_functions.get_interval_length(df))
    
    volume_tariff_rates = tariff_details['Volume ($/MWh)']
    if volume_tariff_rates['Type'] == 'Flat':
        df['Tariff Rate ($/MWh)'] = volume_tariff_rates['Rate(s)']['Flat']
    else:
        for key, value in volume_tariff_rates['Components'].items():
            if key != 'Flat':
                start_month = value['Start Month']
                end_month = value['End Month']
                start_weekday = value['Start Weekday']
                end_weekday = value['End Weekday']
                start_hour = value['Start Hour']
                end_hour = value['End Hour']
                cost = value['Cost ($/MWh)']

                months = (((df.index.month >= start_month) & (df.index.month <= end_month)) | ((start_month > end_month) & ((df.index.month >= start_month) | (df.index.month <= end_month))))
                days = ((df.index.weekday >= start_weekday-1) & (df.index.weekday <= end_weekday-1))
                hours = (((df.index.hour >= start_hour) & (df.index.hour <= end_hour)) | ((start_hour > end_hour) & ((df.index.hour >= start_hour) | (df.index.hour <= end_hour))))

                df.loc[(months & days & hours), 'Tariff Rate ($/MWh)'] = cost
    
    return df


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
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

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
    # TODO: integrate tariff tool and CEEM tariff API for large commercial tariffs??

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
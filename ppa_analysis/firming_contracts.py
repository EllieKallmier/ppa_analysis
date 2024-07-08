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
) -> pd.DataFrame:
    df['Firming price'] = df['RRP'].copy()
    return df


# Partial wholesale exposure:
def part_wholesale_exposure(
        df:pd.DataFrame,
        upper_bound:float,
        lower_bound:float,
) -> pd.DataFrame:
    df['Firming price'] = df['RRP'].copy().clip(upper=upper_bound, lower=lower_bound)
    return df


# Retail tariff contract:
def retail_tariff_contract(
        df:pd.DataFrame,
        tariff_details:dict[str:float]
) -> pd.DataFrame:

    

    return df


def choose_firming_type(
        firming_type:str,
        time_series_data:pd.DataFrame,
        upper_bound:float=None,
        lower_bound:float=None,
        tariff_details:dict[str:float]=None
) -> pd.DataFrame:
    """
    Creates new column named "Firming price" in the time series DataFrame provided, which specifies the time varying
    cost of energy not provided by renewable energy generators.

    If the firming type is set to 'Wholesale exposed', then the firming price is equal to the 'RRP' column in
    time_series_data DataFrame. If the firming type is set to 'Partially wholesale exposed', then the firming price is
    equal to the 'RRP' column in time_series_data DataFrame, but capped at the upper_bound, and with a minimum of
    lower_bound. If the firming type is set to 'Retail', then the time varying firming price is calculated based on the
    time of use or flat rate tariff components.

    :param firming_type: str, specifying how the firming price is to be set. Must be one of 'Wholesale exposed',
        'Partially wholesale exposed', 'Retail'.
    :param time_series_data: DataFrame with a date time index, if 'Wholesale exposed',
        'Partially wholesale exposed' is provided as the firming type then a column specifying the wholesale spot
        price name 'RRP' must also be provided in the time_series_data DataFrame.
    :param upper_bound: float, maximum wholesale spot price exposure in $/MWh, must be provided if firming type is
        'Partially wholesale exposed'. Default is None.
    :param lower_bound: float, minimum wholesale spot price exposure in $/MWh, must be provided if firming type is
        'Partially wholesale exposed'. Default is None.
    :param tariff_details: dict, specifying retail tariff components must be provided if firming type is 'Retail'.
    :return: DataFrame with date time index and column named 'Firming price' specifying the time varying cost of
        purchasing energy not met by the PPA.
    """

    valid_options = ['Wholesale exposed', 'Partially wholesale exposed', 'Retail']

    if firming_type == 'Wholesale exposed':
        firming_cost = total_wholesale_exposure(time_series_data)
    elif firming_type == 'Partially wholesale exposed':
        firming_cost = part_wholesale_exposure(time_series_data, upper_bound, lower_bound)
    elif firming_type == 'Retail':
        firming_cost = retail_tariff_contract(time_series_data, tariff_details)
    else:
        raise ValueError(f'firming_type must be one of {valid_options}')

    return firming_cost.copy()
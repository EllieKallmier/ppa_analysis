# File to hold functions that will assist with testing and validation. 
import pandas as pd
import numpy as np
import logging
from datetime import timedelta, datetime
import os
from collections import Counter

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
    df = df.copy().reset_index()

    first_int = df['DateTime'].iloc[1] - df['DateTime'].iloc[0]
    last_int = df['DateTime'].iloc[-1] - df['DateTime'].iloc[-2]

    if first_int == last_int:
        return int(first_int.total_seconds() / 60)
    else:
        print('Interval lengths are different throughout dataset.\n')
        return int(first_int.total_seconds() / 60)

def _check_interval_consistency(df:pd.DataFrame, mins:int) -> tuple[bool, pd.Timestamp]:
    df = df.copy().reset_index()
    return (df['DateTime'].diff() == timedelta(minutes=mins)).iloc[1:].all()


# Function to check whether a years' worth of data contain a leap year
# If the first day + 365 days != same day (number in month) - it's a leap year
def check_leap_year(
        df:pd.DataFrame
) -> bool:
    day_one = df.index[0]
    day_365 = day_one + timedelta(days=365)

    return day_one.day != day_365.day


# Helper function to create the "shaped" profile based on the defined period and 
# percentile
def get_percentile_profile(
        period_str:str,
        data:pd.DataFrame,
        percentile:float
) -> pd.DataFrame:
    
    if period_str == 'M':
        percentile_profile_period = data.groupby(
            [data.index.month.rename('Month'), 
             data.index.hour.rename('Hour')]
        ).quantile(percentile)

    if period_str == 'Q':
        percentile_profile_period = data.groupby(
            [data.index.quarter.rename('Quarter'), 
             data.index.hour.rename('Hour')]
        ).quantile(percentile)

    if period_str == 'Y':
        percentile_profile_period = data.groupby(
            data.index.hour.rename('Hour')
        ).quantile(percentile)

    return percentile_profile_period


# Helper function to apply the shaped profile across the whole desired timeseries
def concat_shaped_profiles(
        period_str:str,             # define the re-shaping period (one of 'Y', 'M', 'Q')
        shaped_data:pd.DataFrame,   # df containing the shaped 'percentile profile'
        long_data:pd.DataFrame,     # df containing full datetime index: to apply shaped profiles across
) -> pd.DataFrame:
    
    if period_str == 'M':
        long_data['Month'] = long_data.DateTime.dt.month
        long_data['Hour'] = long_data.DateTime.dt.hour

        long_data = long_data.set_index(['Month', 'Hour'])
        long_data = pd.concat([long_data , shaped_data], axis='columns')
        long_data = long_data.reset_index().drop(columns=['Month', 'Hour'])

    if period_str == 'Q':
        long_data['Quarter'] = long_data.DateTime.dt.quarter
        long_data['Hour'] = long_data.DateTime.dt.hour

        long_data = long_data.set_index(['Quarter', 'Hour'])
        long_data = pd.concat([long_data , shaped_data], axis='columns')
        long_data = long_data.reset_index().drop(columns=['Quarter', 'Hour'])

    if period_str == 'Y':
        long_data['Hour'] = long_data.DateTime.dt.hour

        long_data = long_data.set_index('Hour')
        long_data = pd.concat([long_data , shaped_data], axis='columns')
        long_data = long_data.reset_index().drop(columns=['Hour'])

    long_data = long_data.set_index('DateTime')

    return long_data.copy()


# Helper function to calculate yearly indexation:
def yearly_indexation(
        df:pd.DataFrame,
        strike_price:float,
        indexation:float|list[float]
) -> pd.DataFrame:
    
    years = df.index.year.unique()
    
    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if type(indexation) != list:
        indexation = [indexation] * len(years)

    while len(indexation) < len(years):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, year in enumerate(years):
        spi_map[year] = strike_price
        strike_price += strike_price * indexation[i]/100

    spi_map[year] = strike_price

    df_with_strike_price = df.copy()

    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price.index.year
    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price['Strike Price (Indexed)'].map(spi_map)

    return df_with_strike_price['Strike Price (Indexed)']


# Same as above, but for quarterly instance:
def quarterly_indexation(
        df:pd.DataFrame,
        strike_price:float,
        indexation:float|list[float]
) -> pd.DataFrame:
    
    years = df.index.year.unique()

    quarters = [(year, quarter) for year in years for quarter in range(1, 5)]

    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if type(indexation) != list:
        indexation = [indexation] * len(quarters)

    while len(indexation) < len(quarters):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, quarter in enumerate(quarters):
        spi_map[quarter] = strike_price
        strike_price += strike_price * indexation[i]/100

    spi_map[quarter] = strike_price

    df_with_strike_price = df.copy()

    tuples = list(zip(df_with_strike_price.index.year.values, df_with_strike_price.index.quarter.values))

    df_with_strike_price['Strike Price (Indexed)'] = tuples
    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price['Strike Price (Indexed)'].map(spi_map)

    return df_with_strike_price['Strike Price (Indexed)']


def get_data_years(cache_directory):
    """
    Find all the years that have a complete set of generation, pricing and emissions data in the cache
    directory. Assumes that only generation, pricing and emissions are in the cache directory and that
    files are parquet files with the year being the last part of the filename before .parquet
    """

    files_in_cache = os.listdir(cache_directory)
    years_cache = [f[-12:-8] for f in files_in_cache]  # Extract the year from each filename.
    year_counts = Counter(years_cache)  # Count the number of files in the cache for each year.
    # Get all the year that hav three files cached for each year.
    years_with_complete_data = [year for (year, count) in year_counts.items() if count >= 3]
    return years_with_complete_data

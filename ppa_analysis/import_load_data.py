import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from ppa_analysis import helper_functions


# -------------------------------- Get Load Data -------------------------------
#   - check dtypes of columns - should all be float, except datetime col.
#   - update colname(s)
#   - set datetime index
#   - get interval length
#   - check for NaN/missing data
# Must be given in MWh!!

def get_load_data(
    load_file_name:str,
    datetime_col_name:str,
    load_col_name:str,
    day_first:bool,
    period:str='H'
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:

    load_data = pd.read_csv(load_file_name)
    load_data = load_data.rename(columns={datetime_col_name: 'DateTime', load_col_name : 'Load'})
    load_data['Load'] = pd.to_numeric(load_data['Load'], errors='coerce')

    # TODO: consider re-formatting datetime col here for consistency
    load_data['DateTime'] = pd.to_datetime(load_data['DateTime'], infer_datetime_format=True, dayfirst=day_first)
    
    # check all intervals are same length here:
    ## CHANGED ASSUMPTION: EVERYTHING NOW GOES TO HOURLY INTERVALS
    # interval = get_interval_length(load_data)

    # if not _check_interval_consistency(load_data, interval):
    #     print('Time intervals are not consistent throughout dataset. Resampling to 30 minutes.\n')
    #     load_data = load_data.resample('30min').sum(numeric_only=True)

    load_data = load_data.set_index('DateTime')

    # Check for missing or NaN data and fill with zeros:
    load_data = helper_functions._check_missing_data(load_data)

    # Finally make sure no outliers or values that don't make sense (negative)
    load_data = load_data.clip(lower=0.0)

    load_data = load_data.resample(period).sum(numeric_only=True)

    start_date = load_data.first_valid_index()
    end_date = load_data.last_valid_index()

    return (load_data, start_date, end_date)
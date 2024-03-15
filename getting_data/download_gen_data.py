import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import List
from nemosis import dynamic_data_compiler, static_table
from helper_functions import _check_missing_data


# Fetch the generation data using NEMOSIS: this function acts as a wrapper that
# filters for region and technology type, and checks the earliest data for each
# matching generator against start date.
def get_generation_data(
        cache:str, 
        region:str, 
        technology_type_s:List[str],
        sampling_interval_minutes:str,
        start_date:pd.Timestamp,
        end_date:pd.Timestamp
        ) -> pd.DataFrame:
    
    # First: get the static table for generators and scheduled loads. This is used
    # to filter for generators by region and type.
    # Credit for example code snippets: https://github.com/UNSW-CEEM/NEMOSIS/blob/master/examples/generator_bidding_data.ipynb (Nick Gorman)
    dispatch_units = static_table(table_name='Generators and Scheduled Loads', 
                                raw_data_location=cache,
                                update_static_file=True)
    
    # Get only relevant columns
    dispatch_units = dispatch_units[['Station Name', 'Region', 'Technology Type - Descriptor', 'DUID']]

    # Filter for region and technology type, then get a list of the unique remaining DUIDs
    dispatch_units = dispatch_units[
        (dispatch_units['Region'] == region) &
        (dispatch_units['Technology Type - Descriptor'].str.upper().isin(technology_type_s))
    ]
    duids_to_check = dispatch_units['DUID'].values

    # Need to convert start and end dates to strings and collect all scada data 
    # from nemosis:
    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M:%S')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M:%S')
    scada_data = dynamic_data_compiler(start_time=start_date_str,
                                    end_time=end_date_str,
                                    table_name='DISPATCH_UNIT_SCADA',
                                    raw_data_location=cache)
    
    useable_scada_data = pd.DataFrame()
    for duid in duids_to_check:
        scada_data_gen = scada_data[scada_data['DUID'] == duid].copy()
        tech_type = dispatch_units[dispatch_units['DUID'] == duid]['Technology Type - Descriptor'].values[0]

        scada_data_gen = scada_data.rename(columns={'SCADAVALUE' : duid + ': ' + tech_type})
        scada_data_gen = scada_data_gen.drop(columns=['DUID'])
        scada_data_gen['DateTime'] = pd.to_datetime(scada_data_gen['SETTLEMENTDATE'])
        scada_data_gen = scada_data_gen.set_index('DateTime')
        scada_data_gen = scada_data_gen.sort_values(by='DateTime')
        scada_data_gen = scada_data_gen.resample(f'{sampling_interval_minutes}min').mean(numeric_only=True)

        non_nan_scada = scada_data_gen.dropna(how='any', axis='rows').copy()
        
        if not non_nan_scada.empty:
            if non_nan_scada.first_valid_index() <= start_date:
                useable_scada_data = pd.concat([useable_scada_data, scada_data_gen], axis='columns')

    gen_data = _check_missing_data(useable_scada_data)
    gen_data = gen_data.clip(lower=0.0)

    return gen_data
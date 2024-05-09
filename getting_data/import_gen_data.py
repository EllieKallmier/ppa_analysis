import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from typing import List
from nemosis import dynamic_data_compiler, static_table
from helper_functions import _check_missing_data, get_interval_length, _check_interval_consistency


# Fetch the generation data using NEMOSIS: this function acts as a wrapper that
# filters for region and technology type, and checks the earliest data for each
# matching generator against start date.
def get_generation_data(
        cache:str, 
        region:list, 
        technology_type_s:list[str],
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        period:str='H'
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
                                    raw_data_location=cache,
                                    select_columns=['DUID', 'SCADAVALUE', 'SETTLEMENTDATE'],
                                    filter_cols=['DUID'],
                                    filter_values=([duids_to_check]))
    
    useable_scada_data = pd.DataFrame()
    for duid in duids_to_check:
        tech_type = dispatch_units[dispatch_units['DUID'] == duid]['Technology Type - Descriptor'].values[0]

        scada_data_gen = scada_data[scada_data['DUID'] == duid].copy()

        if not scada_data_gen.empty:
            scada_data_gen = scada_data_gen.rename(columns={'SCADAVALUE' : duid + ': ' + tech_type})
            scada_data_gen = scada_data_gen.drop(columns=['DUID'])
            scada_data_gen['DateTime'] = pd.to_datetime(scada_data_gen['SETTLEMENTDATE'])

            scada_data_gen = scada_data_gen.set_index('DateTime').drop(columns=['SETTLEMENTDATE'])
            scada_data_gen = scada_data_gen.sort_values(by='DateTime')

            # SCADA data is given in MW. To convert to MWh we need to know the time 
            # interval:
            int_length = get_interval_length(scada_data_gen)
            
            if _check_interval_consistency(scada_data_gen, int_length):
                # Convert from MW to MWh:
                scada_data_gen *= (int_length / 60)
                scada_data_gen = scada_data_gen.resample(period).sum(numeric_only=True)
            
            else:
                print('The interval lengths are different across this generator data.')
                pass

            # Check to make sure that the earliest data for each gen start at or before
            # the load start date.
            non_nan_scada = scada_data_gen.dropna(how='any', axis='rows').copy()
            
            if not non_nan_scada.empty:
                if non_nan_scada.first_valid_index().date() <= start_date.date():
                    useable_scada_data = pd.concat([useable_scada_data, scada_data_gen], axis='columns')

    gen_data = _check_missing_data(useable_scada_data)
    gen_data = gen_data.clip(lower=0.0)

    return gen_data
from datetime import datetime

import nemed
import pandas as pd
from nemosis import static_table, dynamic_data_compiler

from ppa_analysis import helper_functions
from ppa_analysis.helper_functions import get_interval_length, _check_interval_consistency, _check_missing_data


# Fetch the generation data using NEMOSIS: this function acts as a wrapper that
# filters for region and technology type, and checks the earliest data for each
# matching generator against start date.
def get_generation_data(
        cache:str,
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
        # (dispatch_units['Region'] == region) &
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
                                    filter_values=([duids_to_check]),
                                    fformat='parquet',
                                    keep_csv=False)

    useable_scada_data = []
    for duid in duids_to_check:
        tech_type = dispatch_units[dispatch_units['DUID'] == duid]['Technology Type - Descriptor'].values[0]
        region = dispatch_units[dispatch_units['DUID'] == duid]['Region'].values[0]

        scada_data_gen = scada_data[scada_data['DUID'] == duid].copy()

        if not scada_data_gen.empty:
            scada_data_gen['DateTime'] = pd.to_datetime(scada_data_gen['SETTLEMENTDATE'])
            scada_data_gen['REGIONID'] = region
            scada_data_gen['DUID'] = duid + ': ' + tech_type
            scada_data_gen = scada_data_gen.rename(columns={'DUID': 'UNIT'})

            scada_data_gen = scada_data_gen.set_index('DateTime').drop(columns=['SETTLEMENTDATE'])
            scada_data_gen = scada_data_gen.sort_values(by='DateTime')

            # SCADA data is given in MW. To convert to MWh we need to know the time
            # interval:
            int_length = get_interval_length(scada_data_gen)

            if _check_interval_consistency(scada_data_gen, int_length):
                # Convert from MW to MWh:
                scada_data_gen['SCADAVALUE'] *= (int_length / 60)
                scada_data_gen['SCADAVALUE'] = scada_data_gen.resample(period).sum(numeric_only=True)

                # Check to make sure that the earliest data for each gen start at or before
                # the load start date.
                non_nan_scada = scada_data_gen.dropna(how='any', axis='rows').copy()

                if not non_nan_scada.empty:
                    if non_nan_scada.first_valid_index().date() <= start_date.date():
                        useable_scada_data.append(non_nan_scada)
            else:
                print('The interval lengths are different across this generator data.')
                pass

    gen_data = pd.concat(useable_scada_data)
    gen_data = _check_missing_data(gen_data)
    gen_data['SCADAVALUE'] = gen_data['SCADAVALUE'].clip(lower=0.0)

    return gen_data


def get_preprocessed_gen_data(file, regions):
    gen_data = pd.read_parquet(file)
    gen_data = gen_data[gen_data['REGIONID'].isin(regions)]
    gen_data['UNIT'] = gen_data['UNIT'].str.upper()
    gen_data = gen_data.pivot(columns='UNIT', values='SCADAVALUE')
    return gen_data


def get_generator_options(file, regions):
    gen_data = pd.read_parquet(file)
    gen_data = gen_data[gen_data['REGIONID'].isin(regions)]
    gen_data['UNIT'] = gen_data['UNIT'].str.upper()
    gen_options = gen_data['UNIT'].unique()
    return gen_options


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


# The following retrieval code is heavily based on examples given in the NEMED
# documentation, found here: https://nemed.readthedocs.io/en/latest/examples/total_emissions.html


# Rationale for all 3 options:
# There will be times where speed is important, and where you may only need/want
# to look at the impacts of marginal or average emissions. Keeping the two calls
# separate where possible and a third option to combine maximises speed and removes
# the chance of making unnecessary calls to nemed.

# TODO: add check for regions - make sure it's always passed a LIST.
def get_avg_emissions_intensity(
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        cache:str,
        period:str='H'
        ) -> pd.DataFrame:


    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M')
    nemed_result = nemed.get_total_emissions(start_time = start_date_str,
                                             end_time = end_date_str,
                                             cache = cache,
                                             by = None,                         # don't aggregate using inbuilt NEMED functionality - can't do 30 min increments
                                             assume_energy_ramp=True,           # can set this to False for faster computation / less accuracy
                                             generation_sent_out=False          # currently NOT considering auxiliary load factors (from static tables)
                                             )

    # Create empty df to fill with columns corresponding to each region, containing the emissions intensity index:
    # emissions_df = pd.DataFrame()
    nemed_result = nemed_result.reset_index()
    nemed_result['DateTime'] = pd.to_datetime(nemed_result['TimeEnding'])
    emissions_df = nemed_result.drop(columns=['TimeEnding'])
    emissions_df = emissions_df.pivot_table(columns='Region', values='Intensity_Index', index='DateTime')

    emissions_df = emissions_df.resample(period).mean(numeric_only=True)

    emissions_df = emissions_df.rename(columns={
        col : 'AEI: ' + col for col in emissions_df.columns
    })

    return emissions_df


# TODO: decide if this even needs to be here before updating.
# Function to get marginal emissions intensity:
def get_marginal_emissions_intensity(start, end, cache, regions, period=None):
    nemed_result = nemed.get_marginal_emissions(start_time = start,
                                                end_time = end,
                                                cache = cache
                                                )
    emissions_df = pd.DataFrame()
    emissions_df['DateTime'] = pd.to_datetime(nemed_result['Time'])

    if regions == None:
        regions = ['NSW1', 'VIC1', 'SA1', 'QLD1', 'TAS1']

    for region in regions:
        emissions_df[region] = nemed_result[nemed_result['Region']==region]['Intensity_Index']

    # if end < '2017/11/01 00:00':
    #     min_period = '15T'
    # else:
    #     min_period = '5T'

    if period != None:
        emissions_df = emissions_df.set_index('DateTime').resample(period).mean()
        emissions_df = emissions_df.reset_index()

    return emissions_df


# Get both types of emissions and return as a combined df.
def get_both_emissions(start, end, cache, regions, period=None):
    average_emissions = get_avg_emissions_intensity(start, end, cache, regions, period)
    average_emissions = average_emissions.rename(columns={col : col+'_average' for col in average_emissions.columns if col != 'DateTime'})

    marginal_emissions = get_marginal_emissions_intensity(start, end, cache, regions, period)
    marginal_emissions = marginal_emissions.rename(columns={col : col+'_marginal' for col in marginal_emissions.columns if col != 'DateTime'})

    emissions_df = average_emissions.merge(marginal_emissions, how='outer', on='DateTime')

    return emissions_df


def get_preprocessed_avg_intensity_emissions_data(file, regions):
    emissions_data = pd.read_parquet(file)
    regions = list(set(regions))
    emissions_data = emissions_data.loc[:, ['AEI: ' + region for region in regions]]
    return emissions_data


# Wrapper function to import dispatch pricing data using NEMOSIS (credit: Nick Gorman).
def get_wholesale_price_data(
    start_date:pd.Timestamp,
    end_date:pd.Timestamp,
    cache:str,
    period:str='H'
) -> pd.DataFrame:

    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M:%S')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M:%S')

    price_data = dynamic_data_compiler(start_time=start_date_str,
                                   end_time=end_date_str,
                                   table_name='DISPATCHPRICE',
                                   raw_data_location=cache,
                                   select_columns=['REGIONID', 'SETTLEMENTDATE', 'RRP'],
                                   fformat='parquet',
                                   keep_csv=False
                                   )

    price_data = price_data.reset_index()
    price_data['DateTime'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
    price_data = price_data.drop(columns=['SETTLEMENTDATE'])
    price_data = price_data.pivot_table(columns='REGIONID', values='RRP', index='DateTime')
    price_data = price_data.resample(period).mean(numeric_only=True)

    price_data = price_data.rename(columns={
        col : 'RRP: ' + col for col in price_data.columns
    })

    return price_data


# Wrapper function to import pre-dispatch pricing data using NEMSEER (credit: Abhijith Prakash)
def get_predispatch_prices(
    start_date:pd.Timestamp,
    end_date:pd.Timestamp,
    cache:str,
    regions:list[str],
    period:str='H'
) -> pd.DataFrame:
    return


def get_preprocessed_price_data(file, regions):
    price_data = pd.read_parquet(file)
    regions = list(set(regions))
    price_data = price_data.loc[:, ['RRP: ' + region for region in regions]]
    return price_data

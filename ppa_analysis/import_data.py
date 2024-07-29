"""
Functions for reading load data files getting NEM generation, spot price, and emissions data.

Generation load functions:
    - get_load_data: reads load data from disk

Generation data functions:
    - get_generation_data: downloads and returns timeseries generation data for specified technologies
    - get_preprocessed_gen_data: reads from disk generation data in the format produced by get_generation_data and
        reformats for compatibility with ppa analysis work flow
    - get_generator_options: gets the list of generators in a data file saved to disk in the format produced by
        get_generation_data.

Emissions data functions:
    - get_avg_emissions_intensity: downloads and returns timeseries emission intensity data.
    - get_preprocessed_avg_intensity_emissions_data: reads from disk generation data in the format produced by
      get_avg_emissions_intensity

Price data functions:
    - get_wholesale_price_data: downloads and returns timeseries wholesale spot price data.
    - get_preprocessed_price_data: reads from disk generation data in the format produced by
      get_wholesale_price_data
"""
from datetime import datetime

import nemed
import pandas as pd
from nemosis import static_table, dynamic_data_compiler

from ppa_analysis import helper_functions
from ppa_analysis.helper_functions import get_interval_length, _check_interval_consistency, _check_missing_data


def get_generation_data(
        cache:str,
        technology_type_s:list[str],
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        period:str='H'
        ) -> pd.DataFrame:
    """
    Downloads and process generation data from AEMO.

    Fetches generation from AEMO using data using the NEMOSIS package, then filters for region and technology type,
    resample the data according to the specified period and converts from MW to MWH. Also, checks the earliest data for
    each matching generator against start date, filtering out generators where data starts after the start date to make
    sure only generators with data for the whole time period are returned.

    :param cache: str, the directory where raw data from AEMO file are to be cached (required by NEMOSIS).
    :param technology_type_s: list[str], technologies type to fetch generation data for.
    :param start_date: str, '%Y/%m/%d %H:%M:%S', start time to return data from.
    :param end_date: str, '%Y/%m/%d %H:%M:%S', end time to return data uptill.
    :param period: periodicity to resample data to, 'H' for hourly, '30min' for half-hourly, etc. Original is 5minutely,
        so resample should be '5min' or greater.
    :return: pd.Dataframe, with columns:
        - 'DateTime': the datetime at the end of the period.
        - 'UNIT': combines unit DUID and technology type into a label of format '<DUID>: <technology type>'
        - 'REGIONID': specifies where the generator is located.
        - 'SCADAVALUE': number of MWh generated in the time period.

    """

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
                scada_data_gen['SCADAVALUE'] = scada_data_gen.resample(period, label='right', closed='right').sum(
                    numeric_only=True)

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
    """
    Read data returned by get_generation_data saved to disk in parquet format, additionally filter by region and
    reformat such that UNIT names are columns names.

    :param file: str or pathlib.Path, path to file of saved parquet file.
    :param regions: list[str], regions to filter data by.
    :return: pd.Dataframe containing unit generation time series data, with a datetime index, and columns for each unit
        formated like '<DUID>: <technology type>.
    """
    gen_data = pd.read_parquet(file)
    gen_data = gen_data[gen_data['REGIONID'].isin(regions)]
    gen_data['UNIT'] = gen_data['UNIT'].str.upper()
    gen_data = gen_data.pivot(columns='UNIT', values='SCADAVALUE')
    return gen_data


def get_generator_options(file, regions):
    """
    Fetches the list of generators with in a DataFrame returned by get_generation_data and saved to disk in parquet
    format, and additionally filter by region.

    :param file: str or pathlib.Path, path to file of saved parquet file.
    :param regions: list[str], regions to filter data by.
    :return:list[str] of unit names in the format '<DUID>: <technology type>'.
    """
    gen_data = pd.read_parquet(file)
    gen_data = gen_data[gen_data['REGIONID'].isin(regions)]
    gen_data['UNIT'] = gen_data['UNIT'].str.upper()
    gen_options = gen_data['UNIT'].unique()
    return gen_options


def get_load_data(
    load_file_name:str,
    datetime_col_name:str,
    load_col_name:str,
    day_first:bool,
    units:str='kWh',
    period:str='H'
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Reads load data saved to disk in CSV format and carries out data preprocessing.

        - CSV should have two columns, one specifying the datetime and the other load volume in kWh or MWh. Datetime can
          be in a variety of formats that pandas will automatically pass.
        - load data will be processed to replace NaN and negative values with zeros.
        - load data will be resampled according to the specified period

    :param load_file_name: str or pathlib.Path, path to file of saved CSV file.
    :param datetime_col_name: str, name of column with datetime data.
    :param load_col_name: str, name of column with load data.
    :param day_first: bool, specify if the day comes at the start of the datetime format.
    :param units: the units that the load data is in should be 'kWh' or 'MWh', default is 'kWh'.
    :param period: periodicity to resample data to, 'H' for hourly, '30min' for half-hourly, etc. Original is 5minutely,
        so resample should be '5min' or greater.
    :return: tuple(pd.DataFrame, datetime, datetime)
        - DataFrame with datetime index at periodicity specified by the period argument, and the column 'Load'
          specifying the load in MWh for the period.
        - the first datetime in the data
        - the last datetime in the data
    """

    load_data = pd.read_csv(load_file_name)
    load_data = load_data.rename(columns={datetime_col_name: 'DateTime', load_col_name : 'Load'})
    load_data['Load'] = pd.to_numeric(load_data['Load'], errors='coerce')

    if units == 'kWh':
        load_data['Load'] = load_data['Load']/1000.0

    # TODO: consider re-formatting datetime col here for consistency
    load_data['DateTime'] = pd.to_datetime(load_data['DateTime'], infer_datetime_format=True, dayfirst=day_first)

    load_data = load_data.set_index('DateTime')

    # Check for missing or NaN data and fill with zeros:
    load_data = helper_functions._check_missing_data(load_data)

    # Finally make sure no outliers or values that don't make sense (negative)
    load_data = load_data.clip(lower=0.0)

    load_data = load_data.resample(period, label='right', closed='right').sum(numeric_only=True)

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
def get_avg_emissions_intensity_data(
        start_date:pd.Timestamp,
        end_date:pd.Timestamp,
        cache:str,
        period:str='H'
        ) -> pd.DataFrame:
    """
    Downloads and process emissions data from AEMO.

    - Fetches generation from AEMO using data using the NEMED package.
    - Resamples the data according to the specified period.

    :param start_date: str, '%Y/%m/%d %H:%M:%S', start time to return data from.
    :param end_date: str, '%Y/%m/%d %H:%M:%S', end time to return data till.
    :param cache: str, the directory where raw data from AEMO file are to be cached (required by NEMOSIS).
    :param period: periodicity to resample data to, 'H' for hourly, '30min' for half-hourly, etc. Original is 5minutely,
        so resample should be '5min' or greater.
    :return: pd.Dataframe, with:
        -  a datetime index, specifying the end of the period the data covers.
        -  a column 'REGIONID' specifying the NEM region and a column 'AEI' specifying the average emissions intensity
        for the period.
    """

    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M')
    nemed_result = nemed.get_total_emissions(start_time = start_date_str,
                                             end_time = end_date_str,
                                             cache = cache,
                                             by = None,                         # don't aggregate using inbuilt NEMED functionality - can't do 30 min increments
                                             assume_energy_ramp=True,           # can set this to False for faster computation / less accuracy
                                             generation_sent_out=False          # currently NOT considering auxiliary load factors (from static tables)
                                             )

    nemed_result['DateTime'] = pd.to_datetime(nemed_result['TimeEnding'])
    emissions_df = nemed_result.drop(columns=['TimeEnding'])
    emissions_df = emissions_df.rename(columns={'Region': 'REGIONID'})
    emissions_df = emissions_df.sort_values(by='DateTime')
    emissions_df = emissions_df.set_index('DateTime')
    emissions_df = emissions_df.groupby("REGIONID").resample(
        period, label='right', closed='right').mean(numeric_only=True).reset_index(level="REGIONID")
    emissions_df = emissions_df.rename(columns={'Intensity_Index': 'AEI'})
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
        emissions_df = emissions_df.set_index('DateTime').resample(period, label='right', closed='right').mean()
        emissions_df = emissions_df.reset_index()

    return emissions_df


# Get both types of emissions and return as a combined df.
def get_both_emissions(start, end, cache, regions, period=None):
    average_emissions = get_avg_emissions_intensity_data(start, end, cache, regions, period)
    average_emissions = average_emissions.rename(columns={col : col+'_average' for col in average_emissions.columns if col != 'DateTime'})

    marginal_emissions = get_marginal_emissions_intensity(start, end, cache, regions, period)
    marginal_emissions = marginal_emissions.rename(columns={col : col+'_marginal' for col in marginal_emissions.columns if col != 'DateTime'})

    emissions_df = average_emissions.merge(marginal_emissions, how='outer', on='DateTime')

    return emissions_df


def get_preprocessed_avg_intensity_emissions_data(file, region):
    """
    Read data returned by get_avg_emissions_intensity saved to disk in parquet format, and additionally filter by region.

    :param file: str or pathlib.Path, path to file of saved parquet file.
    :param region: str, region's data to return.
    :return: pd.Dataframe containing regional emissions time series data, with a datetime index,
     and columns named 'AEI' containing the average emission intensity data.
    """
    emissions_data = pd.read_parquet(file)
    emissions_data = emissions_data[emissions_data['REGIONID']==region].copy()
    emissions_data = emissions_data.loc[:, ['AEI']]
    return emissions_data


def get_wholesale_price_data(
    start_date:pd.Timestamp,
    end_date:pd.Timestamp,
    cache:str,
    period:str='H'
) -> pd.DataFrame:
    """
    Downloads and process wholesale spot price data from AEMO.

    - Fetches generation from AEMO using data using the NEMOSIS package.
    - Resamples the data according to the specified period.

    :param start_date: str, '%Y/%m/%d %H:%M:%S', start time to return data from.
    :param end_date: str, '%Y/%m/%d %H:%M:%S', end time to return data till.
    :param cache: str, the directory where raw data from AEMO file are to be cached (required by NEMOSIS).
    :param period: periodicity to resample data to, 'H' for hourly, '30min' for half-hourly, etc. Original is 5minutely,
        so resample should be '5min' or greater.
    :return: pd.Dataframe, with:
        -  a datetime index, specifying the end of the period the data covers.
        -  a column 'REGIONID' specifying the NEM region and a column 'RRP' specifying average price for the period
    """

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
    price_data['DateTime'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
    price_data = price_data.drop(columns=['SETTLEMENTDATE'])
    price_data = price_data.sort_values(by='DateTime')
    price_data = price_data.set_index('DateTime', drop=True)
    price_data = price_data.groupby("REGIONID").resample(
        period, label='right', closed='right').mean(numeric_only=True).reset_index(level="REGIONID")
    return price_data


def get_preprocessed_price_data(file, region):
    """
    Read data returned by get_wholesale_price_data saved to disk in parquet format, and additionally filter by region.

    :param file: str or pathlib.Path, path to file of saved parquet file.
    :param region: str, the region's data to return.
    :return: pd.Dataframe containing regional wholesale spot price data, with a datetime index,
     and columns named 'RRP' containing the price data ($/MWh).
    """
    price_data = pd.read_parquet(file)
    price_data = price_data[price_data['REGIONID']==region].copy()
    price_data = price_data.loc[:, ['RRP']]
    return price_data

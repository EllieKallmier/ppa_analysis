# File to hold functions that will assist with testing, validation or other small formatting and calculation tasks that are secondary to the main functionality of the tool.
import pandas as pd
import os
import json
import copy
import holidays
from datetime import timedelta
from collections import Counter
from ppa_analysis import advanced_settings
from sunspot_bill_calculator import add_other_charges_to_tariff, convert_network_tariff_to_retail_tariff


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
    import os
    print(os.getcwd())
    print(cache_directory)
    files_in_cache = os.listdir(cache_directory)
    years_cache = [f[-12:-8] for f in files_in_cache]  # Extract the year from each filename.
    year_counts = Counter(years_cache)  # Count the number of files in the cache for each year.
    # Get all the year that hav three files cached for each year.
    years_with_complete_data = [year for (year, count) in year_counts.items() if count >= 3]
    return years_with_complete_data


# Calculate LCOE from user inputs/predetermined values
# Function takes in the  generator LCOE info dictionary, and calculates LCOE
# for only one generator with each call.
# Returns LCOE value in $/MW
def calculate_lcoe(
        generator_info: dict[str:object]
) -> float:
    # Baseline assumptions:


    capital_cost = generator_info['Capital ($/kW)']
    numerator, denominator = 0, 0
    for year in range(1, advanced_settings.LIFETIME_YEARS + 1):
        kwh_in_year_n = generator_info['Capacity Factor'] * (365 * 24) # Note: this doesn't currently account for leap years!
        numerator += (generator_info['Fixed O&M ($/kW)']
                      + generator_info['Variable O&M ($/kWh)'] * kwh_in_year_n) / \
                     ((1 + advanced_settings.DISCOUNT_RATE) ** year)
        denominator += (kwh_in_year_n) / ((1 + advanced_settings.DISCOUNT_RATE) ** year)
    numerator += capital_cost

    return (numerator / denominator) * 1000


# ----- Fetch inputs and set up info_dict data to pass to later functions:
def get_all_lcoes(
        generator_data_editor: dict[str:dict[str:object]]
) -> dict[str:float]:
    all_generator_lcoes = {}
    for gen, gen_info in generator_data_editor.items():
        if gen != 'out':
            gen_lcoe = calculate_lcoe(gen_info)
            all_generator_lcoes[gen] = gen_lcoe

    return all_generator_lcoes


# TODO: Ellie to add docstrings here
# Helper function to read in json files (for network tariff selection)
def read_json_file(filename):
    f = open(f'{filename}.json', 'r')
    data = json.loads(f.read())
    f.close()
    return data


# Add an extra column to load_and_gen_data df that holds a string value denoting the season associated with the index timestamp.
def get_seasons(
    load_and_gen:pd.DataFrame
) -> pd.DataFrame:
    seasonal_load_and_gen = load_and_gen.copy()
    seasonal_load_and_gen['Season'] = ''

    seasonal_load_and_gen.loc[seasonal_load_and_gen.index.month.isin([1,2,12]), 'Season'] = 'Summer'
    seasonal_load_and_gen.loc[seasonal_load_and_gen.index.month.isin([3, 4, 5]), 'Season'] = 'Autumn'
    seasonal_load_and_gen.loc[seasonal_load_and_gen.index.month.isin([6, 7, 8]), 'Season'] = 'Winter'
    seasonal_load_and_gen.loc[seasonal_load_and_gen.index.month.isin([9, 10, 11]), 'Season'] = 'Spring'

    return seasonal_load_and_gen


# Helper function: get weekday/weekends
# Sets regional public holidays as 'weekend' alongside Sat/Sun
def get_weekends(load_and_gen, region):
    weekend_load_and_gen = load_and_gen.copy()
    holiday_dates = pd.Series(
        holidays.country_holidays(
            'AU', 
            subdiv=region[:-1], 
            years=range(weekend_load_and_gen.index.min().year, weekend_load_and_gen.index.max().year+1)
        )
    )
    weekend_load_and_gen['Date'] = weekend_load_and_gen.index.date
    weekend_load_and_gen['Weekday'] = weekend_load_and_gen.index.dayofweek
    weekend_load_and_gen['Weekend'] = (weekend_load_and_gen['Date'].isin(holiday_dates.index)) | \
        (weekend_load_and_gen['Weekday'].isin([5,6]))
    weekend_load_and_gen['Weekend'] = weekend_load_and_gen['Weekend'].astype(int)
    weekend_load_and_gen = weekend_load_and_gen.drop(columns=['Date', 'Weekday'])

    return weekend_load_and_gen


# Format other charges: used to re-format "other_charges" collected from user input widgets.
# Re-formats to match required structure for sunspot bill_calculator.
def format_other_charges(
        extra_charges_collector:dict
) -> dict:
    other_charges = { 
        "Customer Type": "Commercial",
        "Energy Charges": {
            "Peak Rate": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['peak_rate'].value
            },
            "Shoulder Rate": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['shoulder_rate'].value
            },
            "Off-Peak Rate": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['off_peak_rate'].value
            },
            "Retailer Demand Charge": {
            "Unit": "c/kVA/day",
            "Value": extra_charges_collector['retailer_demand_charge'].value
            }
        },
        "Metering Charges": {
            "Meter Provider/Data Agent Charges": {
            "Unit": "$/Day",
            "Value": extra_charges_collector['meter_provider_charge'].value
            },
            "Other Meter Charges": {
            "Unit": "$/Day",
            "Value": extra_charges_collector['other_meter_charge'].value
            }
        },
        "Environmental Charges": {
            "LREC Charge": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['lrec_charge'].value
            },
            "SREC Charge": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['srec_charge'].value
            },
            "State Environment Charge": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['state_env_charge'].value
            }
        },
        "AEMO Market Charges": {
            "AEMO Participant Charge": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['participant_charge'].value
            },
            "AEMO Ancillary Services Charge": {
            "Unit": "c/kWh",
            "Value": extra_charges_collector['ancillary_services_charge'].value
            }
        },
        "Other Variable Charges": {
            "Other Variable Charge 1": {
            "Unit": "$/kWh",
            "Value": extra_charges_collector['other_charge_one'].value
            },
            "Other Variable Charge 2": {
            "Unit": "$/kWh",
            "Value": extra_charges_collector['other_charge_two'].value
            },
            "Other Variable Charge 3": {
            "Unit": "$/kWh",
            "Value": extra_charges_collector['other_charge_three'].value
            }
        },
        "Other Fixed Charges": {
            "Total GST": {
            "Unit": "$/Bill",
            "Value": extra_charges_collector['total_gst'].value
            },
            "Other Fixed Charge 1": {
            "Unit": "$/Bill",
            "Value": extra_charges_collector['other_fixed_charge_one'].value
            },
            "Other Fixed Charge 2": {
            "Unit": "$/Bill",
            "Value": extra_charges_collector['other_fixed_charge_two'].value
            }
        }
    }

    return other_charges


# Gets the 'chunk' of load data for one settlement period
def get_load_data_chunk(
    load_data:pd.DataFrame,
    end_date:pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """ 
    Gets and returns two dataframes from an initial one, separated at a specified date.

    :param load_data: Dataframe with datetime index and a column containing load data (MWh)
        named 'Load'.
    :param end_date: pd.Timestamp denoting the end of the chunk to be returned. A timedelta day
        is added to end_date to ensure that all data up to the end of the desired period is returned.
    :return:
        - pd.DataFrame, containing the contents of the original dataframe up to and including the specified end date.
        - pd.DataFrame, containing the contents of the original dataframe starting from the day after the specified end date to the end of the timeseries.
    """
    
    end_date_plus_one_day = end_date + timedelta(days=1)
    chunk = load_data[load_data.index <= end_date_plus_one_day].copy()
    remainder = load_data[load_data.index > end_date_plus_one_day].copy()
    
    return chunk, remainder


def get_selected_tariff(
        tariff_collector:dict,
        extra_charges_collector:dict,
        retail:bool=False
) -> dict:
    """ 
    Selects and returns the user's chosen network tariff formatted to be passed to CEEM's SunSPOT bill_calculator function as a retail tariff. If extra charges are supplied and the tariff type is retail, the extra charges are also added to the tariff structure and returned.

    :param tariff_collector: dict, containing the name of the user's chosen tariff as a str.
    :param extra_charges_collector: dict, containing user inputs for extra charges 
        as defined in user_inputs.launch_extra_charges_collector().
    :param retail: bool, indicates whether the tariff selected to be returned is 
        a retail tariff and therefore needs other charges to be added before reformatting.
    :return: dict, the selected tariff returned reformatted and with other charges 
        added as necessary.
    """
    all_tariffs = read_json_file(advanced_settings.COMMERCIAL_TARIFFS_FN)
    all_tariffs = all_tariffs[0]['Tariffs']
    selected_tariff_name = tariff_collector['tariff_name'].value
    selected_tariff = {}
    for tariff in all_tariffs:
        if tariff['Name'] == selected_tariff_name:
            selected_tariff = copy.deepcopy(tariff)

    if retail:
        other_charges = format_other_charges(extra_charges_collector)
        selected_tariff = add_other_charges_to_tariff(selected_tariff, other_charges)

    selected_tariff_as_retail = convert_network_tariff_to_retail_tariff(selected_tariff)

    return selected_tariff_as_retail
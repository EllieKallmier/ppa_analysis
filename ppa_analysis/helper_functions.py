# File to hold functions that will assist with testing, validation or other small formatting and calculation tasks that are secondary to the main functionality of the tool.
import pandas as pd
import os
import json
import copy
import holidays
from datetime import timedelta
from collections import Counter
from ppa_analysis import advanced_settings
from ppa_analysis.tariffs import add_other_charges_to_tariff, convert_network_tariff_to_retail_tariff


# Test help functions:
def _check_missing_data(df:pd.DataFrame) -> pd.DataFrame:
    """Checks and fills missing data in the DataFrame.

    This function checks if the DataFrame contains missing values (NaN). If missing
    data is found, it fills the missing values with zeros and prints a message indicating that the data has been filled. If the DataFrame is empty, it prints a warning. 

    Args:
        df (pd.DataFrame): A DataFrame to check for missing data.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled with zeros, if any.

    Note:
        The function modifies the original DataFrame by filling NaN values with 
        zeros and prints warnings for empty or incomplete data.
    """
    if df.empty:
        print('DataFrame is empty.')
    
    nan_df = df.copy().dropna(how='any')

    if nan_df.shape != df.shape:
        print('Some missing data found. Filled with zeros.')
        df = df.fillna(0.0)

    return df


# Returns an integer representing minutes in the interval
def get_interval_length(df:pd.DataFrame) -> int:
    """Returns the time interval length in minutes between consecutive timestamps.

    This function calculates the time difference between the first and second 
    timestamps, and the last and second-to-last timestamps in the 'DateTime' column 
    of the DataFrame. If the intervals are consistent, the function returns the 
    interval length in minutes. If the intervals differ, it prints a warning and 
    returns the interval length based on the first timestamp difference.

    Args:
        df (pd.DataFrame): A DataFrame with a 'DateTime' column containing 
            timestamp values.

    Returns:
        int: The time interval length in minutes between consecutive timestamps.

    Note:
        If the interval lengths are inconsistent, the function prints a warning 
        and returns the interval based on the first two timestamps.
    """
    df = df.copy().reset_index()

    first_int = df['DateTime'].iloc[1] - df['DateTime'].iloc[0]
    last_int = df['DateTime'].iloc[-1] - df['DateTime'].iloc[-2]

    if first_int == last_int:
        return int(first_int.total_seconds() / 60)
    else:
        print('Interval lengths are different throughout dataset.\n')
        return int(first_int.total_seconds() / 60)


def _check_interval_consistency(df:pd.DataFrame, mins:int) -> tuple[bool, pd.Timestamp]:
    """Checks if the time intervals in the DataFrame are consistent.

    This function checks whether the difference between consecutive timestamps 
    in the 'DateTime' column of the DataFrame is consistently equal to the 
    specified interval (in minutes).

    Args:
        df (pd.DataFrame): A DataFrame with a datetime index called 'DateTime'.
        mins (int): The expected time difference (in minutes) between consecutive 
            timestamps.

    Returns:
        tuple: A tuple containing:
            - bool: True if all intervals between consecutive timestamps are 
              consistent with the specified `mins`, False otherwise.
            - pd.Timestamp: The first timestamp where the interval inconsistency 
              was detected, or the last valid timestamp if intervals are consistent.
    """
    df = df.copy().reset_index()
    return (df['DateTime'].diff() == timedelta(minutes=mins)).iloc[1:].all()


def check_leap_year(
        df:pd.DataFrame
) -> bool:
    """Checks if the data given occurs in a leap year.

    This function checks whether a given set of data is recorded for a leap year
    by verifying if adding 365 days to the first day of the dataset results in the 
    same day of the month. If not, the year contains a leap day (February 29).

    Args:
        df (pd.DataFrame): A DataFrame with a DateTime index.

    Returns:
        bool: True if the year contains a leap day, False otherwise.
    """
    day_one = df.index[0]
    day_365 = day_one + timedelta(days=365)

    return day_one.day != day_365.day


def convert_gen_editor_to_dict(
        editor:dict
) -> dict:
    """
    Convert generator_data_editor to simple nested dict.

    Helper function to convert the nested dict with widget objects to a simple 
    nested dict in order to pass generator info to further functions.
    :param editor: a generator_data_editor object which is a nested dict with key
        value pairs of generator name, generator info. Values are dictionaries 
        containing widget objects.
    
    :return: new_dict, a simpler nested dict containing the same structure but instead
        of widget objects, the values of each input widget are returned.
    """
    new_dict = {}
    for key, value in editor.items():
        if key != 'out':
            new_dict[key] = {}
            for key_2, value_2 in value.items():
                if key_2 != 'label':
                    new_dict[key][key_2] = value_2.value
    return new_dict


# Helper function to create the "shaped" profile based on the defined period and 
# percentile
def get_percentile_profile(
        period_str:str,
        data:pd.DataFrame,
        percentile:float
) -> pd.DataFrame:
    """Generates a shaped profile based on a specified percentile and period.

    This function calculates a "shaped" profile by grouping the data based on a 
    specified period ('Y', 'M', or 'Q'), and then computes the given percentile 
    (e.g., 0.5 for median) for each group. The result is a DataFrame with the 
    percentile values for each period and hour.

    Args:
        period_str (str): The period to group the data by. Can be one of:
            - 'Y' for yearly,
            - 'M' for monthly,
            - 'Q' for quarterly.
        data (pd.DataFrame): A DataFrame containing a datetime index and column 
            containing generation data, where the values to compute percentiles 
            are located.
        percentile (float): The percentile to calculate (between 0 and 1), e.g., 
            0.5 for the median.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated percentile profile, 
            indexed by the specified period and hour.
    """
    
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


def concat_shaped_profiles(
        period_str:str,
        shaped_data:pd.DataFrame,
        long_data:pd.DataFrame,
) -> pd.DataFrame:
    """Applies a daily "shaped" profile across a full datetime timeseries.

    This function extends a given daily shaped profile (percentile-based load profile) 
    to a timeseries DataFrame. The shaped profile is concatenated with the long 
    timeseries data according to the period on which it was shaped, allowing the 
    profile to be applied across the full datetime range.

    Args:
        period_str (str): The period over which to reshape the data. Must be one of:
            - 'Y' for yearly,
            - 'M' for monthly,
            - 'Q' for quarterly.
        shaped_data (pd.DataFrame): A DataFrame containing the shaped 'percentile profile' 
            to be applied across the timeseries. This DataFrame should have hourly
            profiles for a full day in each specified period. 
        long_data (pd.DataFrame): A DataFrame containing a full datetime index, 
            with a 'DateTime' column with hourly intervals, to which the shaped 
            profile will be added.

    Returns:
        pd.DataFrame: A DataFrame with the shaped profile added as a new column, 
            indexed by 'DateTime', and containing the original data along with 
            the reshaped profile.
    """
    
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


def yearly_indexation(
        df: pd.DataFrame,
        strike_price: float,
        indexation: float | list[float]
) -> pd.DataFrame:
    """
    Helper function to calculate yearly indexation.

    The function takes a dataframe with an index of type datetime and returns the same dataframe with an additional
    column named 'Strike Price (Indexed)'. For each year after the initial year in the index, the strike price is
    increased by the specified indexation rate. If the indexation rate is provided as a float then the same rate is
    used for all years. If a list is then each year uses the next indexation rate in the list and if there are more
    years than rates in the list then last rate is reused.

    :param df: with datetime index
    :param strike_price: in $/MW/h
    :param indexation: as percentage i.e. 5 is an indexation rate of 5 %
    :return: The input dataframe with an additional column named 'Strike Price (Indexed)'
    """

    years = df.index.year.unique()

    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if not isinstance(indexation, list):
        indexation = [indexation] * len(years)

    while len(indexation) < len(years):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, year in enumerate(years):
        spi_map[year] = strike_price
        strike_price += strike_price * indexation[i] / 100

    spi_map[year] = strike_price

    df_with_strike_price = df.copy()

    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price.index.year
    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price['Strike Price (Indexed)'].map(spi_map)

    return df_with_strike_price['Strike Price (Indexed)']


def quarterly_indexation(
        df: pd.DataFrame,
        strike_price: float,
        indexation: float | list[float]
) -> pd.DataFrame:
    """
    Helper function to calculate quarterly indexation.

    The function takes a dataframe with an index of type datetime and returns the same dataframe with an additional
    column named 'Strike Price (Indexed)'. For each quarter after the initial quarter in the index, the strike price is
    increased by the specified indexation rate. If the indexation rate is provided as a float then the same rate is
    used for all quarters. If a list is given then each quarter uses the next indexation rate in the list and if there
    are more quarters than rates in the list then last rate is reused.

    :param df: with datetime index
    :param strike_price: in $/MWh
    :param indexation: as percentage i.e. 5 is an indexation rate of 5 %
    :return: The input dataframe with an additional column named 'Strike Price (Indexed)'
    """

    years = df.index.year.unique()

    quarters = [(year, quarter) for year in years for quarter in range(1, 5)]

    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if not isinstance(indexation, list):
        indexation = [indexation] * len(quarters)

    while len(indexation) < len(quarters):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, quarter in enumerate(quarters):
        spi_map[quarter] = strike_price
        strike_price += strike_price * indexation[i] / 100

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
# Returns LCOE value in $/MWh
def calculate_lcoe(
    generator_info:dict[str:object]        
) -> float:
    """
    Calculate LCOE for chosen generator. 

    This function takes inputs for one renewable energy generator and calculates
    the levelised cost of energy (LCOE) for that generator in $/MWh. 

    :param generator_info: dict containing key value pairs of generator information
        as follows:
        generator_info = {
            'Capital ($/kW)' : float,
            'Construction Time (years)' : float,
            'Economic Life (years)' : float,
            'Fixed O&M ($/kW)' : float,
            'Variable O&M ($/kWh)' : float,
            'Capacity Factor' : float
        }
    
    :return: float, the calculated LCOE value for this generator in $/MWh.
    """
    
    # Baseline assumptions:
    discount_rate = advanced_settings.DISCOUNT_RATE

    capital = generator_info['Capital ($/kW)']*1000
    construction_years = generator_info['Construction Time (years)']
    economic_life = generator_info['Economic Life (years)']
    fixed_om = generator_info['Fixed O&M ($/kW)']
    variable_om = generator_info['Variable O&M ($/kWh)']
    capacity_factor = generator_info['Capacity Factor']

    first_capital_sum = (capital*(1+discount_rate)**construction_years * \
                         discount_rate * (1+discount_rate)**economic_life) / \
                            (((1+discount_rate)**economic_life)-1)/(8760*capacity_factor)

    op_and_main = variable_om * ((fixed_om*1000)/(8760*capacity_factor))

    lcoe = first_capital_sum + op_and_main

    return lcoe


# ----- Fetch inputs and set up info_dict data to pass to later functions:
def get_all_lcoes(
        generator_data_dict:dict
) -> dict[str:float]:
    
    """ 
    Calculate LCOE value for all selected renewable energy generators.

    Acts as a wrapper function to call calculate_lcoe repeatedly for each of the
    selected generators. This function is expected to be primarily called from interface.ipynb
    or api_examples.ipynb, but is public and can be utilised anywhere. Returns from
    this function are structured to be passed directly to create_hybrid_generation().

    :param generator_data_dict: a nested dictionary containing a set of key value pairs
        with generator information to be fed into the lcoe calculator.

        dictionary structure should be as follows:
        generator_data_dict = {
        'Generator Name 1 ' : {
            'Capital ($/kW)' : float,
            'Construction Time (years)' : float,
            'Economic Life (years)' : float,
            'Fixed O&M ($/kW)' : float,
            'Variable O&M ($/kWh)' : float,
            'Capacity Factor' : float
        }, 
        'Generator Name 2' : {
            ...
        }, 
        ...
    }

    :return: dict containing key:value pairs of generator name (str) and LCOE
        value for the corresponding generator (float) in $/MWh.
    """

    all_generator_lcoes = {}
    for gen, gen_info in generator_data_dict.items():
        if gen != 'out':
            gen_lcoe = calculate_lcoe(gen_info)
            all_generator_lcoes[gen] = gen_lcoe
    
    return all_generator_lcoes


# TODO: Ellie to add docstrings here
# Helper function to read in json files (for network tariff selection)
def read_json_file(filename):
    """Reads a JSON file and returns its contents.

    Args:
        filename (str): The name of the JSON file (without the '.json' extension) 
            to be read.

    Returns:
        dict: The data from the JSON file, parsed into a Python dictionary.

    Note:
        The function assumes the file is located in the current working directory 
        and has the '.json' extension.
    """
    f = open(f'{filename}.json', 'r')
    data = json.loads(f.read())
    f.close()
    return data


# Add an extra column to load_and_gen_data df that holds a string value denoting the season associated with the index timestamp.
def get_seasons(
    load_and_gen:pd.DataFrame
) -> pd.DataFrame:
    """Adds a season column to a DataFrame based on the index timestamp.

    This function adds a new column, 'Season', to the provided DataFrame, 
    assigning a season label ('Summer', 'Autumn', 'Winter', 'Spring') to each 
    row based on the month of the timestamp in the index.

    Args:
        load_and_gen (pd.DataFrame): A DataFrame with a datetime index.

    Returns:
        pd.DataFrame: A new DataFrame with the added 'Season' column. The season 
            is determined by the month of the timestamp.
    """

    seasonal_load_and_gen = load_and_gen.copy()
    seasonal_load_and_gen['Season'] = ''

    seasonal_load_and_gen.loc[
        seasonal_load_and_gen.index.month.isin([1,2,12]), 'Season'] = 'Summer'
    seasonal_load_and_gen.loc[
        seasonal_load_and_gen.index.month.isin([3,4,5]), 'Season'] = 'Autumn'
    seasonal_load_and_gen.loc[
        seasonal_load_and_gen.index.month.isin([6,7,8]), 'Season'] = 'Winter'
    seasonal_load_and_gen.loc[
        seasonal_load_and_gen.index.month.isin([9,10,11]), 'Season'] = 'Spring'

    return seasonal_load_and_gen


# Helper function: get weekday/weekends
# Sets regional public holidays as 'weekend' alongside Sat/Sun
def get_weekends(
        load_and_gen:pd.DataFrame, 
        region:str
    ):
    """Adds a 'Weekend' column to a DataFrame, marking weekends and public holidays.

    This function identifies weekends (Saturday and Sunday) and public holidays 
    for a given region and adds a 'Weekend' column to the provided DataFrame. Public 
    holidays are determined using the `holidays` library, with holidays in the 
    specified region being treated as weekends.

    Args:
        load_and_gen (pd.DataFrame): A DataFrame with a datetime index containing 
            the load and generation data.
        region (str): The region code (e.g., 'NSW' for New South Wales) to determine 
            public holidays in Australia. At the moment, Australia is the hard-coded
            country in this function.

    Returns:
        pd.DataFrame: A new DataFrame with the 'Weekend' column (int), where weekends 
            (Saturday and Sunday) and public holidays are 1 (True), and weekdays 
            are 0 (False).

    Note:
        The function assumes that the DataFrame has a datetime index. Public holidays 
        are determined based on the 'AU' (Australia) country holidays, and weekends 
        are based on Saturday (5) and Sunday (6).
    """
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
    """Formats user-filled extra charges into a structured format for bill calculation.

    This function takes a dictionary of extra charges collected from user input, 
    re-formats it into a structured dictionary that matches the required format 
    for use in the Sunspot bill calculator. If no charges are provided, the function 
    returns default zero values for all charge categories.

    Args:
        extra_charges_collector (dict): A dictionary containing the user input for 
            various charge categories. Each key corresponds to a specific charge type 
            and its value holds the corresponding user input (such as a rate or charge value).

    Returns:
        dict: A dictionary formatted to match the expected structure for the bill calculator, 
            with fields for energy, metering, environmental, market, variable, and fixed charges.
            Each charge includes the unit and value (either from the input or set to zero by default).
    """
    
    if len(extra_charges_collector) > 0:
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
    else:
        other_charges = {
            "Energy Charges": {
                "Peak Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "Shoulder Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "Off-Peak Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "Retailer Demand Charge": {
                    "Unit": "$/kVA/day",
                    "Value": 0.0
                }
            },
            "Metering Charges": {
                "Meter Provider/Data Agent Charges": {
                    "Unit": "$/Day",
                    "Value": 0.0
                },
                "Other Meter Charges": {
                    "Unit": "$/Day",
                    "Value": 0.0
                }
            },
            "Environmental Charges": {
                "LREC Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "SREC Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "State Environment Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                }
            },
            "AEMO Market Charges": {
                "AEMO Participant Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                },
                "AEMO Ancillary Services Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.0
                }
            },
            "Other Variable Charges": {
                "Other Variable Charge 1": {
                    "Unit": "$/kWh",
                    "Value": ""
                },
                "Other Variable Charge 2": {
                    "Unit": "$/kWh",
                    "Value": ""
                },
                "Other Variable Charge 3": {
                    "Unit": "$/kWh",
                    "Value": ""
                }
            },
            "Other Fixed Charges": {
                "Total GST": {
                    "Unit": "$/Bill",
                    "Value": ""
                },
                "Other Fixed Charge 1": {
                    "Unit": "$/Bill",
                    "Value": ""
                },
                "Other Fixed Charge 2": {
                    "Unit": "$/Bill",
                    "Value": ""
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
        chosen_tariff:dict|str,
        extra_charges_collector:dict,
        retail:bool=False
) -> dict:
    """ 
    Selects and returns the user's chosen network tariff formatted to be passed to CEEM's SunSPOT bill_calculator function as a retail tariff. If extra charges are supplied and the tariff type is retail, the extra charges are also added to the tariff structure and returned.

    :param chosen_tariff: dict, containing the name of the user's chosen tariff as a str
        with the key 'tariff_name'.
    :param extra_charges_collector: dict, containing user inputs for extra charges 
        as defined in user_inputs.launch_extra_charges_collector() or hard-coded.
        Pass as empty dict if no extra charges are to be considered.
    :param retail: bool, indicates whether the tariff selected to be returned is 
        a retail tariff and therefore needs other charges to be added before reformatting.
    :return: dict, the selected tariff returned reformatted and with other charges 
        added as necessary.
    """
    all_tariffs = read_json_file(advanced_settings.COMMERCIAL_TARIFFS_FN)
    all_tariffs = all_tariffs[0]['Tariffs']
    if isinstance(chosen_tariff, dict):
        selected_tariff_name = chosen_tariff['tariff_name'].value
    else:
        selected_tariff_name = chosen_tariff

    selected_tariff = {}
    for tariff in all_tariffs:
        if tariff['Name'] == selected_tariff_name:
            selected_tariff = copy.deepcopy(tariff)

    if retail:
        other_charges = format_other_charges(extra_charges_collector)
        selected_tariff = add_other_charges_to_tariff(selected_tariff, other_charges)

    selected_tariff_as_retail = convert_network_tariff_to_retail_tariff(selected_tariff)

    return selected_tariff_as_retail
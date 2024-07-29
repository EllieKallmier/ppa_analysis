import pandas as pd

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


# straight from sunspot bill calculator function...
# but renamed variables for clarity, changed return
# and slightly altered functionality (now fills in a column 'Firming' at the 
# chosen/specified times with the tariff rate corresponding).
def tariff_firming_col_fill(
        load_and_gen_data:pd.DataFrame, 
        tariff_component_details:dict
) -> pd.DatetimeIndex:
    """ 
    For a component of a TOU tariff, adds that component's value (or rate) in $/MWh
    to the rows corresponding to times in which that component applies.

    :param load_and_gen_data: Dataframe with datetime index, and at least a column named 'Firming' that is either filled with zeroes or partially filled with tariff rates.
    :param tariff_component_details: a dictionary with the following structure:
            tariff_component_details = {
                "Month": [1, 2, 12],        # list of integers representing months
                "TimeIntervals": {
                    "T1": [
                    start_time (string),
                    end_time (string)
                    ]
                },
                "Unit": "$/kWh",
                "Value": float,
                "Weekday": bool,
                "Weekend": bool
            }
    """
    load_and_gen_data = load_and_gen_data.copy()
    data_at_tariff_selected_intervals = pd.DataFrame()
    for label, time_value, in tariff_component_details['TimeIntervals'].items():
        if time_value[0][0:2] == '24':
            time_value[0] = time_value[1].replace("24", "00")
        if time_value[1][0:2] == '24':
            time_value[1] = time_value[1].replace("24", "00")
        if time_value[0] != time_value[1]:
            data_between_times = load_and_gen_data.between_time(start_time=time_value[0], end_time=time_value[1],
                                                            include_start=False, include_end=True)
        else:
            data_between_times = load_and_gen_data.copy()

        if not tariff_component_details['Weekday']:
            data_between_times = data_between_times.loc[data_between_times.index.weekday >= 5].copy()

        if not tariff_component_details['Weekend']:
            data_between_times = data_between_times.loc[data_between_times.index.weekday < 5].copy()

        data_between_times = data_between_times.loc[data_between_times.index.month.isin(tariff_component_details['Month']), :].copy()

        data_at_tariff_selected_intervals = pd.concat([data_at_tariff_selected_intervals, data_between_times])
    
    index_for_selected_times = data_at_tariff_selected_intervals.index
    load_and_gen_data.loc[index_for_selected_times, 'Firming price'] += tariff_component_details['Value'] * 1000

    return load_and_gen_data


# Retail tariff contract:
def retail_tariff_contract(
        df:pd.DataFrame,
        tariff_details:dict
) -> pd.DataFrame:
    df_with_firming = df.copy()
    df_with_firming['Firming price'] = 0

    for component_name, info in tariff_details['Parameters']['NUOS'].items():
        if 'FlatRate' in component_name:
            df_with_firming['Firming price'] += info['Value'] * 1000
        if 'TOU' in component_name:
            for tou_component, tou_info in info.items():
                df_with_firming = tariff_firming_col_fill(df_with_firming, tou_info)

    if len(df_with_firming[df_with_firming['Firming price']==0]) == len(df_with_firming):
        df_with_firming['Firming price'] = df_with_firming[f'RRP'].copy()

    return df_with_firming


# Function to choose which firming contract to apply:
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
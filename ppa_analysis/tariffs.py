"""
Module containing functions for calculating bill outcomes under retail and network
tariffs for large commercial customers.

This module was originally written for CEEM tool TDA Python: https://github.com/UNSW-CEEM/TDA_Python,
and is used in the APVI's SunSPOT tool: https://solarcalculator.sunspot.org.au/?i=f20a67e6b11744db9718385841aa38dc.
Functions have been adapted to match data type inputs of this PPA Analysis tool
and in the case of add_other_charges_to_tariff() changed significantly to be fit
for purpose here.

The following functions are called in bill_calc.py module to calculate either
network or retail bills as required, but are also public if the user wishes to use
them otherwise.
"""

import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")


def time_select(
    load_profile_s: pd.DataFrame, tariff_component_details: dict
) -> pd.DataFrame:
    """Filters a load profile DataFrame based on specified time intervals and days
    of the week/month from tariff component details.

    Args:
        load_profile_s: A DataFrame containing the load profile data with
             a DateTime index.
        tariff_component_details: A dictionary containing the time intervals, weekdays,
            weekends, and months for filtering. The dictionary must have the following key/value
            pairs
            - TimeIntervals: A dictionary where each key is an interval ID
                and each value is a list of two time strings (start and end).
                Time strings in 'TimeIntervals' should be in 'HH:MM' format.
                Time intervals starting at '24:00' are adjusted to '00:00'
                for proper filtering.
            - Weekday: A boolean indicating whether weekdays are included in this
                tariff component.
            - Weekend: A boolean indicating whether weekends are included in this
                tariff component.
            - Month: A list of integers representing the months included in this
                component (e.g., [1, 2, 3] for January, February, March).
            Dict structure looks like:
            tariff_component_details = {
                "Month": [
                    1,
                    2,
                    12
                ],
                "TimeIntervals": {
                    "T1": [
                        "22:00",
                        "07:00"
                    ]
                },
                "Weekday": true,
                "Weekend": false
            }

    Returns:
        load_profile_selected_times: A DataFrame filtered to
            include only the rows that fall within the specified time intervals,
            and match the specified weekday/weekend and month criteria for the
            given tariff component.

    """
    load_profile_selected_times = pd.DataFrame()
    for (
        interval_id,
        times,
    ) in tariff_component_details["TimeIntervals"].items():
        if times[0][0:2] == "24":
            times[0] = times[1].replace("24", "00")
        if times[1][0:2] == "24":
            times[1] = times[1].replace("24", "00")
        if times[0] != times[1]:
            lp_between_times = load_profile_s.between_time(
                start_time=times[0], end_time=times[1], inclusive="right"
            )
        else:
            lp_between_times = load_profile_s.copy()

        if not tariff_component_details["Weekday"]:
            lp_times_and_days = lp_between_times.loc[
                lp_between_times.index.weekday >= 5
            ].copy()
        elif not tariff_component_details["Weekend"]:
            lp_times_and_days = lp_between_times.loc[
                lp_between_times.index.weekday < 5
            ].copy()
        else:
            lp_times_and_days = lp_between_times.copy()
        lp_times_days_months = lp_times_and_days.loc[
            lp_times_and_days.index.month.isin(tariff_component_details["Month"]), :
        ].copy()

        load_profile_selected_times = pd.concat(
            [load_profile_selected_times, lp_times_days_months]
        )
    return load_profile_selected_times


def calculate_daily_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the total daily charges for a bill.

    Args:
        load_profile: A DataFrame containing the load profile data in kWh
            with a DateTime index.
        tariff_component: A dictionary containing tariff details. It should
            include a 'Daily' key with a nested dictionary that has a 'Value' key
            specifying the daily charge rate as follows:
            tariff_component = {
                ...
                'Daily' : {
                    'Unit' : '$/Day',
                    'Value' : 10.0
                }
                ...
            }
        results: dict, not used here, included to simplify control logic.
        tariff_category: A string representing the tariff category, one of 'NUOS'
            or 'Retailer' (not used here, included to simplify control logic).

    Returns:
        float: The bill's total daily charge in dollars ($), calculated as
            num_days_in_load_profile * daily_charge_value.

    """
    num_days = len(load_profile.index.normalize().unique()) - 1
    daily_charge = num_days * tariff_component["Daily"]["Value"]
    return daily_charge


def calculate_fixed_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Returns the total fixed charges for a bill.

    Args:
        load_profile: A DataFrame containing the load profile data
            with a DateTime index (not used here, included to simplify control
            logic).
        tariff_component: A dictionary containing tariff details. It should
            include a 'Fixed' key with a nested dictionary that has a 'Value' key
            specifying the fixed rate per bill as follows:
            tariff_component = {
                ...
                'Fixed' : {
                    'Unit' : '$/Bill',
                    'Value' : 100.0
                }
                ...
            }
            - 'Unit' must be '$/Bill'
        results: dict, not used here, included to simplify control logic.
        tariff_category: A string representing the tariff category, one of 'NUOS'
            or 'Retailer' (not used here, included to simplify control logic).

    Returns:
        float: The bill's total fixed charge in dollars ($).

    """
    return tariff_component["Fixed"]["Value"]


def calculate_flatrate_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the total of all flat rate charges for a bill.

    Args:
        load_profile: DataFrame not used here, included to simplify control logic.
        tariff_component: A dictionary containing tariff details. It should
            include a 'FlatRate' key with a nested dictionary that has a 'Value' key
            specifying the daily charge rate as follows:
            tariff_component = {
                ...
                'FlatRate' : {
                    'Unit' : '$/kWh',
                    'Value' : 0.55
                }
                ...
            }
        results: A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key tariff_category with a pd.DataFrame that stores tariff component
            results.
            Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                tariff_category : pd.DataFrame()
            }
        tariff_category: str, not used here, included to simplify control logic.

    Returns:
        float: The bill's total daily charge in dollars ($), calculated as
            num_days_in_load_profile * daily_charge_value.

    """
    flat_rate_charge = (
        results["LoadInfo"]["Annual_kWh"] * tariff_component["FlatRate"]["Value"]
    )
    return flat_rate_charge


def calculate_annual_block_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the total of all annual block charges for a bill.

    For each block described in the tariff component, energy usage is compared
    against the bounds of the block. Usage up to the upper bound of the block
    is charged at the block's set rate, and the remaining energy use is charged
    under the next block's rate (and so on). For example, with an annual usage
    of 1000kWh and an upper bound of 800kWh for the first block at $0.5/kWh
    and no upper bound for the second block at $0.8/kWh, the annual charge
    is calculated as 800 * 0.5 + 200 * 0.8 = $560.

    Args:
        load_profile: DataFrame not used here, included to simplify control
            logic.
        tariff_component: A dictionary containing tariff details. It should
            include a 'BlockAnnual' key with a nested dictionary with the following
            structure:
            tariff_component = {
            ...
                'BlockAnnual' : {
                    'Block1' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.20,
                        'HighBound' : 60
                    },
                    'Block2' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.55,
                        'HighBound' : Infinity
                    },
                    ...
                }
                ...
            }
        results: A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key tariff_category with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                tariff_category : pd.DataFrame()
            }
        tariff_category: str, not used here, included to simplify control logic.

    Returns:
        float: The bill's total daily charge in dollars ($), calculated as
            num_days_in_load_profile * daily_charge_value.

    """
    block_use = results["LoadInfo"][["Annual_kWh"]].copy()
    block_use_charge = block_use.copy()
    lim = 0
    for k, v in tariff_component["BlockAnnual"].items():
        block_use[k] = block_use["Annual_kWh"]
        block_use[k][block_use[k] > float(v["HighBound"])] = float(v["HighBound"])
        block_use[k] = block_use[k] - lim
        block_use[k][block_use[k] < 0] = 0
        lim = float(v["HighBound"])
        block_use_charge[k] = block_use[k] * v["Value"]
    del block_use["Annual_kWh"]
    del block_use_charge["Annual_kWh"]
    annual_block_charge = block_use_charge.sum(axis=1)

    return annual_block_charge


def calculate_quarterly_block_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the quarterly block charge based on the load profile and
    tariff component details.

    This function calculates quarterly consumption for each of the four quarters,
    applies the block tariff charges based on consumption within each block, and
    sums up the charges for each quarter. This total charge is returned as a float.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a 'BlockQuarterly' key with a nested dictionary where each
            key represents a block and each value is a dictionary with 'HighBound'
            and 'Value' specifying the upper bound and charge rate for that block:
            tariff_component = {
            ...
                'BlockQuarterly' : {
                    'Block1' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.20,
                        'HighBound' : 60
                    },
                    'Block2' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.55,
                        'HighBound' : Infinity
                    },
                    ...
                }
                ...
            }
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }

        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The total quarterly block charge calculated from the load profile
            and tariff component details.

    Notes:
        - Quarterly periods are defined as:
            Q1: January - March
            Q2: April - June
            Q3: July - September
            Q4: October - December
    """
    # first: get quarterly consumption and save in the results 'LoadInfo' df:
    for Q in range(1, 5):
        lp_quarterly = load_profile.loc[
            load_profile.index.month.isin(list(range((Q - 1) * 3 + 1, Q * 3 + 1))), :
        ]
        results["LoadInfo"]["kWh_Q" + str(Q)] = [
            np.nansum(lp_quarterly[col].values[lp_quarterly[col].values > 0])
            for col in lp_quarterly.columns
        ]

    # then get the charge for each quarter:
    for Q in range(1, 5):
        block_use = results["LoadInfo"][["kWh_Q" + str(Q)]].copy()
        block_use_charge = block_use.copy()
        lim = 0
        for k, v in tariff_component["BlockQuarterly"].items():
            block_use[k] = block_use["kWh_Q" + str(Q)]
            block_use[k][block_use[k] > float(v["HighBound"])] = float(v["HighBound"])
            block_use[k] = block_use[k] - lim
            block_use[k][block_use[k] < 0] = 0
            lim = float(v["HighBound"])
            block_use_charge[k] = block_use[k] * v["Value"]
        del block_use["kWh_Q" + str(Q)]
        del block_use_charge["kWh_Q" + str(Q)]

        results[tariff_category]["C_BlockQuarterly_" + str(Q)] = block_use_charge.sum(
            axis=1
        )

    quarterly_block_charge = results[tariff_category][
        ["C_BlockQuarterly_" + str(Q) for Q in range(1, 5)]
    ].sum(axis=1)
    return quarterly_block_charge


def calculate_monthly_block_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the monthly block charge based on the load profile and
    tariff component details.

    This function calculates consumption within each month, applies the block tariff
    charges based on consumption within each block, and sums up the charges for each
    month. This total charge is returned as a float.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a 'BlockMonthly' key with a nested dictionary where each
            key represents a block and each value is a dictionary with 'HighBound'
            and 'Value' specifying the upper bound and charge rate for that block:
            tariff_component = {
            ...
                'BlockMonthly' : {
                    'Block1' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.20,
                        'HighBound' : 60
                    },
                    'Block2' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.55,
                        'HighBound' : Infinity
                    },
                    ...
                }
                ...
            }
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }

        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The total monthly block charge calculated from the load profile
            and tariff component details.

    """
    # Get monthly consumtion and store in results 'LoadInfo' df:
    for m in range(1, 13):
        lp_monthly = load_profile.loc[load_profile.index.month == m, :]
        results["LoadInfo"]["kWh_m" + str(m)] = [
            np.nansum(lp_monthly[col].values[lp_monthly[col].values > 0])
            for col in lp_monthly.columns
        ]

    # then calculate the charge for each month:
    for m in range(1, 13):
        block_use = results["LoadInfo"][["kWh_m" + str(m)]].copy()
        block_use_charge = block_use.copy()
        lim = 0
        for k, v in tariff_component["BlockMonthly"].items():
            block_use[k] = block_use["kWh_m" + str(m)]
            block_use[k][block_use[k] > float(v["HighBound"])] = float(v["HighBound"])
            block_use[k] = block_use[k] - lim
            block_use[k][block_use[k] < 0] = 0
            lim = float(v["HighBound"])
            block_use_charge[k] = block_use[k] * v["Value"]
        del block_use["kWh_m" + str(m)]
        del block_use_charge["kWh_m" + str(m)]
        results[tariff_category]["C_BlockMonthly_" + str(m)] = block_use_charge.sum(
            axis=1
        )

    monthly_block_charge = results[tariff_category][
        ["C_BlockMonthly_" + str(m) for m in range(1, 13)]
    ].sum(axis=1)
    return monthly_block_charge


def calculate_daily_block_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the daily block charge based on the load profile and
    tariff component details.

    This function calculates consumption within each month, applies the block tariff
    charges based on consumption within each block, and sums up the charges for each
    month. This total charge is returned as a float.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a 'BlockDaily' key with a nested dictionary where each
            key represents a block and each value is a dictionary with 'HighBound'
            and 'Value' specifying the upper bound and charge rate for that block:
            tariff_component = {
            ...
                'BlockDaily' : {
                    'Block1' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.20,
                        'HighBound' : 60
                    },
                    'Block2' : {
                        'Unit' : '$/kWh',
                        'Value' : 0.55,
                        'HighBound' : Infinity
                    },
                    ...
                }
                ...
            }
        results (dict): dict, not used here, included to simplify control logic.
        tariff_category (str): str, not used here, included to simplify control logic.

    Returns:
        float: The total daily block charge calculated from the load profile
            and tariff component details.

    """

    # First, resample the load profile to get daily usage:
    daily_kwh_usage = load_profile.resample("D").sum()
    block_use_temp_charge = daily_kwh_usage.copy()
    block_use_temp_charge.iloc[:, :] = 0
    lim = 0
    # then apply the daily blocks to find daily charges:
    for block, details in tariff_component["BlockDaily"].items():
        block_use_temp = daily_kwh_usage.copy()
        block_use_temp[block_use_temp > float(details["HighBound"])] = float(
            details["HighBound"]
        )
        block_use_temp = block_use_temp - lim
        block_use_temp[block_use_temp < 0] = 0
        lim = float(details["HighBound"])
        block_use_temp_charge = (
            block_use_temp_charge + block_use_temp * details["Value"]
        )
    daily_block_charge = block_use_temp_charge.sum(axis=0)
    return daily_block_charge


def calculate_time_of_use_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the total time of use energy charge based on the load profile and
    tariff component details.

    This function calculates consumption within each defined time of use period,
    applies the tariff rate based on consumption within each period, and sums up
    all time of use charges.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a 'TOU' key with a nested dictionary where each key represents
            a charging period and each value is a dictionary with details specifying
            month, time and weekdays during which the charge applies, as well as
            the units ($/kWh) and rate of the charge itself.
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }
        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The total TOU charge calculated from the load profile and tariff
            component details.

    """
    # First set up temporary dfs to hold interim results:
    time_of_use_consumption = pd.DataFrame()
    time_of_use_consumption_charge = pd.DataFrame()
    # Loop over each TOU component (e.g. Peak, Weekend Off-Peak, Shoulder etc)
    # and fill in any missing details with default values
    for tou_component, details in tariff_component["TOU"].items():
        details_copy = details.copy()
        if "Weekday" not in details_copy:
            details_copy["Weekday"] = True
            details_copy["Weekend"] = True
        if "TimeIntervals" not in details_copy:
            details_copy["TimeIntervals"] = {"T1": ["00:00", "00:00"]}
        if "Month" not in details_copy:
            details_copy["Month"] = list(range(1, 13))

        # Then call time_select to get the load_profile for times during which
        # this charge component applies. Calculate usage then total charge for
        # this period:
        lp_time_of_use = time_select(load_profile, details_copy)
        time_of_use_consumption[tou_component] = lp_time_of_use.sum()
        results[tariff_category]["kWh_" + tou_component] = time_of_use_consumption[
            tou_component
        ].copy()
        time_of_use_consumption_charge[tou_component] = (
            details_copy["Value"] * time_of_use_consumption[tou_component]
        )
        results[tariff_category]["TOU_" + tou_component] = (
            time_of_use_consumption_charge[tou_component].copy()
        )

    time_of_use_charge = time_of_use_consumption_charge.sum(axis=1)
    return time_of_use_charge


def calc_dem_(
    dem_component_details: dict,
    num_peaks: int,
    load_profile_selected_times: pd.DataFrame,
    tariff_category: str,
    demand_component: str,
    results: dict,
) -> float:
    """Calculate the demand charge based on demand component details and load profile data.

    This function computes the demand charge based on the provided demand component details,
    the number of peaks to consider, and the load profile. It updates a results DataFrame
    with the average demand and the total demand charge for the given tariff category and
    demand component.

    Args:
        dem_component_details: A dictionary containing details about the demand component,
            such as minimum demand and charge values. Expected keys are 'Value', 'Unit' ($/kW/day),
            'Min Demand (kW)' and 'Min Demand Charge ($)'.
        num_peaks: The number of peaks to consider when calculating the demand charge.
        load_profile_selected_times: DataFrame containing the load profile with
            datetime index and at least one column named 'kWh' containing half-hourly
            load data. This dataframe will contain load data for selected periods
            based on the tariff component, calculated before being passed to this function.
        tariff_category: A string representing the tariff category, used
            to store the charges in the results dictionary.
        demand_component:A string naming the demand charge component, used
            to store the charges in the results dictionary.
        results:A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results.

    Returns:
        float: The total demand charge calculated based on the demand component details and load
            profile data.

    Notes:
        - The function updates the `results[<tariff_category>]` DataFrame with two new columns for the specified
          `demand_component`:
            - 'Avg_kW_Dem_<demand_component>': The average demand in kW.
            - 'Demand_<demand_component>': The total demand charge in dollars.
    """

    # Get any value(s) for min demand present in the tariff definition:
    min_demand = 0
    min_demand_from_charge = 0
    if "Min Demand (kW)" in dem_component_details:
        min_demand = dem_component_details["Min Demand (kW)"]
    if "Min Demand Charge ($)" in dem_component_details:
        if dem_component_details["Value"] > 0:
            min_demand_from_charge = (
                dem_component_details["Min Demand Charge ($)"]
                / dem_component_details["Value"]
            )

    # Set up an empty array to hold peak values:
    average_peaks_all = np.empty((0, load_profile_selected_times.shape[1]), dtype=float)

    # Loop through each month present in the tariff component definition to find
    # peaks
    for m in dem_component_details["Month"]:
        arr = (
            load_profile_selected_times.loc[
                load_profile_selected_times.index.month == m, :
            ]
            .copy()
            .values
        )
        arr.sort(axis=0)
        arr = arr[::-1]

        # 2 * -> to change units from kWh to kW. Get the average of the peaks (if
        # the number of peaks is > 1)
        average_peaks_all = np.append(
            average_peaks_all, [2 * arr[:num_peaks, :].mean(axis=0)], axis=0
        )

    # If there is a minimum demand set in the tariff component, depending on the
    # type of minimum set, apply here:
    if min_demand_from_charge > 0:
        # If the minimum demand comes from min_demand_from_charge, apply it as
        # a clipping value
        average_peaks_all = np.clip(
            average_peaks_all, a_min=min_demand_from_charge, a_max=None
        )
    else:
        # Otherwise, if it's coming from min_demand (or is <= 0), any 'peak' value
        # less than the min_demand is set to zero.
        average_peaks_all[average_peaks_all < min_demand] = 0

    # Sum all average peaks form each month together and use this sum to calculate
    # the total demand charge for this bill.
    average_peaks_all_sum = average_peaks_all.sum(axis=0)
    results[tariff_category]["Avg_kW_Dem_" + demand_component] = (
        average_peaks_all_sum / len(dem_component_details["Month"])
    )
    results[tariff_category]["Demand_" + demand_component] = (
        average_peaks_all_sum * dem_component_details["Value"] * 365 / 12
    )  # the charges in demand charge should be in $/kW/day

    dem_charge = average_peaks_all_sum * dem_component_details["Value"] * 365 / 12

    return dem_charge


def calculate_demand_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculate the total demand charge for all `Demand` tariff components.

    This function acts as a wrapper for `calc_dem()`, passing each component of a
    `Demand` tariff individually to calculate the charge for that component only.
    It also calls `time_select()` to pass only relevant parts of the load profile
    to calculate the demand charge.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a `Demand` key with a nested dictionary where each key represents
            a demand period and each value is a dictionary with details specifying
            month, time and weekdays during which the charge applies, as well as
            the units ($/kW/day) and rate of the charge itself.
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }
        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The sum of demand charges calculated based on the sum of component
            charges.

    """
    demand_charge_total = 0.0
    for demand_component, demand_component_details in tariff_component[
        "Demand"
    ].items():
        if "Number of Peaks" not in demand_component_details:
            num_of_peaks = 1
        else:
            num_of_peaks = demand_component_details["Number of Peaks"]

        lp = load_profile.copy()
        lp_selected_times = time_select(lp, demand_component_details)

        demand_charge_total += calc_dem_(
            demand_component_details,
            num_of_peaks,
            lp_selected_times,
            tariff_category,
            demand_component,
            results,
        )

    return demand_charge_total


def calculate_off_peak_demand_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculate the total demand charge for all `Off-Peak Demand` tariff components.

    This function acts as a wrapper for `calc_dem()`, first compiling the load profile
    during all off-peak periods by repeatedly calling `time_select()` for each
    `Off-Peak Demand` component, then passing the resulting load profile to calculate
    the total off-peak demand charge.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include an `Off_Peak Demand` key with a nested dictionary where each
            key represents a demand period and each value is a dictionary with details
            specifying month, time and weekdays during which the charge applies, as
            well as the units ($/kW/day) and rate of the charge itself.
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }
        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The total sum of off-peak demand charges.

    """
    lp_off_peak_dem = pd.DataFrame()
    for opd_component, opd_component_details in tariff_component[
        "Off Peak Demand"
    ].items():
        lp = load_profile.copy()
        lp_selected_times = time_select(lp, opd_component_details)
        lp_off_peak_dem = pd.concat([lp_off_peak_dem, lp_selected_times], axis="rows")

    if "Number of Peaks" not in opd_component_details:
        num_of_peaks = 1
    else:
        num_of_peaks = opd_component_details["Number of Peaks"]

    demand_charge = calc_dem_(
        opd_component_details,
        num_of_peaks,
        lp_off_peak_dem,
        tariff_category,
        "Off_Peak",
        results,
    )

    return demand_charge


def calculate_12_month_demand_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculate the total rolling 12-month demand charge based on a load profile
    and tariff component details.

    This function finds the rolling 12-month peak for each month present in the
    given load profile, and uses these values to calculate a monthly demand charge.
    Where only 12 months of load profile data is supplied (default), each month
    the peak will be calulated from the start of the load data to the end of the
    month.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a `Demand - last 12 Months` key with a nested dictionary where each
            key represents a demand period and each value is a dictionary with details
            specifying month, time and weekdays during which the charge applies, as
            well as the units ($/kW/day) and rate of the charge itself.
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }
        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The sum of all rolling charges based on rolling monthly peaks and
            supplied tariff rates.

    """
    for demand_component, demand_component_details in tariff_component[
        "Demand - last 12 Months"
    ].items():
        lp = load_profile.copy()

        # get minimum demand specifications:
        min_demand = 0
        min_demand_from_charge = 0
        if "Min Demand (kW)" in demand_component_details:
            min_demand = demand_component_details["Min Demand (kW)"]
        if "Min Demand Charge ($)" in demand_component_details:
            if demand_component_details["Value"] > 0:
                min_demand_from_charge = (
                    demand_component_details["Min Demand Charge ($)"]
                    / demand_component_details["Value"]
                )

        # Get the overall start and end dates of the given load profile, and set
        # rolling_date to the start date.
        start_date = lp.index.min()
        end_date = lp.index.max()
        rolling_date = start_date
        total_charge = 0.0
        # While the rolling_date is before (or equal to) the end date, calculate
        # the rolling 12-month peak (during periods covered by this demand charge
        # component)
        while rolling_date <= end_date:
            # Move the rolling date to the start of next month:
            month_days = rolling_date.daysinmonth
            month = rolling_date.month
            rolling_date = rolling_date + pd.to_timedelta(month_days, "D")

            # Get the 12 months prior to rolling_date, including the current month:
            monthly_rolling_lp = lp[
                (lp.index < rolling_date)
                & (lp.index >= rolling_date + relativedelta(years=-1))
            ]

            monthly_rolling_lp_dem_times = time_select(
                monthly_rolling_lp, demand_component_details
            )

            # Get the maximum value and *2 to convert to demand assuming half
            # hourly time stamps
            max_rolling_monthly = monthly_rolling_lp_dem_times.max().max() * 2

            # If the minimum demand comes from min_demand_from_charge, apply it
            # as a clip (any value lower is set to min_demand_from_charge). Otherwise,
            # apply as a condition (any value lower is set to zero).
            if min_demand_from_charge > 0:
                max_rolling_monthly = (
                    max_rolling_monthly
                    if max_rolling_monthly > min_demand_from_charge
                    else min_demand_from_charge
                )
            else:
                max_rolling_monthly = (
                    max_rolling_monthly if max_rolling_monthly > min_demand else 0.0
                )

            # Calculate the charge for this month and sum the rolling total charge
            month_charge = (
                max_rolling_monthly * month_days * demand_component_details["Value"]
            )

            if month in demand_component_details["Month"]:
                total_charge = np.nansum([total_charge, month_charge])

        results[tariff_category]["12MonDemand_" + demand_component] = total_charge

    demand_charge_12_months = results[tariff_category][
        [col for col in results[tariff_category] if col.startswith("12MonDemand_")]
    ].sum(axis=1)

    return demand_charge_12_months


def calculate_13_month_demand_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculate the total rolling 13-month demand charge based on a load profile
    and tariff component details.

    This function finds the rolling 13-month peak for each month present in the
    given load profile, and uses these values to calculate a monthly demand charge.
    Where only 12 months of load profile data is supplied (default), each month
    the peak will be calulated from the start of the load data to the end of the
    month.

    Args:
        load_profile (pd.DataFrame): A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component (dict): A dictionary containing tariff details. It should
            include a `Demand - last 13 Months` key with a nested dictionary where each
            key represents a demand period and each value is a dictionary with details
            specifying month, time and weekdays during which the charge applies, as
            well as the units ($/kW/day) and rate of the charge itself.
        results (dict): A dict containing key 'LoadInfo' with a pd.DataFrame
            value that has column 'Annual_kWh' with a single entry at index
            'kWh' that holds the annual energy usage of the given load profile,
            and key <tariff_category> with a pd.DataFrame that stores tariff component
            results. Structured as follows:
            results = {
                'LoadInfo' : pd.DataFrame(
                    columns=['Annual_kWh'],
                    index=['kWh'],
                    data=[6758021.922]
                ),
                <tariff_category> : pd.DataFrame()
            }
        tariff_category (str): A string representing the tariff category, used
            to store the charges in the results dictionary.

    Returns:
        float: The sum of all rolling charges based on rolling monthly peaks and
            supplied tariff rates.

    """
    for demand_component, demand_component_details in tariff_component[
        "Demand - last 13 Months"
    ].items():
        lp = load_profile.copy()
        min_demand = 0
        min_demand_from_charge = 0
        if "Min Demand (kW)" in demand_component_details:
            min_demand = demand_component_details["Min Demand (kW)"]
        if "Min Demand Charge ($)" in demand_component_details:
            if demand_component_details["Value"] > 0:
                min_demand_from_charge = (
                    demand_component_details["Min Demand Charge ($)"]
                    / demand_component_details["Value"]
                )

        start_date = lp.index.min()
        end_date = lp.index.max()
        rolling_date = start_date
        total_charge = 0.0
        while rolling_date <= end_date:
            month_days = rolling_date.daysinmonth
            rolling_date = rolling_date + pd.to_timedelta(month_days, "D")
            thirteen_months_in_days = 365 + month_days

            # the timedelta here may need to be changed for leap years??
            monthly_rolling_lp = lp[
                (lp.index < rolling_date)
                & (
                    lp.index
                    >= rolling_date - pd.to_timedelta(thirteen_months_in_days, "D")
                )
            ]  # should be rolling 13 months - using days for the timedelta

            monthly_rolling_lp_dem_times = time_select(
                monthly_rolling_lp, demand_component_details
            )

            max_rolling_monthly = monthly_rolling_lp_dem_times.max().max() * 2

            # At the moment: min_dem is set like a clip. SOME of the LC tariffs have
            # this behaviour explicitly written: if demand is below minimum, user
            # is charged for the minimum (e.g. CitiPower, ). BUT this might not be the case for everyone.
            if min_demand_from_charge > 0:
                max_rolling_monthly = (
                    max_rolling_monthly
                    if max_rolling_monthly > min_demand_from_charge
                    else min_demand_from_charge
                )
            else:
                max_rolling_monthly = (
                    max_rolling_monthly if max_rolling_monthly > min_demand else 0.0
                )

            month_charge = (
                max_rolling_monthly * month_days * demand_component_details["Value"]
            )
            total_charge = np.nansum([total_charge, month_charge])

        results[tariff_category]["13MonDemand_" + demand_component] = total_charge

        demand_charge_13_months = results[tariff_category][
            [col for col in results[tariff_category] if col.startswith("13MonDemand_")]
        ].sum(axis=1)

    return demand_charge_13_months


def calculate_excess_demand_charge(
    load_profile: pd.DataFrame,
    tariff_component: dict,
    results: dict,
    tariff_category: str,
) -> float:
    """Calculates the total demand charge for an Excess Demand tariff.

    This function finds the demand peaks during specified `Excess Demand` periods and
    `Peak Demand` periods, and uses the positive difference between the two peaks
    to calculate an excess demand charge (where a positive difference exists). For
    example, a load profile with a `Peak Demand` max value of 20kW and an `Excess
    Demand` max value of 25kW will be charged for 5kW of excess demand.

    Args:
        load_profile: A DataFrame containing half-hourly timeseries
            data with a DateTime index. It should have at least one column named 'kWh'
            containing energy usage (load) values for the corresponding half-hour
            up to index.
        tariff_component: A dictionary containing tariff details. It should
            include an `Excess Demand` key with a nested dictionary where each
            key represents a demand period and each value is a dictionary with details
            specifying month, time and weekdays during which the charge applies, as
            well as the units ($/kW/day) and rate of the charge itself.
        results: dict, not used here, included to simplify control logic.
        tariff_category: str, not used here, included to simplify control logic.

    Returns:
        float: The sum of all rolling charges based on rolling monthly peaks and
            supplied tariff rates.

    """
    lp_excess_dem = pd.DataFrame()

    # Create a load profile containing all data during specified Excess Demand
    # periods:
    for exc_component, exc_component_details in tariff_component[
        "Excess Demand"
    ].items():
        lp = load_profile.copy()
        lp_selected_times = time_select(lp, exc_component_details)
        lp_excess_dem = pd.concat([lp_excess_dem, lp_selected_times], axis="rows")

    if "Number of Peaks" not in exc_component_details:
        num_of_peaks = 1
    else:
        num_of_peaks = exc_component_details["Number of Peaks"]

    # Now that the dataframe with 'Excess' demand (all non-Peak demand times)
    # has been created, we need to find the monthly peaks in these data and
    # compare against the monthly peaks during 'Peak' time.

    # To find 'Peak' time load profile, use the inverse of the 'Excess' load profile:
    non_excess_dem = load_profile[~load_profile.index.isin(lp_excess_dem.index)].copy()

    excess_demand_charge = 0.0
    for m in lp_excess_dem.index.month.unique():
        # Get 'Peak' period peaks:
        non_excess_month = non_excess_dem.loc[non_excess_dem.index.month == m, :].copy()
        num_days_in_month = non_excess_month.index[0].daysinmonth

        # *2 to go form kWh -> kW (half-hourly)
        non_excess_month = non_excess_month.values * 2
        non_excess_month.sort(axis=0)
        non_excess_month = non_excess_month[::-1]

        non_excess_month_peaks = non_excess_month[:num_of_peaks, :]

        # Get 'Excess' period peaks:
        excess_month = (
            lp_excess_dem.loc[lp_excess_dem.index.month == m, :].copy().values * 2
        )
        excess_month.sort(axis=0)
        excess_month = excess_month[::-1]
        excess_month_peaks = excess_month[:num_of_peaks, :]

        # Find the positive difference between excess - peak. Where peak > excess,
        # set difference to 0.0.
        difference = excess_month_peaks - non_excess_month_peaks
        difference[difference < 0] = 0.0

        excess_dem_charge = np.sum(
            difference * exc_component_details["Value"] * num_days_in_month
        )
        excess_demand_charge += excess_dem_charge

    return excess_demand_charge


def tariff_bill_calculator(load_profile: pd.DataFrame, tariff: dict) -> dict:
    """
    Calculate the billing charges for large commercial tariffs based on the load profile and
    tariff details.

    This function computes the energy bill for a large commercial load profile, including
    daily, fixed, flat rate, block, time-of-use (TOU), demand, and excess demand charges. It also
    handles retailer-specific tariff adjustments and calculates energy charges. The results are
    stored in a dictionary with detailed billing information for each tariff component.

    This function was originally written for the following tools and has been adapted
    for use in SunSPOT.
    - CEEM Bill_calculator github - https://github.com/UNSW-CEEM/Bill_Calculator
    - CEEM tariff tool github - https://github.com/UNSW-CEEM/TDA_Python

    Args:
        load_profile: DataFrame containing the load profile data for one year.
            It should have two columns: 'TS' (timestamp) and 'kWh' (kilowatt-hours).
        tariff: Dictionary containing tariff details, including tariff parameters and
            types of charges.
        network_load: optional, DataFrame containing network load data. This
            parameter is not used in this function but included for consistency with other tools.
        fit: optional, flag indicating whether to include the Feed-in Tariff (FiT) rebate
            in the calculations. Defaults to True.

    Returns:
        results: A dictionary containing billing results for each tariff component. The dictionary
            includes
                - `'LoadInfo'`: DataFrame with annual consumption information.
                - `'NUOS'`: DataFrame with charges calculated for each component present in the
                    chosen tariff.

    Notes:
        - The function uses a dictionary of functions to calculate different types of charges based
          on the tariff parameters.
        - If the tariff provider is a retailer, it adjusts the tariff parameters accordingly.
        - The function handles a variety of tariff charge types and calculates the total bill and
          energy charges for each tariff component.
        - The 'LoadInfo' DataFrame in the results dictionary provides annual kWh consumption data
          and, if applicable, the annual kWh exported.
        - The function assumes that the load profile data is provided in half-hourly intervals.

    Examples:
        >>> load_profile = pd.DataFrame({
        ... 'TS': pd.date_range(start='2023-01-01', periods=8760, freq='30T'),
        ... 'kWh': np.random.rand(8760)})
        >>> tariff = {
        ... "CustomerType": "Commercial",
        ... "Date_accessed": "2024-02",
        ... "Distributor": "Ausgrid",
        ... "Name": "Example Tariff",
        ... "Parameters": {"NUOS": {<tariff_details>}},
        ... "ProviderType": "Network",
        ... "State": "VIC",
        ... "Tariff ID": "ID01",
        ... "Type": "TOU",
        ... "Year": "2024"}

        >>> results = bill_calculator_large_commercial_tariffs(load_profile, tariff)
        >>> results['LoadInfo']
                Annual_kWh  Annual_kWh_exp
        kWh     6758021.92            -0.0

        >>> results['NUOS']
            Charge_Daily  ...         Bill    energy_charge
        kWh     68860.43  ... 1.469779e+06        202740.66

    """
    load_profile = load_profile[["TS", "kWh"]].copy()
    load_profile.set_index("TS", inplace=True)
    load_profile = load_profile.fillna(0)

    ## Set up "results" dictionary to store calculated consumption and tariff
    ## charges.
    results = {}

    # Calculate imports and exports
    temp_import = load_profile.values
    temp_export = temp_import.copy()
    temp_import[temp_import < 0] = 0
    temp_export[temp_export > 0] = 0

    lp_net_import = pd.DataFrame(
        temp_import, columns=load_profile.columns, index=load_profile.index
    )

    # Store annual consumption information in results dict, as a dataframe
    # under the key 'LoadInfo':
    results["LoadInfo"] = pd.DataFrame(
        index=[col for col in load_profile.columns],
        data=np.sum(lp_net_import.values, axis=0),
        columns=["Annual_kWh"],
    )

    # If it is retailer put retailer as a component to make it similar to network tariffs
    if tariff["ProviderType"] == "Retailer":
        tariff_temp = tariff.copy()
        del tariff_temp["Parameters"]
        tariff_temp["Parameters"] = {"Retailer": tariff["Parameters"]}
        tariff = tariff_temp.copy()

    func_dict = {
        "Daily": (calculate_daily_charge, "Charge_Daily"),
        "Fixed": (calculate_fixed_charge, "Charge_Fixed"),
        "FlatRate": (calculate_flatrate_charge, "Charge_FlatRate"),
        "BlockAnnual": (calculate_annual_block_charge, "Charge_BlockAnnual"),
        "BlockQuarterly": (calculate_quarterly_block_charge, "Charge_BlockQuarterly"),
        "BlockMonthly": (calculate_monthly_block_charge, "Charge_BlockMonthly"),
        "BlockDaily": (calculate_daily_block_charge, "Charge_BlockDaily"),
        "TOU": (calculate_time_of_use_charge, "Charge_TOU"),
        "Demand": (calculate_demand_charge, "Charge_Demand"),
        "Off Peak Demand": (calculate_off_peak_demand_charge, "Charge_Off_Peak_Demand"),
        "Demand - last 12 Months": (
            calculate_12_month_demand_charge,
            "Charge_12_Mon_Demand",
        ),
        "Demand - last 13 Months": (
            calculate_13_month_demand_charge,
            "Charge_13_Mon_Demand",
        ),
        "Excess Demand": (calculate_excess_demand_charge, "Charge_Excess_Demand"),
    }

    # Set up another entry to results dict to contain charge/bill results for
    # the network component (called "Retailer" for Large Comms for consistency)
    # with small business/residential.
    for component_type, component_details in tariff["Parameters"].items():
        results[component_type] = pd.DataFrame(index=results["LoadInfo"].index)
        results[component_type]["Charge_FiT_Rebate"] = 0

        # Loop through each charge component in the tariff (e.g. TOU, Demand)
        # and calculate the amount to be charged under this component
        for charge_type in component_details.keys():
            results[component_type][func_dict[charge_type][1]] = func_dict[charge_type][
                0
            ](lp_net_import, component_details, results, component_type)

    energy_comp_list = [
        "BlockAnnual",
        "BlockQuarterly",
        "BlockMonthly",
        "BlockDaily",
        "FlatRate",
        "TOU",
    ]
    for k, v in results.items():
        if k != "LoadInfo":
            results[k]["Bill"] = results[k][
                [col for col in results[k].columns if col.startswith("Charge")]
            ].sum(axis=1)
            results[k]["energy_charge"] = results[k][
                [
                    col
                    for col in results[k].columns
                    if (
                        col.startswith("Charge")
                        and col.endswith(tuple(energy_comp_list))
                    )
                ]
            ].sum(axis=1)

    return results


def add_other_charges_to_tariff(tariff: dict, other_charges: dict) -> dict:
    """
    Add user-supplied retail charges to network tariff structure.

    This function takes in a chosen network tariff and a set of other retail charges
    and replaces underlying network charge values with the new user-supplied
    rates in order to separately calculate network and retail bills.

    :param tariff: nested dict containing the chosen network tariff structure and
        values, formatted according to the correspoding tariff type structure. Example
        tariff structure templates can be found in templates/tariff_structure_template.json.
    :param other_charges: nested dict containing additional retail charge rates
        as supplied by the user. Default rates and units are charge-specific, and
        structured as follows:

        other_charges = {
            "Energy Charges": {
                "Peak Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.06
                },
                "Shoulder Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.06
                },
                "Off-Peak Rate": {
                    "Unit": "$/kWh",
                    "Value": 0.04
                },
                "Retailer Demand Charge": {
                    "Unit": "$/kVA/day",
                    "Value": 0.000
                }
            },
            "Metering Charges": {
                "Meter Provider/Data Agent Charges": {
                    "Unit": "$/Day",
                    "Value": 2.000
                },
                "Other Meter Charges": {
                    "Unit": "$/Day",
                    "Value": 0.000
                }
            },
            "Environmental Charges": {
                "LREC Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.008
                },
                "SREC Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.004
                },
                "State Environment Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.002
                }
            },
            "AEMO Market Charges": {
                "AEMO Participant Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.00036
                },
                "AEMO Ancillary Services Charge": {
                    "Unit": "$/kWh",
                    "Value": 0.00018
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


    :return: nested dict containing the other retail charge rates in the chosen
        network tariff structure.
    """
    if "TOU" in tariff["Parameters"]["NUOS"]:
        for tou_component_name in tariff["Parameters"]["NUOS"]["TOU"].keys():
            if "shoulder" in tou_component_name or "shoulder" in tou_component_name:
                tariff["Parameters"]["NUOS"]["TOU"][tou_component_name]["Value"] = (
                    other_charges["Energy Charges"]["Shoulder Rate"]["Value"]
                )
            elif (
                "Off" in tou_component_name
                or "off" in tou_component_name
                or "Non" in tou_component_name
                or "non" in tou_component_name
            ):
                tariff["Parameters"]["NUOS"]["TOU"][tou_component_name]["Value"] = (
                    other_charges["Energy Charges"]["Off-Peak Rate"]["Value"]
                )
            else:
                tariff["Parameters"]["NUOS"]["TOU"][tou_component_name]["Value"] = (
                    other_charges["Energy Charges"]["Peak Rate"]["Value"]
                )

    if "Demand" not in tariff["Parameters"]["NUOS"]:
        tariff["Parameters"]["NUOS"]["Demand"] = {}

    if len(tariff["Parameters"]["NUOS"]["Demand"]) > 1:
        for demand_component in tariff["Parameters"]["NUOS"]["Demand"].keys():
            if (
                "Peak" in demand_component and "Off" not in demand_component
            ) or "Non" not in demand_component:
                tariff["Parameters"]["NUOS"]["Demand"][demand_component]["Value"] = (
                    other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"]
                )
            else:
                tariff["Parameters"]["NUOS"]["Demand"][demand_component]["Value"] = 0.0
    elif len(tariff["Parameters"]["NUOS"]["Demand"]) == 1:
        demand_component = next(iter(tariff["Parameters"]["NUOS"]["Demand"]))
        tariff["Parameters"]["NUOS"]["Demand"][demand_component]["Value"] = (
            other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"]
        )
    else:
        tariff["Parameters"]["NUOS"]["Demand"]["Peak"] = {
            "Value": other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"],
            "Unit": "$/kW/Day",
            "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Weekday": True,
            "Weekend": True,
            "Based on Network Peak": False,
            "Number of Peaks": 1,
            "Demand Window Length": 1,
            "Min Demand (kW)": 0,
            "Min Demand Charge ($)": 0,
            "Day Average": False,
            "TimeIntervals": {"T1": ["00:00", "24:00"]},
        }

    if "Daily" not in tariff["Parameters"]["NUOS"]:
        tariff["Parameters"]["NUOS"]["Daily"] = {"Value": 0.0, "Unit": "$/day"}

    tariff["Parameters"]["NUOS"]["Daily"]["Value"] = other_charges["Metering Charges"][
        "Meter Provider/Data Agent Charges"
    ]["Value"]
    tariff["Parameters"]["NUOS"]["Daily"]["Value"] += other_charges["Metering Charges"][
        "Other Meter Charges"
    ]["Value"]

    if "FlatRate" not in tariff["Parameters"]["NUOS"]:
        tariff["Parameters"]["NUOS"]["FlatRate"] = {"Value": 0.0, "Unit": "$/kWh"}

    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] = other_charges[
        "Environmental Charges"
    ]["LREC Charge"]["Value"]
    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] += other_charges[
        "Environmental Charges"
    ]["SREC Charge"]["Value"]
    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] += other_charges[
        "Environmental Charges"
    ]["State Environment Charge"]["Value"]

    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] = other_charges[
        "AEMO Market Charges"
    ]["AEMO Participant Charge"]["Value"]
    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] += other_charges[
        "AEMO Market Charges"
    ]["AEMO Ancillary Services Charge"]["Value"]

    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] = other_charges[
        "Other Variable Charges"
    ]["Other Variable Charge 1"]["Value"]
    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] += other_charges[
        "Other Variable Charges"
    ]["Other Variable Charge 2"]["Value"]
    tariff["Parameters"]["NUOS"]["FlatRate"]["Value"] += other_charges[
        "Other Variable Charges"
    ]["Other Variable Charge 3"]["Value"]

    if "Fixed" not in tariff["Parameters"]["NUOS"]:
        tariff["Parameters"]["NUOS"]["Fixed"] = {"Value": 0.0, "Unit": "$/Bill"}

    tariff["Parameters"]["NUOS"]["Fixed"]["Value"] = other_charges[
        "Other Fixed Charges"
    ]["Other Fixed Charge 1"]["Value"]
    tariff["Parameters"]["NUOS"]["Fixed"]["Value"] += other_charges[
        "Other Fixed Charges"
    ]["Other Fixed Charge 2"]["Value"]
    tariff["Parameters"]["NUOS"]["Fixed"]["Value"] += other_charges[
        "Other Fixed Charges"
    ]["Total GST"]["Value"]

    return tariff


# ------------- Convert the large commerical network tariffs to a retail tariff structure -------------
def convert_network_tariff_to_retail_tariff(tariff: dict) -> dict:
    """
    Converts a network tariff with only the NUOS components to the structure that
    tariff_bill_calculator expects from a retail tariff.

    :param tariff: nested dict containing the chosen tariff, formatted according
        to the correspoding tariff type structure. Example
        tariff structure templates can be found in templates/tariff_structure_template.json.

    """
    tariff["Parameters"] = tariff["Parameters"]["NUOS"]
    tariff["ProviderType"] = "Retailer"
    return tariff

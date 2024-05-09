# Helper script to load in commercial data for GAN development/testing
# This script loads and returns all commercial data that meets criteria
# (>100000 kWh/year, correct length of data) combined with half hourly 
# data (average for the region) including temperature, wet bulb temp and 
# humidity. 

import pandas as pd
import numpy as np
import json
import calendar
import holidays
import math
from datetime import datetime

# Define type dict for importing station data:
dtypes = {
    'Site' : int,
    'Name' : str,
    'Lat' : float,
    'Lon': float,
    'Years' : float,
    '%' : int,
    'Obs' : float,
    'AWS' : str
}

# Helper function to read in data from json files:
def read_json_file(filename):
    f = open(f'{filename}.json', 'r')
    data = json.loads(f.read())
    f.close()
    return data


# Import csv data files for old (2014-15) data and 'future' (2018-22) data
def get_average_weather_data():
    future_bom_data = pd.read_csv(
        '/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/GAN_Testing/bom_data_qld58979564.csv', 
        usecols=[1, 8, 9, 11, 13, 17, 67], 
        parse_dates=['datetime'], 
        dayfirst=False
    )

    past_bom_data = pd.read_csv(
        '/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/GAN_Testing/bom_data_qld31694126.csv', 
        usecols=[1, 8, 9, 11, 13, 17, 67], 
        parse_dates=['datetime'], 
        dayfirst=False
    )

    print(past_bom_data.columns)

    # Import data about weather stations
    station_data = pd.read_csv(
        '/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/GAN_Testing/weather_stations.csv', 
        dtype=dtypes
    )

    station_data = station_data.loc[station_data['AWS'] == 'Y']
    station_data = station_data.set_index('Name')

    weather_station_ = read_json_file('/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/GAN_Testing/regions')

    # Create dictionary of station name : region data to map onto weather data later
    station_region_dict = {}
    for region, station_list in weather_station_.items():
        for station in station_list:
            if station.upper() in station_data.index:
                station_num = station_data.loc[station.upper(), 'Site']

                if type(station_num) == np.int64:
                    station_region_dict[station_num] = region
                else:
                    for site_number in list(station_data.loc[station.upper(), 'Site']):
                        station_region_dict[site_number] = region

    # Map regions to station data and filter for stations that don't exist in the dataset:
    filt_past_bom_data = past_bom_data.copy()
    filt_past_bom_data['region'] = ''
    filt_past_bom_data['region'] = filt_past_bom_data['station_number']\
        .map(station_region_dict)

    filt_past_bom_data = filt_past_bom_data[~filt_past_bom_data['region']\
                                            .isna()].drop(columns=['region'])
    filt_past_bom_data = filt_past_bom_data.set_index('datetime')
    unique_sites = filt_past_bom_data['station_number'].unique()

    # Group the data by station number (get dfs split up for each station):
    grouped_past_bom_data = filt_past_bom_data\
        .groupby(filt_past_bom_data['station_number'])

    grouped_dfs = {
        region : pd.DataFrame() for region in weather_station_.keys()
    }

    # For each unique weather station number, get the grouped df and concat to the
    # regional df
    for num in unique_sites:
        num_df = grouped_past_bom_data.get_group(num)
        num_df = num_df.drop(columns=['station_number'])
        rename_dict = {col : col + '_' + str(num) for col in num_df.columns}
        num_df = num_df.rename(columns=rename_dict)

        region = station_region_dict[num]

        temp_df = grouped_dfs[region].copy()
        temp_df = pd.concat([temp_df, num_df], axis='columns')

        grouped_dfs[region] = temp_df

    # For each region and weather attribute, calculate average half-hourly values
    average_weather_by_region = pd.DataFrame()

    for key, val in grouped_dfs.items():
        # Air temperature
        air_temp_data = val[[col for col in val.columns if 'air_temp' in col]]
        avg_air_temp = air_temp_data.mean(axis='columns')
        avg_air_temp.name = f'{key}_Temp'

        average_weather_by_region = pd.concat(
            [average_weather_by_region, avg_air_temp], axis='columns')
        
        # Wet bulb temperature
        wet_bulb_data = val[[col for col in val.columns if 'wet_bulb' in col]]
        avg_wet_bulb = wet_bulb_data.mean(axis='columns')
        avg_wet_bulb.name = f'{key}_Bulb'

        average_weather_by_region = pd.concat(
            [average_weather_by_region, avg_wet_bulb], axis='columns')

        # Relative humidity
        humidity_data = val[[col for col in val.columns if 'humidity' in col]]
        avg_humidity = humidity_data.mean(axis='columns')
        avg_humidity.name = f'{key}_Humidity'

        average_weather_by_region = pd.concat(
            [average_weather_by_region, avg_humidity], axis='columns')
        
        # Precipitation
        precipitation_data = val[[col for col in val.columns if 'precip' in col]]
        avg_precipitation = precipitation_data.mean(axis='columns')
        avg_precipitation.name = f'{key}_Rainfall'

        average_weather_by_region = pd.concat(
            [average_weather_by_region, avg_precipitation], axis='columns')
        

        # Pressure
        pressure_data = val[[col for col in val.columns if 'pressure' in col]]
        avg_pressure = pressure_data.mean(axis='columns')
        avg_pressure.name = f'{key}_Pressure'

        average_weather_by_region = pd.concat(
            [average_weather_by_region, avg_pressure], axis='columns')
    
    return average_weather_by_region


def get_load_data():
    commercials_info = pd.read_csv(
        '/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/Commercial loads for SunSPoT/c&i_profile_ids.csv'
    )

    # Filter commercials by annual usage and standardise labels:
    # This is just the info about each commercial - to find the region and name
    commercials_info = commercials_info[(commercials_info['CUSTOMER TYPE (kWh/year)']\
                                         .str.contains('>1000000') | \
                                            commercials_info['CUSTOMER TYPE (kWh/year)']\
                                                .str.contains('> 1000000'))]

    commercials_info['Type'] = commercials_info['DESCRIPTION'].str.split(' - ')\
        .str[0] + '_' + commercials_info['ID'].astype(str)
    commercials_info['Name'] = commercials_info['DESCRIPTION'].str.split(' - ')\
        .str[0].str.lower()
    commercials_info['Region'] = commercials_info['DESCRIPTION'].str.split(' QLD')\
        .str[0]
    commercials_info['Region'] = commercials_info['Region'].str.split(' - ')\
        .str[-1]
    commercials_info['REGION'] = commercials_info['RegionID'] + ' - ' \
        + commercials_info['Region']

    commercials_info = commercials_info[['ID', 'Type', 'Name', 'REGION', 'RegionID']]\
        .replace({'FN - North':'NQ - North', 'SW - South' : 'SE - South'})
    commercials_info = commercials_info.set_index('ID')

    profile_types = {
        (commercials_info.loc[i, 'Type'], commercials_info.loc[i, 'RegionID']) \
            : i for i in commercials_info.index
    }

    # Load in the commercial data that meets filtering criteria and store in 
    # a new df:
    parse_dates = ['CurrentDate']
    all_commercials = pd.DataFrame()
    for name_region, number in profile_types.items():
        name = name_region[0]
        region = name_region[1]

        data = pd.read_csv(
            f'/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/Commercial loads for SunSPoT/{number}.txt', 
            sep = ',', 
            dtype = dtypes, 
            parse_dates = parse_dates,
            dayfirst=True
        )

        if len(data) == 17520:
            data = data[['CurrentDate', 'KW']]\
                .rename(columns={
                    'CurrentDate':'DateTime', 
                    'KW':f'{name}_{region}'
                })\
                .set_index('DateTime')\
                .copy()

            if all_commercials.empty:
                all_commercials = data.copy()
            else:
                all_commercials = pd.merge(
                    all_commercials, 
                    data, 
                    how='outer', 
                    left_index=True, 
                    right_index=True
                )

    all_commercials = all_commercials.reset_index()

    # Add holidays/weekends column:
    holiday_dates = holidays.country_holidays('AU', subdiv='QLD')
    all_commercials['Holiday'] = False
    for i in range(len(all_commercials)):
        all_commercials.loc[i, 'Holiday'] = all_commercials.loc[i, 'DateTime'] in holiday_dates

    all_commercials['Weekend'] = np.where(
        (all_commercials['DateTime'].dt.dayofweek.isin([5, 6]) | all_commercials['Holiday'] == True),
        1, 0
    )

    all_commercials = all_commercials.drop('Holiday', axis=1)
    all_commercials = all_commercials.set_index('DateTime')

    return all_commercials


def combine_load_weather_data():
    commercial_load_and_weather_dict = {}

    weather_data = get_average_weather_data()
    load_data = get_load_data()

    for commercial in [col for col in load_data.columns if col != 'Weekend']:
        commercial_load = load_data[[commercial, 'Weekend']]
        region = commercial.split('_')[-1]
        # comm_name = commercial.split('_')[0] + '_' + commercial.split('_')[1]
        regional_weather = weather_data[[col for col in weather_data.columns if region in col]]

        combined_load_and_weather = pd.concat([commercial_load, regional_weather], axis=1, join='inner')

        commercial_load_and_weather_dict[commercial] = combined_load_and_weather
              
    return commercial_load_and_weather_dict
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import nemed

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
        regions:list[str],
        period:str='H'
        ) -> pd.DataFrame:
    
    
    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M')
    nemed_result = nemed.get_total_emissions(start_time = start_date_str,
                                             end_time = end_date_str, 
                                             cache = cache,
                                             filter_regions = regions,
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
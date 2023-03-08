import pandas as pd
import numpy as np
import nemed

# The following retrieval code is heavily based on examples given in the NEMED
# documentation, found here: https://nemed.readthedocs.io/en/latest/examples/total_emissions.html

def get_emissions_intensity(start, end, cache, regions, period):
    nemed_result = nemed.get_total_emissions(start_time = start,
                                             end_time = end, 
                                             cache = cache,
                                             filter_regions = regions,
                                             by = None,                         # don't aggregate using inbuilt NEMED functionality - can't do 30 min increments
                                             assume_energy_ramp=True,           # can set this to False for faster computation / less accuracy
                                             generation_sent_out=False          # currently NOT considering auxiliary load factors (from static tables)
                                             )
    
    # Create empty df to fill with columns corresponding to each region, containing the emissions intensity index:
    emissions_df = pd.DataFrame()
    emissions_df['DateTime'] = pd.to_datetime(nemed_result['TimeEnding'])
    for region in regions:
        emissions_df[region] = nemed_result[nemed_result['Region']==region]['Intensity_Index']

    # Make sure to resample for the given period:
    if period != None:
        emissions_df = emissions_df.set_index('DateTime').resample(period).mean()
        emissions_df = emissions_df.reset_index()

    return emissions_df


def firming_emissions_calc(residual_profiles):
    temp = residual_profiles.copy()
    # Avg. emissions intensity given as tCO2-e/MWh - need to /1000 as firming energy given in kWh
    temp['Emissions'] = (temp['Black']/1000) * temp['Average Emissions Intensity']
    firming_emissions = np.sum(temp['Emissions'])
    return firming_emissions
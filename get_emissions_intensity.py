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
                                             by = period,
                                             assume_energy_ramp=True,           # can set this to False for faster computation / less accuracy
                                             generation_sent_out=False          # currently NOT considering auxiliary load factors (from static tables)
                                             )
    
    # Create empty df to fill with columns corresponding to each region, containing the emissions intensity index:
    emissions_df = pd.DataFrame()
    emissions_df['DateTime'] = pd.to_datetime(nemed_result['TimeEnding'])
    for region in regions:
        emissions_df[region] = emissions_df[emissions_df['Region']==region]['Intensity_Index']

    return emissions_df
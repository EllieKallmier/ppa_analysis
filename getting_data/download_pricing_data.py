import pandas as pd
import numpy as np
from datetime import timedelta
from nemosis import static_table, dynamic_data_compiler
import plotly.express as px

# Set the dates you want pricing data for and the region. 
START_DATE = '2020/12/31 23:30:00'
END_DATE = '2022/09/30 23:30:00'
REGIONIDS = ['QLD1', 'NSW1']
SAMPLE_TIME = '30min'

raw_data_cache = 'getting_data/pricing_cache'
# dispatch_units = static_table(table_name='Generators and Scheduled Loads', 
#                               raw_data_location=raw_data_cache,
#                               update_static_file=False)     # change to 'True' to force an updated download

price_data = dynamic_data_compiler(start_time=START_DATE,
                                   end_time=END_DATE,
                                   table_name='TRADINGPRICE',
                                   raw_data_location=raw_data_cache)

price_data = price_data[price_data['REGIONID'].isin(REGIONIDS)]


price_data['ts'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
price_data = price_data.set_index('ts')
price_data = price_data.sort_values(by='ts')#.reset_index()

sorted_price_data = pd.DataFrame()
for region in REGIONIDS:
    region_data = price_data[price_data['REGIONID']==region]
    sorted_price_data[region] = region_data['RRP']

sorted_price_data = sorted_price_data.resample(SAMPLE_TIME).mean(numeric_only=True)


sorted_price_data.to_csv('pricing_data.csv')
import pandas as pd
import numpy as np
from datetime import timedelta
from nemosis import dynamic_data_compiler
import plotly.express as px

# Set dates that you want the data between
START_DATE = '2016/01/01 00:00:00'
END_DATE = '2022/12/31 23:30:00'

# Set up a cache or point this to your preferred cache:
raw_data_cache = 'getting_data/gen_data_cache'
scada_data = dynamic_data_compiler(start_time=START_DATE,
                                   end_time=END_DATE,
                                   table_name='DISPATCH_UNIT_SCADA',
                                   raw_data_location=raw_data_cache)

# This list can be updated depending on the generator(s) you want data from. 
# These generators are a mix of solar, wind and pumped hydro from QLD.
generators = [
    'BAKING1','BARCSF1','BARRON-1','BARRON-2','BLUEGSF1','CHILDSF1','CLARESF1','CLERMSF1',\
    'COLUMSF1','CSPVPS1','COOPGWF1','COOPGWF1','COOPGWF1','COOPGWF1','DDSF1','DAYDSF1','DAYDSF2',\
    'EDENVSF1','EMERASF1','GANGARR1','HAMISF1','HAUGHT11','HAYMSF1','HUGSF1','KABANWF1','KAREEYA1',\
    'KAREEYA2','KAREEYA3','KAREEYA4','KAREEYA5','KEPSF1','KEPWF1','KSP1','LILYSF1','LRSF1',\
    'MARYRSF1','MIDDLSF1','MEWF1','MEWF1','MOUSF1','OAKEY1SF','OAKEY2SF','RRSF1','RUGBYR1','SMCSF1',\
    'SMCSF1','SRSF1','VALDORA1','WARWSF1','WARWSF2','WDGPH1','WDGPH1','WHITSF1','WHILL1','PUMP1',\
    'PUMP2',"W/HOE#1","W/HOE#2",'WIVENSH','WOOLGSF1','YARANSF1'
]


for duid in generators:
    scada_data_gen = scada_data[scada_data['DUID'] == duid]
    scada_data_gen['ts'] = pd.to_datetime(scada_data_gen['SETTLEMENTDATE'])
    scada_data_gen = scada_data_gen.set_index('ts')
    scada_data_gen = scada_data_gen.sort_values(by='ts')
    scada_data_gen = scada_data_gen.resample('30min').mean()
    print(scada_data_gen.head())

    # Export data for this generator to csv file.
    scada_data_gen.to_csv(f'{duid}_generation_data.csv')
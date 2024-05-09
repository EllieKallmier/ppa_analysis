import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import timedelta, datetime
from nemosis import dynamic_data_compiler
# from nemseer import compile_data, download_raw_data, generate_runtimes

logging.getLogger("nemosis").setLevel(logging.WARNING)
# logging.getLogger("nemseer").setLevel(logging.ERROR)

# Wrapper function to import dispatch pricing data using NEMOSIS (credit: Nick Gorman).
def get_wholesale_price_data(
    start_date:pd.Timestamp, 
    end_date:pd.Timestamp, 
    cache:str, 
    regions:list[str],
    period:str='H'
) -> pd.DataFrame:
    
    start_date_str = datetime.strftime(start_date, '%Y/%m/%d %H:%M:%S')
    end_date_str = datetime.strftime(end_date, '%Y/%m/%d %H:%M:%S')

    price_data = dynamic_data_compiler(start_time=start_date_str,
                                   end_time=end_date_str,
                                   table_name='DISPATCHPRICE',
                                   raw_data_location=cache,
                                   select_columns=['REGIONID', 'SETTLEMENTDATE', 'RRP'],
                                   filter_cols=['REGIONID'],
                                   filter_values=(regions,)
                                   )
    
    price_data = price_data.reset_index()
    price_data['DateTime'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
    price_data = price_data.drop(columns=['SETTLEMENTDATE'])
    price_data = price_data.pivot_table(columns='REGIONID', values='RRP', index='DateTime')
    price_data = price_data.resample(period).mean(numeric_only=True)

    price_data = price_data.rename(columns={
        col : 'RRP: ' + col for col in price_data.columns
    })

    return price_data


# Wrapper function to import pre-dispatch pricing data using NEMSEER (credit: Abhijith Prakash)
def get_predispatch_prices(
    start_date:pd.Timestamp, 
    end_date:pd.Timestamp, 
    cache:str, 
    regions:list[str],
    period:str='H'
) -> pd.DataFrame:



    return
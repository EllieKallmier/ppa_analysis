import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

# Define options for different firming contracts: 

# 1. Wholesale exposure: fully risk exposed, no retail contract
# 2. Partial wholesale exposure (cap, swap or collar)
# 3. Retail contract

# There could be a mix of all three, but for the moment sticking to simpler structures


# Total wholesale exposure:
def total_wholesale_exposure(
        df:pd.DataFrame,
        regions:list[str]
) -> pd.DataFrame:

    for region in regions:
        df[f'Firming price: {region}'] = df[f'RRP: {region}'].copy()

    return df


# Partial wholesale exposure:
def part_wholesale_exposure(
        df:pd.DataFrame,
        regions:list[str],
        upper_bound:float,
        lower_bound:float
) -> pd.DataFrame:

    for region in regions:
        df[f'Firming price: {region}'] = df[f'RRP: {region}'].copy()\
            .clip(upper=upper_bound, lower=lower_bound)

    return df


# Retail tariff contract:
def retail_tariff_contract():
    # TODO: integrate tariff tool and CEEM tariff API for large commercial tariffs??

    return
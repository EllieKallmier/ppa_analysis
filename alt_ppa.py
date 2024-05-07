# Functions to calculate the PPA cost outcomes under each contract type:
def calculate_ppa(
        df:pd.DataFrame,
        load_region:str,
        strike_price:float,
        settlement_period:str,
        indexation:float|list[float]=0, # set a default for no indexation, for simplicity
        index_period:str='Y',   # and default period is yearly for slightly faster calcs just in case
        floor_price:float=-1000.0,  # default value is market floor
) -> pd.DataFrame:

    # add indexed strike_price column to df:
    indexation_calc = {'Y':yearly_indexation, 'Q':quarterly_indexation}
    strike_prices_indexed = indexation_calc[index_period](df, strike_price, indexation)
    
    df['Strike Price (Indexed)'] = strike_prices_indexed.copy()

    # settle around the contracted energy: strike_price - max(RRP, floor_price)
    df['Price'] = (df['Strike Price (Indexed)'] - np.maximum(df[f'RRP: {load_region}'], floor_price))
    df['PPA Settlement'] = df['Contracted Energy'] * (df['Strike Price (Indexed)'] - np.maximum(df[f'RRP: {load_region}'], floor_price))
    df['PPA Value'] = df['Strike Price (Indexed)'] * df['Contracted Energy']

    df_resamp = df.resample(settlement_period).sum(numeric_only=True).copy()

    return df_resamp


# Returns a TIMESERIES df
def calculate_excess_electricity(
        df:pd.DataFrame,
        load_region:str,
        settlement_period:str,
        excess_price:float|str='Wholesale'  # need to validate this input
) -> pd.DataFrame:
        # then selling any excess energy (contracted, but not used by the buyer) - keeping LGCs associated if bundled!!!
    # excess price is determined as either wholesale prices or a fixed price.
    if type(excess_price) == str and excess_price != 'Wholesale':
        raise ValueError('excess_price should be one of "Wholesale" or a float representing the fixed on-sell price.')

    if excess_price == 'Wholesale':
        df['Excess Price'] = df[f'RRP: {load_region}'].copy()
    else:
        df['Excess Price'] = excess_price

    df['Excess Energy'] = (df['Contracted Energy'] - df['Load']).clip(lower=0.0)
    df['Excess Energy Revenue'] = df['Excess Energy'] * df['Excess Price']

    df_resamp = df.resample(settlement_period).sum(numeric_only=True).copy()

    return df_resamp


# Returns a RESAMPLED DF to the settlement period: not interval timeseries data
def calculate_shortfall(
        df:pd.DataFrame,
        settlement_period:str,
        contract_type:str,
        shortfall_penalty:float,       # Penalty to be paid for shortfall energy - "shortfall damages"
        guaranteed_percent:float,     # A percentage of load in each settlement period to contract.
) -> pd.DataFrame:
    
    allowed_periods = {'Y', 'Q', 'M'}
    if settlement_period not in allowed_periods:
        raise ValueError(f'settlement_period should be one of {allowed_periods}.')

    # resample to each period
    # find the difference between contracted amount and delivered amount in each period
    df_resamp = df.resample(settlement_period).sum(numeric_only=True).copy()

    if contract_type in ['Pay as Produced', 'Pay as Consumed']:
        df_resamp['Guaranteed Energy'] = df_resamp['Load'] * guaranteed_percent/100
        df_resamp['Shortfall'] = (df_resamp['Guaranteed Energy'] - df_resamp['Contracted Energy']).clip(lower=0.0)
        df_resamp['Shortfall'] *= shortfall_penalty

    elif contract_type in ['Baseload', 'Shaped']:
        df['Shortfall'] = (df['Contracted Energy'] - df['Hybrid']).clip(lower=0.0)
        df['Shortfall'] *= shortfall_penalty         # IF the seller contracts other "Replacement Energy" - set shortfall penalty to zero.
        df_resamp = df.resample(settlement_period).sum(numeric_only=True).copy()
        
    else:
        # 24/7 PPA shortfall is based on the match % (CFE score) - if actual match < contracted % on average in each period, penalty applies to the missing %
        df_resamp_247 = df.copy()
        df_resamp_247['Match %'] = 0
        df_resamp_247['Match %'] = np.where(df_resamp_247['Load'] == 0, 100, np.minimum(df_resamp_247['Contracted Energy'] / df_resamp_247['Load'] * 100, 100))
        df_resamp_247 = df_resamp_247.resample(settlement_period).mean(numeric_only=True)
        df_resamp_247['Shortfall %'] = (guaranteed_percent - df_resamp_247['Match %']).clip(lower=0.0)

        df_resamp = pd.concat([df_resamp, df_resamp_247['Shortfall %']], axis='columns')
        df_resamp['Shortfall'] = df_resamp['Load'] * df_resamp['Shortfall %']/100 * shortfall_penalty

    return df_resamp


# Returns RESAMPLED df to settlement period.
def calculate_lgcs(
    df:pd.DataFrame,
    settlement_period:str,
    lgc_buy_price:float,
    lgc_sell_price:float,
    guaranteed_percent:float
) -> pd.DataFrame:

    allowed_periods = {'Y', 'Q', 'M'}
    if settlement_period not in allowed_periods:
        raise ValueError(f'settlement_period should be one of {allowed_periods}.')

    df_to_check = df[['Load', 'Contracted Energy']].copy()

    # Find the short and long positions for each settlement period - this informs
    # the need for penalties, purchases or sales.
    # Note: checking against the contract
    df_to_check = df_to_check.resample(settlement_period).sum(numeric_only=True)
    df_to_check['Volume Difference'] = df_to_check['Load']*(guaranteed_percent/100) - df_to_check['Contracted Energy']
    
    # Volume difference tells how much under/over the amount of contracted load the generation in this period is
    # The handling of these values depends on the PPA type!
    df_to_check['LGC Undersupply'] = np.where(df_to_check['Volume Difference'] > 0, df_to_check['Volume Difference'], 0.0)
    df_to_check['LGC Oversupply'] = np.where(df_to_check['Volume Difference'] < 0, df_to_check['Volume Difference'], 0.0)

    df_to_check['LGC Undersupply'] *= lgc_buy_price
    df_to_check['LGC Oversupply'] *= lgc_sell_price

    return df_to_check


def calculate_bill(
        df:pd.DataFrame,
        settlement_period:str,
        contract_type:str,
        load_region:str,
        strike_price:float,
        lgc_buy_price:float=0.0,                # Default values for all shortfall and LGC sale/purchase is $0 (so default is no penalties, no balancing).
        lgc_sell_price:float=0.0,
        shortfall_penalty:float=0.0,
        guaranteed_percent:float=100.0,
        excess_price:float|str='Wholesale',
        indexation:float|list[float]=0,         # set a default for no indexation, for simplicity
        index_period:str='Y',                   # and default period is yearly for slightly faster calcs just in case
        floor_price:float=-1000.0,              # default value is market floor
) -> pd.DataFrame:
    # put everything together here:
    # 1. PPA settlement costs
    # 2. Firming costs 
    # 3. Revenue from on-sold excess RE
    # 4. Purchase of extra LGCs
    # 5. Sale of excess LGCs
    # 6. Any shortfall penalty payments for LGCs or generation

    results = pd.DataFrame()

    # 1. PPA settlement costs: 
    ppa_costs = calculate_ppa(df, load_region, strike_price, settlement_period, indexation, index_period, floor_price)
    results['PPA Value'] = ppa_costs['PPA Value'].copy()
    results['PPA Settlement'] = ppa_costs['PPA Settlement'].copy()

    # 2. Firming costs: TODO: call FIRMING_FUNCTION here!!!
    firming_costs = df.copy()
    firming_costs['Unmatched Energy'] = (firming_costs['Load'] - firming_costs['Contracted Energy']).clip(lower=0.0)
    firming_costs['Firming Costs'] = firming_costs['Unmatched Energy'] * firming_costs[f'Firming price: {load_region}']
    firming_costs = firming_costs.resample(settlement_period).sum(numeric_only=True)

    results['Firming Costs'] = firming_costs['Firming Costs'].copy()

    # 3. Revenue from on-sold excess RE:
    excess_val = calculate_excess_electricity(df, load_region, settlement_period, excess_price)
    results['Revenue from on-sold RE'] = -1 * excess_val['Excess Energy Revenue'].copy()

    # 4. and 5. Revenue/Cost of LGCs
    lgc_balance = calculate_lgcs(df, settlement_period, lgc_buy_price, lgc_sell_price, guaranteed_percent)
    results['Revenue from excess LGCs'] = lgc_balance['LGC Oversupply'].copy()
    results['Cost of shortfall LGCs'] = lgc_balance['LGC Undersupply'].copy()

    # 6. Shortfall payments for energy - applied based on contract type.
    shortfall_payment_received = calculate_shortfall(df, settlement_period, contract_type, shortfall_penalty, guaranteed_percent)
    results['Shortfall Payments Received'] = -1 * shortfall_payment_received['Shortfall']

    results['Total'] = results.sum(axis='columns')

    return results

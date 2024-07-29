"""
Module containing functions for calculating the components of bills associated with PPA contracts. The functionality is
primarily intend to be used through the function calculate_bill, although the other functions are also public if a user
wishes to utilise them.

Examples:

Generally the data for calculating a PPA bill would be continuous data covering the period of a year, but to
demonstrate the use of monthly settlement period we will use data with two interval from January and February.

>>> volume_and_price = pd.DataFrame({
... 'datetime': ['2023/01/01 00:30:00', '2023/01/01 01:00:00', '2023/02/01 00:30:00', '2023/02/01 01:00:00'],
... 'Load': [100.0, 100.0, 100.0, 100.0],
... 'Contracted Energy': [100.0, 100.0, 80.0, 80.0],
... 'RRP': [50.0, 50.0, 50.0, 50.0],
... 'Firming price': [80.0, 80.0, 80.0, 80.0]
... })
>>> volume_and_price['datetime'] = pd.to_datetime(volume_and_price['datetime'])
>>> volume_and_price = volume_and_price.set_index(keys='datetime', drop=True)

>>> calculate_bill(
... volume_and_price=volume_and_price,
... settlement_period='M',
... contract_type='Pay as Produced',
... strike_price=75.0,
... lgc_buy_price=10.0,
... lgc_sell_price=10.0,
... shortfall_penalty=50.0,
... guaranteed_percent=90.0,
... excess_price='Wholesale',
... indexation=1.0,
... index_period='Y',
... floor_price= -1000.0)
            PPA Value  PPA Settlement  ...  Shortfall Payments Received    Total
datetime                               ...
2023-01-31    15150.0          5150.0  ...                         -0.0  20100.0
2023-02-28    12120.0          4120.0  ...                      -1000.0  18640.0
<BLANKLINE>
[2 rows x 8 columns]
"""
import pandas as pd
import numpy as np
import ppa_analysis.tariffs as tariffs
from ppa_analysis import helper_functions

def yearly_indexation(
        df: pd.DataFrame,
        strike_price: float,
        indexation: float | list[float]
) -> pd.DataFrame:
    """
    Helper function to calculate yearly indexation.

    The function takes a dataframe with an index of type datetime and returns the same dataframe with an additional
    column named 'Strike Price (Indexed)'. For each year after the initial year in the index, the strike price is
    increased by the specified indexation rate. If the indexation rate is provided as a float then the same rate is
    used for all years. If a list is then each year uses the next indexation rate in the list and if there are more
    years than rates in the list then last rate is reused.

    :param df: with datetime index
    :param strike_price: in $/MW/h
    :param indexation: as percentage i.e. 5 is an indexation rate of 5 %
    :return: The input dataframe with an additional column named 'Strike Price (Indexed)'
    """

    years = df.index.year.unique()

    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if type(indexation) != list:
        indexation = [indexation] * len(years)

    while len(indexation) < len(years):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, year in enumerate(years):
        spi_map[year] = strike_price
        strike_price += strike_price * indexation[i] / 100

    spi_map[year] = strike_price

    df_with_strike_price = df.copy()

    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price.index.year
    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price['Strike Price (Indexed)'].map(spi_map)

    return df_with_strike_price['Strike Price (Indexed)']


def quarterly_indexation(
        df: pd.DataFrame,
        strike_price: float,
        indexation: float | list[float]
) -> pd.DataFrame:
    """
    Helper function to calculate quarterly indexation.

    The function takes a dataframe with an index of type datetime and returns the same dataframe with an additional
    column named 'Strike Price (Indexed)'. For each quarter after the initial quarter in the index, the strike price is
    increased by the specified indexation rate. If the indexation rate is provided as a float then the same rate is
    used for all quarters. If a list is given then each quarter uses the next indexation rate in the list and if there
    are more quarters than rates in the list then last rate is reused.

    :param df: with datetime index
    :param strike_price: in $/MWh
    :param indexation: as percentage i.e. 5 is an indexation rate of 5 %
    :return: The input dataframe with an additional column named 'Strike Price (Indexed)'
    """

    years = df.index.year.unique()

    quarters = [(year, quarter) for year in years for quarter in range(1, 5)]

    # If the value given for indexation is just a float, or the list isn't as long
    # as the number of periods, keep adding the last element of the list until
    # the length is correct.
    if type(indexation) != list:
        indexation = [indexation] * len(quarters)

    while len(indexation) < len(quarters):
        indexation.append(indexation[-1])

    spi_map = {}
    for i, quarter in enumerate(quarters):
        spi_map[quarter] = strike_price
        strike_price += strike_price * indexation[i] / 100

    spi_map[quarter] = strike_price

    df_with_strike_price = df.copy()

    tuples = list(zip(df_with_strike_price.index.year.values, df_with_strike_price.index.quarter.values))

    df_with_strike_price['Strike Price (Indexed)'] = tuples
    df_with_strike_price['Strike Price (Indexed)'] = df_with_strike_price['Strike Price (Indexed)'].map(spi_map)

    return df_with_strike_price['Strike Price (Indexed)']


def calculate_ppa(
        price_and_load: pd.DataFrame,
        strike_price: float,
        settlement_period: str,
        indexation: float | list[float] = 0,  # set a default for no indexation, for simplicity
        index_period: str = 'Y',  # and default period is yearly for slightly faster calcs just in case
        floor_price: float = -1000.0,  # default value is market floor
) -> pd.DataFrame:
    """
    Calculates the cost associated with the settlement of the PPA contract for difference.

    Before settlement is calculated indexation is applied to the strike price, and the floor price is applied to the
    wholesale spot price(i.e. where the wholesale price is lower than the floor price the wholesale price is set to the
    floor price).

    :param price_and_load: Dataframe with datetime index, a column named 'RRP' specifying the wholesale spot price in
       the load region, and a column specifying the contracted energy name 'Contracted Energy'.
    :param strike_price: The strike price of the contract in $/MWh
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yeary, 'Q'
        for quarterly and 'M' for monthly.
    :param indexation: as percentage i.e. 5 is an indexation rate of 5 %
    :param index_period: How frequently to index the strike price as a st. 'Y' for yearly or 'Q' for quarterly.
    :param floor_price: Minimum wholesale price to use for settlement calculation i.e. for each interval the minimum of
        the floor_price and the wholesale price is taken, and then the resulting price is used to calculate settlement.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and the column 'PPA Settlement' specifying the cost of the PPA, the total energy traded
        through the contract through each period is provided in the column 'Contracted Energy' and the value of
        contracted energy at the indexed PPA strike price is provided in the column 'PPA Value'. The column labelled
        'PPA Final Cost' contains the cost to the buyer after settlement of the contracted energy.
    """

    # add indexed strike_price column to df:
    indexation_calc = {'Y': yearly_indexation, 'Q': quarterly_indexation}
    strike_prices_indexed = indexation_calc[index_period](price_and_load, strike_price, indexation)

    price_and_load['Strike Price (Indexed)'] = strike_prices_indexed.copy()

    # settle around the contracted energy: strike_price - max(RRP, floor_price)
    price_and_load['Price'] = (price_and_load['Strike Price (Indexed)'] - np.maximum(price_and_load['RRP'], floor_price))
    price_and_load['Wholesale Cost'] = price_and_load['Contracted Energy'] * price_and_load['RRP']
    price_and_load['PPA Settlement'] = price_and_load['Contracted Energy'] * (
            price_and_load['Strike Price (Indexed)'] - np.maximum(price_and_load['RRP'], floor_price))

    price_and_load['PPA Value'] = price_and_load['Strike Price (Indexed)'] * price_and_load['Contracted Energy']
    price_and_load['PPA Final Cost'] = price_and_load[['Wholesale Cost', 'PPA Settlement']].sum(axis=1)

    df_resamp = price_and_load.resample(settlement_period).sum(numeric_only=True).copy()

    return df_resamp


def calculate_tariff_bill(
        load_and_gen_data:pd.DataFrame,
        settlement_period:str,
        tariff_details:dict,
        bill_type:str
) -> pd.DataFrame:
    """ 
    Calculates the costs associated with energy use under either Network or Retail tariffs.

    The network bill is calculated for all PPA and firming types, and the retail bill is only 
    applied to energy not provided through the PPA if the firming type chosen is 'Retail'.

    :param load_and_gen_data: Dataframe with datetime index, a column specifying 
        the load (MWh) named 'Load' and, if bill_type == 'Retail' also a column specifying the load in intervals where the load is higher than the contracted energy (MWh) named 'Unmatched Energy'.
    :param settlement_period: str, The settlement period as a str in the pandas 
        period alias format e.g. 'Y' for yearly, 'Q' for quarterly and 'M' for monthly.
    :param tariff_details: dict containing two tariff structures labelled by the 
        keys 'Network' and 'Retail'. The tariff structures are nested dictionaries containing tariff components defined by the user selected network tariff.
    :param bill_type: str, one of 'Network' or 'Retail'. Indicates which tariff 
        to apply in the bill calculation and which energy trace to apply the calculation to, as above.
    :return: Results are returned in a dataframe on settlement period basis with 
        the index specifying the end of the settlement period and the column '{bill_type} Bill ($)' specifying the cost of the energy defined by bill_type.
    """
    resampled_load_index = load_and_gen_data.copy().resample(settlement_period).sum(numeric_only=True)
    resampled_load_index = resampled_load_index.index

    # Load in all the tariffs and find the selected tariff (network charges):
    tariff_bill_results = pd.DataFrame(index=resampled_load_index, columns=[f'{bill_type} Bill ($)'])

    if bill_type == 'Retail':
        col_name_to_calc = 'Unmatched Energy'
    else:
        col_name_to_calc = 'Load'
    
    selected_tariff = tariff_details[bill_type]

    # Separate out the firming energy as a new column (to apply retail charges onto):
    remainder = load_and_gen_data.copy()
    for end_date in resampled_load_index:
        load_chunk, remainder = helper_functions.get_load_data_chunk(remainder, end_date)
        load_chunk[col_name_to_calc] *= 1000  # convert the load data to kWh from MWh
        load_chunk = load_chunk.reset_index()

        # network charges applied to all load:
        network_load = load_chunk[['DateTime',col_name_to_calc]].copy()\
            .rename(columns={'DateTime':'TS', col_name_to_calc:'kWh'})
        network_bill = tariffs.tariff_bill_calculator(network_load, selected_tariff)
        tariff_bill_results.loc[end_date, f'{bill_type} Bill ($)'] = float(network_bill['Retailer']['Bill'][0])

    return tariff_bill_results

def calculate_firming(
        volume_and_price: pd.DataFrame,
        firming_type:str,
        tariff_details:dict,
        settlement_period: str
) -> pd.DataFrame:
    """
    Calculates the cost associated with buying energy not provided through the PPA.

    Firming energy required is calculated on an interval by interval basis as the difference between the 'Load' and the
    'Contracted Energy' multiplied by the 'Firming Costs'. Note firming costs are only applied in intervals where the
    load is greater than the contracted energy.

    :param volume_and_price: Dataframe with datetime index, a column specifying the load  (MWh) named 'Load' and
        a column specifying the contracted energy (MWh) name 'Contracted Energy', and a column specifying the firming
        price ($/MWh) (name formatted like 'Firming price: NSW1').
    :param firming_type: str, one of 'Wholesale exposed', 'Partially wholesale exposed', 'Retail'.
        Defines the calculation to be performed on the firming energy (load above the contracted energy).
    :param tariff_details: dict containing two tariff structures labelled by the 
        keys 'Network' and 'Retail'. The tariff structures are nested dictionaries containing tariff components defined by the user selected network tariff.
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yearly, 'Q'
        for quarterly and 'M' for monthly.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and the column 'Firming Costs' specifying the cost of the firming energy, and a column
        called 'Unmatched Energy' specifying the volume of firming energy procured.
    """

    firming_costs = volume_and_price.copy()
    firming_costs['Unmatched Energy'] = (firming_costs['Load'] - firming_costs['Contracted Energy']).clip(lower=0.0)

    if firming_type == 'Retail':
        firming_costs = calculate_tariff_bill(firming_costs, settlement_period, tariff_details, 'Retail')
        firming_costs = firming_costs.rename(columns={f'Retail Bill ($)':'Firming Costs'})
    else:
        firming_costs['Firming Costs'] = firming_costs['Unmatched Energy'] * firming_costs[f'Firming price']

        firming_costs = firming_costs.resample(settlement_period).sum(numeric_only=True)

    return firming_costs


def calculate_excess_electricity(
        load_contracted_volume: pd.DataFrame,
        settlement_period: str,
        excess_price: float | str = 'Wholesale'  # need to validate this input
) -> pd.DataFrame:
    """
    Calculates the contracted energy in excess of the load and the value of this energy.

    The excess energy is calculated on an interval by interval basis and then summed across each settlement period.

    :param load_contracted_volume: Dataframe with datetime index, a column specifying the load  (MWh) named 'Load' and
    a column specifying the contracted energy (MWh) name 'Contracted Energy', if the excess_price is calculated using
    the wholesale spot price (by specifying excess_price='Wholesale') then a column named 'RRP' specifying the wholesale
    spot price ($/MWh) is also required.
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yeary, 'Q'
        for quarterly and 'M' for monthly.
    :param excess_price: a float specifying the price for excess energy in $/MWh or string ('Wholesale') specifying
        that the wholesale spot price should be used to calculate the value of the excess energy.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and additional columns 'Excess Energy' (MWh), 'Excess Energy Revenue' ($). All columns
        contain values summed across the settlement period.
    """

    # then selling any excess energy (contracted, but not used by the buyer) - keeping LGCs associated if bundled!!!
    # excess price is determined as either wholesale prices or a fixed price.
    if type(excess_price) == str and excess_price != 'Wholesale':
        raise ValueError('excess_price should be one of "Wholesale" or a float representing the fixed on-sell price.')

    if excess_price == 'Wholesale':
        load_contracted_volume['Excess Price'] = load_contracted_volume['RRP'].copy()
    else:
        load_contracted_volume['Excess Price'] = excess_price

    load_contracted_volume['Excess Energy'] = (load_contracted_volume['Contracted Energy'] -
                                               load_contracted_volume['Load']).clip(lower=0.0)
    load_contracted_volume['Excess Energy Revenue'] = (load_contracted_volume['Excess Energy'] *
                                                       load_contracted_volume['Excess Price'])

    resampled = load_contracted_volume.resample(settlement_period).sum(numeric_only=True).copy()

    return resampled


def calculate_shortfall(
        volume: pd.DataFrame,
        settlement_period: str,
        contract_type: str,
        shortfall_penalty: float,
        guaranteed_percent: float,
) -> pd.DataFrame:
    """
    Calculates the total penalty associated with an energy shortfall on a settlement period basis.

    For 'Pay as Produced' and 'Pay as Consumed' contracts, the shortfall is calculated as the difference between the
    guaranteed percentage of the load and the contracted energy. For 'Baseload' and 'Shaped', the shortfall is
    calculated as difference between the 'Hybrid' (combined renewable energy generator profiles) and
    the 'Contracted Energy'. For '24/7' contracts the percentage of load covered by the contracted energy is calculated
    on an interval by interval basis (matching percentage), and then the mean of the percentage is taken on a settlement
    period basis. A shortfall percentage is then calculated as the difference between the guaranteed_percentage and the
    average monthly matching percentage, with the shortfall volume then calculated as the shortfall percentage
    multiplied by the settlement period load. Shortfalls cannot be negative for a settlement period for any contract
    types.

    :param volume: Dataframe with datetime index, the columns 'Load' and 'Contracted Energy', if a contract_type of
        'Baseload' or 'Shaped' is specified then a column 'Hybrid' is also required.
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yeary, 'Q'
        for quarterly and 'M' for monthly.
    :param contract_type: 
    :param shortfall_penalty: a float specifying the penalty for a shortfall in $/MWh
    :param guaranteed_percent: a float specifying the percentage of load in each settlement period to guaranteed to be
        met by the contract. Should be a number between 0 - 100.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and an additional column 'Shortfall' specifying the total shortfall penalty in $.
    """


    allowed_periods = {'Y', 'Q', 'M'}
    if settlement_period not in allowed_periods:
        raise ValueError(f'settlement_period should be one of {allowed_periods}.')

    # resample to each period
    # find the difference between contracted amount and delivered amount in each period
    df_resamp = volume.resample(settlement_period).sum(numeric_only=True).copy()

    if contract_type in ['Pay as Produced', 'Pay as Consumed']:
        df_resamp['Guaranteed Energy'] = df_resamp['Load'] * guaranteed_percent / 100
        df_resamp['Shortfall'] = (df_resamp['Guaranteed Energy'] - df_resamp['Contracted Energy']).clip(lower=0.0)
        df_resamp['Shortfall'] *= shortfall_penalty

    elif contract_type in ['Baseload', 'Shaped']:
        volume['Shortfall'] = (volume['Contracted Energy'] - volume['Hybrid']).clip(lower=0.0)
        volume['Shortfall'] *= shortfall_penalty  # IF the seller contracts other "Replacement Energy" - set shortfall penalty to zero.
        df_resamp = volume.resample(settlement_period).sum(numeric_only=True).copy()


    else:
        # 24/7 PPA shortfall is based on the match % (CFE score) - if actual match < contracted % on average in each
        # period, penalty applies to the missing %
        df_resamp_247 = volume.copy()
        df_resamp_247['Match %'] = 0
        df_resamp_247['Match %'] = np.where(df_resamp_247['Load'] == 0, 100,
                                            np.minimum(df_resamp_247['Contracted Energy'] / df_resamp_247['Load'] * 100,
                                                       100))
        df_resamp_247 = df_resamp_247.resample(settlement_period).mean(numeric_only=True)
        df_resamp_247['Shortfall %'] = (guaranteed_percent - df_resamp_247['Match %']).clip(lower=0.0)

        df_resamp = pd.concat([df_resamp, df_resamp_247['Shortfall %']], axis='columns')
        df_resamp['Shortfall'] = df_resamp['Load'] * df_resamp['Shortfall %'] / 100 * shortfall_penalty

    return df_resamp


def calculate_lgcs(
        volume: pd.DataFrame,
        settlement_period: str,
        lgc_buy_price: float,
        lgc_sell_price: float,
        guaranteed_percent: float
) -> pd.DataFrame:
    """
    Calculates the cost of buying LGCs if they are under supplied and the revenue from selling if they are oversupplied.

    The difference between the expected LGCs based on the guaranteed percent of load to be met by the contract and the
    actual volume supplied ('Contracted Energy') is calculated on a settlement period basis. Where the volume exceeds
    the guaranteed volume an excess of LGCs is produced and the oversupply revenue is calculated, where the volume is
    less than the guaranteed volume there is a deficit of LGCs and the cost of buying LGCs to meet the deficit is
    calculated.

    :param volume: Dataframe with datetime index, the columns 'Load' (MWh) and 'Contracted Energy' (MWh).
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yeary, 'Q'
        for quarterly and 'M' for monthly.
    :param lgc_buy_price: a float specifying the price ($/MWh) of buying LCGs to meet a deficit.
    :param lgc_sell_price: a float specifying the price ($/MWh) at which excess LCGs can be sold.
    :param guaranteed_percent: a float specifying the percentage of load in each settlement period to guaranteed to be
        met by the contract. Should be a number between 0 - 100.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and an additional column 'LGC Oversupply' specifying the revenue from selling excess
        LGCs and a column 'LGC Undersupply' specifying the cost of buying LGCs to meet a deficit.
    """

    allowed_periods = {'Y', 'Q', 'M'}
    if settlement_period not in allowed_periods:
        raise ValueError(f'settlement_period should be one of {allowed_periods}.')

    df_to_check = volume[['Load', 'Contracted Energy']].copy()

    # Find the short and long positions for each settlement period - this informs
    # the need for penalties, purchases or sales.
    # Note: checking against the contract
    df_to_check = df_to_check.resample(settlement_period).sum(numeric_only=True)
    df_to_check['Volume Difference'] = (df_to_check['Load'] * (guaranteed_percent / 100) -
                                        df_to_check['Contracted Energy'])

    # Volume difference tells how much under/over the amount of contracted load the generation in this period is
    # The handling of these values depends on the PPA type!
    df_to_check['LGC Undersupply'] = np.where(df_to_check['Volume Difference'] > 0, df_to_check['Volume Difference'],0.0)
    df_to_check['LGC Oversupply'] = np.where(df_to_check['Volume Difference'] < 0, df_to_check['Volume Difference'],0.0)

    df_to_check['LGC Undersupply'] *= lgc_buy_price
    df_to_check['LGC Oversupply'] *= lgc_sell_price

    return df_to_check


# TODO: add docstring
# Function to calculate a hypothetical bill under total wholesale exposure, with
# no PPA or firming arrangements.
# Includes wholesale purchase of LGCs for equivalent 'annual matching'
# Assumes LGC prices have been given?
def calculate_wholesale_bill(
        df:pd.DataFrame,
        settlement_period:str,
        lgc_buy_price:float=0.0,
) -> pd.DataFrame:
    data = df.copy()
    data['Wholesale Cost'] = data['RRP'] * data['Load']

    wholesale_bill = data[['Load', 'Wholesale Cost']].resample(settlement_period)\
        .sum(numeric_only=True)

    wholesale_bill['LGC Cost'] = wholesale_bill['Load'] * lgc_buy_price
    wholesale_bill['Total'] = wholesale_bill['LGC Cost'] + wholesale_bill['Wholesale Cost']
    wholesale_bill = wholesale_bill[['Wholesale Cost', 'LGC Cost', 'Total']]

    return wholesale_bill



def calculate_bill(
        volume_and_price: pd.DataFrame,
        settlement_period: str,
        contract_type: str,
        firming_type:str,
        tariff_details:dict,
        strike_price: float,
        lgc_buy_price: float = 0.0,
        # Default values for all shortfall and LGC sale/purchase is $0 (so default is no penalties, no balancing).
        lgc_sell_price: float = 0.0,
        shortfall_penalty: float = 0.0,
        guaranteed_percent: float = 100.0,
        excess_price: float | str = 'Wholesale',
        indexation: float | list[float] = 0,  # set a default for no indexation, for simplicity
        index_period: str = 'Y',  # and default period is yearly for slightly faster calcs just in case
        floor_price: float = -1000.0,  # default value is market floor
) -> pd.DataFrame:
    """
    Calculates the components of the bills associated with a PPA contract.

    Costs are calculated for each component as per the documentation for corresponding function in this module:
        1. PPA settlement costs: calculate_ppa
        2. Firming costs: calculate_firming
        3. Network costs: calculate_tariff_bill
        4. Revenue from on-sold excess RE: calculate_excess_electricity
        5. Purchase of extra LGCs: calculate_lgcs
        6. Sale of excess LGCs: calculate_lgcs
        7. Any shortfall penalty payments for generation: calculate_shortfall

    Examples:

    Generally the data for calculating a PPA bill would be continuous data covering the period of a year, but to
    demonstrate the use of monthly settlement period we will use data with two interval from January and February.

    >>> volume_and_price = pd.DataFrame({
    ... 'datetime': ['2023/01/01 00:30:00', '2023/01/01 01:00:00', '2023/02/01 00:30:00', '2023/02/01 01:00:00'],
    ... 'Load': [100.0, 100.0, 100.0, 100.0],
    ... 'Contracted Energy': [100.0, 100.0, 80.0, 80.0],
    ... 'RRP': [50.0, 50.0, 50.0, 50.0],
    ... 'Firming price': [80.0, 80.0, 80.0, 80.0]
    ... })
    >>> volume_and_price['datetime'] = pd.to_datetime(volume_and_price['datetime'])
    >>> volume_and_price = volume_and_price.set_index(keys='datetime', drop=True)

    >>> calculate_bill(
    ... volume_and_price=volume_and_price,
    ... settlement_period='M',
    ... contract_type='Pay as Produced',
    ... firming_type='Wholesale exposed',
    ... tariff_details={'Retail':{...}, 'Network':{...}},
    ... strike_price=75.0,
    ... lgc_buy_price=10.0,
    ... lgc_sell_price=10.0,
    ... shortfall_penalty=50.0,
    ... guaranteed_percent=90.0,
    ... excess_price='Wholesale',
    ... indexation=1.0,
    ... index_period='Y',
    ... floor_price= -1000.0)
                Wholesale Cost  PPA Settlement  ...  Shortfall Payments Received    Total
    datetime                                    ...
    2023-01-31         15150.0          5150.0  ...                         -0.0  20100.0
    2023-02-28         12120.0          4120.0  ...                      -1000.0  18640.0
    <BLANKLINE>
    [2 rows x 8 columns]

    :param volume_and_price: Dataframe with datetime index, a column specifying the load  (MWh) named 'Load' and
        a column specifying the contracted energy (MWh) name 'Contracted Energy', a column named 'RRP' specifying the
        wholesale, a column named 'Firming price' specifying the firming price ($/MWh), and a column specifying the
        combined renewable energy generation profiles named 'Hybrid' if a contract_type of 'Baseload' or 'Shaped' is
        specified. An optional column named 'Fixed ($/day)' can be passed if the firming type is retail tariff with a
        daily charge.
    :param settlement_period: The settlement period as a str in the pandas period alias format e.g. 'Y' for yeary, 'Q'
        for quarterly and 'M' for monthly.
    :param contract_type: str,
    :param firming_type: str, one of 'Wholesale exposed', 'Partially wholesale exposed',
        'Retail'.
    :param tariff_details: dict containing two tariff structures labelled by the 
        keys 'Network' and 'Retail'. The tariff structures are nested dictionaries containing tariff components defined by the user selected network tariff.
    :param strike_price: The strike price of the contract in $/MWh
    :param lgc_buy_price: a float specifying the price ($/MWh) of buying LCGs to meet a deficit.
    :param lgc_sell_price: a float specifying the price ($/MWh) at which excess LCGs can be sold.
    :param shortfall_penalty:
    :param guaranteed_percent: a float specifying the percentage of load in each settlement period to guaranteed to be
        met by the contract. Should be a number between 0 - 100.
    :param excess_price: a float specifying the price for excess energy in $/MWh or string ('Wholesale') specifying
        that the wholesale spot price should be used to calculate the value of the excess energy.
    :param indexation:
    :param index_period: How frequently to index the strike price as a st. 'Y' for yearly or 'Q' for quarterly.
    :param floor_price: Minimum wholesale price to use for settlement calculation i.e. for each interval the minimum of
        the floor_price and the wholesale price is taken, and then the resulting price is used to calculate settlement.
    :return: Results are returned in a dataframe on settlement period basis with the index specifying the end of
        the settlement period and an additional columns: 'PPA Value' (value of
        contracted energy at the indexed PPA strike price), 'PPA Settlement' (the cost of settling the PPA),
        'Firming Costs', 'Network Costs', 'Revenue from on-sold RE', 'Revenue from excess LGCs', 'Cost of shortfall LGCs',
        'Shortfall Payments Received', 'Total'.
    """
    results = pd.DataFrame()

    # 1. PPA settlement costs:
    ppa_costs = calculate_ppa(volume_and_price, strike_price, settlement_period, indexation, index_period,
                              floor_price)
    results['PPA Value'] = ppa_costs['PPA Value'].copy()
    results['PPA Settlement'] = ppa_costs['PPA Settlement'].copy()
    results['PPA Final Cost'] = ppa_costs['PPA Final Cost'].copy()

    # 2. Firming costs:
    firming_costs = calculate_firming(volume_and_price, firming_type, tariff_details, settlement_period)
    results['Firming Costs'] = firming_costs['Firming Costs'].copy()

    # 3. Network costs:
    network_costs = calculate_tariff_bill(volume_and_price, settlement_period, tariff_details, 'Network')
    results['Network Costs'] = network_costs['Network Bill ($)'].copy()

    # 4. Revenue from on-sold excess RE:
    excess_val = calculate_excess_electricity(volume_and_price, settlement_period, excess_price)
    results['Revenue from on-sold RE'] = -1 * excess_val['Excess Energy Revenue'].copy()

    # 5. and 6. Revenue/Cost of LGCs
    lgc_balance = calculate_lgcs(volume_and_price, settlement_period, lgc_buy_price, lgc_sell_price, guaranteed_percent)
    results['Revenue from excess LGCs'] = lgc_balance['LGC Oversupply'].copy()
    results['Cost of shortfall LGCs'] = lgc_balance['LGC Undersupply'].copy()

    # 7. Shortfall payments for energy - applied based on contract type.
    shortfall_payment_received = calculate_shortfall(volume_and_price, settlement_period, contract_type, shortfall_penalty,
                                                     guaranteed_percent)
    results['Shortfall Payments Received'] = -1 * shortfall_payment_received['Shortfall']

    results['Total'] = results[['PPA Final Cost', 'Firming Costs', 'Network Costs', 'Revenue from on-sold RE', 'Revenue from excess LGCs', 'Cost of shortfall LGCs', 'Shortfall Payments Received']].sum(axis='columns')

    return results

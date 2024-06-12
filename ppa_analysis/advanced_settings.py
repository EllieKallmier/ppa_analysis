from pathlib import Path

NEM_REGIONS = ['QLD1', 'NSW1', 'VIC1', 'SA1', 'TAS1']

LOAD_COL_NAME = 'Load'
LOAD_DATETIME_COL_NAME = 'TS'
LOAD_TIMEZONE = ''
LOAD_DATA_DIR = Path('data_caches/c_and_i_customer_loads')

GEN_DATETIME_COL_NAME = ''
DAY_FIRST = True

YEARLY_DATA_CACHE = Path('data_caches/yearly_data_files/')

RAW_DATA_CACHE = 'data_caches/gen_data_cache'
EMISSIONS_CACHE = 'data_caches/nemed_cache'
PRICING_CACHE = 'data_caches/pricing_cache'
GEN_TECH_TYPE_S = ['WIND - ONSHORE', 'PHOTOVOLTAIC FLAT PANEL']

CONTRACT_TYPES = [
    'Pay as Produced',
    'Pay as Consumed',
    'Shaped',
    'Baseload',
    '24/7'
]

FIRMING_CONTRACT_TYPES = [
    'Wholesale exposed',
    'Partially wholesale exposed',
    'Retail'
]

SETTLEMENT_PERIODS = [
    'Y',
    'M',
    'Q'
]

TIME_SERIES_INTERVALS = [
    '60'
]

REDEFINE_PERIODS = [
    'Y',
    'Q',
    'M',
]

INDEX_PERIODS = [
    'Y'
]

GEN_COST_DATA = {
    'GenCost 2018 Low': {
        'Wind': {
            'Fixed O&M ($/kW)': 36.0,
            'Variable O&M ($/kWh)': 2.7 / 1000,
            'Capital ($/kW)': 2005,
            'Capacity Factor': 0.35
        },
        'Photovoltaic Flat Panel': {
            'Fixed O&M ($/kW)': 14.4,
            'Variable O&M ($/kWh)': 0.0,
            'Capital ($/kW)': 1280,
            'Capacity Factor': 0.22
        }
    }
}
"""
This module defines the settings used by the user interface notebook and the associated functionality that supports
the interface. The average user will probably not need to modify these settings but a more advance user who wishes to
modify the interface functionality may need to.

Two main types of settings are defined here:

1. Path settings that specify where different datasets are stored.
2. Options settings that define which options are available for users to select through the interface.

"""
from pathlib import Path


# Path settings:

LOAD_DATA_DIR = Path('data_caches/c_and_i_customer_loads')
YEARLY_DATA_CACHE = Path('data_caches/yearly_data_files/')
RAW_DATA_CACHE = 'data_caches/gen_data_cache'
EMISSIONS_CACHE = 'data_caches/nemed_cache'
PRICING_CACHE = 'data_caches/pricing_cache'

# User interface options:

NEM_REGIONS = ['QLD1', 'NSW1', 'VIC1', 'SA1', 'TAS1']

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
    'Y'
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

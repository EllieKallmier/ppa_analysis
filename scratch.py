import os
import functools
from collections import OrderedDict

import ipywidgets as widgets
from IPython.display import display, HTML
from nemosis import static_table

import helper_functions
import advanced_settings
from getting_data import import_gen_data


def get_unit_capcity(unit):
    duid = unit.split(':')[0]
    registered_capacity = static_table(
        table_name='Generators and Scheduled Loads',
        raw_data_location=advanced_settings.RAW_DATA_CACHE,
        select_columns=['DUID', 'Reg Cap (MW)'],
        filter_cols=['DUID'],
        filter_values=[(duid,)]
    ) ['Reg Cap (MW)'].values[0]
    return float(registered_capacity) # str(registered_capacity * 1000)

x = get_unit_capcity('CSPVPS1: PHOTOVOLTAIC FLAT PANEL')
x = 1

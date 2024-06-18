import os
import functools
import logging

import ipywidgets as widgets
from IPython.display import display, HTML
from nemosis import static_table

from ppa_analysis import helper_functions, advanced_settings, import_data

logging.getLogger("nemosis").setLevel(logging.WARNING)

def launch_input_collector():

    display(HTML(
        '''
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        '''
    ))

    display(HTML(
        '''
        <h3>Historical data selection:</h3>
        '''
    ))

    input_collector = {}

    years_with_cached_data = helper_functions.get_data_years(advanced_settings.YEARLY_DATA_CACHE)
    input_collector['year'] = widgets.Dropdown(
        options=years_with_cached_data,
        value=years_with_cached_data[0],
        description='Year:',
        disabled=False,
    )
    display(input_collector['year'])

    input_collector['generator_region'] = widgets.Dropdown(
        options=advanced_settings.NEM_REGIONS,
        value=advanced_settings.NEM_REGIONS[0],
        description='Generator region:',
        disabled=False,
    )
    display(input_collector['generator_region'])

    input_collector['load_region'] = widgets.Dropdown(
        options=advanced_settings.NEM_REGIONS,
        value=advanced_settings.NEM_REGIONS[0],
        description='Load region:',
        disabled=False,
    )
    display(input_collector['load_region'])

    input_collector['load_data_file'] = widgets.Dropdown(
        options=os.listdir(advanced_settings.LOAD_DATA_DIR),
        description='Load data file:',
        disabled=False,
    )
    display(input_collector['load_data_file'])

    def get_generator_options():
        year = input_collector['year'].value
        gen_data_file = advanced_settings.YEARLY_DATA_CACHE / f"gen_data_{year}.parquet"
        gen_regions = [input_collector['generator_region'].value]
        gen_options = import_data.get_generator_options(gen_data_file, gen_regions)
        return gen_options

    gen_options = get_generator_options()
    input_collector['generators'] = widgets.SelectMultiple(
        options=gen_options,
        value=tuple(gen_options[:4]),
        description='Generators:',
        disabled=False
    )
    display(input_collector['generators'])

    def update_generator_options(change):
        print('update gens')
        if change['new'] != change['old']:
            input_collector['generators'].options = get_generator_options()

    input_collector['year'].observe(update_generator_options)
    input_collector['generator_region'].observe(update_generator_options)

    display(HTML(
        '''
        <h3>Contract paramters:</h3>
        '''
    ))

    display(HTML(
        '''
        <h4> - All contracts:</h4>
        '''
    ))

    input_collector['contract_type'] = widgets.Dropdown(
        options=advanced_settings.CONTRACT_TYPES,
        value=advanced_settings.CONTRACT_TYPES[0],
        description='Contract type:',
        disabled=False,
    )
    display(input_collector['contract_type'])

    input_collector['firming_contract_type'] = widgets.Dropdown(
        options=advanced_settings.FIRMING_CONTRACT_TYPES,
        value=advanced_settings.FIRMING_CONTRACT_TYPES[0],
        description='Firming contract type:',
        disabled=False,
    )
    display(input_collector['firming_contract_type'])

    input_collector['settlement_period'] = widgets.Dropdown(
        options=advanced_settings.SETTLEMENT_PERIODS,
        value=advanced_settings.SETTLEMENT_PERIODS[0],
        description='Settlment period:',
        disabled=False,
    )
    display(input_collector['settlement_period'])

    input_collector['contract_amount'] = widgets.BoundedFloatText(
        value=100.0,
        min=0,
        max=100,
        description='Contract amount (%):',
    )
    display(input_collector['contract_amount'])

    input_collector['strike_price'] = widgets.FloatText(
        value=100.0,
        description='Strike price ($/MW/h):',
    )
    display(input_collector['strike_price'])

    input_collector['lgc_buy_price'] = widgets.FloatText(
        value=35.0,
        description='LGC buy price ($/MW/h):',
    )
    display(input_collector['lgc_buy_price'])

    input_collector['lgc_sell_price'] = widgets.FloatText(
        value=20.0,
        description='LGC sell price ($/MW/h):',
    )
    display(input_collector['lgc_sell_price'])

    input_collector['shortfall_penalty'] = widgets.FloatText(
        value=25.0,
        description='Short fall penalty ($/MW/h):',
    )
    display(input_collector['shortfall_penalty'])

    input_collector['guaranteed_percent'] = widgets.BoundedFloatText(
        value=85.0,
        min=0,
        max=100,
        description='Guaranteed percentage (%):',
    )
    display(input_collector['guaranteed_percent'])

    input_collector['floor_price'] = widgets.FloatText(
        value=65.0,
        description='Floor price ($/MW/h):',
    )
    display(input_collector['floor_price'])

    input_collector['excess_price'] = widgets.FloatText(
        value=65.0,
        description='Excess price ($/MW/h):',
    )
    display(input_collector['excess_price'])

    input_collector['indexation'] = widgets.BoundedFloatText(
        value=1.0,
        min=0,
        max=100,
        description='Indexation (%):',
    )
    display(input_collector['indexation'])

    input_collector['index_period'] = widgets.Dropdown(
        options=advanced_settings.INDEX_PERIODS,
        value=advanced_settings.INDEX_PERIODS[0],
        description='Index period:',
        disabled=False,
    )
    display(input_collector['index_period'])

    display(HTML(
        '''
        <h4> - Shaped and baseload contracts only:</h4>
        '''
    ))

    input_collector['redefine_period'] = widgets.Dropdown(
        options=advanced_settings.REDEFINE_PERIODS,
        value=advanced_settings.REDEFINE_PERIODS[2],
        description='Redefine period:',
        disabled=False,
    )
    display(input_collector['redefine_period'])

    display(HTML(
        '''
        <h4> - Shaped contracts only:</h4>
        '''
    ))

    input_collector['matching_percentile'] = widgets.BoundedFloatText(
        value=1.0,
        min=0,
        max=100,
        description='Matching percentile:',
    )
    display(input_collector['matching_percentile'])

    display(HTML(
        '''
        <h4> - Wholesale exposure only:</h4>
        '''
    ))

    input_collector['exposure_upper_bound'] = widgets.FloatText(
        value=300.0,
        description='Exposure upper bound ($/MW/h):',
    )
    display(input_collector['exposure_upper_bound'])

    input_collector['exposure_lower_bound'] = widgets.FloatText(
        value=20.0,
        description='Exposure lower bound ($/MW/h):',
    )
    display(input_collector['exposure_lower_bound'])

    display(HTML(
        '''
        <h4> - Analysis paramters:</h4>
        '''
    ))

    input_collector['time_series_interval'] = widgets.Dropdown(
        options=advanced_settings.TIME_SERIES_INTERVALS,
        value=advanced_settings.TIME_SERIES_INTERVALS[0],
        description='Time series interval:',
        disabled=False,
    )
    display(input_collector['time_series_interval'])

    input_collector['generator_data_set'] = widgets.Dropdown(
        options=advanced_settings.GEN_COST_DATA.keys(),
        description='Generator data set:',
        disabled=False,
    )
    display(input_collector['generator_data_set'])

    return input_collector


def get_unit_capcity(unit):
    duid = unit.split(':')[0]
    registered_capacity = static_table(
        table_name='Generators and Scheduled Loads',
        raw_data_location=advanced_settings.RAW_DATA_CACHE,
        select_columns=['DUID', 'Reg Cap (MW)'],
        filter_cols=['DUID'],
        filter_values=[(duid,)]
    )['Reg Cap (MW)'].values[0]
    return float(registered_capacity) * 1000

def add_editor_for_generator(generator_data_editor, generator, input_collector):
    with generator_data_editor['out']:
        generator_data_set_name = input_collector['generator_data_set'].value
        generator_data_set = advanced_settings.GEN_COST_DATA[generator_data_set_name]
        for generator_type in generator_data_set.keys():
            if generator_type.upper() in generator:
                generator_data_editor[f'{generator}'] = {}

                generator_data_editor[f'{generator}']['label'] = HTML(
                    f'''
                    <h5>{generator}:</h5>
                    '''
                )

                capacity = get_unit_capcity(generator)
                generator_data_editor[f'{generator}']['capacity'] = widgets.FloatText(
                    value=capacity,
                    description='Nameplate Capacity (kW)',
                )

                generator_data_editor[f'{generator}']['fixed_om'] = widgets.FloatText(
                    value=generator_data_set[generator_type]['Fixed O&M ($/kW)'],
                    description='Fixed O&M ($/kW)',
                )
 
                generator_data_editor[f'{generator}']['variable_om'] = widgets.FloatText(
                    value=generator_data_set[generator_type]['Variable O&M ($/kWh)'],
                    description='Variable O&M ($/kWh)',
                )

                generator_data_editor[f'{generator}']['capital'] = widgets.FloatText(
                    value=generator_data_set[generator_type]['Capital ($/kW)'],
                    description='Capital ($/kW)',
                )

                generator_data_editor[f'{generator}']['capacity_factor'] = widgets.FloatText(
                    value=generator_data_set[generator_type]['Capacity Factor'],
                    description='Capacity Factor',
                )

def remove_editor_for_generator(generator_data_editor, generator):
    with generator_data_editor['out']:
        generator_data_editor[f'{generator}'].close
        del generator_data_editor[f'{generator}']

def update_generator_data_editor(generator_data_editor, input_collector, change=None):

    if change is None:
        for generator in input_collector['generators'].value:
            add_editor_for_generator(generator_data_editor, generator, input_collector)
    else:
        if isinstance(change['new'], str):
            change_new = [change['new']]
        else:
            change_new = change['new']
        if isinstance(change['old'], str):
            change_old = [change['old']]
        else:
            change_old = change['old']
        for generator in change_new:
            if generator not in change_old:
                add_editor_for_generator(generator_data_editor, generator, input_collector)
        for generator in change_old:
            if generator not in change_new:
                remove_editor_for_generator(generator_data_editor, generator)
                pass
    generator_data_editor['out'].clear_output()
    with generator_data_editor['out']:
        for key, value in generator_data_editor.items():
            if key != 'out':
                for k, v in value.items():
                    display(v)
                    
    display(generator_data_editor['out'])
    return generator_data_editor

def launch_generator_data_editor(input_collector):
    generator_data_editor = {}
    generator_data_editor['out'] = widgets.Output()
    generator_data_editor = update_generator_data_editor(
        generator_data_editor, input_collector
    )
    input_collector['generators'].observe(
        functools.partial(
            update_generator_data_editor,
            generator_data_editor,
            input_collector
        ),
        names='value'
    )
    return generator_data_editor


def launch_battery_input_collector():

    display(HTML(
        '''
        <style>
        .widget-label { min-width: 30ex !important; }
        .widget-select select { min-width: 70ex !important; }
        .widget-dropdown select { min-width: 70ex !important; }
        .widget-floattext input { min-width: 70ex !important; }
        </style>
        '''
    ))

    display(HTML(
        '''
        <h3>Battery inputs:</h3>
        '''
    ))

    battery_input_collector = {}

    battery_input_collector['rated_power_capacity'] = widgets.FloatText(
        value=1.0,
        description='Rated power capacity (MW):',
    )
    display(battery_input_collector['rated_power_capacity'])

    battery_input_collector['size_in_mwh'] = widgets.FloatText(
        value=2.0,
        description='Battery size (MWh):',
    )
    display(battery_input_collector['size_in_mwh'])

    return battery_input_collector
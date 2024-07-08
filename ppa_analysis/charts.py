# File for creating useful charts
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from ppa_analysis import helper_functions

# Set theme for seaborn plots:
sns.set_theme(style = 'white', font_scale = 1)
sns.set_style('darkgrid')

# Suppress logging below warning level
mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)

# Pricing charts


# Emissions comparisons


# Profile plots
def plot_avg_seasonal_load(
    load_and_gen_data:pd.DataFrame,
    input_collector:dict
):
    seasonal_df = helper_functions.get_seasons(load_and_gen_data)
    weekend_df = helper_functions.get_weekends(seasonal_df, input_collector['load_region'].value)

    weekend_df['Weekend'] = weekend_df['Weekend'].map({0:'Weekday', 1:'Weekend'})
    weekend_df['Hour'] = weekend_df.index.hour

    load_name = input_collector['load_data_file'].value.split('.')[0]
    fig = sns.relplot(
        data=weekend_df,
        x = 'Hour', 
        y = 'Load',
        col = 'Weekend', 
        hue = 'Season',
        palette = {
            'Summer' : 'yellowgreen', 
            'Autumn' : 'tomato', 
            'Winter' : 'dodgerblue', 
            'Spring' : 'mediumpurple'
        },
        kind = 'line',
        col_order=['Weekday', 'Weekend']
    )
    fig.set_axis_labels('Hour', 'Average Load (MWh)').set_titles("{col_name}")
    fig.figure.subplots_adjust(top=0.9)
    fig.figure.suptitle(f'{load_name}')
    plt.show()
    

def plot_avg_seasonal_generation(
    load_and_gen_data:pd.DataFrame,
    input_collector:dict,
    generator_data_editor:dict
):
    
    seasonal_df = helper_functions.get_seasons(load_and_gen_data)
    weekend_df = helper_functions.get_weekends(seasonal_df, input_collector['load_region'].value)
    gen_list = []
    for gen, info in generator_data_editor.items():
            if gen != 'out':
                gen_list.append(gen)

    gen_df = weekend_df[gen_list + ['Season']].copy()
    gen_df = gen_df.melt(ignore_index=False, id_vars=['Season'])
    gen_df = gen_df.rename(columns={'variable':'Generator', 'value':'MWh'})  
    gen_df['Hour'] = gen_df.index.hour

    num_gens = len(gen_list)
    if num_gens <= 3:
        col_wrap = num_gens
    else:
        col_wrap = (num_gens % 2 == 0) * 2 + (num_gens % 2 == 1) * 3

    fig = sns.relplot(
        data=gen_df,
        x = 'Hour', 
        y = 'MWh',
        col = 'Generator', 
        hue = 'Season',
        palette = {
            'Summer' : 'gold', 
            'Autumn' : 'tomato', 
            'Winter' : 'darkred', 
            'Spring' : 'darkorange'
        },
        kind = 'line',
        col_wrap = col_wrap
    )
    fig.set_axis_labels('Hour', 'Average Generation (MWh)').set_titles("{col_name}")
    fig.figure.subplots_adjust(top=0.9)
    fig.figure.suptitle(f'Average daily generation')
    plt.show()


def plot_contract_samples(
    load_and_gen_data:pd.DataFrame,
    input_collector:dict,
    columns_to_plot:list[str]
):
    to_plot = load_and_gen_data[columns_to_plot+['Hybrid', 'Contracted Energy']].copy()
    avg_day_to_plot = to_plot.groupby(to_plot.index.hour.rename('Hour')).mean(numeric_only=True)

    load_name = input_collector['load_data_file'].value.split('.')[0]
    contract_type = input_collector['contract_type'].value

    # Plot bars
    fig, ax1 = plt.subplots(figsize=(12,6))
    sns.barplot(avg_day_to_plot['Contracted Energy'], color='skyblue', label='Contracted Energy', ax=ax1)

    # Plot lines
    colours = {'Load':'red','Load with battery':'yellowgreen','Load with flex':'deeppink'}
    for col in columns_to_plot:
        sns.lineplot(avg_day_to_plot[col], marker='o', color=colours[col], linewidth=1, label=col, ax=ax1)

    sns.lineplot(avg_day_to_plot['Hybrid'], marker='o', color='purple', linewidth=1, label='Hybrid', ax=ax1)

    plt.title(f'Average daily profiles\n{contract_type}: {load_name}')
    plt.ylabel('MWh')
    plt.show()
 
# Financial outcomes
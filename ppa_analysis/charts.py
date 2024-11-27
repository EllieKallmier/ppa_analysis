# File for creating useful charts
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from ppa_analysis import helper_functions
from mpl_toolkits.axes_grid1.inset_locator import inset_axes 

# Set theme for seaborn plots:
sns.set_theme(style = 'white', font_scale = 1)
sns.set_style('darkgrid')

# Suppress logging below warning level
mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)


def plot_hybrid_results(
        load_and_gen_data:pd.DataFrame,
        percentage_results:dict,
        generator_capacities:dict
):
    """Plots the results of an optimal hybrid generation mix, showing contracted 
    capacity and energy generation.

    This function visualizes the hybrid generation results, including the contracted 
    capacities of generators based on their percentage of output and the total 
    contracted energy (in MWh). The results are presented in two subplots: one 
    for the total contracted energy in MWh, and a pie chart representing the 
    percentage breakdown of the hybrid mix.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data 
            with load and generation values in MWh.
        percentage_results (dict): A dictionary with generator names as keys and 
            values are sub-dictionaries containing the percentage of hybrid trace 
            and generator output for each generator.
        generator_capacities (dict): A dictionary with generator names as keys and 
            their respective nameplate capacities in MW.
    """
    simplified_percentages_dict = {
        key : val['Percent of hybrid trace'] for key, val in percentage_results.items()
    }

    percentages_as_df = pd.DataFrame()
    percentages_as_df['Name'] = simplified_percentages_dict.keys()
    percentages_as_df['Percentage'] = simplified_percentages_dict.values()

    percentages_as_df = percentages_as_df.set_index('Name')

    for generator, percs in percentage_results.items():
        capacity_mw = generator_capacities[generator] / 1000
        contracted_capacity = capacity_mw * (percs['Percent of generator output'] / 100)
        percentages_as_df.loc[generator, 'Contracted Capacity (MW)'] = contracted_capacity

    percentages_as_df['Labels'] = percentages_as_df.index
    percentages_as_df['Labels'] = percentages_as_df['Labels'].map(
        {name : name.split(': ')[0] + '\n' + name.split(': ')[1] \
        for name in percentages_as_df['Labels'].values}
    )

    total_contract_capacity = percentages_as_df['Contracted Capacity (MW)'].sum(numeric_only=True)
    if total_contract_capacity < 1:
        total_contract_capacity = round(total_contract_capacity*1000, None)
        units = 'kW'
    else:
        units = 'MW'
        total_contract_capacity = round(total_contract_capacity, 2)
    
    list_of_cols = ['Load'] + percentages_as_df.index.to_list()
    contracted_generation = load_and_gen_data[list_of_cols].copy()
    for generator, percs in percentage_results.items():
        contracted_generation[generator] *= (percs['Percent of generator output'] / 100)

    total_energy_sums = pd.DataFrame(contracted_generation.sum(numeric_only=True), columns=['MWh'])

    # Reshape this df to allow stacking bar chart:
    reshaped_sums = pd.DataFrame(columns=['Load', 'Generation'], index=total_energy_sums.index)

    for row in total_energy_sums.index:
        if 'Load' in row:
            reshaped_sums.loc[row, 'Load'] = total_energy_sums.loc[row, 'MWh']
            reshaped_sums.loc[row, 'Generation'] = 0
        else:
            reshaped_sums.loc[row, 'Generation'] = total_energy_sums.loc[row, 'MWh']
            reshaped_sums.loc[row, 'Load'] = 0

    reshaped_sums = reshaped_sums.T

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle('Optimal hybrid results')

    axes[0].set_title('Total MWh Contracted')
    axes[0].set_ylabel('MWh')
    axes[1].set_title(f'Hybrid Mix\nContracted Capacity: {total_contract_capacity}{units}')

    color_palette = sns.mpl_palette('Set2')
    reshaped_sums.plot.bar(stacked=True, ax=axes[0], legend=False, rot=0, color=color_palette)
    axes[0].legend(bbox_to_anchor=(1.25, 0.1), loc='upper left')

    plt.pie(
        data=percentages_as_df, 
        x='Percentage',
        autopct=(lambda p: '{:.1f}%'.format(round(p)) if p > 0 else ''), # formats out labels with 0.0% values!
        colors=color_palette[1:]
    )

    plt.show()



def plot_emissions_bw(
        load_and_gen_data:pd.DataFrame,
):
    """Plots a bow and whisker distribution of emissions (in tCO2-e) due to 
        unmatched load for each hour of the day.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data for 
            load, hybrid generation, contracted energy, and associated emissions factors 
            (e.g., AEI - Average Emissions Intensity). Requires at least columns named
            'AEI', 'Load', and a datetime index. 
    """
    emissions_measure = load_and_gen_data.copy()
    cols_to_plot = ['Time']
    for col in emissions_measure.columns:
        if 'Load' in col:
            emissions_measure[f'Unmatched Energy - {col}'] = (emissions_measure['Load'] - np.minimum(emissions_measure['Hybrid'], emissions_measure['Contracted Energy'])).clip(lower=0.0)
            emissions_measure[f'Emissions (tCO2-e) - {col}'] = emissions_measure['AEI'] * emissions_measure[f'Unmatched Energy - {col}']

            cols_to_plot.append(f'Emissions (tCO2-e) - {col}')


    emissions_measure['Time'] = emissions_measure.index.strftime('%H:%M')
    to_plot_emissions = emissions_measure[cols_to_plot].copy()
    to_plot_emissions = to_plot_emissions.melt(id_vars=['Time']).rename(columns={'value':'Emissions (tCO2-e)'})

    plt.figure(figsize=(16,8))
    sns.boxplot(data=to_plot_emissions, x='Time', y='Emissions (tCO2-e)', hue='variable', flierprops={"marker":'.'}, palette=sns.color_palette('YlGnBu')[0:len(cols_to_plot)*2:2])
    plt.title('Emissions due to unmatched load by hour')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()



def plot_matching_bw(
        load_and_gen_data:pd.DataFrame
):
    """Plots a bow and whisker distribution of hourly time-matching for each hour 
        of the day.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data for 
            load, hybrid generation and contracted energy. Requires at least columns named
            'Load', 'Hybrid', 'Contracted Energy' and a datetime index. 
    """
    matching = load_and_gen_data.copy()
    matching['Delivered Hybrid'] = np.minimum(matching['Hybrid'], matching['Contracted Energy'])

    cols_to_plot = ['Time']
    for col in load_and_gen_data.columns:
        if 'Load' in col:
            matching[f'Hourly match (%) - {col}'] = np.where(matching[col] == 0, 100, np.minimum(matching['Delivered Hybrid'] / matching[col] * 100, 100))
            cols_to_plot.append(f'Hourly match (%) - {col}')

    matching['Time'] = matching.index.strftime('%H:%M')
    to_plot_match = matching[cols_to_plot].copy()
    to_plot_match = to_plot_match.melt(id_vars=['Time']).rename(columns={'value':'Match %'})

    plt.figure(figsize=(16,8))
    sns.boxplot(
        data=to_plot_match, x='Time', y='Match %', hue='variable', 
        palette=sns.mpl_palette('RdYlGn')[0:len(cols_to_plot)*2:2]
    )
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title('Match between load and generation by hour')
    plt.show()



# Emissions comparisons
def plot_emissions_heatmap(
        load_and_gen_data:pd.DataFrame,
        load_column_to_plot:str
):
    """Plots a heatmap of hourly emissions due to unmatched load.

    This function calculates the emissions (in tCO2-e) caused by unmatched energy 
    for each hour of the day, based on the specified load column. The heatmap shows 
    the distribution of emissions across days and hours, allowing for comparison of 
    emissions over time.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data for 
            load, hybrid generation, contracted energy, and emissions factors (e.g., AEI).
            Requires at least columns named 'AEI', 'Load', and a datetime index. 
        load_column_to_plot (str): The column name representing the load data to be 
            used for calculating emissions (e.g., 'Load', 'Load with battery').

    Notes:
        - The emissions are calculated as the product of the Average Emissions Intensity 
          (AEI) and the unmatched energy, which is the difference between the specified 
          load column and the minimum of hybrid generation and contracted energy.
        - The heatmap visualizes emissions over time, with hours on the y-axis and dates 
          on the x-axis.
    """

    emissions_measure = load_and_gen_data.copy()
    emissions_measure['Unmatched Energy'] = (emissions_measure[load_column_to_plot] \
                                - np.minimum(emissions_measure['Hybrid'], \
                                    emissions_measure['Contracted Energy'])).clip(lower=0.0)
    emissions_measure['Emissions (tCO2-e)'] = emissions_measure['AEI'] * \
        emissions_measure['Unmatched Energy']

    em_results = pd.DataFrame(emissions_measure['Emissions (tCO2-e)'])
    em_results['Hour'] = em_results.index.hour
    em_results['Day'] = em_results.index.strftime('%m/%d/%Y')
    em_results = em_results.reset_index(drop=True)

    em_results = em_results.pivot(index='Hour', columns='Day', values='Emissions (tCO2-e)')

    fig, ax = plt.subplots(figsize=(16, 7))
    colorbar_axes = inset_axes(
        ax,
        width = '40%',
        height = '5%',
        loc = 'upper right',
        bbox_to_anchor=(0, 0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax = sns.heatmap(
        em_results, 
        ax=ax, 
        cmap='RdYlGn_r', 
        cbar_ax=colorbar_axes, 
        cbar_kws={'orientation': 'horizontal'},
        yticklabels = 4,
        xticklabels=30
    )

    title_extra = ''
    if 'battery' in load_column_to_plot.lower():
        title_extra = ' - with battery'
    elif 'flex' in load_column_to_plot.lower():
        title_extra = ' - with flex'
    else:
        title_extra = ''

    ax.invert_yaxis()
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha='left')
    fig.suptitle(
        f'Hourly emissions due to unmatched load (tCO2-e){title_extra}', 
        x=0.125, y=0.91, ha = 'left', va = 'bottom'
    )
    plt.show()


# Matching
def plot_matching_heatmap(
        load_and_gen_data:pd.DataFrame,
        load_column_to_plot:str
):
    """Plots a heatmap of the percentage of load matched by hybrid generation.

    This function calculates the hourly percentage of load matched by hybrid generation, 
    and visualizes it as a heatmap across hours and days.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data for 
            load, hybrid generation, contracted energy, and associated time-related 
            columns. Must have at least columns named 'Load', 'Hybrid' and 
            'Contracted Energy'.
        load_column_to_plot (str): The column name representing the load data to be 
            used for calculating the match percentage (e.g., 'Load', 'Load with battery').

    Notes:
        - The "Delivered Hybrid" value is the minimum of hybrid generation and contracted energy.
        - The hourly match percentage is calculated as the ratio of "Delivered Hybrid" to the 
          specified load column, with a cap of 100% match.
        - The heatmap visualizes hourly match percentages across days, with hours on the y-axis 
          and dates on the x-axis.
    """
    matching = load_and_gen_data.copy()
    matching['Delivered Hybrid'] = np.minimum(matching['Hybrid'], matching['Contracted Energy'])
    matching['Hourly match (%)'] = np.where(
        matching[load_column_to_plot] == 0, 
        100, 
        np.minimum(matching['Delivered Hybrid'] / matching[load_column_to_plot] * 100, 100)
    )

    matching_results_df = pd.DataFrame(matching['Hourly match (%)'])
    matching_results_df['Hour'] = matching_results_df.index.hour
    matching_results_df['Day'] = matching_results_df.index.strftime('%m/%d/%Y')
    matching_results_df = matching_results_df.reset_index(drop=True)

    matching_results_df = matching_results_df.pivot(
        index='Hour', columns='Day', values='Hourly match (%)'
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    colorbar_axes = inset_axes(
        ax,
        width = '40%',
        height = '5%',
        loc = 'upper right',
        bbox_to_anchor=(0, 0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    ax = sns.heatmap(
        matching_results_df, 
        ax=ax, 
        cmap='RdBu', 
        cbar_ax=colorbar_axes, 
        cbar_kws={'orientation': 'horizontal'},
        yticklabels = 4,
        xticklabels=30
    )

    title = ''
    if 'battery' in load_column_to_plot.lower():
        title = 'Percentage of load match - with battery'
    elif 'flex' in load_column_to_plot.lower():
        title = 'Percentage of load match - with flex'
    else:
        title = 'Percentage of load match'

    ax.invert_yaxis()
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha='left')
    fig.suptitle(title, x=0.125, y=0.91, ha='left', va='bottom')
    plt.show()


# Profile plots
def plot_avg_seasonal_load(
    load_data:pd.DataFrame,
    load_region:str,
    load_title:str
):
    """Plots an average seasonal load profile for a weekday and weekend day.

    This function generates line plots showing the average load for different seasons 
    (Summer, Autumn, Winter, Spring) across hours of the day, separated by weekday
    and weekend.

    Args:
        load_data (pd.DataFrame): A DataFrame containing time-series load data indexed by date and time.
        load_region (str): The region used to identify public holidays and weekends, affecting load patterns.
        load_title (str): The title of the plot to be displayed.
    """

    seasonal_df = helper_functions.get_seasons(load_data)
    weekend_df = helper_functions.get_weekends(seasonal_df, load_region)

    weekend_df['Weekend'] = weekend_df['Weekend'].map({0:'Weekday', 1:'Weekend'})
    weekend_df['Hour'] = weekend_df.index.hour

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
    fig.figure.suptitle(f'{load_title}')
    plt.show()
    

def plot_avg_seasonal_generation(
    gen_data:pd.DataFrame,
    load_region:str,
    generator_col_names:list
):
    """Plots the average seasonal generation for each generator across hours of the day.

    This function generates line plots of the average generation for each season
    for each specified generator.

    Args:
        gen_data (pd.DataFrame): A DataFrame containing time-series generation, must
            have a datetime index and generation data in MWh. 
        load_region (str): The region used to determine public holidays and weekends.
        generator_col_names (list): A list of generator column names to include 
            in the plot.

    Notes:
        - The function generates one subplot for each generator.
    """
    
    seasonal_df = helper_functions.get_seasons(gen_data)
    weekend_df = helper_functions.get_weekends(seasonal_df, load_region)
    gen_list = []
    for gen in generator_col_names:
        if gen != 'out':
            gen_list.append(gen)

    gen_df = weekend_df[gen_list + ['Season']].copy()
    gen_df = gen_df.melt(ignore_index=False, id_vars=['Season'])

    if 'UNIT' in gen_df.columns:
        gen_df = gen_df.rename(columns={'UNIT':'Generator', 'value':'MWh'})  
    else:
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
    fig.figure.suptitle('Average daily generation')
    plt.show()


def plot_contract_samples(
    load_and_gen_data:pd.DataFrame,
    contract_type:str,
    load_title:str,
    columns_to_plot:list[str]
):
    to_plot = load_and_gen_data[columns_to_plot+['Hybrid', 'Contracted Energy']].copy()
    avg_day_to_plot = to_plot.groupby(to_plot.index.hour.rename('Hour')).mean(numeric_only=True)

    # Plot bars
    fig, ax1 = plt.subplots(figsize=(12,6))
    sns.barplot(avg_day_to_plot['Contracted Energy'], color='skyblue', label='Contracted Energy', ax=ax1)

    # Plot lines
    colours = {'Load':'tomato','Load with battery':'yellowgreen','Load with flex':'orange'}
    for col in columns_to_plot:
        sns.lineplot(avg_day_to_plot[col], marker='o', color=colours[col], linewidth=1, label=col, ax=ax1)

    sns.lineplot(avg_day_to_plot['Hybrid'], marker='o', color='purple', linewidth=1, label='Hybrid', ax=ax1)

    plt.title(f'Average daily profiles\n{contract_type}: {load_title}')
    plt.ylabel('MWh')
    plt.show()


# Financial outcomes
def plot_bill_components(
          bill_results:pd.DataFrame,
          wholesale_bill_results:pd.DataFrame,
          settlement_period:str
):
    """Plots average daily profiles of load, contracted energy, and hybrid energy 
        for different contract types.

    This function generates a bar plot for contracted energy and line plots for 
    specified load types (e.g., load with battery, load with flex) along with 
    hybrid energy across each hour of the day.

    Args:
        load_and_gen_data (pd.DataFrame): A DataFrame containing time-series data 
            for load, hybrid energy, and contracted energy, indexed by datetime.
        contract_type (str): The type of contract being analyzed, used in the plot title.
        load_title (str): The title for the plot, typically representing the 
            load profile or contract details.
        columns_to_plot (list of str): A list of column names representing the 
            different load profiles (e.g., "Load", "Load with battery", 
            "Load with flex") to be plotted as lines.

    Notes:
        - The colors for the line plots are customized for different load profiles:
            - 'tomato' for "Load"
            - 'yellowgreen' for "Load with battery"
            - 'orange' for "Load with flex"
            - 'purple' for "Hybrid"
    """

    # Reformat the bill df here rather than outside of the function:
    bill_results_to_plot = bill_results[['PPA Final Cost', 'Firming Costs', 'Network Costs', 'Revenue from on-sold RE', 'Revenue from excess LGCs', 'Cost of shortfall LGCs', 'Shortfall Payments Received', 'Total']].copy()

    # Add the wholesale bill results for comparison here:
    bill_results_to_plot['No PPA Total'] = wholesale_bill_results['Total'].copy()
    bill_results_to_plot['No PPA Total'] += bill_results_to_plot['Network Costs']

    bill_results_to_plot['Year'] = bill_results_to_plot.index.year

    if settlement_period == 'M':
        bill_results_to_plot['Settlement Period'] = bill_results_to_plot.index.strftime('%b\n%Y')
    elif settlement_period == 'Q':
        bill_results_to_plot['Quarter'] = bill_results_to_plot.index.quarter
        bill_results_to_plot['Settlement Period'] = 'Q' + bill_results_to_plot['Quarter'].astype(str) + '\n' + bill_results_to_plot['Year'].astype(str)
        bill_results_to_plot = bill_results_to_plot.drop(columns=['Quarter'])
    else:
        bill_results_to_plot['Settlement Period'] = bill_results_to_plot.index.year

    bill_results_to_plot = bill_results_to_plot.drop(columns=['Year'])
    bill_results_to_plot = bill_results_to_plot.reset_index(drop=True).set_index('Settlement Period')

    total_values = bill_results_to_plot[['Total', 'No PPA Total']].copy()

    bill_results_to_plot = bill_results_to_plot.drop(columns=['No PPA Total', 'Total'])

    fig, ax = plt.subplots(figsize=(12,8))

    bill_results_to_plot.plot.bar(stacked=True, ax=ax, position=1, width=0.35, color=mpl.colormaps['Dark2'].colors, linewidth=0)
    ax2 = ax.twinx()
    ax2.sharey(ax)

    total_values.plot.bar(ax=ax2, position=0, width=0.35, color=['silver', 'darkgrey'], linewidth=0)
    ax2.grid(visible=False)
    ax2.axes.get_yaxis().set_visible(False)

    plt.title('Bill components by settlement period')
    ax.set_ylabel('Costs and revenues ($)')
    ax.set_xlabel('Settlement period')
    ax.set_xticklabels(bill_results_to_plot.index.values, rotation=0)
    ax.set_xlim(-0.5, len(bill_results_to_plot.index) - 0.5)
    ax.legend(bbox_to_anchor=(1.005,0.91), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.005,1), loc='upper left')

    plt.show()


def plot_cashflow(
        bill_results:pd.DataFrame
): 
    """Plots a waterfall chart representing the annual cashflow from various costs and revenues.

    This function generates a waterfall chart that visualizes the annual cashflow by showing the 
    contributions of different cost and revenue categories, such as PPA costs, firming costs, 
    network costs, revenue from on-sold renewable energy, and other related revenues/shortfalls. 

    Args:
        bill_results (pd.DataFrame): A DataFrame containing time-series data on various cost and revenue 
            categories related to energy bills, indexed by datetime.

    Notes:
        - The function aggregates data by year and sums it for each relevant column.
        - The chart visualizes positive (increasing) values in green, negative 
            (decreasing) values in red, and total values in blue.
    """

    costs_waterfall = bill_results[['PPA Final Cost', 'Firming Costs', 'Network Costs', 'Revenue from on-sold RE', 'Revenue from excess LGCs', 'Cost of shortfall LGCs', 'Shortfall Payments Received', 'Total']].copy()

    fig_w = go.Figure()

    costs_waterfall = costs_waterfall.resample('Y').sum(numeric_only=True)
    costs_waterfall['Total'] = None
    costs_waterfall['Year'] = costs_waterfall.index.year
    costs_waterfall = costs_waterfall.set_index('Year', drop=True)

    costs_waterfall = costs_waterfall.T
    costs_waterfall['Measure'] = np.where(costs_waterfall.index == 'Total', 'total', 'relative')
    costs_waterfall['Category'] = costs_waterfall.index
    costs_waterfall = costs_waterfall.reset_index(drop=True)

    costs_waterfall = costs_waterfall.melt(id_vars=['Category', 'Measure'])
    colour = ['yellowgreen', 'orangered', 'lightskyblue']

    fig_w.add_trace(
        go.Waterfall(
            name = '', 
            orientation = "v",
            measure = costs_waterfall['Measure'],
            x = [costs_waterfall['Year'], costs_waterfall['Category']],
            y = costs_waterfall['value'],
            connector = {"line":{"color":"rgb(63, 63, 63)", "width":1}},
            decreasing = {"marker":{"color":colour[0]}},
            increasing = {"marker":{"color":colour[1]}},
            totals = {"marker":{"color":colour[2], "line":{"color":colour[2], "width":2}}}
        )
    )

    fig_w.update_layout(
        title = "Annual Cashflow",
        waterfallgroupgap = 0.2,
        height=600,
        width=800,
        yaxis_title='Costs and revenues ($)',
        xaxis=dict(showgrid=True, gridwidth=0.1, gridcolor='white')
    )

    fig_w.show()
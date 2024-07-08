"""
Functionality for optimised load shifting to minimise wholesale spot costs.

User facing functionality is provided through the function daily_load_shifting see it's docstring for further info.

"""
import pandas as pd
import numpy as np

from datetime import timedelta
from mip import Model, xsum, minimize, CONTINUOUS, OptimizationStatus, CBC, GUROBI

from ppa_analysis import advanced_settings


def _get_daily_load_sums(
        df: pd.DataFrame  # a pandas df that has DateTime index and 'Load' as a column name
) -> pd.DataFrame:
    return df['Load'].copy().resample('D').sum(numeric_only=True)


def _create_base_days(
        load_profile: pd.DataFrame,
        base_load_quantile: float
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # First get just the load profile from df:
    load_profile = load_profile[['Load', 'Weekend']].copy()

    all_weekdays_only = load_profile[load_profile['Weekend'] == 0].copy()
    all_weekends_only = load_profile[load_profile['Weekend'] == 1].copy()

    base_weekday = all_weekdays_only.groupby(
        [all_weekdays_only.index.month.rename('Month'), all_weekdays_only.index.hour.rename('Hour')])['Load'].quantile(
        base_load_quantile).reset_index()

    base_weekend = all_weekends_only.groupby(
        [all_weekends_only.index.month.rename('Month'), all_weekends_only.index.hour.rename('Hour')])['Load'].quantile(
        base_load_quantile).reset_index()

    return base_weekday, base_weekend


def daily_load_shifting(
        timeseries_data: pd.DataFrame,
        base_load_quantile: float = 0.75,
        lower_price: float = 0.0,
        ramp_up_price: float = 0.01,
        ramp_down_price: float = 0.01
) -> pd.DataFrame:
    """
    Optimises load shifting to minimise cost of purchasing energy not covered by a PPA at the wholesale spot price.

    The optimisation is performed on each day separately and uses mixed integer linear programing. The optimisation
    proceeds as follows:
        1. A base load profile of inflexible consumption is defined for the day. The base load volume for each interval
        in the day is defined as the quantile of load volume across the month specified by the base_load_quantile, i.e.
        if the base_load_quantile is 0.5, then the base load volume for each interval will be the median consumption of
        the load on corresponding hour of the day across the month. Note, that when calculating the quantile weekdays
        and weekends are considered separately. Further, the base load profile is adjusted, such that if the actual load
        profile for the day is lower than the base load for an interval, then the base load volume is set to the actual
        load volume.
        2. The optimisation uses a decision variable for each interval in the day, to decide how much energy above the
        inflexible base load profile to dispatch in each interval.
        3. The optimisation constrains the variables such that the net load profile (inflexible plus dispatched) cannot
        be greater than the maximum consumption in the timeseries data provided.
        4. The sum of the net load profile (after shifting) is constrained to be equal to the sum load profile (before
        shifting).
        5. The objective function is defined as the cost of purchasing energy for the net load profile that is not
        covered by the PPA (i.e. greater than 'Contracting Load') at the wholesale spot price. Lowering (when the
        inflexible and dispatch are less than the original load), ramping up, and down, are also given costs in the
        objective function. The ramping costs can be used to disincentives the creation of sharp changes in the load
        profile, and the lowering costs can be used to disincentives shifting energy when only small reductions in
        the cost of purchasing energy are available.

    :param timeseries_data: pd.DataFrame containing the load, generation, and price timeseries data. Needs have a
        datetime index and contain the columns 'Load', 'Contracted Load' (the energy being purchased from renewable
        energy generators through the PPA), and a wholesale price column named 'RRP'. The data is assumed to be sorted
        in sequential order and be hourly interval data.
    :param base_load_quantile: float, between 0.0 and 1.0, the quantile used to specify the base load volume for each
        interval.
    :param lower_price: float, $/MWh, the cost in the objective function of lowering the load below the value in the
        original load profile. Used to set a threshold on the savings required before energy shifting should be used.
    :param ramp_up_price: float, $/MWh, the cost in the objective function of ramping the load up, used to disincentives
        sharp changes in the load.
    :param ramp_down_price:float, $/MWh, the cost in the objective function of ramping the load down, used to
        disincentives sharp changes in the load.
    :return:
        pd.Dataframe with a datetime index and the columns (all in MWh):
            'Load dispatch': the energy dispatch above the base load profile
            'Contracted Energy': the volume contracted from renewable energy generators
            'Original load': the load profile before shifting
            'Base load': the inflexible base load profile
            'Firming': the load not met by the PPA after load shifting
            'Raised load': the next increase in load
            'Ramp up': the ramp up in net load (base load plus dispatch) from the previous interval
            'Ramp down': the ramp down in net load (base load plus dispatch) from the previous interval
    """

    # Because time stamps are time ending for period, to make sure the midnight time stamp is assigned to the
    # right day we shift all time stamps back slightly and use this shifted time to filter by date.
    timeseries_data = timeseries_data.copy()
    timeseries_data['DateTime'] = timeseries_data.index.to_series() - timedelta(seconds=1)
    timeseries_data = timeseries_data.set_index('DateTime', drop=True)

    timeseries_data['Weekend'] = np.where(timeseries_data.index.to_series().dt.weekday >= 5, 1, 0)

    results_df = pd.DataFrame(
        columns=['Load dispatch', 'Contracted Energy', 'Original load', 'Base load',
                 'Firming', 'Raised load', 'Ramp up', 'Ramp down']
    )

    daily_load_sums = _get_daily_load_sums(timeseries_data)
    base_weekday, base_weekend = _create_base_days(timeseries_data, base_load_quantile)
    all_time_max_load = timeseries_data['Load'].max(numeric_only=True)

    # run optimisation for each day individually to keep constraints:
    for idx, date in enumerate(daily_load_sums.index):

        data_for_one_day = timeseries_data[timeseries_data.index.date == date.date()].copy()

        if data_for_one_day['Weekend'].values[0] == 0:
            weekday_month = base_weekday[base_weekday.Month == date.month]
            base_day = weekday_month['Load'].values
        else:
            weekend_month = base_weekend[base_weekend.Month == date.month]
            base_day = weekend_month['Load'].values

        if len(data_for_one_day) == 24:
            data_for_one_day['Base Day'] = base_day

            # Use the lower of base_day and load values to form the 'base load' for
            # this day
            data_for_one_day['Base Load'] = np.where(
                data_for_one_day['Base Day'] <= data_for_one_day['Load'],
                data_for_one_day['Base Day'],
                data_for_one_day['Load']
            )

            data_for_one_day['Flexible load'] = (data_for_one_day['Load'] - data_for_one_day['Base Load']).clip(
                lower=0.0)

            # the load sum for this day will be a constraint in optimisation:
            load_sum_for_one_day = daily_load_sums.iloc[idx]

            # Transform all traces to arrays for optimisation:
            original_load = data_for_one_day['Load'].values
            base_load = data_for_one_day['Base Load'].values
            contracted_renewables = data_for_one_day['Contracted Energy'].values
            wholesale_prices = data_for_one_day[f'RRP'].clip(lower=0.0).values

            if sum(data_for_one_day['Flexible load']) > 0:
                # TODO: figure out a better way to catch and add 'missing' days that don't fit the 
                # conditions to actuall shift!!

                # Start setting up the model:
                I = range(len(base_load))

                if advanced_settings.solver == 'GUROBI':
                    solver = GUROBI
                elif advanced_settings.solver == 'CBC':
                    solver = CBC
                else:
                    raise ValueError(f'Solver name {advanced_settings.solver} not recognised.')

                m = Model(solver_name=solver)

                load_dispatch = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=all_time_max_load) for i in I]
                unmatched = [m.add_var(var_type=CONTINUOUS, lb=0.0) for i in I]
                raised_load = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=all_time_max_load) for i in I]
                lowered_load = [m.add_var(var_type=CONTINUOUS, lb=-1 * all_time_max_load, ub=0.0) for i in I]

                # Add 'ramp' constraints applied as a penalty term in the optimisation:
                ramp_up = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=np.inf) for i in I]
                ramp_down = [m.add_var(var_type=CONTINUOUS, lb=-np.inf, ub=0.0) for i in I]

                # Set up objective: to minimise unmatched load and associated cost.
                # Included in the objective are ramp penalties to disincentivise big jumps,
                # and a penalty on raising the load above its original value (small, can be set to 0)
                m.objective = minimize(
                    xsum(
                        (unmatched[i] * wholesale_prices[i] + \
                         - lowered_load[i] * lower_price \
                         + ramp_up[i] * ramp_up_price - \
                         ramp_down[i] * ramp_down_price) for i in I
                    )
                )

                # Add defining constraints to optimisation:
                for i in I:
                    # total load in any hour is the sum of load_dispatch + base_load
                    m += unmatched[i] >= (load_dispatch[i] + base_load[i]) - contracted_renewables[i]

                    # raised load is the positive difference between total load and original load:
                    m += raised_load[i] >= (load_dispatch[i] + base_load[i]) - original_load[i]
                    m += lowered_load[i] <= (load_dispatch[i] + base_load[i]) - original_load[i]

                    # Final constraint on the upper limit of total load in any hour:
                    m += load_dispatch[i] + base_load[i] <= all_time_max_load

                # # Add ramping definition as constraints:
                for j in range(len(base_load) - 1):
                    m += ramp_up[j] >= (load_dispatch[j + 1] + base_load[j + 1]) - (load_dispatch[j] + base_load[j])
                    m += ramp_down[j] <= (load_dispatch[j + 1] + base_load[j + 1]) - (load_dispatch[j] + base_load[j])

                # Add constraint on the sum of daily load (can't change):
                # At the moment: there are no allowances for wiggle room here!!
                m += xsum((load_dispatch[i] + base_load[i]) for i in I) >= load_sum_for_one_day
                m += xsum((load_dispatch[i] + base_load[i]) for i in I) <= load_sum_for_one_day

                # Run the optimisation, suppressing excess outputs:
                m.verbose = 0
                status = m.optimize()

                if status == OptimizationStatus.INFEASIBLE:
                    print('Load shifting optimisation infeasible.')
                    m.clear()

                if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
                    # Get results:
                    dispatch = [load_dispatch[i].x for i in I]
                    firm = [unmatched[i].x for i in I]
                    raised = [raised_load[i].x for i in I]
                    r_up = [ramp_up[i].x for i in I]
                    r_down = [ramp_down[i].x for i in I]

                    day_result = pd.DataFrame(
                        {'Load dispatch': dispatch,
                         'Contracted Energy': contracted_renewables,
                         'Original load': original_load,
                         'Base load': base_load,
                         'Firming': firm,
                         'Raised load': raised,
                         'Ramp up': r_up,
                         'Ramp down': r_down},
                        index=data_for_one_day.index)

                    results_df = pd.concat([results_df, day_result], axis='rows')

                    # Now check the results to make sure that they make sense:
                    day_result['Firm real'] = (
                                (day_result['Load dispatch'] + day_result['Base load']) - day_result['Contracted Energy']).clip(
                        lower=0.0)
                    day_result['Firm check'] = (round(day_result['Firm real'], 3) == round(day_result['Firming'], 3))

                    if not day_result[~day_result['Firm check']].empty:
                        print(day_result)
                        raise ValueError('wrong type of error atm but firming isn\'t right')

                    m.clear()

            else:
                day_result = pd.DataFrame(
                    {'Load dispatch': 0.0, 'Contracted Energy': contracted_renewables, 'Original load': original_load,
                     'Base load': base_load}, index=data_for_one_day.index)
                results_df = pd.concat([results_df, day_result], axis='rows')

        else:
            day_result = pd.DataFrame(
                {'Load dispatch': 0.0, 'Contracted Energy': contracted_renewables, 'Original load': original_load,
                 'Base load': base_load}, index=data_for_one_day.index)
            results_df = pd.concat([results_df, day_result], axis='rows')
    
    results_df['Load with flex'] = results_df['Load dispatch'] + results_df['Base load']

    # Re-adjust index back to original values
    results_df['DateTime'] = results_df.index.to_series() + timedelta(seconds=1)
    results_df = results_df.set_index('DateTime', drop=True)

    return results_df

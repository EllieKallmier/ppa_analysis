# Functions for creating and running a load-shifting optimisation model. 

import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, CONTINUOUS, BINARY, OptimizationStatus

# TODO: add documentation here

def get_daily_load_sums(
        df:pd.DataFrame     # a pandas df that has DateTime index and 'Load' as a column name
) -> pd.DataFrame:
    return df['Load'].copy().resample('D').sum(numeric_only=True)


def create_base_days(
        df:pd.DataFrame,
        flexibility_rating:str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # flex rating percentile dictionary:
    # TODO: find/make sure these match the literature or have a decent justification
    flex_dict = {
        'High': 0.5,
        'Medium' : 0.75,
        'Low' : 0.95
    }

    # validate flex rating here? Or outside of this function??
    if flexibility_rating not in flex_dict:
        raise ValueError(f'flexibility_rating must be one of {flex_dict.keys}')
    
    quant = flex_dict[flexibility_rating]
    
    # First get just the load profile from df:
    load_profile = df[['Load', 'Weekend']].copy()

    all_weekdays_only = load_profile[load_profile['Weekend'] == 0].copy()
    all_weekends_only = load_profile[load_profile['Weekend'] == 1].copy()

    base_weekday = all_weekdays_only.groupby(all_weekdays_only.index.hour)['Load'].quantile(quant).reset_index(drop=True)

    base_weekend = all_weekends_only.groupby(all_weekends_only.index.hour)['Load'].quantile(quant).reset_index(drop=True)
    
    return base_weekday, base_weekend


def daily_load_shifting(
        df:pd.DataFrame,
        flexibility_rating:str,
        load_region:str,            # region that the load is in, to select firming trace
        price_weight:float=0.0,     # weight on firming prices in objective
        raise_price:float=1,    # price on 'raising' load above original value
        ramp_up_price:float=100,    # price on ramp: acts as penalty against extreme ramps.
        ramp_down_price:float=100    # price on ramp: acts as penalty against extreme ramps.
) -> pd.DataFrame:
    
    results_df = pd.DataFrame(columns=['Load dispatch','Contract', 'Original load', 'Base load', 'Firming', 'Raised load', 'Ramp up', 'Ramp down'])

    daily_load_sums = get_daily_load_sums(df)
    base_weekday, base_weekend = create_base_days(df, flexibility_rating)
    all_time_max_load = df['Load'].max(numeric_only=True)


    # run optimisation for each day individually to keep constraints:
    for idx, date in enumerate(daily_load_sums.index):
        data_for_one_day = df[df.index.date == date.date()].copy()
        if data_for_one_day['Weekend'].values[0] == 0:
            base_day = base_weekday.values
        else:
            base_day = base_weekend.values

        data_for_one_day['Base Day'] = base_day

        # Use the lower of base_day and load values to form the 'base load' for
        # this day
        data_for_one_day['Base Load'] = np.where(
            data_for_one_day['Base Day'] <= data_for_one_day['Load'], 
            data_for_one_day['Base Day'], 
            data_for_one_day['Load']
        )

        data_for_one_day['Flexible load'] = (data_for_one_day['Load'] - data_for_one_day['Base Load']).clip(lower=0.0)

        # the load sum for this day will be a constraint in optimisation:
        load_sum_for_one_day = daily_load_sums.iloc[idx]

        # Transform all traces to arrays for optimisation:
        original_load = data_for_one_day['Load'].values
        base_load = data_for_one_day['Base Load'].values
        contracted_renewables = data_for_one_day['Contracted Energy'].values
        firming_prices = data_for_one_day[f'Firming price: {load_region}'].clip(lower=0.0).values

        if sum(data_for_one_day['Flexible load']) > 0:
            # Start setting up the model:
            I = range(len(base_load))
            m = Model()

            load_dispatch = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=all_time_max_load) for i in I]
            unmatched = [m.add_var(var_type=CONTINUOUS, lb=0.0) for i in I]
            raised_load = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=all_time_max_load) for i in I]

            # Add 'ramp' constraints applied as a penalty term in the optimisation:
            ramp_up = [m.add_var(var_type=CONTINUOUS, lb=0.0, ub=np.inf) for i in I]
            ramp_down = [m.add_var(var_type=CONTINUOUS, lb=-np.inf, ub=0.0) for i in I]

            # Set up objective: to minimise unmatched load and associated cost.
            # Included in the objective are ramp penalties to disincentivise big jumps,
            # and a penalty on raising the load above its original value (small, can be set to 0)
            m.objective = minimize(
                xsum(
                    (unmatched[i] + unmatched[i]*firming_prices[i]*price_weight + \
                     raised_load[i]*raise_price + ramp_up[i]*ramp_up_price - \
                        ramp_down[i]*ramp_down_price) for i in I
                )
            )
            
            # Add defining constraints to optimisation:
            for i in I:
                # total load in any hour is the sum of load_dispatch + base_load
                m += unmatched[i] >= (load_dispatch[i] + base_load[i]) - contracted_renewables[i]

                # raised load is the positive difference between total load and original load:
                m += raised_load[i] >= (load_dispatch[i] + base_load[i]) - original_load[i]

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

                day_result = pd.DataFrame({'Load dispatch':dispatch,'Contract': contracted_renewables, 'Original load': original_load, 'Base load': base_load, 'Firming':firm, 'Raised load':raised, 'Ramp up':r_up, 'Ramp down':r_down})

                results_df = pd.concat([results_df, day_result], axis='rows')
            
                # Now check the results to make sure that they make sense:
                day_result['Firm real'] = ((day_result['Load dispatch'] + day_result['Base load']) - day_result['Contract']).clip(lower=0.0)
                day_result['Firm check'] = (round(day_result['Firm real'], 3) == round(day_result['Firming'], 3))

                if not day_result[~day_result['Firm check']].empty:
                    print(day_result)
                    raise ValueError('wrong type of error atm but firming isn\'t right')
                
                m.clear()
        
        else:
            day_result = pd.DataFrame({'Load dispatch':base_load,'Contract': contracted_renewables, 'Original load': original_load, 'Base load': base_load})

            results_df = pd.concat([results_df, day_result], axis='rows')
    
    return results_df


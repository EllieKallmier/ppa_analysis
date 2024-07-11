import numpy as np
import pandas as pd
import pickle
import requests
from datetime import datetime
import os
import json
import warnings
import ast
import math

warnings.filterwarnings("ignore")

# When user selects the area the PV profile is being generated
# Also the load profile of user is being generated (with or without user's inputs)
# And the tariff is selected by user as described in bill_calculator
# a set of sample inputs for this function is provided in Testing_files.py
# This file contains four functions:
#  1- bill_calculator calculates the bill for any load profile and tariff.
#     It now calls two different versions based on the input tariff: one for residential/small business tariffs, and one for large commercial ones.
#     The bill calculator for large commercial tariffs is slightly different from residential bill calculator; hence we separated them for now.
#  2- load_estimator estimates the load profile from demographic info and/or previous usages and/or historical load profile
#  3- battery: estimates the net load for a load + PV + battery based on the tariff
#  - if the tariff is flat rate or block rate, it is maximising the self consumption. i.e. always storing the excess PV in battery
#  - if it is TOU, it is maximising the self consumption but also doesn't discharge the battery until peak time.
#  4- Main function to call these.


# -------------------- Bill Calculator functions for large commercial tariffs --------------------
# It is mostly similar to bill_calculator_residential_small_business; might be unified with or replace it later.

def bill_calculator_large_commercial_tariffs(load_profile, tariff, network_load=None, fit=True):
    load_profile = load_profile[['TS', 'kWh']].copy()
    load_profile.set_index('TS', inplace=True)
    load_profile = load_profile.fillna(0)

    def time_select(load_profile_s, par):
        load_profile_s_t_a = pd.DataFrame()
        for k2_1, v2_1, in par['TimeIntervals'].items():
            if v2_1[0][0:2] == '24':
                v2_1[0] = v2_1[1].replace("24", "00")
            if v2_1[1][0:2] == '24':
                v2_1[1] = v2_1[1].replace("24", "00")
            if v2_1[0] != v2_1[1]:
                load_profile_s_t = load_profile_s.between_time(start_time=v2_1[0], end_time=v2_1[1],
                                                               include_start=False, include_end=True)
            else:
                load_profile_s_t = load_profile_s.copy()

            if not par['Weekday']:
                load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.weekday >= 5].copy()

            if not par['Weekend']:
                load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.weekday < 5].copy()

            load_profile_s_t = load_profile_s_t.loc[load_profile_s_t.index.month.isin(par['Month']), :].copy()

            load_profile_s_t_a = pd.concat([load_profile_s_t_a, load_profile_s_t])
        return load_profile_s_t_a

    # Calculate imports and exports
    results = {}

    Temp_imp = load_profile.values
    Temp_exp = Temp_imp.copy()
    Temp_imp[Temp_imp < 0] = 0
    Temp_exp[Temp_exp > 0] = 0
    load_profile_import = pd.DataFrame(Temp_imp, columns=load_profile.columns, index=load_profile.index)
    load_profile_export = pd.DataFrame(Temp_exp, columns=load_profile.columns, index=load_profile.index)

    results['LoadInfo'] = pd.DataFrame(index=[col for col in load_profile.columns],
                                       data=np.sum(load_profile_import.values, axis=0), columns=['Annual_kWh'])
    if fit:
        results['LoadInfo']['Annual_kWh_exp'] = -1 * np.sum(load_profile_export.values, axis=0)
    # If it is retailer put retailer as a component to make it similar to network tariffs
    if tariff['ProviderType'] == 'Retailer':
        tariff_temp = tariff.copy()
        del tariff_temp['Parameters']
        tariff_temp['Parameters'] = {'Retailer': tariff['Parameters']}
        tariff = tariff_temp.copy()

    for TarComp, TarCompVal in tariff['Parameters'].items():
        results[TarComp] = pd.DataFrame(index=results['LoadInfo'].index)

    # Calculate the FiT
    for TarComp, TarCompVal in tariff['Parameters'].items():
        results[TarComp]['Charge_FiT_Rebate'] = 0


    # Check if daily exists and calculate the charge
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Daily' in TarCompVal.keys():
            num_days = (len(load_profile.index.normalize().unique()) - 1)
            break
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Daily' in TarCompVal.keys():
            results[TarComp]['Charge_Daily'] = num_days * TarCompVal['Daily']['Value']

    # Fixed component for bill
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Fixed' in TarCompVal.keys():
            results[TarComp]['Charge_Fixed'] = TarCompVal['Fixed']['Value']

    # Energy
    # Flat Rate:
    # Check if flat rate charge exists and calculate the charge
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'FlatRate' in TarCompVal.keys():
            results[TarComp]['Charge_FlatRate'] = results['LoadInfo']['Annual_kWh'] * TarCompVal['FlatRate']['Value']


    # Block Annual:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockAnnual' in TarCompVal.keys():
            block_use = results['LoadInfo'][['Annual_kWh']].copy()
            block_use_charge = block_use.copy()
            # separating the blocks of usage
            lim = 0
            for k, v in TarCompVal['BlockAnnual'].items():
                block_use[k] = block_use['Annual_kWh']
                block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                block_use[k] = block_use[k] - lim
                block_use[k][block_use[k] < 0] = 0
                lim = float(v['HighBound'])
                block_use_charge[k] = block_use[k] * v['Value']
            del block_use['Annual_kWh']
            del block_use_charge['Annual_kWh']
            results[TarComp]['Charge_BlockAnnual'] = block_use_charge.sum(axis=1)

    # Block Quarterly:
    # check if it has quarterly and if yes calculate the quarterly energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockQuarterly' in TarCompVal.keys():
            for Q in range(1, 5):
                load_profile_q = load_profile_import.loc[
                                 load_profile_import.index.month.isin(list(range((Q - 1) * 3 + 1, Q * 3 + 1))), :]
                results['LoadInfo']['kWh_Q' + str(Q)] = [
                    np.nansum(load_profile_q[col].values[load_profile_q[col].values > 0])
                    for col in load_profile_q.columns]
            break

    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockQuarterly' in TarCompVal.keys():
            for Q in range(1, 5):
                block_use = results['LoadInfo'][['kWh_Q' + str(Q)]].copy()
                block_use_charge = block_use.copy()
                lim = 0
                for k, v in TarCompVal['BlockQuarterly'].items():
                    block_use[k] = block_use['kWh_Q' + str(Q)]
                    block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                    block_use[k] = block_use[k] - lim
                    block_use[k][block_use[k] < 0] = 0
                    lim = float(v['HighBound'])
                    block_use_charge[k] = block_use[k] * v['Value']
                del block_use['kWh_Q' + str(Q)]
                del block_use_charge['kWh_Q' + str(Q)]
                results[TarComp]['C_Q' + str(Q)] = block_use_charge.sum(axis=1)
            results[TarComp]['Charge_BlockQuarterly'] = results[TarComp][
                ['C_Q' + str(Q) for Q in range(1, 5)]].sum(axis=1)

    # Block Monthly:
    # check if it has Monthly and if yes calculate the Monthly energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockMonthly' in TarCompVal.keys():
            for m in range(1, 13):
                load_profile_m = load_profile_import.loc[load_profile_import.index.month == m, :]
                results['LoadInfo']['kWh_m' + str(m)] = [
                    np.nansum(load_profile_m[col].values[load_profile_m[col].values > 0])
                    for col in load_profile_m.columns]
            break

    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockMonthly' in TarCompVal.keys():
            for Q in range(1, 13):
                block_use = results['LoadInfo'][['kWh_m' + str(Q)]].copy()
                block_use_charge = block_use.copy()
                lim = 0
                for k, v in TarCompVal['BlockMonthly'].items():
                    block_use[k] = block_use['kWh_m' + str(Q)]
                    block_use[k][block_use[k] > float(v['HighBound'])] = float(v['HighBound'])
                    block_use[k] = block_use[k] - lim
                    block_use[k][block_use[k] < 0] = 0
                    lim = float(v['HighBound'])
                    block_use_charge[k] = block_use[k] * v['Value']
                del block_use['kWh_m' + str(Q)]
                del block_use_charge['kWh_m' + str(Q)]
                results[TarComp]['C_m' + str(Q)] = block_use_charge.sum(axis=1)
            results[TarComp]['Charge_BlockMonthly'] = results[TarComp][['C_m' + str(Q) for Q in range(1, 13)]].sum(
                axis=1)

    # Block Daily:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'BlockDaily' in TarCompVal.keys():
            DailykWh = load_profile_import.resample('D').sum()
            block_use_temp_charge = DailykWh.copy()
            block_use_temp_charge.iloc[:, :] = 0
            lim = 0
            for k, v in TarCompVal['BlockDaily'].items():
                block_use_temp = DailykWh.copy()
                block_use_temp[block_use_temp > float(v['HighBound'])] = float(v['HighBound'])
                block_use_temp = block_use_temp - lim
                block_use_temp[block_use_temp < 0] = 0
                lim = float(v['HighBound'])
                block_use_temp_charge = block_use_temp_charge + block_use_temp * v['Value']
            results[TarComp]['Charge_BlockDaily'] = block_use_temp_charge.sum(axis=0)


    # TOU energy
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'TOU' in TarCompVal.keys():
            load_profile_ti = pd.DataFrame()
            load_profile_ti_charge = pd.DataFrame()
            for k, v in TarCompVal['TOU'].items():
                this_part = v.copy()
                if 'Weekday' not in this_part:
                    this_part['Weekday'] = True
                    this_part['Weekend'] = True
                if 'TimeIntervals' not in this_part:
                    this_part['TimeIntervals'] = {'T1': ['00:00', '00:00']}
                if 'Month' not in this_part:
                    this_part['Month'] = list(range(1, 13))
                load_profile_t_a = time_select(load_profile_import, this_part)
                load_profile_ti[k] = load_profile_t_a.sum()
                results[TarComp]['kWh_' + k] = load_profile_ti[k].copy()
                load_profile_ti_charge[k] = this_part['Value'] * load_profile_ti[k]
                results[TarComp]['TOU_' + k] = load_profile_ti_charge[k].copy()
            results[TarComp]['Charge_TOU'] = load_profile_ti_charge.sum(axis=1)

    # Demand charge:
    for TarComp, TarCompVal in tariff['Parameters'].items():
        if 'Demand' in TarCompVal.keys():
            for DemCharComp, DemCharCompVal in TarCompVal['Demand'].items():
                if 'Demand Window Length' not in DemCharCompVal:
                    ts_num = 1
                else:   
                    ts_num = DemCharCompVal['Demand Window Length']  # number of timestamp
                
                if 'Number of Peaks' not in DemCharCompVal:
                    num_of_peaks = 1
                else:
                    num_of_peaks = DemCharCompVal['Number of Peaks']
                if ts_num > 1:
                    load_profile_r = load_profile_import.rolling(ts_num, min_periods=1).mean()
                else:
                    load_profile_r = load_profile_import.copy()
                load_profile_f = time_select(load_profile_r, DemCharCompVal)

                # if capacity charge is applied meaning the charge only applies when you exceed the capacity for
                #  a certain number of times
                if 'Capacity' in DemCharCompVal:
                    # please note the capacity charge only works with user's demand peak (not coincident peak)
                    # Customers can exceed their capacity level on x separate days per month during each interval
                    # (day or night). If they exceed more than x times, they will be charged for the highest
                    # exceedance of their capacity the capacity charge (if they don't exceed) is already included
                    # in the fixed charge so they only pay for the difference
                    capacity = DemCharCompVal['Capacity']['Value']
                    if 'Capacity Exceeded No' in DemCharCompVal:
                        cap_exc_no = DemCharCompVal['Capacity Exceeded No']
                    else:
                        cap_exc_no = 0
                    load_profile_f = load_profile_f - (capacity / 2)
                    load_profile_f = load_profile_f.clip(lower=0)
                    load_profile_f_g = load_profile_f.groupby(load_profile_f.index.normalize()).max()
                    for m in range(1, 13):
                        arr = load_profile_f_g.loc[load_profile_f_g.index.month == m, :].copy().values
                        cap_exc_no_val = np.sum(arr > 0, axis=0)
                        load_profile_f.loc[load_profile_f.index.month == m, cap_exc_no_val <= cap_exc_no] = 0
                    load_profile_f2 = load_profile_f.copy()
                else:
                    load_profile_f2 = load_profile_f.copy()
                based_on_network_peak = False
                if 'Based on Network Peak' in DemCharCompVal:
                    if DemCharCompVal['Based on Network Peak']:
                        based_on_network_peak = True
                # minimum demand or demand charge
                min_dem1 = 0
                min_dem2 = 0
                if 'Min Demand (kW)' in DemCharCompVal:
                    min_dem1 = DemCharCompVal['Min Demand (kW)']
                if 'Min Demand Charge ($)' in DemCharCompVal:
                    if DemCharCompVal['Value'] > 0:
                        min_dem2 = DemCharCompVal['Min Demand Charge ($)'] / DemCharCompVal['Value']
                min_dem = min(min_dem1, min_dem2)
                if based_on_network_peak:
                    new_load = pd.merge(load_profile_f2, network_load, left_index=True, right_index=True)
                    average_peaks_all = np.empty((0, new_load.shape[1] - 1), dtype=float)
                    for m in DemCharCompVal['Month']:
                        new_load2 = new_load.loc[new_load.index.month == m, :].copy()
                        new_load2.sort_values(by='NetworkLoad', inplace=True, ascending=False)
                        average_peaks_all = np.append(average_peaks_all,
                                                      [new_load2.iloc[:num_of_peaks, :-1].values.mean(axis=0)],
                                                      axis=0)
                    average_peaks_all = np.clip(average_peaks_all, a_min=min_dem, a_max=None)
                    average_peaks_all_sum = average_peaks_all.sum(axis=0)
                else:
                    average_peaks_all = np.empty((0, load_profile_f.shape[1]), dtype=float)
                    for m in DemCharCompVal['Month']:
                        arr = load_profile_f.loc[load_profile_f.index.month == m, :].copy().values
                        arr.sort(axis=0)
                        arr = arr[::-1]

                        if arr.size != 0:
                            # arr = np.array([[np.nan]])
                            # print('sliced:')
                            # print(np.nanmean(arr[:num_of_peaks, :], axis=0))



                            # Removed a multiple of 2 here that was used to convert from kWh->kW (script previously used exclusively for half-hourly data)
                            average_peaks_all = np.append(average_peaks_all, [np.nanmean(arr[:num_of_peaks, :], axis=0)], axis=0)
                        
                    average_peaks_all = np.clip(average_peaks_all, a_min=min_dem, a_max=None)

                    average_peaks_all_sum = np.nansum(average_peaks_all, axis=0)
                results[TarComp]['Avg_kW_' + DemCharComp] = average_peaks_all_sum / len(DemCharCompVal['Month'])


                # Sum of all the peak demand amounts for each month (kW*month) * tariff ($/kW/day) * days/months
                # imagine 2 months of data, one 80kW peak and one 40kW peak. * $10/kW/day * days per month.
                results[TarComp]['Demand_' + DemCharComp] = average_peaks_all_sum * DemCharCompVal['Value']*365/12  # the charges in demand charge should be in $/kW/day
                results[TarComp]['Charge_Demand'] = results[TarComp][
                    [col for col in results[TarComp] if col.startswith('Demand_')]].sum(axis=1)
                
                
        ###### NEW BIT #######
        if 'Demand - last 12 Months'in TarCompVal.keys(): 
            for DemCharComp, DemCharCompVal in TarCompVal['Demand - last 12 Months'].items():
                if 'Demand Window Length' not in DemCharCompVal:
                    ts_num = 1
                else:   
                    ts_num = DemCharCompVal['Demand Window Length']  # number of timestamp
                
                if 'Number of Peaks' not in DemCharCompVal:
                    num_of_peaks = 1
                else:
                    num_of_peaks = DemCharCompVal['Number of Peaks']
                if ts_num > 1:
                    load_profile_r = load_profile_import.rolling(ts_num, min_periods=1).mean()
                else:
                    load_profile_r = load_profile_import.copy()

                last_12_months_load = load_profile_r.iloc[-24*2*30*12:, :] #assuming months have 30 days

                load_profile_f = time_select(last_12_months_load, DemCharCompVal)

                max_peak = load_profile_f.max().max()  # Maximum peak across all timestamps
            
                # Calculate the charge based on the maximum peak
                charge = max_peak * DemCharCompVal['Value'] * 365 # Charge in $/kW/day            
                results[TarComp]['Charge_12MonDemand'] = charge

        if 'Demand - last 13 Months'in TarCompVal.keys(): 
            for DemCharComp, DemCharCompVal in TarCompVal['Demand - last 13 Months'].items():
                if 'Demand Window Length' not in DemCharCompVal:
                    ts_num = 1
                else:   
                    ts_num = DemCharCompVal['Demand Window Length']  # number of timestamp
                
                if 'Number of Peaks' not in DemCharCompVal:
                    num_of_peaks = 1
                else:
                    num_of_peaks = DemCharCompVal['Number of Peaks']
                if ts_num > 1:
                    load_profile_r = load_profile_import.rolling(ts_num, min_periods=1).mean()
                else:
                    load_profile_r = load_profile_import.copy()

                last_13_months_load = load_profile_r.iloc[-24*2*30*13:, :] #assuming months have 30 days

                load_profile_f = time_select(last_13_months_load, DemCharCompVal)

                max_peak = load_profile_f.max().max()  # Maximum peak across all timestamps
            
                # Calculate the charge based on the maximum peak
                charge = max_peak * DemCharCompVal['Value'] * 365  # Charge in $/kW/day
                results[TarComp]['Charge_13MonDemand'] = charge

        if "Excess Demand" in TarCompVal.keys(): 
            peaks = []
            for DemCharComp, DemCharCompVal in TarCompVal['Excess Demand'].items():
                if 'Demand Window Length' not in DemCharCompVal:
                    ts_num = 1
                else:   
                    ts_num = DemCharCompVal['Demand Window Length']  # number of timestamp
                
                if 'Number of Peaks' not in DemCharCompVal:
                    num_of_peaks = 1
                else:
                    num_of_peaks = DemCharCompVal['Number of Peaks']
                if ts_num > 1:
                    load_profile_r = load_profile_import.rolling(ts_num, min_periods=1).mean()
                else:
                    load_profile_r = load_profile_import.copy()
                load_profile_f = time_select(load_profile_r, DemCharCompVal)

                ex_average_peaks_all = np.empty((0, load_profile_f.shape[1]), dtype=float)
                for m in DemCharCompVal['Month']:
                    arr = load_profile_f.loc[load_profile_f.index.month == m, :].copy().values
                    arr.sort(axis=0)
                    arr = arr[::-1]
                    ex_average_peaks_all = np.append(ex_average_peaks_all, [arr[:num_of_peaks, :].mean(axis=0)],axis=0)
                    ex_average_peaks_all = np.clip(ex_average_peaks_all, a_min=min_dem, a_max=None)

                peaks.append(ex_average_peaks_all)
            max_peaks = np.max(peaks, axis=0)
            excess_peaks = np.maximum(max_peaks - ex_average_peaks_all, 0)
            average_peaks_all_sum = excess_peaks.sum(axis=0)
            results[TarComp]['Avg_kW_' + DemCharComp] = average_peaks_all_sum / len(DemCharCompVal['Month'])
            results[TarComp]['C_' + DemCharComp] = average_peaks_all_sum * DemCharCompVal['Value']*365/12  # the charges in demand charge should be in $/kW/day
            results[TarComp]['Charge_Excess_Demand'] = results[TarComp][
                [col for col in results[TarComp] if col.startswith('C_')]].sum(axis=1)
            ###### NEW BIT #######


            
    energy_comp_list = ['BlockAnnual', 'BlockQuarterly', 'BlockMonthly', 'BlockDaily', 'FlatRate', 'TOU']
    for k, v in results.items():
        if k != 'LoadInfo':
            results[k]['Bill'] = results[k][[col for col in results[k].columns if col.startswith('Charge')]].sum(axis=1)
            results[k]['energy_charge'] = results[k][[col for col in results[k].columns if (col.startswith('Charge') and col.endswith(tuple(energy_comp_list)))]].sum(axis=1)
    tariff_comp_list = []
    for TarComp, TarCompVal in tariff['Parameters'].items():
        for TarComp2, TarCompVal2 in tariff['Parameters'][TarComp].items():
            tariff_comp_list.append(TarComp2)
    tariff_comp_list = list(set(tariff_comp_list))
    energy_lst = [value for value in tariff_comp_list if value in energy_comp_list]
    return results



# ----------------------- Bill calculator function -----------------------
def bill_calculator(load_profile, tariff, network_load=None, fit=True):
    # Based on the CustomerType of the input tariff decides which bill calculator to call.
     
    return bill_calculator_large_commercial_tariffs(load_profile, tariff, network_load, fit)


# ------------- Add other commercial charges to large commercial tariffs -------------
# Ellie changes: 
# Replaced all initial instances of "+=" with just "=". This way the retail and network
# components can be considered separately, but the underlying structure of the 
# chosen tariff remains the same.
# Also updated the structure for PPA bill purposes: this replaces some values with
# zeroes to avoid double counting network charges in retail bills.
def add_other_charges_to_tariff(tariff, other_charges):
    if 'TOU' in tariff['Parameters']['NUOS']:
        for tou_component_name in tariff['Parameters']['NUOS']['TOU'].keys():
            if 'shoulder' in tou_component_name or 'shoulder' in tou_component_name:
                tariff['Parameters']['NUOS']['TOU'][tou_component_name]["Value"] = \
                    other_charges["Energy Charges"]["Shoulder Rate"]["Value"]
            elif 'Off' in tou_component_name or "off" in tou_component_name or \
                'Non' in tou_component_name or 'non' in tou_component_name:
                tariff['Parameters']['NUOS']['TOU'][tou_component_name]["Value"] = \
                    other_charges["Energy Charges"]["Off-Peak Rate"]["Value"]
            else:
                tariff['Parameters']['NUOS']['TOU'][tou_component_name]["Value"] = \
                    other_charges["Energy Charges"]["Peak Rate"]["Value"]

    if "Demand" not in tariff['Parameters']['NUOS']:
        tariff['Parameters']['NUOS']["Demand"] = {}

    if len(tariff['Parameters']['NUOS']["Demand"]) > 1:
        for demand_component in tariff['Parameters']['NUOS']["Demand"].keys():
            if ('Peak' in demand_component and 'Off' not in demand_component) or 'Non' not in demand_component:
                tariff['Parameters']['NUOS']["Demand"][demand_component]["Value"] = \
                    other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"]
            else:
                tariff['Parameters']['NUOS']["Demand"][demand_component]["Value"] = 0.0
    elif len(tariff['Parameters']['NUOS']["Demand"]) == 1:
        demand_component = next(iter(tariff['Parameters']['NUOS']["Demand"]))
        tariff['Parameters']['NUOS']["Demand"][demand_component]["Value"] = \
            other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"]
    else:
        tariff['Parameters']['NUOS']["Demand"]["Peak"] = {
            "Value": other_charges["Energy Charges"]["Retailer Demand Charge"]["Value"],
            "Unit": "$/kW/Day",
            "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Weekday": True, "Weekend": True,
            "Based on Network Peak": False,
            "Number of Peaks": 1,
            "Demand Window Length": 1,
            "Min Demand (kW)": 0,
            "Min Demand Charge ($)": 0,
            "Day Average": False,
            "TimeIntervals": {"T1": ["00:00", "24:00"]}
        }

    if 'Daily' not in tariff['Parameters']['NUOS']:
        tariff['Parameters']['NUOS']['Daily'] = {"Value": 0.0, "Unit": "$/day"}

    tariff['Parameters']['NUOS']['Daily']['Value'] = \
        other_charges["Metering Charges"]["Meter Provider/Data Agent Charges"]["Value"]
    tariff['Parameters']['NUOS']['Daily']['Value'] += \
        other_charges["Metering Charges"]["Other Meter Charges"]["Value"]

    if 'FlatRate' not in tariff['Parameters']['NUOS']:
        tariff['Parameters']['NUOS']['FlatRate'] = {"Value": 0.0, "Unit": "$/kWh"}

    tariff['Parameters']['NUOS']['FlatRate']['Value'] = \
        other_charges["Environmental Charges"]["LREC Charge"]["Value"]
    tariff['Parameters']['NUOS']['FlatRate']['Value'] += \
        other_charges["Environmental Charges"]["SREC Charge"]["Value"]
    tariff['Parameters']['NUOS']['FlatRate']['Value'] += \
        other_charges["Environmental Charges"]["State Environment Charge"]["Value"]

    tariff['Parameters']['NUOS']['FlatRate']['Value'] = \
        other_charges["AEMO Market Charges"]["AEMO Participant Charge"]["Value"]
    tariff['Parameters']['NUOS']['FlatRate']['Value'] += \
        other_charges["AEMO Market Charges"]["AEMO Ancillary Services Charge"]["Value"]

    tariff['Parameters']['NUOS']['FlatRate']['Value'] = \
        other_charges["Other Variable Charges"]["Other Variable Charge 1"]["Value"]
    tariff['Parameters']['NUOS']['FlatRate']['Value'] += \
        other_charges["Other Variable Charges"]["Other Variable Charge 2"]["Value"]
    tariff['Parameters']['NUOS']['FlatRate']['Value'] += \
        other_charges["Other Variable Charges"]["Other Variable Charge 3"]["Value"]

    if 'Fixed' not in tariff['Parameters']['NUOS']:
        tariff['Parameters']['NUOS']['Fixed'] = {"Value": 0.0, "Unit": "$/Bill"}

    tariff['Parameters']['NUOS']['Fixed']['Value'] = \
        other_charges["Other Fixed Charges"]["Other Fixed Charge 1"]["Value"]
    tariff['Parameters']['NUOS']['Fixed']['Value'] += \
        other_charges["Other Fixed Charges"]["Other Fixed Charge 2"]["Value"]
    tariff['Parameters']['NUOS']['Fixed']['Value'] += \
        other_charges["Other Fixed Charges"]["Total GST"]["Value"]

    return tariff


# ------------- Convert the large commerical network tariffs to a retail tariff structure -------------
def convert_network_tariff_to_retail_tariff(tariff):
    """
    Converts a network tariff with only the NUOS components to the structure the bill calculator expects from a
    retail tariff.
    """
    tariff['Parameters'] = tariff['Parameters']['NUOS']
    tariff['ProviderType'] = 'Retailer'
    return tariff
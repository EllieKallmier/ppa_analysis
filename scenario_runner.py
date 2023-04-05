import pandas as pd
import residuals
import tariffs
import emissions
import hybrid
import ppa
import numpy as np
 
# TODO: add intake for emissions profiles here
# TODO: add field for scaling factor (for generation profile) - default == 1, float
# TODO: incorporate hybrids here? Could change type of 'Generator ID' to match the function call for hybrids? - Or add more than 1 'Generator ID' field and create hybrid here?
# --> idea: add 'Hybrids' input: list of gen ids to hybridise w/ percentages and use Generator ID as the name for new hybrid.


def run_scenario_from_row(scenario_row, price_profiles, load_profiles, charge_set, emissions_profiles):
   """
   Calculate retail and ppa costs for a given row in the scenario table

   :param scenario_row: pandas DataFrame row with following columns 
      'Scenario_ID'
      'PPA', 
      'PPA_Volume',
      'Wholesale_Exposure_Volume',
      'PPA_Price', 
      'Excess_RE_Purchase_Price',
      'Excess_RE_Sale_Price', 
      'LGC_Volume_Type', 
      'LGC_Purhcase_Volume',
      'LGC_Purchase_Price', 
      'Load_MLF', 
      'Load_DLF', 
      'Generator_MLF',
      'Generator_DLF', 
      'Target_Period', 
      'Yearly_Target_MWh', 
      'Yearly_Short_Fall_Penalty_MWh',
      'Yearly_LGC_target_LGC', 
      'Yearly_LGC_short_fall_penalty_LGC', 
      'Average_Wholesale_Price',
      'Wholesale_Price_ID',
      'Load_ID', 
      'Generator_ID',    ---> gen_id should hold a list?
      'Emissions_Region_ID', 
      'Scaling_Period',
      'Scaling_Factor',
      'Scale_to_ID',
      'Hybrid_Profiles',
      'Hybrid_Mix_Name', 
      'TOU_1', 'TOU_2', 'TOU_3', 'TOU_4', 'TOU_5', 'TOU_6', 'TOU_7', 'TOU_8', 'TOU_9', 'TOU_10',
      'Flat_1', 'Flat_2', 'Flat_3', 'Flat_4', 'Flat_5', 'Flat_6', 'Flat_7', 'Flat_8'
   :param price_profiles:
   :param load_profiles: The set of possible load profiles
   :param emissions_profiles: The set of possible emissions profiles
   :param charge_set: The set of retail charge details
   :return: retail_cost, ppa_cost: float
   """

   load_id = scenario_row['Load_ID']
   generator_id = scenario_row['Generator_ID']
   price_id = scenario_row['Wholesale_Price_ID']
   emissions_id = scenario_row['Emissions_Region_ID']
   scaling_period = scenario_row['Scaling_Period']
   scaling_factors = scenario_row['Scaling_Factor']
   scale_to_id = scenario_row['Scale_to_ID']
   hybrids = scenario_row['Hybrid_Profiles']
   hybrid_name = scenario_row['Hybrid_Mix_Names']

   load_profiles['DateTime'] = pd.to_datetime(load_profiles["DateTime"])#, format="%d/%m/%Y %H:%M")
   load_profiles[load_id] = pd.to_numeric(load_profiles[load_id])

   ppa_contract_volume = scenario_row['PPA'] 

   load_profiles[generator_id] = pd.to_numeric(load_profiles[generator_id])
   load_profiles['Average Emissions Intensity'] = pd.to_numeric(emissions_profiles[emissions_id])

   # Add in the scaling function here - scaling has to come before hybrid function
   load_profiles = hybrid.scale_gen_profile(load_profiles, generator_id, ppa_contract_volume, load_id, scaling_period=scaling_period, scaling_factor=scaling_factors, scale_to_id=scale_to_id)

   # Add hybrid function here if flagged
   # TODO: make this more robust.
   if hybrids != None:
      load_profiles = hybrid.create_hybrid(load_profiles, hybrids, hybrid_name)

   price_profiles['DateTime'] = pd.to_datetime(price_profiles['DateTime'])
   price_profiles[price_id] = pd.to_numeric(price_profiles[price_id])


   residual_profiles = residuals.calc(load_profiles=load_profiles, load_id=load_id, generator_id=generator_id)
   retail_costs = tariffs.calc_tou_set(tou_set=charge_set, load_profiles=residual_profiles, contract_type=ppa_contract_volume, wholesale_volume=scenario_row['Wholesale_Exposure_Volume'])
   retail_cost = retail_costs['Cost'].sum()
   ppa_cost = ppa.calc_by_row(scenario_row, price_profiles[price_id], residual_profiles)
   firming_emissions = emissions.firming_emissions_calc(residual_profiles)

   # TODO: add plotting capability here?
   # Consider what to pass: either gen_id or hybrid_name to plot profile of
   # If a hybrid has been made, assume this is what to use?

   return retail_cost, ppa_cost, firming_emissions


def plot_avg_week(residuals, price_profiles, ):

   return


def plot_avg_week_emissions():

   return
import pandas as pd
import residuals
import tariffs
import emissions
import hybrid
import ppa
import numpy as np


def run_scenario_from_row(scenario_row, price_profiles, load_profiles, charge_set, emissions_profiles):
   """
   Calculate retail and ppa costs for a given row in the scenario table

   :param scenario_row: pandas DataFrame row with following columns 
      'Scenario_ID'
      'PPA', 
      'PPA_Volume',
      'Wholesale_Exposure_Volume',
      'PPA_Price', 
      'Floor_Price',
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
      'Generator_ID_1',    ---> can keep adding more of these for each generator you want to hybridise
      'Emissions_Region_ID', 
      'Scaling_Period',
      'Scaling_Factor',
      'Scale_to_ID',
      'Hybrid_Percent_1',  ---> same as above: add more for each generator you want to hybridise
      'Hybrid_Mix_Name', 
      'TOU_1', 'TOU_2', 'TOU_3', 'TOU_4', 'TOU_5', 'TOU_6', 'TOU_7', 'TOU_8', 'TOU_9', 'TOU_10',
      'Flat_1', 'Flat_2', 'Flat_3', 'Flat_4', 'Flat_5', 'Flat_6', 'Flat_7', 'Flat_8'
   :param price_profiles:
   :param load_profiles: The set of possible load profiles
   :param emissions_profiles: The set of possible emissions profiles
   :param charge_set: The set of retail charge details
   :return: retail_cost, ppa_cost: float
   """
   # TODO: add input type and value validation

   load_id = scenario_row['Load_ID']
   generator_id_list = [scenario_row[col] for col in scenario_row.index if ('Generator_ID' in col) & (scenario_row[col] != None)]
   price_id = scenario_row['Wholesale_Price_ID']
   emissions_id = scenario_row['Emissions_Region_ID']
   scaling_period = scenario_row['Scaling_Period']
   scaling_factors = [scenario_row['Scaling_Factor']]
   scale_to_id = scenario_row['Scale_to_ID']
   hybrid_percent_list = [scenario_row[col] for col in scenario_row.index if ('Hybrid_Percent' in col) & (scenario_row[col] != None)]
   hybrid_name = scenario_row['Hybrid_Mix_Name']
   hybrid_name = scenario_row['Hybrid_Mix_Name']

   # Set the gen_id to the hybrid name OR to the first element in the gen id list:
   # TODO: consider adding a flag (make_hybrid = True/False)
   generator_id = ""
   if len(generator_id_list) > 1:
      generator_id = hybrid_name
   else:
      generator_id = generator_id_list[0]

   load_profiles['DateTime'] = pd.to_datetime(load_profiles["DateTime"])
   load_profiles[load_id] = pd.to_numeric(load_profiles[load_id])

   ppa_contract_volume = scenario_row['PPA']
   load_profiles[generator_id_list] = load_profiles[generator_id_list].apply(pd.to_numeric, errors='coerce')
   load_profiles['Average Emissions Intensity'] = pd.to_numeric(emissions_profiles[emissions_id])
   
   # Get the scaled profile to use for this scenario:
   load_profiles = hybrid.scale_gen_profile(load_profiles, generator_id_list, ppa_contract_volume, load_id, scaling_period=scaling_period, scaling_factor=scaling_factors, scale_to_id=scale_to_id)

   # TODO: make this more robust - maybe add flag as input or create flag column (as mentioned above)
   if len(hybrid_percent_list) > 1:
      for percent in hybrid_percent_list:
         percent = float(percent)
      hybrids = list(zip(generator_id_list, hybrid_percent_list))
      load_profiles = hybrid.create_hybrid(load_profiles, hybrids, hybrid_name)

   price_profiles['DateTime'] = pd.to_datetime(price_profiles['DateTime'])
   price_profiles[price_id] = pd.to_numeric(price_profiles[price_id])
   residual_profiles = residuals.calc(load_profiles=load_profiles, load_id=load_id, generator_id=generator_id)
   retail_costs = tariffs.calc_tou_set(tou_set=charge_set, load_profiles=residual_profiles, contract_type=ppa_contract_volume, wholesale_volume=scenario_row['Wholesale_Exposure_Volume'])
   retail_cost = retail_costs['Cost'].sum()
   ppa_cost = ppa.calc_by_row(scenario_row, price_profiles[price_id], residual_profiles)
   firming_emissions = np.sum(residual_profiles['Firming Emissions'])

   # Return the percentage of load matched by each scenario:
   matched_percent = (1 - (np.sum(residual_profiles['Black']) / np.sum(residual_profiles['Load']))) * 100

   # TODO: add plotting capability here? (Now considering creating new file for plotting)

   return retail_cost, ppa_cost, firming_emissions, matched_percent


def plot_avg_week(id_to_plot, residuals, price_profiles):

   # TODO: Plot: Line chart w/ two y axes. One shows energy (kWh) produced/consumed
   # and the other showing avg. wholesale prices in $/MWh


   return


def plot_avg_week_emissions(residuals):

   # TODO: Plot showing an average week of emissions due to firming
   # * and net emissions?

   return
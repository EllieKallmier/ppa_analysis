import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nemed
from nemed.downloader import download_current_aemo_cdeii_summary
import pprint
import ppa, residuals, tariffs

# format data types for load profile csv input:
type_dict = {
    'DateTime' : 'str',
    'total_load' : 'float',
    'Wind' : 'float',
    'Solar' : 'float',
    '20/80' : 'float',
    '30/70' : 'float',
    '40/60' : 'float',
    '50/50' : 'float',
    '60/40' : 'float',
    '70/30' : 'float',
    '80/20' : 'float'
}
parse_dates = ['DateTime']

# 1. One year of load and generation data
profiles = pd.read_csv('data/cleaned_profiles.csv', dtype=type_dict, parse_dates=parse_dates)

# 2. One year of wholesale price data
price_data = pd.read_csv('data/qld_prices_2021.csv')
price_data['ts'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
price_data = price_data.set_index('ts').resample('30min').mean()
price_data['DateTime'] = pd.date_range("2021-01-01 00:30", periods=17521, freq="30min")
price_data = price_data.reset_index()
prices = price_data[['DateTime', 'RRP']]

# 3. Calculate the residual profiles need for PPA and tariff calculations
residuals_dict = {}
re_gen_ids = ['wind', 'Solar', '20/80', '30/70', '40/60', '50/50', '60/40', '70/30', '80/20']
for id in re_gen_ids:
    residual_profiles = residuals.calc(profiles, load_id='total_load', generator_id=id)
    resi = pd.DataFrame(data=residual_profiles).set_index('DateTime')
    residuals_dict[id] = resi

#resi.to_csv('data/residuals.csv')
# print('\n Load profiles used for cost calculations:')
# print(residual_profiles)

# 3(a). Plot Residuals
for id, data in residuals_dict.items():
    day_one = data.head(48)
    plt.plot(day_one.index, day_one['Black'], label=id)

plt.legend()
plt.show()

# 3(b). Find and plot total firming energy under different RE generator profiles
total_load_2021 = np.sum(profiles['total_load'])
firming_by_re_gen = {}
#percent_load_not_covered = {}
for id, data in residuals_dict.items():
    total_firming = np.sum(residuals_dict[id]['Black'])
    firming_by_re_gen[id] = total_firming / 1000
    #percent_load_not_covered[id] = total_firming / total_load_2021

plt.bar(range(len(firming_by_re_gen)), list(firming_by_re_gen.values()), align='center')
plt.xticks(range(len(firming_by_re_gen)), list(firming_by_re_gen.keys()))
plt.xlabel('RE Generator')
plt.ylabel('Black Energy (MWh)')
plt.show()

# 4. Calculate PPA costs
ppa_costs = ppa.calc(contract_type='Off-site - Contract for Difference',
                     ppa_volume='RE Uptill Load',  # In each 30 min interval the PPA volume is the lesser of generator
                                                   # or load volume.
                     contract_price=50.0,
                     wholesale_volume='RE Uptill Load',  # In each 30 min interval the volume bought from the wholesale
                                                         # market is less of generator or load volume.
                     residual_profiles=residual_profiles,
                     price_profile=prices['RRP']
                     )
print('\n PPA cost summary:')
pprint.pprint(ppa_costs)

# 5. Define tariffs
applicable_tariffs = pd.DataFrame({
    'Charge name': ['peak', 'off_peak'],
    'Charge Type': ['Energy', 'Energy'],
    'Volume Type': ['Energy ($/MWh)', 'Energy ($/MWh)'],
    'Rate': [85.0,65.0],
    'MLF': [1.0, 1.0],
    'DLF': [1.0, 1.0],
    'Start Month': [1, 1],
    'End Month': [12, 12],
    'Start Weekday': [1, 1],
    'End Weekday': [7, 7],
    'Start Hour': [7, 21],
    'End Hour': [20, 6]
})

# 6. Calculate Tariff costs
tariff_costs = tariffs.calc_tou_set(tou_set=applicable_tariffs, contract_type='Off-site - Contract for Difference',
                                    wholesale_volume='RE Uptill Load', load_profiles=residual_profiles)

print('\n Tariffs and costs summary:')
print(tariff_costs)
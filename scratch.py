# And then we can run the load shifting optimisation.
import pandas as pd
from ppa_analysis import load_flex

time_series_data_with_battery = pd.read_parquet('time_series_data_with_battery.parquet')



time_series_data_with_battery = load_flex.daily_load_shifting(
        timeseries_data=time_series_data_with_battery,
        base_load_quantile=0.75,
        lower_price=0.0,
        ramp_up_price=0.01,
        ramp_down_price=0.01
)
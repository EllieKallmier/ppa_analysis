import pandas as pd
import numpy as np
from nemseer import compile_data, download_raw_data, generate_runtimes

# GET PRE-DISPATCH PRICES USING NEMSEER
# This is just for paper purposes and will not form part of the tool.
# October 2022 is missing from AEMO database so collected and shared by Dylan McConnell.
# Credit Abhijith Prakash for examples of using NEMSEER, including function below.


def predispatch_prices_for_paper():
    GENERATOR_REGION = 'QLD1'
    LOAD_REGION = 'NSW1'
    analysis_start = "2019/01/01 00:30:00"
    analysis_end = "2023/12/31 23:30:00"

    def get_forecast_price_data(ftype: str) -> pd.DataFrame:
        """
        Get price forecast data for the analysis period given a particular forecast type

        Args:
            ftype: 'P5MIN' or 'PREDISPATCH'
        Returns:
            DataFrame with price forecast data
        """
        # ftype mappings
        table = {"PREDISPATCH": "PRICE", "P5MIN": "REGIONSOLUTION"}
        run_col = {"PREDISPATCH": "PREDISPATCH_RUN_DATETIME", "P5MIN": "RUN_DATETIME"}
        forecasted_col = {"PREDISPATCH": "DATETIME", "P5MIN": "INTERVAL_DATETIME"}
        # get run times
        forecasts_run_start, forecasts_run_end = generate_runtimes(
            analysis_start, analysis_end, ftype
        )
        df = compile_data(
            forecasts_run_start,
            forecasts_run_end,
            analysis_start,
            analysis_end,
            ftype,
            table[ftype],
            "nemseer_cache/",
        )[table[ftype]]
        # remove intervention periods
        df = df.query("INTERVENTION == 0")
        # rename run and forecasted time cols
        df = df.rename(
            columns={
                run_col[ftype]: "run_time",
                forecasted_col[ftype]: "forecasted_time",
            }
        )
        # ensure values are sorted by forecasted and run times for nth groupby operation
        return df[["run_time", "forecasted_time", "REGIONID", "RRP"]].sort_values(
            ["forecasted_time", "run_time"]
        )

    # Collect predispatch data up to October 2022:
    analysis_start = "2019/01/01 00:30:00"
    analysis_end = "2022/09/30 23:30:00"

    nemseer_raw_data = download_raw_data(
        "PREDISPATCH",
        "PRICE",
        "nemseer_cache/",
        forecasted_start=analysis_start,
        forecasted_end=analysis_end,
    )

    forecast_data_before = get_forecast_price_data('PREDISPATCH')

    # Collect predispatch data after October 2022:
    analysis_start = "2022/11/30 00:30:00"
    analysis_end = "2023/12/31 23:30:00"

    nemseer_raw_data = download_raw_data(
        "PREDISPATCH",
        "PRICE",
        "nemseer_cache/",
        forecasted_start=analysis_start,
        forecasted_end=analysis_end,
    )

    forecast_data_after = get_forecast_price_data('PREDISPATCH')

    forecast_data_before['Diff'] = forecast_data_before['forecasted_time'] - forecast_data_before['run_time']
    forecast_data_before['Diff'] = forecast_data_before['Diff'].dt.total_seconds() / 60 / 60
    forecast_data_before = forecast_data_before[forecast_data_before['Diff'] == 1.0].drop(columns=['Diff', 'run_time'])


    forecast_data_after['Diff'] = forecast_data_after['forecasted_time'] - forecast_data_after['run_time']
    forecast_data_after['Diff'] = forecast_data_after['Diff'].dt.total_seconds() / 60 / 60
    forecast_data_after = forecast_data_after[forecast_data_after['Diff'] == 1.0].drop(columns=['Diff', 'run_time'])


    forecast_data_before = forecast_data_before.pivot_table(values='RRP', columns='REGIONID', index='forecasted_time')
    forecast_data_after = forecast_data_after.pivot_table(values='RRP', columns='REGIONID', index='forecasted_time')


    filepath = '/Users/elliekallmier/Desktop/RA_Work/247_TRUZERO/247_ppa/PREDISPATCH_REGION_PRICES-OCT22-NOV22.csv'

    forecast_data_oct = pd.read_csv(filepath, parse_dates=['DATETIME'])
    forecast_data_oct = forecast_data_oct[forecast_data_oct['PERIODID'] == 2]
    forecast_data_oct = forecast_data_oct.rename(columns={'DATETIME':'forecasted_time'})

    forecast_data_oct['REGIONID'] = forecast_data_oct['REGIONID'].map({val : f'{val}1' for val in forecast_data_oct['REGIONID'].unique()})

    forecast_data_oct = forecast_data_oct.pivot_table(values='RRP', columns='REGIONID', index='forecasted_time')

    # Concat all 3 predispatch dfs together:
    all_predispatch_data = pd.concat([forecast_data_before, forecast_data_oct, forecast_data_after], axis='rows')
    all_predispatch_data = all_predispatch_data.resample('H').mean(numeric_only=True)
    all_predispatch_data = all_predispatch_data.rename(columns={col : f'Predispatch: {col}' for col in all_predispatch_data.columns})


    chosen_predispatch = all_predispatch_data[[col for col in all_predispatch_data.columns if col[-4:] in [LOAD_REGION, GENERATOR_REGION]]]

    chosen_predispatch = chosen_predispatch.reset_index().rename(columns={'forecasted_time':'DateTime'}).set_index('DateTime')
    chosen_predispatch = chosen_predispatch[chosen_predispatch.index >= '2019-01-01 23:00:00']

    return chosen_predispatch

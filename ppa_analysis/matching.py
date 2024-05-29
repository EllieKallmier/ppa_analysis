import pandas as pd
import numpy as np


# Functions to calculate the other metrics: matching % hourly, annually, against hybrid and contracted traces, emissions outcomes

def calc_hourly_match(
        df: pd.DataFrame,
        col_to_match_to: str,
        resample_period: str,
        load_region: str
) -> pd.DataFrame:
    matching = df.copy()
    matching['Hourly Match %'] = 0
    matching['Hourly Match %'] = np.where(matching['Load'] == 0, 100,
                                          np.minimum(matching[col_to_match_to] / matching['Load'] * 100, 100))

    matching = matching.resample(resample_period).mean(numeric_only=True)

    return matching['Hourly Match %']


def calc_bulk_match(
        df: pd.DataFrame,
        col_to_match_to: str,
        resample_period: str,
        load_region: str
) -> pd.DataFrame:
    resampled = df.resample(resample_period).sum(numeric_only=True).copy()
    resampled['Bulk Match %'] = resampled[col_to_match_to] / resampled['Load'] * 100

    return resampled['Bulk Match %']


def calc_unmatched_emissions(
        df: pd.DataFrame,
        col_to_match_to: str,
        resample_period: str,
        load_region: str
) -> pd.DataFrame:
    emissions_df = df.copy()
    emissions_df['Emissions'] = (emissions_df['Load'] - emissions_df[col_to_match_to]).clip(lower=0.0) * emissions_df[
        f'AEI: {load_region}']
    emissions_df = emissions_df.resample(resample_period).sum(numeric_only=True)

    return emissions_df['Emissions']

# calc_unmatched_emissions(combined_data_firming, 'Contracted Energy', 'Y', LOAD_REGION)
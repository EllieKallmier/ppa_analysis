import numpy as np
import pandas as pd

# Functions to calculate the other metrics: matching % hourly, annually, against hybrid and contracted traces, emissions outcomes


def calc_hourly_match(
    df: pd.DataFrame, col_to_match_to: str, resample_period: str, load_region: str
) -> pd.Series:
    """Calculates each row's match % (capped at 100%), then averages over resample_period.

    Match % is Load == 0 -> 100%, otherwise min(col_to_match_to / Load * 100, 100). Because
    the cap is applied per row before resampling, this is a mean of already-capped values —
    reflecting the mean of the actual hourly match achieved in a resample_period.

    I/O Example:
        df (resample_period='D'):
            DateTime           Load  Contracted Energy
            2026-01-01 00:00   100   50                  # 50%
            2026-01-01 12:00   100   150                 # 150% uncapped, capped to 100%

        returns:
            DateTime    Hourly Match %
            2026-01-01  75.0                             # mean(50, 100)
    """
    # TODO: load_region is accepted but never used in this function — check whether it's
    # leftover from a simplified implementation or was intended for future use.
    matching = df.copy()
    matching["Hourly Match %"] = 0
    matching["Hourly Match %"] = np.where(
        matching["Load"] == 0,
        100,
        np.minimum(matching[col_to_match_to] / matching["Load"] * 100, 100),
    )

    avg_hourly_match = (
        matching["Hourly Match %"].resample(resample_period).mean(numeric_only=True)
    )

    return avg_hourly_match.copy()


def calc_bulk_match(
    df: pd.DataFrame, col_to_match_to: str, resample_period: str, load_region: str
) -> pd.Series:
    """Sums Load and col_to_match_to over resample_period first, then divides (uncapped).

    Contrast with calc_hourly_match, which caps each row at 100% before resampling. Here the
    ratio is taken after summing, so it can exceed 100% and won't match calc_hourly_match's
    output on the same input. Reflects the 'bulk' match over the resample_period, comparing
    total energy volumes instead of time-stamped volumes.

    I/O Example:
        df (resample_period='D'):
            DateTime           Load  Contracted Energy
            2026-01-01 00:00   100   50
            2026-01-01 12:00   100   150

        returns:
            DateTime    Bulk Match %
            2026-01-01  100.0                  # sum(200) / sum(200) * 100
    """
    # TODO: load_region is accepted but never used in this function — check whether it's
    # leftover from a simplified implementation or was intended for future use.
    resampled = (
        df[["Load", col_to_match_to]]
        .resample(resample_period)
        .sum(numeric_only=True)
        .copy()
    )
    resampled["Bulk Match %"] = resampled[col_to_match_to] / resampled["Load"] * 100

    return resampled["Bulk Match %"].copy()


def calc_unmatched_emissions(
    df: pd.DataFrame, col_to_match_to: str, resample_period: str, load_region: str
) -> pd.Series:
    """Emissions from load left unmatched by col_to_match_to, summed over resample_period.

    Per row: max(Load - col_to_match_to, 0) * AEI, using the f"AEI: {load_region}" column.
    Clipped at 0 so a surplus (col_to_match_to > Load) never produces negative emissions.

    I/O Example:
        df (resample_period='D', load_region='NSW1'):
            DateTime           Load  Contracted Energy  AEI: NSW1
            2026-01-01 00:00   100   60                 0.5        # shortfall 40 -> 20
            2026-01-01 12:00   100   120                0.5        # surplus -> clipped to 0

        returns:
            DateTime    Emissions
            2026-01-01  20.0
    """
    emissions_df = df.copy()
    emissions_df["Emissions"] = (
        emissions_df["Load"] - emissions_df[col_to_match_to]
    ).clip(lower=0.0) * emissions_df[f"AEI: {load_region}"]
    total_emissions = (
        emissions_df["Emissions"].resample(resample_period).sum(numeric_only=True)
    )

    return total_emissions.copy()

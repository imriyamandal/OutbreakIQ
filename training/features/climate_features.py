import numpy as np
def create_climate_features(df):

    climate_cols = [ "temp", "precipitation", "lai", ]
    for col in climate_cols:

        df[f"{col}_change"] = ( df.groupby(["state_ut", "district"] )[col] .diff())
    return df


def create_seasonality_features(df):
    df["month_sin"] = np.sin( 2 * np.pi * df["month"] / 12 )
    df["month_cos"] = np.cos( 2 * np.pi * df["month"] / 12 )
    return df


def create_trend_features(df):
    df["time_index"] = ( df["year"] * 12 + df["month"] )
    return df


def create_interaction_features(df):
    df["temp_precip_interaction"] = ( df["temp"] * df["precipitation"] )
    return df   
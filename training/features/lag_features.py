from .constants import LAGS
def create_lag_features(df):
    group = df.groupby( ["state_ut", "district", "disease"] )["cases"]

    for lag in LAGS:
        df[f"lag_{lag}"] = group.shift(lag)
    return df


def create_rolling_features(df):
    group = df.groupby( ["state_ut", "district", "disease"] )["cases"]

    df["roll_3_mean"] = ( group.transform( lambda x: x.shift(1).rolling(3).mean()))

    df["roll_6_mean"] = ( group.transform( lambda x: x.shift(1).rolling(6).mean()))

    df["roll_3_std"] = ( group.transform( lambda x: x.shift(1).rolling(3).std()))

    df["roll_6_std"] = ( group.transform( lambda x: x.shift(1).rolling(6).std()))
    return df
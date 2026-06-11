def create_epidemic_features(df):

    df["spike_ratio"] = ( df["cases"] / (df["roll_3_mean"] + 1) )
    df["acceleration"] = ( df["lag_1"] - df["lag_2"] )
    df["yoy_change"] = ((df["lag_1"] - df["lag_12"]) / (df["lag_12"] + 1) )
    df["growth_rate"] = ( (df["lag_1"] - df["lag_3"]) / (df["lag_3"] + 1) )
    return df


def create_outbreak_frequency(df):
    threshold = ( df["cases"] .quantile(0.75) )

    df["outbreak_flag"] = ( df["cases"] > threshold ).astype(int)

    df["outbreak_frequency_12m"] = ( df.groupby( ["state_ut", "district", "disease"] )["outbreak_flag"].transform( lambda x: x.shift(1).rolling(12).sum()))
    return df
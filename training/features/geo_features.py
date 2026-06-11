import pandas as pd
def create_geo_features(df):

    state_districts = ( df.groupby("state_ut")["district"].nunique().to_dict())

    df["district_count"] = ( df["state_ut"].map(state_districts))
    return df


def create_spread_features(df):

    temp = df.copy()
    temp["is_affected"] = ( temp["cases"] > 0 ).astype(int)

    spread = (temp.groupby( ["state_ut", "disease", "date"] )["is_affected"].sum().reset_index())

    spread.columns = [ "state_ut", "disease","date","spread_raw",]

    spread["geo_spread"] = ( spread.groupby( ["state_ut", "disease"] )["spread_raw"].shift(1))

    spread["geo_spread_prev2"] = ( spread.groupby( ["state_ut", "disease"] )["spread_raw"].shift(2))

    spread["district_expansion"] = (spread["geo_spread"] - spread["geo_spread_prev2"])

    df = pd.merge(
        df,
        spread[
            [
                "state_ut",
                "disease",
                "date",
                "geo_spread",
                "district_expansion",
            ]
        ],
        on=["state_ut", "disease", "date"],
        how="left",
    )

    return df
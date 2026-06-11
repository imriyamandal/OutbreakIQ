import logging
import pandas as pd
import numpy as np

from training.preprocessing.validation import validate_columns
from training.features.constants import ( REQUIRED_COLUMNS, NUMERIC_COLUMNS, )
from training.features.lag_features import ( create_lag_features, create_rolling_features, )
from training.features.climate_features import ( create_climate_features, create_seasonality_features, create_trend_features, create_interaction_features,)
from training.features.epidemic_features import ( create_epidemic_features, create_outbreak_frequency, )
from training.features.geo_features import ( create_geo_features, create_spread_features,)
from training.utils.config import ( PROCESSED_DATA_PATH, FINAL_DATASET_PATH, )

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", )
logger = logging.getLogger(__name__)


def main():
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info( f"Loading dataset from: {PROCESSED_DATA_PATH}" )
    df = pd.read_csv(PROCESSED_DATA_PATH)

    df.columns = ( df.columns.str.strip().str.lower())
    validate_columns( df, REQUIRED_COLUMNS,)

    logger.info( f"Initial Dataset Shape: {df.shape}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric( df[col], errors="coerce",)

    df = df.sort_values(
        by=[  "state_ut",  "district", "disease", "year", "month", ] )
    df["date"] = pd.to_datetime( df["year"].astype(str) + "-" + df["month"].astype(str) + "-01", errors="coerce",)

    logger.info("Date column created")
    logger.info("Creating lag features")

    df = create_lag_features(df)

    logger.info("Creating rolling statistics")
    df = create_rolling_features(df)

    logger.info("Creating climate features")
    df = create_climate_features(df)

    logger.info("Creating seasonality features")
    df = create_seasonality_features(df)

    logger.info("Creating trend features")
    df = create_trend_features(df)

    logger.info("Creating interaction features")
    df = create_interaction_features(df)

    logger.info("Creating epidemic features")
    df = create_epidemic_features(df)

    logger.info("Creating outbreak frequency features")
    df = create_outbreak_frequency(df)

    logger.info("Creating geographic features")
    df = create_geo_features(df)

    logger.info("Creating spread features")
    df = create_spread_features(df)

    logger.info("Creating target variables")
    df["log_cases"] = np.log1p(df["cases"])

    threshold = df["cases"].quantile(0.75)

    df["outbreak"] = ( df["cases"] > threshold ).astype(int)

    logger.info("Handling missing values")
    numeric_columns = df.select_dtypes( include=["number"]).columns

    df[numeric_columns] = ( df[numeric_columns].fillna(0))

    before_rows = len(df)

    df.drop_duplicates( inplace=True )

    removed_rows = ( before_rows - len(df) )

    logger.info( f"Removed {removed_rows} duplicate rows" )
    logger.info( f"Final Dataset Shape: {df.shape}" )
    logger.info( f"Total Features Generated: {len(df.columns)}" )

    FINAL_DATASET_PATH.parent.mkdir( parents=True, exist_ok=True, )

    df.to_csv( FINAL_DATASET_PATH, index=False,)

    logger.info(f"Dataset saved successfully to:")
    logger.info(f"{FINAL_DATASET_PATH}")
    logger.info("FEATURE ENGINEERING COMPLETED")

if __name__ == "__main__":
    main()
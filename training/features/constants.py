LAGS = [1, 2, 3, 6, 12]

ROLLING_WINDOWS = [3, 6]

NUMERIC_COLUMNS = [
    "cases",
    "deaths",
    "temp",
    "precipitation",
    "lai",
    "month",
    "year",
    "latitude",
    "longitude",
]

REQUIRED_COLUMNS = [
    "state_ut",
    "district",
    "disease",
    "year",
    "month",
    "cases",
]

GROUP_COLS = [
    "state_ut",
    "district",
    "disease",
]
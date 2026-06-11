from pydantic import BaseModel

class PredictionRequest(BaseModel):
    day:int
    month:int
    year:int

    latitude:float
    longitude:float

    precipitation:float
    lai:float
    temp:float

    death_rate:float

    lag_1:float
    lag_2:float
    lag_3:float
    lag_6:float
    lag_12:float

    roll_3_mean:float
    roll_6_mean:float
    roll_3_std:float
    roll_6_std:float

    month_sin:float
    month_cos:float

    temp_change:float
    precipitation_change:float
    lai_change:float

    spike_ratio:float
    acceleration:float
    growth_rate:float

    outbreak_frequency_12m:float

    state_ut:str
    district:str
    disease:str
    disease_category:str
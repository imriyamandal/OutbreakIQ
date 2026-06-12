from pydantic import BaseModel
from typing import Optional
class PredictionRequest(BaseModel):

    state_ut: str
    district: str
    disease: str
    disease_category: Optional[str] = None

    day: int
    month: int
    year: int

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    precipitation: float
    lai: float
    temp: float
    death_rate: float = 0.0

    lag_1: float = 0.0
    lag_2: float = 0.0
    lag_3: float = 0.0
    lag_6: float = 0.0
    lag_12: float = 0.0

    roll_3_mean: float = 0.0
    roll_6_mean: float = 0.0

    roll_3_std: float = 0.0
    roll_6_std: float = 0.0

    month_sin: float = 0.0
    month_cos: float = 0.0

    temp_change: float = 0.0
    precipitation_change: float = 0.0
    lai_change: float = 0.0

    spike_ratio: float = 0.0
    acceleration: float = 0.0
    growth_rate: float = 0.0

    outbreak_frequency_12m: float = 0.0
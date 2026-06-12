import joblib
import pandas as pd
import numpy as np
import shap
import logging
from pathlib import Path
from typing import Dict, Any

from training.models.predict import DiseasePredictor, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_COLS

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        self.predictor = DiseasePredictor()
        
        # Load historical data to autocomplete lag features
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_path = self.base_dir / "ml" / "data" / "final" / "ml_data.csv"
        if self.data_path.exists():
            try:
                self.df_history = pd.read_csv(self.data_path)
                self.df_history.columns = self.df_history.columns.str.strip().str.lower()
                logger.info(f"Predictor loaded {len(self.df_history)} rows of historical data.")
            except Exception as e:
                logger.error(f"Error loading historical data in Predictor: {e}")
                self.df_history = pd.DataFrame()
        else:
            logger.warning(f"Historical data not found at {self.data_path}")
            self.df_history = pd.DataFrame()

        # Initialize SHAP explainer on the regressor model
        try:
            self.shap_explainer = shap.TreeExplainer(self.predictor.regressor)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Could not initialize SHAP Explainer: {e}")
            self.shap_explainer = None

    def autocomplete_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich input data with historical lag features, coordinates, and seasonal features.
        """
        enriched = data.copy()
        
        state_ut = enriched.get("state_ut", "")
        district = enriched.get("district", "")
        disease = enriched.get("disease", "")
        month = int(enriched.get("month", 1))
        year = int(enriched.get("year", 2024))
        
        # 1. Fill coordinates if missing or default
        lat = float(enriched.get("latitude", 0.0) or 0.0)
        lon = float(enriched.get("longitude", 0.0) or 0.0)
        
        # 2. Look up latest history for lags
        history_row = None
        filtered = pd.DataFrame()
        if not self.df_history.empty:
            filtered = self.df_history[
                (self.df_history["state_ut"].str.lower() == state_ut.lower()) &
                (self.df_history["district"].str.lower() == district.lower()) &
                (self.df_history["disease"].str.lower() == disease.lower())
            ]
            if not filtered.empty:
                filtered_sorted = filtered.sort_values(by=["year", "month", "day"])
                history_row = filtered_sorted.iloc[-1]
                
                if lat == 0.0:
                    lat = float(history_row.get("latitude", 22.9734))
                if lon == 0.0:
                    lon = float(history_row.get("longitude", 78.6569))

        enriched["latitude"] = lat
        enriched["longitude"] = lon

        # 3. Fill disease category
        disease_lower = disease.strip().lower()
        category_mapping = {
            "acute diarrhoeal disease": "Diarrheal Diseases",
            "acute encephalitis syndrome": "Encephalitis",
            "acute gastroenteritis": "Diarrheal Diseases",
            "chikungunya": "Chikungunya",
            "chikungunya/ dengue": "Dengue-Chikungunya",
            "chikungunya/dengue": "Dengue-Chikungunya",
            "cholera": "Diarrheal Diseases",
            "dengue": "Dengue",
            "dengue and chikungunya": "Dengue-Chikungunya",
            "dengue and malaria": "Dengue And Malaria",
            "dengue chikungunya": "Dengue-Chikungunya",
            "dengue fever": "Dengue",
            "dengue/chikungunya": "Dengue-Chikungunya",
            "diarrhea": "Diarrheal Diseases",
            "gastroenteritis": "Diarrheal Diseases",
            "malaria": "Malaria",
            "malaria (pv)": "Malaria",
            "suspected chikungunya": "Chikungunya",
            "suspected cholera": "Diarrheal Diseases",
            "suspected dengue": "Dengue",
            "suspected dengue and chikungunya": "Dengue-Chikungunya",
            "pyrexia of unknown origin": "Fever of Unknown Origin"
        }
        
        if disease_lower in category_mapping:
            enriched["disease_category"] = category_mapping[disease_lower]
        else:
            if "diarrh" in disease_lower or "gastro" in disease_lower or "cholera" in disease_lower:
                enriched["disease_category"] = "Diarrheal Diseases"
            elif "dengue" in disease_lower and "chikungunya" in disease_lower:
                enriched["disease_category"] = "Dengue-Chikungunya"
            elif "dengue" in disease_lower and "malaria" in disease_lower:
                enriched["disease_category"] = "Dengue And Malaria"
            elif "dengue" in disease_lower:
                enriched["disease_category"] = "Dengue"
            elif "malaria" in disease_lower:
                enriched["disease_category"] = "Malaria"
            elif "chikungunya" in disease_lower:
                enriched["disease_category"] = "Chikungunya"
            elif "encephalitis" in disease_lower:
                enriched["disease_category"] = "Encephalitis"
            elif "pyrexia" in disease_lower or "fever" in disease_lower:
                enriched["disease_category"] = "Fever of Unknown Origin"
            else:
                enriched["disease_category"] = history_row.get("disease_category", "Dengue") if history_row is not None else "Dengue"

        # 4. Compute temporal features
        enriched["month_sin"] = float(np.sin(2 * np.pi * month / 12))
        enriched["month_cos"] = float(np.cos(2 * np.pi * month / 12))

        # 5. Populate Lags and Rolling statistics
        if history_row is not None:
            last_year = int(history_row.get("year", 2022))
            last_month = int(history_row.get("month", 6))
            is_future = (year > last_year) or (year == last_year and month > last_month)
            
            if is_future:
                # We iteratively project future lag features up to the target date!
                filtered_sorted = filtered.sort_values(by=["year", "month", "day"])
                cases_seq = list(filtered_sorted.tail(12)["cases"].values)
                overall_cases = filtered_sorted["cases"].mean() if len(filtered_sorted) > 0 else 0.0
                cases_seq = [overall_cases] * (12 - len(cases_seq)) + cases_seq
                
                # Seasonal monthly average & trend
                monthly_avg = filtered_sorted.groupby("month")["cases"].mean().to_dict()
                recent_cases = filtered_sorted.tail(6)["cases"].mean() if len(filtered_sorted) >= 6 else overall_cases
                trend_factor = (recent_cases / overall_cases) if overall_cases > 0 else 1.0
                trend_factor = np.clip(trend_factor, 0.5, 2.0)
                
                months_diff = (year - last_year) * 12 + (month - last_month)
                curr_month = last_month
                curr_year = last_year
                for _ in range(months_diff):
                    curr_month += 1
                    if curr_month > 12:
                        curr_month = 1
                        curr_year += 1
                    
                    base_cases = monthly_avg.get(curr_month, overall_cases)
                    pred_val = base_cases * trend_factor
                    pred_val = max(0.0, pred_val)
                    cases_seq.append(pred_val)
                
                lag_1 = float(cases_seq[-1])
                lag_2 = float(cases_seq[-2])
                lag_3 = float(cases_seq[-3])
                lag_6 = float(cases_seq[-6])
                lag_12 = float(cases_seq[-12])
                
                roll_3_mean = float(np.mean(cases_seq[-3:]))
                roll_6_mean = float(np.mean(cases_seq[-6:]))
                roll_3_std = float(np.std(cases_seq[-3:]))
                roll_6_std = float(np.std(cases_seq[-6:]))
                
                growth_rate = float((lag_1 - lag_2) / lag_2) if lag_2 > 0 else 0.0
                spike_ratio = float(lag_1 / roll_3_mean) if roll_3_mean > 0 else 1.0
                acceleration = float(lag_1 - 2 * lag_2 + lag_3)
                outbreak_frequency_12m = float(history_row.get("outbreak_frequency_12m", 0.0))
                death_rate = float(history_row.get("death_rate", 0.0))
            else:
                # Historical prediction lags
                lag_1 = float(history_row.get("cases", 0.0))
                lag_2 = float(history_row.get("lag_1", 0.0))
                lag_3 = float(history_row.get("lag_2", 0.0))
                lag_6 = float(history_row.get("lag_6", 0.0))
                lag_12 = float(history_row.get("lag_12", 0.0))
                
                roll_3_mean = (lag_1 + lag_2 + lag_3) / 3.0
                roll_6_mean = float(history_row.get("roll_6_mean", roll_3_mean))
                roll_3_std = float(np.std([lag_1, lag_2, lag_3]))
                roll_6_std = float(history_row.get("roll_6_std", roll_3_std))
                
                growth_rate = float(history_row.get("growth_rate", 0.0))
                spike_ratio = float(history_row.get("spike_ratio", 1.0))
                acceleration = float(history_row.get("acceleration", 0.0))
                outbreak_frequency_12m = float(history_row.get("outbreak_frequency_12m", 0.0))
                death_rate = float(history_row.get("death_rate", 0.0))
            
            # Climate changes
            temp = float(enriched.get("temp", 300.0))
            precip = float(enriched.get("precipitation", 2.0))
            lai = float(enriched.get("lai", 15.0))
            
            temp_change = temp - float(history_row.get("temp", temp))
            precip_change = precip - float(history_row.get("precipitation", precip))
            lai_change = lai - float(history_row.get("lai", lai))
        else:
            # Fallback values if no history exists for this combination
            lag_1 = lag_2 = lag_3 = lag_6 = lag_12 = 0.0
            roll_3_mean = roll_6_mean = roll_3_std = roll_6_std = 0.0
            temp_change = precip_change = lai_change = 0.0
            growth_rate = acceleration = 0.0
            spike_ratio = 1.0
            outbreak_frequency_12m = 0.0
            death_rate = 0.0

        # Assign to enriched payload
        for k, v in {
            "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_6": lag_6, "lag_12": lag_12,
            "roll_3_mean": roll_3_mean, "roll_6_mean": roll_6_mean,
            "roll_3_std": roll_3_std, "roll_6_std": roll_6_std,
            "temp_change": temp_change, "precipitation_change": precip_change, "lai_change": lai_change,
            "growth_rate": growth_rate, "spike_ratio": spike_ratio, "acceleration": acceleration,
            "outbreak_frequency_12m": outbreak_frequency_12m, "death_rate": death_rate
        }.items():
            if enriched.get(k) is None or enriched.get(k) == 0.0:
                enriched[k] = v

        return enriched

    def predict(self, data: Dict[str, Any], calculate_shap: bool = True) -> Dict[str, Any]:
        # 1. Autocomplete/enrich the features first
        enriched_data = self.autocomplete_features(data)
        
        # 2. Get features vector
        X = self.predictor._prepare_input(enriched_data)
        
        # 3. Predict cases & outbreak probability
        log_cases = float(self.predictor.regressor.predict(X)[0])
        log_cases = max(log_cases, 0.0)
        predicted_cases = int(round(np.expm1(log_cases)))
        
        outbreak_prob = float(self.predictor.classifier.predict_proba(X)[0][1])
        risk_level = self.predictor.predict(enriched_data)["risk_level"]
        
        # 4. Calculate confidence score
        # Confidence score represents how confident the model is in its binary classification (outbreak vs no outbreak)
        confidence_score = float(outbreak_prob if outbreak_prob >= 0.5 else (1.0 - outbreak_prob))
        
        # 5. Generate SHAP explanation contributions
        shap_contributors = []
        if calculate_shap and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X)
                # shap_values could be a 1D/2D array depending on shap version
                if len(shap_values.shape) > 1:
                    shap_row = shap_values[0]
                else:
                    shap_row = shap_values
                    
                for name, val in zip(ALL_FEATURES, shap_row):
                    shap_contributors.append({
                        "feature": name,
                        "shap_value": float(val)
                    })
                # Sort by absolute SHAP value
                shap_contributors = sorted(shap_contributors, key=lambda x: abs(x["shap_value"]), reverse=True)
            except Exception as e:
                logger.error(f"Error calculating SHAP in real-time prediction: {e}")
                
        return {
            "predicted_cases": predicted_cases,
            "outbreak_probability": round(outbreak_prob, 4),
            "risk_level": risk_level,
            "confidence_score": round(confidence_score, 4),
            "shap_top_contributors": shap_contributors[:8] if calculate_shap else [],
            "enriched_input": enriched_data
        }
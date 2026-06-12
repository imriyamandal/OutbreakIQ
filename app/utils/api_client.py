import requests
import streamlit as st
import time
from typing import Optional, Dict, Any

API_URL = "http://127.0.0.1:8000"

class APIClient:
    @staticmethod
    def _request(method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Optional[Any]:
        url = f"{API_URL}{endpoint}"
        for attempt in range(1, retries + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=params, timeout=15)
                elif method.upper() == "POST":
                    response = requests.post(url, json=json_data, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    err_msg = f"API Error {response.status_code}: {response.text}"
                    st.error(err_msg)
                    return None
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    st.toast(f"Failed to connect to backend after {retries} attempts.", icon="🚨")
                    st.error(f"Network error: Unable to connect to the backend server. {e}")
                    return None
                else:
                    st.toast(f"Connection attempt {attempt} failed. Retrying...", icon="⚠️")
                    time.sleep(1)
        return None

    @classmethod
    def get(cls, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        return cls._request("GET", endpoint, params=params)

    @classmethod
    def post(cls, endpoint: str, json_data: Dict[str, Any]) -> Optional[Any]:
        return cls._request("POST", endpoint, json_data=json_data)

import sqlite3
from pathlib import Path
from datetime import datetime

DB_DIR = Path(__file__).resolve().parents[2] / "app" / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "history.db"

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_ut TEXT,
            district TEXT,
            disease TEXT,
            disease_category TEXT,
            day INTEGER,
            month INTEGER,
            year INTEGER,
            precipitation REAL,
            lai REAL,
            temp REAL,
            predicted_cases INTEGER,
            outbreak_probability REAL,
            risk_level TEXT,
            confidence_score REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(data: dict, result: dict):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (
            state_ut, district, disease, disease_category,
            day, month, year, precipitation, lai, temp,
            predicted_cases, outbreak_probability, risk_level, confidence_score,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("state_ut"),
        data.get("district"),
        data.get("disease"),
        data.get("disease_category"),
        data.get("day"),
        data.get("month"),
        data.get("year"),
        data.get("precipitation"),
        data.get("lai"),
        data.get("temp"),
        result.get("predicted_cases"),
        result.get("outbreak_probability"),
        result.get("risk_level"),
        result.get("confidence_score"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_history(search_query: str = None, disease_filter: str = None, state_filter: str = None):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM history WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND (district LIKE ? OR disease LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    if disease_filter:
        query += " AND disease = ?"
        params.append(disease_filter)
        
    if state_filter:
        query += " AND state_ut = ?"
        params.append(state_filter)
        
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

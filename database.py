import sqlite3
from datetime import datetime
import json
import os

DB_PATH = "mindguard.db"

def init_db():
    """Initialize database with tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create screenings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            risk_level TEXT,
            risk_probability REAL,
            predicted_class INTEGER,
            breakdown TEXT,
            guidance TEXT,
            created_at TIMESTAMP,
            age INTEGER,
            gender TEXT,
            academic_pressure INTEGER,
            work_pressure INTEGER,
            cgpa REAL,
            study_satisfaction INTEGER,
            job_satisfaction INTEGER,
            sleep_duration TEXT,
            dietary_habits TEXT,
            work_study_hours REAL,
            financial_stress INTEGER,
            family_history TEXT,
            ip_address TEXT
        )
    ''')
    
    # Create stats table for daily aggregates
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_screenings INTEGER DEFAULT 0,
            low_risk_count INTEGER DEFAULT 0,
            moderate_risk_count INTEGER DEFAULT 0,
            high_risk_count INTEGER DEFAULT 0,
            avg_risk_score REAL DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {os.path.abspath(DB_PATH)}")

def save_screening(screening_data):
    """Save a screening result to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO screenings (
            session_id, risk_level, risk_probability, predicted_class,
            breakdown, guidance, created_at, age, gender,
            academic_pressure, work_pressure, cgpa, study_satisfaction,
            job_satisfaction, sleep_duration, dietary_habits,
            work_study_hours, financial_stress, family_history, ip_address
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        screening_data.get('session_id'),
        screening_data.get('risk_level'),
        screening_data.get('risk_probability'),
        screening_data.get('predicted_class'),
        json.dumps(screening_data.get('breakdown', {})),
        screening_data.get('guidance'),
        screening_data.get('created_at'),
        screening_data.get('age'),
        screening_data.get('gender'),
        screening_data.get('academic_pressure'),
        screening_data.get('work_pressure'),
        screening_data.get('cgpa'),
        screening_data.get('study_satisfaction'),
        screening_data.get('job_satisfaction'),
        screening_data.get('sleep_duration'),
        screening_data.get('dietary_habits'),
        screening_data.get('work_study_hours'),
        screening_data.get('financial_stress'),
        screening_data.get('family_history'),
        screening_data.get('ip_address', 'unknown')
    ))
    
    conn.commit()
    conn.close()
    print(f"💾 Screening saved to database (ID: {cursor.lastrowid})")

def get_all_screenings(limit=100):
    """Get all screenings from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM screenings 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Get column names
    cursor.execute("PRAGMA table_info(screenings)")
    columns = [col[1] for col in cursor.fetchall()]
    
    results = []
    for row in rows:
        result = dict(zip(columns, row))
        if result.get('breakdown'):
            result['breakdown'] = json.loads(result['breakdown'])
        results.append(result)
    
    return results

def get_statistics():
    """Get aggregate statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END) as low_count,
            SUM(CASE WHEN risk_level = 'Moderate' THEN 1 ELSE 0 END) as moderate_count,
            SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_count,
            AVG(risk_probability) as avg_risk,
            AVG(age) as avg_age
        FROM screenings
    ''')
    
    stats = cursor.fetchone()
    conn.close()
    
    return {
        'total': stats[0] or 0,
        'low_count': stats[1] or 0,
        'moderate_count': stats[2] or 0,
        'high_count': stats[3] or 0,
        'avg_risk': round(stats[4], 4) if stats[4] else 0,
        'avg_age': round(stats[5], 1) if stats[5] else 0
    }

# Initialize database when module loads
if __name__ == "__main__":
    init_db()
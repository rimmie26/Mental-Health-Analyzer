from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import os
import traceback
import sqlite3
import json
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ==================== DATABASE SETUP ====================

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
    
    # Get column names
    cursor.execute("PRAGMA table_info(screenings)")
    columns = [col[1] for col in cursor.fetchall()]
    
    conn.close()
    
    results = []
    for row in rows:
        result = dict(zip(columns, row))
        if result.get('breakdown'):
            try:
                result['breakdown'] = json.loads(result['breakdown'])
            except:
                result['breakdown'] = {}
        results.append(result)
    
    return results

def get_statistics():
    """Get aggregate statistics from database"""
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

# Initialize database
init_db()

# ==================== MODEL LOADING ====================

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Templates folder exists: {os.path.exists('templates')}")

# Load models with error handling
try:
    model = joblib.load("mental_health_risk_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    selected_features = joblib.load("selected_features.pkl")
    print(f"✅ Selected features loaded: {len(selected_features)} features")
except Exception as e:
    print(f"❌ Error loading selected_features: {e}")
    selected_features = None

try:
    risk_thresholds = joblib.load("risk_thresholds.pkl")
    print(f"✅ Risk thresholds loaded: {risk_thresholds}")
except Exception as e:
    print(f"❌ Error loading risk_thresholds: {e}")
    risk_thresholds = {"low_max": 0.35, "moderate_max": 0.65}

def map_probability_to_level(prob):
    """Map probability to risk level based on thresholds"""
    if prob < risk_thresholds["low_max"]:
        return "Low"
    elif prob < risk_thresholds["moderate_max"]:
        return "Moderate"
    return "High"

def get_guidance(level):
    """Get guidance text based on risk level"""
    if level == "Low":
        return (
            "Your responses suggest a lower current level of mental health risk. "
            "Continue maintaining healthy routines, balanced sleep, and regular support. "
            "Remember to check in with yourself regularly."
        )
    elif level == "Moderate":
        return (
            "Your responses suggest a moderate level of mental health risk. "
            "It may help to monitor your well-being and consider speaking with a counselor "
            "or trusted support person if distress continues. "
            "Consider trying stress management techniques like mindfulness or exercise."
        )
    return (
        "Your responses suggest a higher level of mental health risk. "
        "Reaching out to a trusted counselor, mental health professional, "
        "or support service may be helpful. You are not alone - help is available. "
        "Please consider contacting a mental health professional."
    )

# ==================== ROUTES ====================

@app.route("/")
def landing():
    """Serve the landing page"""
    try:
        return render_template("landing page.html")
    except Exception as e:
        print(f"Error rendering landing page: {e}")
        return jsonify({"error": "Landing page not found"}), 404

@app.route("/screening")
def screening():
    """Serve the screening page"""
    try:
        return render_template("screening page.html")
    except Exception as e:
        print(f"Error rendering screening page: {e}")
        return jsonify({"error": "Screening page not found"}), 404

@app.route("/dashboard")
def dashboard():
    """Serve the dashboard page"""
    try:
        return render_template("dashboard page.html")
    except Exception as e:
        print(f"Error rendering dashboard page: {e}")
        return jsonify({"error": "Dashboard page not found"}), 404

# Direct file routes for backward compatibility
@app.route("/landing page.html")
def landing_direct():
    return render_template("landing page.html")

@app.route("/screening page.html")
def screening_direct():
    return render_template("screening page.html")

@app.route("/dashboard page.html")
def dashboard_direct():
    return render_template("dashboard page.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    """Make a prediction based on submitted data"""
    
    # Handle GET requests (for testing)
    if request.method == "GET":
        return jsonify({
            "message": "Send a POST request with JSON data to get predictions",
            "example": {
                "Gender": "Male",
                "Age": 20,
                "Academic Pressure": 3,
                "Work Pressure": 2,
                "CGPA": 7.5,
                "Study Satisfaction": 3,
                "Job Satisfaction": 3,
                "Sleep Duration": "7-8 hours",
                "Dietary Habits": "Moderate",
                "Have you ever had suicidal thoughts ?": "No",
                "Work/Study Hours": 5,
                "Financial Stress": 2,
                "Family History of Mental Illness": "No"
            }
        })
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        print(f"Received prediction request: {data}")
        
        # Map sleep duration to expected format
        sleep_map = {
            "Less than 5 hours": "Less than 5 hours",
            "5-6 hours": "5-6 hours",
            "7-8 hours": "7-8 hours",
            "More than 8 hours": "More than 8 hours"
        }
        
        # Map dietary habits
        diet_map = {
            "Healthy": "Healthy",
            "Moderate": "Moderate",
            "Unhealthy": "Unhealthy"
        }
        
        # Prepare input data with correct types
        input_data = {
            "Gender": str(data.get("Gender", "Male")),
            "Age": float(data.get("Age", 20)),
            "Academic Pressure": float(data.get("Academic Pressure", 2)),
            "Work Pressure": float(data.get("Work Pressure", 1)),
            "CGPA": float(data.get("CGPA", 7)),
            "Study Satisfaction": float(data.get("Study Satisfaction", 3)),
            "Job Satisfaction": float(data.get("Job Satisfaction", 3)),
            "Sleep Duration": sleep_map.get(data.get("Sleep Duration", "7-8 hours"), "7-8 hours"),
            "Dietary Habits": diet_map.get(data.get("Dietary Habits", "Moderate"), "Moderate"),
            "Have you ever had suicidal thoughts ?": str(data.get("Have you ever had suicidal thoughts ?", "No")),
            "Work/Study Hours": float(data.get("Work/Study Hours", 4)),
            "Financial Stress": float(data.get("Financial Stress", 2)),
            "Family History of Mental Illness": str(data.get("Family History of Mental Illness", "No"))
        }
        
        print(f"Processed input data: {input_data}")
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all selected features are present
        if selected_features is not None:
            for feature in selected_features:
                if feature not in df.columns:
                    print(f"Warning: Feature '{feature}' not found, adding with default value")
                    if feature in ["Gender", "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]:
                        df[feature] = "Unknown"
                    else:
                        df[feature] = 0
            
            df = df[selected_features]
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Make prediction
        if model is not None:
            risk_probability = float(model.predict_proba(df)[0][1])
            predicted_class = int(model.predict(df)[0])
        else:
            # Fallback if model not loaded
            risk_probability = 0.3
            predicted_class = 0
        
        risk_level = map_probability_to_level(risk_probability)
        
        # Calculate breakdown percentages
        if risk_level == "Low":
            breakdown = {
                "low": round((1 - risk_probability) * 100),
                "moderate": round(risk_probability * 40),
                "high": round(risk_probability * 20)
            }
        elif risk_level == "Moderate":
            breakdown = {
                "low": round((1 - risk_probability) * 35),
                "moderate": 60,
                "high": round(risk_probability * 35)
            }
        else:
            breakdown = {
                "low": round((1 - risk_probability) * 15),
                "moderate": round((1 - risk_probability) * 25),
                "high": round(risk_probability * 100)
            }
        
        # Ensure total is 100
        total = breakdown["low"] + breakdown["moderate"] + breakdown["high"]
        if total != 100:
            breakdown["moderate"] += 100 - total
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Get client IP (optional, for analytics)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        
        # Prepare response
        response = {
            "success": True,
            "risk_probability": round(risk_probability, 4),
            "predicted_class": predicted_class,
            "risk_level": risk_level,
            "guidance": get_guidance(risk_level),
            "breakdown": breakdown,
            "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": "This result is a screening estimate only and not a clinical diagnosis. Please consult a mental health professional for proper evaluation."
        }
        
        # 💾 SAVE TO DATABASE
        screening_record = {
            'session_id': session_id,
            'risk_level': risk_level,
            'risk_probability': risk_probability,
            'predicted_class': predicted_class,
            'breakdown': breakdown,
            'guidance': get_guidance(risk_level),
            'created_at': datetime.now(),
            'age': input_data['Age'],
            'gender': input_data['Gender'],
            'academic_pressure': input_data['Academic Pressure'],
            'work_pressure': input_data['Work Pressure'],
            'cgpa': input_data['CGPA'],
            'study_satisfaction': input_data['Study Satisfaction'],
            'job_satisfaction': input_data['Job Satisfaction'],
            'sleep_duration': input_data['Sleep Duration'],
            'dietary_habits': input_data['Dietary Habits'],
            'work_study_hours': input_data['Work/Study Hours'],
            'financial_stress': input_data['Financial Stress'],
            'family_history': input_data['Family History of Mental Illness'],
            'ip_address': client_ip
        }
        
        save_screening(screening_record)
        
        print(f"Prediction successful: {risk_level} risk with probability {risk_probability}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "An error occurred while processing your request"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": selected_features is not None,
        "thresholds_loaded": risk_thresholds is not None,
        "database_connected": os.path.exists(DB_PATH),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/info", methods=["GET"])
def api_info():
    """Get API information"""
    return jsonify({
        "name": "MindGuard AI API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Landing page",
            "/screening": "Screening page",
            "/dashboard": "Dashboard page",
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/api/info": "GET - API information",
            "/api/stats": "GET - Database statistics",
            "/api/screenings": "GET - View all screenings"
        },
        "expected_features": selected_features if selected_features else []
    })

@app.route("/api/stats", methods=["GET"])
def get_db_stats():
    """Get statistics from database"""
    try:
        stats = get_statistics()
        return jsonify({
            "success": True,
            "statistics": stats,
            "database_path": os.path.abspath(DB_PATH)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/screenings", methods=["GET"])
def get_all_screenings_api():
    """Get all screenings from database"""
    try:
        limit = request.args.get('limit', 50, type=int)
        screenings = get_all_screenings(limit)
        return jsonify({
            "success": True,
            "count": len(screenings),
            "screenings": screenings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not Found",
        "message": "The requested URL was not found on the server"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal Server Error",
        "message": "Something went wrong on the server"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Starting MindGuard AI Server...")
    print(f"📍 Server will run on: http://localhost:{port}")
    print(f"📁 Templates directory: {os.path.abspath('templates')}")
    print(f"💾 Database path: {os.path.abspath(DB_PATH)}")
    print(f"🔧 Debug mode: ON")
    print("\nAvailable endpoints:")
    print(f"  - http://localhost:{port}/              (Landing Page)")
    print(f"  - http://localhost:{port}/screening     (Screening Tool)")
    print(f"  - http://localhost:{port}/dashboard     (Dashboard)")
    print(f"  - http://localhost:{port}/health        (Health Check)")
    print(f"  - http://localhost:{port}/api/info      (API Info)")
    print(f"  - http://localhost:{port}/api/stats     (Database Statistics)")
    print(f"  - http://localhost:{port}/api/screenings (View All Screenings)")
    print("\n✨ Server is ready! Press Ctrl+C to stop\n")
    
    app.run(host="0.0.0.0", port=port, debug=True)
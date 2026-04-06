from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("mental_health_risk_model.pkl")
selected_features = joblib.load("selected_features.pkl")
risk_thresholds = joblib.load("risk_thresholds.pkl")


def map_probability_to_level(prob):
    if prob < risk_thresholds["low_max"]:
        return "Low"
    elif prob < risk_thresholds["moderate_max"]:
        return "Moderate"
    return "High"


def get_guidance(level):
    if level == "Low":
        return (
            "Your responses suggest a lower current level of mental health risk. "
            "Continue maintaining healthy routines, balanced sleep, and regular support."
        )
    elif level == "Moderate":
        return (
            "Your responses suggest a moderate level of mental health risk. "
            "It may help to monitor your well-being and consider speaking with a counselor "
            "or trusted support person if distress continues."
        )
    return (
        "Your responses suggest a higher level of mental health risk. "
        "Reaching out to a trusted counselor, mental health professional, "
        "or support service may be helpful."
    )


@app.route("/")
def landing():
    return render_template("landing page.html")


@app.route("/screening")
def screening():
    return render_template("screening page.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard page.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = {
            "Gender": data.get("Gender"),
            "Age": data.get("Age"),
            "Academic Pressure": data.get("Academic Pressure"),
            "Work Pressure": data.get("Work Pressure"),
            "CGPA": data.get("CGPA"),
            "Study Satisfaction": data.get("Study Satisfaction"),
            "Job Satisfaction": data.get("Job Satisfaction"),
            "Sleep Duration": data.get("Sleep Duration"),
            "Dietary Habits": data.get("Dietary Habits"),
            "Have you ever had suicidal thoughts ?": data.get("Have you ever had suicidal thoughts ?"),
            "Work/Study Hours": data.get("Work/Study Hours"),
            "Financial Stress": data.get("Financial Stress"),
            "Family History of Mental Illness": data.get("Family History of Mental Illness")
        }

        df = pd.DataFrame([input_data])[selected_features]

        risk_probability = float(model.predict_proba(df)[0][1])
        predicted_class = int(model.predict(df)[0])
        risk_level = map_probability_to_level(risk_probability)

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

        total = breakdown["low"] + breakdown["moderate"] + breakdown["high"]
        if total != 100:
            breakdown["moderate"] += 100 - total

        return jsonify({
            "risk_probability": round(risk_probability, 4),
            "predicted_class": predicted_class,
            "risk_level": risk_level,
            "guidance": get_guidance(risk_level),
            "breakdown": breakdown,
            "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": "This result is a screening estimate only and not a clinical diagnosis."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
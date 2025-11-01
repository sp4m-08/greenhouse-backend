import os
import pickle
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from fastapi.middleware.cors import CORSMiddleware 

FIREBASE_DB_URL = "https://control-greenhouse-default-rtdb.asia-southeast1.firebasedatabase.app"
SENSOR_DATA_PATH = f"{FIREBASE_DB_URL}/iot/device.json"
MODEL_FILENAME = 'random_forest_model.pkl'

SMTP_SERVER = "smtp.gmail.com" 
SMTP_PORT = 587
SENDER_EMAIL = "roysaunok@gmail.com" 
SENDER_PASSWORD = "lwlg gniz xxms dmia" 
RECIPIENT_EMAIL = "theholyknight460@gmail.com" 

LAST_ALERTED_LABEL = 2

app = FastAPI(title="Greenhouse ML Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

MODEL = None

class PredictionResponse(BaseModel):
    status: str
    code: int
    health_status: str
    confidence: float
    recommendation: str

FEATURE_NAMES = ['temperature', 'humidity', 'soil_pct', 'light_raw', 'gas_raw']

HEALTH_LABELS = {
    0: "Urgent Intervention",
    1: "Monitoring Required",
    2: "Optimal Growth"
}

@app.on_event("startup")
def load_model():
    global MODEL
    try:
        if not os.path.exists(MODEL_FILENAME):
            print(f"ERROR: Model file '{MODEL_FILENAME}' not found.")
            MODEL = None
            return

        with open(MODEL_FILENAME, 'rb') as file:
            MODEL = pickle.load(file)
            print(f"Random Forest Model Loaded: {MODEL_FILENAME}")
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None

def fetch_sensor_data_from_firebase() -> Dict[str, Any]:
    try:
        response = requests.get(SENSOR_DATA_PATH)
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("Firebase returned null.")

        return data

    except requests.exceptions.RequestException as e:
        print(f"Firebase Request Error: {e}")
        raise HTTPException(status_code=503, detail="Could not connect to Firebase.")
    except Exception as e:
        print(f"JSON Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Error reading sensor data from Firebase.")

def send_alert_email(health_status: str, recommendation: str, current_data: Dict[str, Any]):
    
    sensor_details = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in current_data.items()])
    
    email_body = f"""
Greenhouse CRITICAL ALERT! 🚨

Health Status: {health_status}
Confidence: {current_data.get('confidence', 'N/A')}%

--- RECOMMENDATION ---
{recommendation} 

--- REAL-TIME SENSOR DATA ---
{sensor_details}
"""

    msg = MIMEText(email_body)
    msg['Subject'] = f"ALERT: Greenhouse Health Status - {health_status}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls() 
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print(f"Email Alert Sent: Status {health_status}.")
    except Exception as e:
        print(f"Failed to send email alert. Check SMTP config/credentials: {e}")

@app.get("/predict", response_model=PredictionResponse)
def predict_health():
    global LAST_ALERTED_LABEL 

    if MODEL is None:
        raise HTTPException(status_code=500, detail="ML model not loaded.")

    firebase_data = fetch_sensor_data_from_firebase()

    try:
        input_data = [firebase_data[feature] for feature in FEATURE_NAMES]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing sensor field in Firebase: {e}")

    input_array = np.array([input_data]) 

    prediction_label = MODEL.predict(input_array)[0]
    prediction_proba = MODEL.predict_proba(input_array)[0]
    confidence = prediction_proba[prediction_label]

    health_status = HEALTH_LABELS.get(prediction_label, "Unknown")

    if prediction_label == 2:
        recommendation = (
            "OPTIMAL ENVIRONMENT: Everything is within desired parameters.\n\n"
            "1. Maintenance: Keep monitoring environment and plant growth.\n"
            "2. Watering: Ensure scheduled watering (PID) is running effectively.\n"
            "3. Note: No immediate changes or interventions are required."
        )
    elif prediction_label == 1:
        recommendation = (
            "MONITORING REQUIRED: Minor environmental stress detected. Review parameters.\n\n"
            "1. Soil: Verify current **soil_pct** is near the **moistureSetpoint**.\n"
            "2. Humidity: If humidity is high (e.g., > 70%), consider increasing ventilation (Fan/Window).\n"
            "3. Temperature: If temperature is slightly high, ensure the **fan_trigger** is active.\n"
            "4. Light: Check if **light_raw** is sufficient for the current growth stage.\n"
            "5. Action: No immediate shutdown, but adjustments may prevent a future critical state."
        )
    else: 
        recommendation = (
            "CRITICAL - IMMEDIATE ACTION REQUIRED! Severe stress or resource depletion detected.\n\n"
            "1. Ventilation: Immediately ensure high-power **ventilation** is active (Fan/Exhaust).\n"
            "2. Gas/Temp: If **gas_raw** or **temperature** is very high, this is a priority for human intervention.\n"
            "3. Watering: Check the water supply and pump, especially if **soil_pct** is critically low.\n"
            "4. Light: Adjust shade or supplementary light if **light_raw** is extremely high or low.\n"
            "5. Device Check: Verify the physical operation of all sensors and actuators (pump/fan)."
        )
        
    is_alert_required = prediction_label <= 1
    is_state_changed = prediction_label != LAST_ALERTED_LABEL
    
    if is_alert_required and is_state_changed:
        alert_data = firebase_data.copy()
        alert_data['confidence'] = round(confidence * 100, 2)
        send_alert_email(health_status, recommendation, alert_data)
        
        LAST_ALERTED_LABEL = prediction_label
        
    elif prediction_label == 2 and is_state_changed:
        LAST_ALERTED_LABEL = 2

    return PredictionResponse(
        status="Prediction Successful",
        code=int(prediction_label),
        health_status=health_status,
        confidence=round(confidence * 100, 2),
        recommendation=recommendation
    )

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Keep Mental_Health_Model.pkl beside main.py."
    )

# Load the trained pipeline once when the API starts.
model = joblib.load(MODEL_PATH)

# These categories match the grouped-country feature used during training.
TOP_COUNTRIES = {
    "India", "USA", "Canada", "Australia", "UK",
    "Germany", "Mexico", "Turkey", "France"
}

app = FastAPI(
    title="Mansik Santulan Score API",
    description="Student lifestyle data to an ML-estimated wellbeing score. Not a diagnosis.",
    version="1.0.0",
)


class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    country: str = Field(..., min_length=1, max_length=100)
    academic_level: Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform: Literal[
        "Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "YouTube",
        "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"
    ]
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal["Medium", "Low", "Very High", "High"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    grouped_country = data.country if data.country in TOP_COUNTRIES else "Other"

    # Keep exactly the feature names and order used to train the saved pipeline.
    input_row = pd.DataFrame([{
        "Study_Hours": data.study_hours,
        "Age": data.age,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Gender": data.gender,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Grouped_country": grouped_country,
    }])

    try:
        prediction = float(model.predict(input_row)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The model could not process these inputs. Check the model/scikit-learn versions."
        ) from exc

    # This is a model-estimated score, not a clinical assessment.
    prediction = max(0.0, min(10.0, prediction))
    return PredictionResponse(predicted_mental_health_score=round(prediction, 2))


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve the CSS and JavaScript from the same origin as the API.
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")

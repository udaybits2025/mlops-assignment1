from statistics import LinearRegression

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.logger_config import log
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, start_http_server

class HousingInput(BaseModel):
    medinc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0)
    ave_rooms: float = Field(..., ge=0)
    ave_bedrms: float = Field(..., ge=0)
    population: float = Field(..., ge=0)
    ave_occup: float = Field(..., ge=0)
    latitude: float
    longitude: float
    households: float = Field(..., ge=0)
    rooms_per_household: float = Field(..., ge=0)
    bedrooms_per_room: float = Field(..., ge=0)
    population_per_household: float = Field(..., ge=0)
    medinc_log: float

class RetrainData(BaseModel):
    data: list[HousingInput]
    target: list[float]

app = FastAPI()
Instrumentator().instrument(app).expose(app)
# Load the model
try:
    model = joblib.load("models/best_model.pkl")
except FileNotFoundError:
    log.error("Could not find a model to load.")
    raise HTTPException(status_code=500, detail="Model file not found.")

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total number of requests received", ["endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests in seconds", ["endpoint"])

# Start Prometheus metrics server
start_http_server(8001)

@app.post("/predict")
def predict(input_data: HousingInput):
    REQUEST_COUNT.labels(endpoint="predict").inc()
    with REQUEST_LATENCY.labels(endpoint="predict").time():
        log.info("Received request to predict.")
        try:
            features = input_data.dict()
            input_df = pd.DataFrame([features])
            prediction = model.predict(input_df)
            log.info(f"Prediction: {prediction}")
            return {"prediction": prediction.tolist()}
        except Exception as e:
            log.error(e)
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain")
def retrain(retrain_data: RetrainData):
    REQUEST_COUNT.labels(endpoint="retrain").inc()
    with REQUEST_LATENCY.labels(endpoint="retrain").time():
        log.info("Received request to retrain model.")
        try:
            df = pd.DataFrame([item.dict() for item in retrain_data.data])
            target = retrain_data.target
            model = LinearRegression()
            model.fit(df, target)
            joblib.dump(model, "models/best_model.pkl")
            log.info("Model retrained and saved.")
            return {"status": "Model retrained successfully"}
        except Exception as e:
            log.error(e)
            raise HTTPException(status_code=400, detail=f"Retraining failed: {str(e)}")

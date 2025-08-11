from statistics import LinearRegression
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging to both console and file
log = logging.getLogger("api_logger")
log.setLevel(logging.INFO)

# Prevent duplicate logs
if not log.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    log.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("prediction_logs.log")
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    log.addHandler(file_handler)

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the validation error
    error_details = exc.errors()
    log.error(
        f"Pydantic validation failed for request {request.url.path}: "
        f"{error_details}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "message": "Pydantic validation failed",
            "detail": error_details
        },
    )

Instrumentator().instrument(app).expose(app)
# Load the model
try:
    model = joblib.load("models/best_model.pkl")
except FileNotFoundError:
    log.error("Could not find a model to load.")
    raise HTTPException(status_code=500, detail="Model file not found.")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests received",
    ["endpoint"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"]
)

# Start Prometheus metrics server
start_http_server(8001)

@app.post("/predict")
def predict(input_data: HousingInput):
    log.info("SUCCESS! Pydantic Validation Passed")
    REQUEST_COUNT.labels(endpoint="predict").inc()
    with REQUEST_LATENCY.labels(endpoint="predict").time():
        log.info("Received request to predict.")
        # Log incoming request data to file
        log.info(f"Incoming request data: {input_data.dict()}")
        try:
            features = input_data.dict()
            input_df = pd.DataFrame([features])
            prediction = model.predict(input_df)
            # Log model output to file
            log.info(f"Model output: {prediction.tolist()}")
            return {
                "prediction": prediction.tolist()[0],
                "input_features": input_data.dict(),
                "pydantic_validation_status": (
                    "SUCCESS! Pydantic Validation Passed"
                )
            }
        except Exception as e:
            log.error(e)
            raise HTTPException(
                status_code=400,
                detail=f"Prediction failed: {str(e)}"
            )

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
            joblib.dump(model, "/models/best_model.pkl")
            log.info("Model retrained and saved.")
            return {"status": "Model retrained successfully"}
        except Exception as e:
            log.error(e)
            raise HTTPException(
                status_code=400,
                detail=f"Retraining failed: {str(e)}"
            )

# MLOps Assignment: California Housing Prediction

## Project Overview
This project aims to predict California housing prices using machine learning models. It implements an end-to-end pipeline for data preprocessing, model training, evaluation, and deployment using FastAPI. The project follows MLOps best practices, including version control, containerization, model serving, and monitoring.

---

## Architecture

### 1. **Data Pipeline**
- **Source**: The California housing dataset is fetched using `sklearn.datasets.fetch_california_housing`.
- **Preprocessing**: The dataset is preprocessed in `data_service.py`:
  - Columns are renamed for clarity.
  - Derived features are created (e.g., `rooms_per_household`, `bedrooms_per_room`).
  - Log transformation is applied to skewed features.
  - StandardScaler is used for feature scaling, with handling for NaNs and Infs.
- **Output**: Preprocessed data is saved as CSV files:
  - `california_housing_preprocessed.csv`
  - `california_housing_train.csv`
  - `california_housing_test.csv`

### 2. **Model Training**
- **Script**: `train.py`
- **Models**: Linear Regression and Decision Tree Regressor are evaluated.
- **Evaluation**:
  - Metrics: RMSE and R².
  - Best model selection based on RMSE.
- **Logging**: MLflow is used for experiment tracking.
- **Output**:
  - Best model is saved locally as `models/best_model.pkl`.
  - Best model is registered in MLflow.

### 3. **Model Deployment**
- **API**: `main.py`
- **Framework**: FastAPI
- **Endpoints**:
  - `/predict`: Accepts input features as JSON and returns predictions.
  - `/retrain`: Accepts new data and retrains the model.
- **Model Loading**: The saved model (`best_model.pkl`) is loaded for inference.
- **Input Validation**: Pydantic schemas are used to validate input data.

### 4. **Monitoring**
- **Integration**: Prometheus is integrated for monitoring.
- **Metrics**:
  - API request counts.
  - Latency.
  - Error rates.
- **Dashboard**: Grafana is used to visualize metrics.
  - Sample dashboard JSON is provided in `grafana/dashboards/fastapi_metrics.json`.

### 5. **Containerization**
- **Dockerfile**:
  - Base image: `python:3.12-slim`
  - Dependencies: Installed via `requirements.txt`.
  - Application: FastAPI server exposed on port 8000.
- **Command**:
  ```bash
  docker build -t california-housing .
  docker run -p 8000:8000 california-housing
  ```

---

## Folder Structure
```
mlops-assignment/
├── Dockerfile
├── LICENSE
├── logger_config.py
├── README.md
├── requirements.txt
├── sample.curl
├── app/
│   ├── data_service.py
│   ├── main.py
│   ├── train.py
│   ├── __pycache__/
│   └── mlruns/
├── data/
│   ├── california_housing_preprocessed.csv
│   ├── california_housing_train.csv
│   ├── california_housing_test.csv
├── models/
│   ├── best_model.pkl
├── grafana/
│   ├── dashboards/
│   │   ├── dashboard.yml
│   │   ├── fastapi_metrics.json
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   ├── dashboard.yml
│   │   │   ├── fastapi_metrics.json
│   │   ├── datasources/
│   │   │   ├── datasource.yml
```

---

## How to Run

### 1. **Setup Environment**
```bash
pip install -r requirements.txt
```

### 2. **Run Preprocessing**
```bash
python app/data_service.py
```

### 3. **Train Models**
```bash
python app/train.py
```

### 4. **Start FastAPI Server**
```bash
uvicorn app.main:app --reload
```

### 5. **Test API**
Use the sample `curl` command:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medinc": 112.3252,
    "house_age": 41.0,
    "ave_rooms": 6.9841,
    "ave_bedrms": 1.0238,
    "population": 322.0,
    "ave_occup": 2.5556,
    "latitude": 37.88,
    "longitude": -122.23,
    "households": 126.0,
    "rooms_per_household": 5.5,
    "bedrooms_per_room": 0.15,
    "population_per_household": 2.5,
    "medinc_log": 2.12
}'
```

---

## Key Features
- **MLOps Best Practices**:
  - Modular code structure.
  - Experiment tracking with MLflow.
  - Containerization with Docker.
  - Monitoring with Prometheus and Grafana.
- **Scalable Deployment**:
  - FastAPI for serving predictions.
  - Docker for portability.
- **Robust Preprocessing**:
  - Handling of NaNs, Infs, and extreme values.
- **Retraining**:
  - `/retrain` endpoint for model retraining on new data.

---

## Deliverables

### GitHub Repository
The complete project is hosted on GitHub:
[https://github.com/arjunrajeev/mlops-assignment](https://github.com/arjunrajeev/mlops-assignment)

### CI/CD Pipeline
The GitHub Actions pipeline for building and pushing the Docker image can be found here:
[https://github.com/arjunrajeev/mlops-assignment/actions](https://github.com/arjunrajeev/mlops-assignment/actions)

### Docker Hub Repository
The Docker image for the project is available on Docker Hub:
[https://hub.docker.com/repository/docker/arjajje/california-housing-predictor/general](https://hub.docker.com/repository/docker/arjajje/california-housing-predictor/general)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributors
- **Arjun Rajeev**

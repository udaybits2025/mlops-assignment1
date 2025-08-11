import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# Load preprocessed training and test sets
train_df = pd.read_csv("data/california_housing_train.csv")
test_df = pd.read_csv("data/california_housing_test.csv")

X_train = train_df.drop("medhouseval", axis=1)
y_train = train_df["medhouseval"]
X_test = test_df.drop("medhouseval", axis=1)
y_test = test_df["medhouseval"]

# Start an MLflow experiment
mlflow.set_experiment("Californiadataset_Regression")


def evaluate_model(model, model_name, params={}):
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("features", X_train.columns.tolist())

        # Add tags for better organization
        mlflow.set_tag("model_type", model.__class__.__name__)
        mlflow.set_tag(
            "training_data_shape", f"{len(X_train)}x{len(X_train.columns)}"
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Create and log a prediction plot artifact
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            '--r',
            linewidth=2
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs. Predicted Values for {model_name}")

        # Save plot to a file and log it as an artifact
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir, f"{model_name.replace(' ', '_')}_predictions.png"
        )
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Log model
        mlflow.sklearn.log_model(
            model,
            name=model_name.lower().replace(" ", "_"),
            input_example=X_train.head(1),
            signature=mlflow.models.infer_signature(
                X_train, model.predict(X_train)
            )
        )

        print(f"{model_name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return {
            "name": model_name,
            "rmse": rmse,
            "r2": r2,
            "run_id": mlflow.active_run().info.run_id
        }


# Run experiments
results = []
results.append(evaluate_model(LinearRegression(), "Linear Regression"))
results.append(evaluate_model(
    DecisionTreeRegressor(random_state=42),
    "Decision Tree",
    {"random_state": 42}
))

# Select best model
best_model = min(results, key=lambda x: x["rmse"])
print(
    f"\n🏆 Best model: {best_model['name']} "
    f"(RMSE = {best_model['rmse']:.4f}) "
    f"[run_id: {best_model['run_id']}]"
)
best_run_id = best_model["run_id"]
# Register the best model
mlflow.register_model(
    f"runs:/{best_run_id}/{best_model['name'].lower().replace(' ', '_')}",
    "BestCaliforniaHousingModel - " + best_model["name"]
)

# Save the best model locally for FastAPI endpoint
model = mlflow.sklearn.load_model(
    f"runs:/{best_run_id}/{best_model['name'].lower().replace(' ', '_')}"
)
joblib.dump(model, "models/best_model.pkl")
print("✅ Best model saved locally for FastAPI endpoint.")

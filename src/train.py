
import pandas as pd
import mlflow
import os
import joblib

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Set dynamic project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "Advertising.csv")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# 1. Setup tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment_name = "Advertising_Sales_Regression"
registered_model_name = "Sales_Prediction_Model"

mlflow.set_experiment(experiment_name)


# 2. Data Preparation
df = pd.read_csv(DATA_PATH)

x = df[["TV", "radio", "newspaper"]]
y = df["sales"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Train candidate models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        max_depth=5,
        random_state=42
    )
}


batch_runs = []


for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name) as run:

        # Train
        model.fit(x_train, y_train)

        # Prediction
        y_pred = model.predict(x_test)

        # RMSE
        rmse = root_mean_squared_error(
            y_test,
            y_pred
        )

        # Log parameter
        mlflow.log_param(
            "model_type",
            model_name
        )

        # Log metric
        mlflow.log_metric(
            "test_rmse",
            rmse
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # Store run ID + RMSE
        batch_runs.append(
            (run.info.run_id, rmse)
        )


# 4. Find the single best model
batch_runs.sort(key=lambda x: x[1])

best_run_id, best_rmse = batch_runs[0]


# 5. Register winning model as Challenger
client = MlflowClient()

challenger_model = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=registered_model_name
)

challenger_version = challenger_model.version


# 6. Assign Challenger alias
client.set_registered_model_alias(
    registered_model_name,
    "challenger",
    challenger_version
)


print(
    f"Best batch run {best_run_id} registered as "
    f"'challenger' (v{challenger_version}, "
    f"RMSE: {best_rmse:.4f})"
)


# 7. Challenger vs Champion evaluation
try:

    champion_info = client.get_model_version_by_alias(
        registered_model_name,
        "champion"
    )

    champion_run = client.get_run(
        champion_info.run_id
    )

    champion_rmse = champion_run.data.metrics["test_rmse"]

    champion_version = champion_info.version

    print(
        f"Current champion is v{champion_version} "
        f"with RMSE: {champion_rmse:.4f}"
    )


    # 8. Compare Challenger vs Champion
    if best_rmse < champion_rmse:

        client.set_registered_model_alias(
            registered_model_name,
            "champion",
            challenger_version
        )

        print(
            f"Title change! Challenger "
            f"(v{challenger_version}) defeated "
            f"Champion (v{champion_version})"
        )

    else:

        print(
            f"Champion (v{champion_version}) "
            f"retains its title."
        )


except Exception:

    # First time running — no champion exists
    client.set_registered_model_alias(
        registered_model_name,
        "champion",
        challenger_version
    )

    print(
        f"No existing champion found. "
        f"Version {challenger_version} crowned "
        f"as first champion!"
    )


# Load current champion from MLflow Registry
champion_model_uri = (
    f"models:/{registered_model_name}@champion"
)

try:
    champion_model = mlflow.sklearn.load_model(
        champion_model_uri
    )
except Exception:
    print("Champion artifact unavailable. Using current challenger model.")
    champion_model = mlflow.sklearn.load_model(
        f"runs:/{best_run_id}/model"
    )
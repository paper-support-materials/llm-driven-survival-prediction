import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lifelines.utils import concordance_index
from sksurv.util import Surv


def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def prepare_data(csv_path, cols_to_keep, categorical_vars, feature_path=None):
    df = pd.read_csv(csv_path, sep=";")
    # print("CSV columns:", df.columns.tolist())
    # print("Expected:", cols_to_keep)
    df = df[cols_to_keep]

    y = Surv.from_dataframe("os", "time_to_os", df)

    x = df.drop(columns=["os", "time_to_os"])
    x = pd.get_dummies(x, columns=categorical_vars, drop_first=True)
    x = x.fillna(0)

    # Align with training features if provided
    if feature_path is not None:
        feature_columns = joblib.load(feature_path)
        for col in feature_columns:
            if col not in x:
                x[col] = 0
        x = x[feature_columns]

    return x, y


def predict_survival_days(model, scaler, x):
    x_scaled = scaler.transform(x)
    survival_funcs = model.predict_survival_function(x_scaled)
    y_pred = np.array([np.trapezoid(surv.y, surv.x) for surv in survival_funcs])
    return y_pred


def compute_metrics(y_true, y_pred, event):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    c_index = concordance_index(y_true, np.array(y_pred), event.astype(bool))

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "c_index": round(float(c_index), 4)
    }


def save_results(output_path, y_true, y_pred, event, metrics):
    results = {
        "predicted": [float(x) for x in y_pred],
        "real": [float(x) for x in y_true],
        "os": [int(x) for x in event],
        "metrics": metrics
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_path}")


def evaluate(model_path, scaler_path, csv_path, output_json, feature_path=None):
    cols_to_keep = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]
    categorical_vars = ["gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad"]

    print("Loading model...")
    model, scaler = load_model(model_path, scaler_path)

    print("Preparing data...")
    x, y = prepare_data(csv_path, cols_to_keep, categorical_vars, feature_path)

    print("Predicting...")
    y_pred = predict_survival_days(model, scaler, x)

    y_true = y["time_to_os"]
    event = y["os"]

    print("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, event)

    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("C-index:", metrics["c_index"])

    print("Saving results...")
    save_results(output_json, y_true, y_pred, event, metrics)


if __name__ == "__main__":
    evaluate(
        model_path="gradient_boosting/gb_full_model.joblib",
        scaler_path="gradient_boosting/gb_full_scaler.joblib",
        feature_path="gradient_boosting/full_features.joblib",  
        csv_path="csv_data/real_data.csv",
        output_json="gradient_boosting/gb_real_data_results.json"
    )

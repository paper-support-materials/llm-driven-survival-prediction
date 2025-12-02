import json
import os
import pickle

import pandas as pd
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt


def cox(fold, check=False, test=True):
    df = pd.read_csv(f"./Folds_Dimitris/fold_{fold}_train_data.csv", sep=",")
    cols_to_keep = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]

    df = df[cols_to_keep]
    categorical_vars = ["gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad"]

    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    df.dropna(inplace=True)

    # Define the features and the outcome variables
    cph = CoxPHFitter()

    # Fit the model
    cph.fit(df, duration_col="time_to_os", event_col="os")

    # save the model
    with open(f"cox_model_fold_{fold}.pkl", "wb") as f:
        pickle.dump(cph, f)

    # load later
    with open(f"cox_model_fold_{fold}.pkl", "rb") as f:
        cph_loaded = pickle.load(f)

    # Print summary
    cph.print_summary()

    if check:
        cph.check_assumptions(df, p_value_threshold=0.005)
        cph.plot()
        plt.show()

    if test:
        test_df = pd.read_csv(f"./Folds_Dimitris/fold_{fold}_test_data.csv", sep=",")
        test_df_original = test_df.copy()

        test_df = pd.get_dummies(test_df, columns=categorical_vars, drop_first=True)
        test_df.dropna(inplace=True)

        x_test = test_df.reindex(columns=df.columns.drop(['time_to_os', 'os']), fill_value=0)

        res = cph.predict_expectation(x_test)

        res_json = {
            "predicted": res.tolist(),
            "os": test_df_original["os"].tolist(),  # keep a copy of original test_df before dummies/dropna
            "real": test_df_original["time_to_os"].tolist()
        }

        # res.to_json(f"fold_{fold}_cox_result.json")
        with open(f"cox_result_fold_{fold}.json", "w") as f:
            json.dump(res_json, f)


def train_full_cox_model():
    # Read train0 and test0
    train_df = pd.read_csv("./Folds_Dimitris/fold_0_train_data.csv", sep=",")
    test_df = pd.read_csv("./Folds_Dimitris/fold_0_test_data.csv", sep=",")

    # Concatenate to get full dataset
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Columns to keep
    cols_to_keep = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]
    categorical_vars = ["gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad"]

    full_df = full_df[cols_to_keep]

    # One-hot encode categorical variables
    full_df = pd.get_dummies(full_df, columns=categorical_vars, drop_first=True)
    full_df.dropna(inplace=True)

    # Save feature names for alignment later
    import joblib
    feature_names = list(full_df.columns.drop(['time_to_os', 'os']))
    joblib.dump(feature_names, "new_cox/cox_full_features.joblib", compress=3)
    print("Saved feature names to 'cox_full_features.joblib'")

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(full_df, duration_col="time_to_os", event_col="os")

    # Print summary
    cph.print_summary()

    # Save the model
    with open("new_cox/cox_model_full.pkl", "wb") as f:
        pickle.dump(cph, f)

    print("Cox model trained on train0 + test0 and saved as 'cox_model_full.pkl'.")


def predict_with_cox_save(model_path: str, data_csv: str, output_json: str):
    """
    Load a saved Cox model, predict on a CSV dataset, and save results as JSON.
    """

    # Columns to keep and categorical variables (hard-coded)
    cols_to_keep = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]
    categorical_vars = ["gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad"]

    # Load model
    with open(model_path, "rb") as f:
        cph = pickle.load(f)

    # Load test data
    df = pd.read_csv(data_csv, sep=",")

    # Keep only relevant columns
    df = df[cols_to_keep]

    # Keep original for os and time_to_os
    df_original = df.copy()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    # Align columns with model
    df = df.reindex(columns=list(cph.params_.index), fill_value=0)

    # Predict expected survival
    predicted = cph.predict_expectation(df)

    # Save JSON
    res_json = {
        "predicted": predicted.tolist(),
        "os": df_original["os"].tolist(),
        "real": df_original["time_to_os"].tolist()
    }
    with open(output_json, "w") as f:
        json.dump(res_json, f, indent=4)

    print(f"Predictions saved to '{output_json}'")


if __name__ == '__main__':
    # for i in range(5):
    #     cox(i)
    train_full_cox_model()
    # predict_with_cox_save(
    #     model_path="cox/cox_model_full.pkl",
    #     data_csv="./Folds_Dimitris/fold_4_test_data.csv",
    #     output_json="fold_4_cox_result.json"
    # )

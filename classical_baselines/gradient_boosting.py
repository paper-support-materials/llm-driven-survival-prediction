import os

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from files_utilities import *


def perform_decision(base_data_dir="Folds_Dimitris", result_folder="gradient_boosting"):
    os.makedirs(result_folder, exist_ok=True)
    folds = ["fold_" + str(x) for x in range(5)]
    df_train = {}
    df_test = {}
    for fold in folds:
        train_csv = os.path.join(base_data_dir, f"{fold}_train_data.csv")
        test_csv = os.path.join(base_data_dir, f"{fold}_test_data.csv")

        print(train_csv)

        # Load CSV files for the current fold
        df_train[fold] = pd.read_csv(train_csv)
        df_test[fold] = pd.read_csv(test_csv)

        x_train = df_train[fold]
        x_test = df_test[fold]

        y_train = Surv.from_dataframe("os", "time_to_os", df_train[fold])
        y_test = Surv.from_dataframe("os", "time_to_os", df_test[fold])

        x_train = x_train.drop(columns=["os", "time_to_os"])

        events = x_test["os"]
        real_times = x_test["time_to_os"]
        x_test = x_test.drop(columns=["os", "time_to_os"])

        categorical_vars = ["gesl", "tumsoort", "diag_basis", "topo_sublok", "later",
                            "morf", "gedrag", "diffgrad", "stadium", "uitgebr_chir_code",
                            "multifoc"]

        x_train = pd.get_dummies(x_train, columns=categorical_vars, drop_first=True)
        x_train.dropna(inplace=True)

        x_test = pd.get_dummies(x_test, columns=categorical_vars, drop_first=True)
        x_test.dropna(inplace=True)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=5, random_state=42)
        print("qui")

        model.fit(x_train_scaled, y_train)

        survival_funcs = model.predict_survival_function(x_test_scaled)

        y_pred = np.array([np.trapz(surv.y, surv.x) for surv in survival_funcs])

        survival_results = {
            "predicted": [int(x) for x in y_pred],
            "real": [int(x) for x in y_test["time_to_os"]],
            "os": [int(x) for x in y_test["os"]]
        }

        write_json(os.path.join(result_folder, f"{fold}.json"), survival_results)


if __name__ == '__main__':
    perform_decision()

import os

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from scipy.stats import norm
from lifelines.utils import concordance_index

from files_utilities import *


def metrics(predictions: list, labels: list) -> dict:
    pred_days = np.array(predictions)
    true_days = np.array(labels)
    mse = ((pred_days - true_days) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(pred_days - true_days).mean()
    return {"mse": mse, "rmse": rmse, "mae": mae}


# def brier_score_calculation(predictions: list, survival_train: np.ndarray, survival_test: np.ndarray):
#     pred_survival_times = predictions
#     pred_std = np.std(pred_survival_times)
#     time_points = np.linspace(365, 3650, 10)
#     estimate = np.array([
#         1 - norm.cdf(time_points, loc=mu, scale=pred_std)
#         for mu in pred_survival_times
#     ])
#
#     brier_scores = brier_score(survival_train, survival_test, estimate, time_points)
#     return {"average": float(np.mean(brier_scores[1])), "scores": list(brier_scores[1]), "times": list(brier_scores[0])}


def brier_score_calculation(predictions: list, survival_train: np.ndarray, survival_test: np.ndarray):
    pred_survival_times = np.array(predictions)
    pred_std = np.std(pred_survival_times)

    time_field = survival_train.dtype.names[1]
    max_time = float(survival_train[time_field].max())

    # maschera per tagliare il test
    mask = survival_test[time_field] < max_time

    survival_test = survival_test[mask]
    pred_survival_times = pred_survival_times[mask]  # <-- QUI IL FIX

    # griglia del tempo
    time_points = np.linspace(365, 3650, 10)
    time_points = time_points[time_points < max_time]

    estimate = np.array([
        1 - norm.cdf(time_points, loc=mu, scale=pred_std)
        for mu in pred_survival_times
    ])

    print("max_train:", max_time)
    print("max_test_trimmed:", survival_test[time_field].max())
    print("predictions:", pred_survival_times.shape)
    print("test:", survival_test.shape)

    brier_scores = brier_score(survival_train, survival_test, estimate, time_points)

    return {
        "average": float(np.mean(brier_scores[1])),
        "scores": list(brier_scores[1]),
        "times": list(brier_scores[0])
    }


def eval_cox_files(base_res_dir, base_data_dir="Folds_Dimitris", result_file="cox_5_folds_results.json"):
    folds = ["fold_" + str(x) for x in range(5)]
    df_train = {}
    df_test = {}
    results_metrics = {}
    averages = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "brier_score": [],
        "c-index": [],
        "c-index-ipcw": [],
        "c-index-lifeline": [],
        "c-index-lifeline_not_censored": [],
        "AUC": []
    }

    for fold in folds:
        train_csv = os.path.join(base_data_dir, f"{fold}_train_data.csv")
        test_csv = os.path.join(base_data_dir, f"{fold}_test_data.csv")

        # Load CSV files for the current fold
        df_train[fold] = pd.read_csv(train_csv)
        df_test[fold] = pd.read_csv(test_csv)

    res_dir_files = list(os.listdir(base_res_dir))
    res_dir_files = [x for x in res_dir_files if x.endswith(".json")]
    for i, file in enumerate(res_dir_files):
        res = load_json(os.path.join(base_res_dir, file))
        predictions = [res[x] for x in res]
        labels = df_test[folds[i]]["time_to_os"].values
        results_metrics[folds[i]] = metrics(predictions, labels)
        averages["mse"].append(results_metrics[folds[i]]["mse"])
        averages["rmse"].append(results_metrics[folds[i]]["rmse"])
        averages["mae"].append(results_metrics[folds[i]]["mae"])
        event_indicator = df_test[folds[i]]["os"].values
        event_indicator = event_indicator.astype(bool)
        survival_train = Surv.from_dataframe("os", "time_to_os", df_train[folds[i]])
        survival_test = Surv.from_dataframe("os", "time_to_os", df_test[folds[i]])
        results_metrics[folds[i]]["brier_scores"] = brier_score_calculation(predictions, survival_train, survival_test)
        averages["brier_score"].append(results_metrics[folds[i]]["brier_scores"]["average"])

        predicted_times = np.array([-int(x) for x in predictions])  # Model predictions
        # Compute C-index considering censoring
        c_index = concordance_index_censored(event_indicator, labels, predicted_times)
        assert isinstance(c_index, tuple)
        results_metrics[folds[i]]["c-index"] = {
            "score": float(c_index[0]),
            "concordant": int(c_index[1]),
            "discordant": int(c_index[2]),
            "tied_risk": int(c_index[3]),
            "tied_time": int(c_index[4]),
        }
        averages["c-index"].append(float(c_index[0]))

        try:
            c_index = concordance_index_ipcw(survival_train, survival_test, predicted_times)
            assert isinstance(c_index, tuple)
            results_metrics[folds[i]]["c-index-ipcw"] = {
                "score": float(c_index[0]),
                "concordant": int(c_index[1]),
                "discordant": int(c_index[2]),
                "tied_risk": int(c_index[3]),
                "tied_time": int(c_index[4]),
            }
            averages["c-index-ipcw"].append(float(c_index[0]))

            # print(f"Fold {folds[i]} Test set C-index - with sksurv --> ipcw: {c_index_sksurv_ipcw[0]:.4f}")
        except Exception as e:
            print(f"An error occurred in {folds[i]} on concordance_ipcw calculation: {e}")

        true_survival = df_test[folds[i]]["time_to_os"].values
        pred_survival = np.array(predictions)
        c_index_lifelines_censored = concordance_index(true_survival, pred_survival, event_observed=event_indicator)

        print(f"Fold {folds[i]} Test set C-index - with lifelines --> censored: {c_index_lifelines_censored:.4f}")
        results_metrics[folds[i]]["c-index-lifeline"] = c_index_lifelines_censored
        averages["c-index-lifeline"].append(c_index_lifelines_censored)
        c_index_lifelines_all_data_not_censored = concordance_index(true_survival, pred_survival, event_observed=None)

        print(
            f"Fold {folds[i]} Test set C-index - with lifelines --> AllData_notCensored: {c_index_lifelines_all_data_not_censored:.4f}")
        results_metrics[folds[i]]["c-index-lifeline_not_censored"] = c_index_lifelines_all_data_not_censored
        averages["c-index-lifeline_not_censored"].append(c_index_lifelines_all_data_not_censored)

        try:
            time_points = np.arange(0, 3650, 365)  # example: every year up to 10 years
            pred_prob_matrix = np.array([1 - pred_survival[i] / true_survival.max() for i in range(len(pred_survival))])
            pred_prob_matrix = np.tile(pred_prob_matrix, (len(time_points), 1)).T
            cum_auc = cumulative_dynamic_auc(survival_train, survival_test, pred_prob_matrix, time_points)

            if not np.isnan(cum_auc[1]):
                averages["AUC"].append(round(cum_auc[1], 4))
                results_metrics[folds[i]]["AUC"] = round(cum_auc[1], 4)
            print(
                f"Fold {folds[i]} Test set Area under the ROC curve (AUC): {cum_auc[1]:.4f}")
        except Exception as e:
            print(f"An error occurred on RAC-AUC calculation: {e}")

    average_values = {}
    for metric in averages:
        score_value = np.array(averages[metric]).mean()
        average_values[metric] = score_value
        print(metric, score_value)

    results_metrics["fold_key_scores"] = averages
    results_metrics["fold_averages"] = average_values
    write_json(result_file, results_metrics)


def compute_metrics(base_data_dir="Folds_Dimitris",
                    base_pred_dir="llama_results",
                    base_pred_file_name="llama_results_",
                    result_file="llama_5_folds_results.json"):
    folds = ["fold_" + str(x) for x in range(5)]
    results_metrics = {}
    averages = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "brier_score": [],
        "c-index": [],
        "c-index-ipcw": [],
        "c-index-lifeline": [],
        "c-index-lifeline_not_censored": [],
        "AUC": []
    }

    for fold in folds:
        train_csv = os.path.join(base_data_dir, f"{fold}_train_data.csv")
        test_csv = os.path.join(base_data_dir, f"{fold}_test_data.csv")

        # Load CSV files for the current fold
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        # Load results
        pred_file = os.path.join(base_pred_dir, f"{base_pred_file_name}{fold}.json")
        eval_pred = load_json(pred_file)

        if eval_pred == {}:
            continue

        predictions, labels = [int(x) for x in eval_pred["predicted"]], [int(x) for x in eval_pred["real"]]

        pred_days = np.array(predictions)
        true_days = np.array(labels)

        results_metrics[fold] = metrics(predictions, labels)

        averages["mse"].append(results_metrics[fold]["mse"])
        averages["rmse"].append(results_metrics[fold]["rmse"])
        averages["mae"].append(results_metrics[fold]["mae"])
        event_indicator = df_test["os"].values
        event_indicator = event_indicator.astype(bool)
        survival_train = Surv.from_dataframe("os", "time_to_os", df_train)
        survival_test = Surv.from_dataframe("os", "time_to_os", df_test)
        results_metrics[fold]["brier_scores"] = brier_score_calculation(predictions, survival_train, survival_test)
        averages["brier_score"].append(results_metrics[fold]["brier_scores"]["average"])

        predicted_times = np.array([-int(x) for x in predictions])  # Model predictions
        # Compute C-index considering censoring
        c_index = concordance_index_censored(event_indicator, labels, predicted_times)
        assert isinstance(c_index, tuple)
        results_metrics[fold]["c-index"] = {
            "score": float(c_index[0]),
            "concordant": int(c_index[1]),
            "discordant": int(c_index[2]),
            "tied_risk": int(c_index[3]),
            "tied_time": int(c_index[4]),
        }
        averages["c-index"].append(float(c_index[0]))

        try:
            c_index = concordance_index_ipcw(survival_train, survival_test, predicted_times)
            assert isinstance(c_index, tuple)
            results_metrics[fold]["c-index-ipcw"] = {
                "score": float(c_index[0]),
                "concordant": int(c_index[1]),
                "discordant": int(c_index[2]),
                "tied_risk": int(c_index[3]),
                "tied_time": int(c_index[4]),
            }
            averages["c-index-ipcw"].append(float(c_index[0]))

            # print(f"Fold {folds[i]} Test set C-index - with sksurv --> ipcw: {c_index_sksurv_ipcw[0]:.4f}")
        except Exception as e:
            print(f"An error occurred in {fold} on concordance_ipcw calculation: {e}")

        true_survival = df_test["time_to_os"].values
        pred_survival = np.array(predictions)
        c_index_lifelines_censored = concordance_index(true_survival, pred_survival, event_observed=event_indicator)

        print(f"Fold {fold} Test set C-index - with lifelines --> censored: {c_index_lifelines_censored:.4f}")
        results_metrics[fold]["c-index-lifeline"] = c_index_lifelines_censored
        averages["c-index-lifeline"].append(c_index_lifelines_censored)
        c_index_lifelines_all_data_not_censored = concordance_index(true_survival, pred_survival, event_observed=None)

        print(
            f"Fold {fold} Test set C-index - with lifelines --> AllData_notCensored: {c_index_lifelines_all_data_not_censored:.4f}")
        results_metrics[fold]["c-index-lifeline_not_censored"] = c_index_lifelines_all_data_not_censored
        averages["c-index-lifeline_not_censored"].append(c_index_lifelines_all_data_not_censored)

        try:
            time_points = np.arange(0, 3650, 365)  # every year up to 10 years
            pred_prob_matrix = np.array([1 - pred_survival[i] / true_survival.max() for i in range(len(pred_survival))])
            pred_prob_matrix = np.tile(pred_prob_matrix, (len(time_points), 1)).T
            cum_auc = cumulative_dynamic_auc(survival_train, survival_test, pred_prob_matrix, time_points)
            if not np.isnan(cum_auc[1]):
                averages["AUC"].append(round(cum_auc[1], 4))
                results_metrics[fold]["AUC"] = round(cum_auc[1], 4)
            print(
                f"Fold {fold} Test set Area under the ROC curve (AUC): {cum_auc[1]:.4f}")
        except Exception as e:
            print(f"An error occurred on RAC-AUC calculation: {e}")

    average_values = {}
    for metric in averages:
        if len(averages[metric]) > 0:
            score_value = round(np.array(averages[metric]).mean(), 4)
            average_values[metric] = score_value
            print(metric, score_value)

    results_metrics["fold_key_scores"] = averages
    results_metrics["fold_averages"] = average_values
    write_json(result_file, results_metrics)


def single_compute_metrics(
        base_pred_dir="real_data",
        pred_file_name="biobert_results_real_data_full.json",
        result_file="real_data_results_full.json"):
    averages = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "brier_score": [],
        "c-index": [],
        "c-index-ipcw": [],
        "c-index-lifeline": [],
        "c-index-lifeline_not_censored": [],
        "AUC": []
    }

    # Load results
    pred_file = os.path.join(base_pred_dir, pred_file_name)
    eval_pred = load_json(pred_file)

    if eval_pred == {}:
        return

    predictions, labels, event_indicator = ([int(x) for x in eval_pred["predicted"]],
                                            [int(x) for x in eval_pred["real"]],
                                            [x == 1 for x in eval_pred["os"]])

    # pred_days = np.array(predictions)
    # true_days = np.array(labels)

    results_metrics = metrics(predictions, labels)

    df_eval = pd.DataFrame({
        "os": event_indicator,
        "time_to_os": labels,
        "predicted_days": predictions
    })

    full_dataset = load_from_disk("datasets/full_data")
    df_train = full_dataset.to_pandas()

    survival_train = Surv.from_dataframe("os", "time_to_os", df_train)
    survival_test = Surv.from_dataframe("os", "time_to_os", df_eval)
    results_metrics["brier_scores"] = brier_score_calculation(predictions, survival_train, survival_test)

    predicted_times = np.array([-int(x) for x in predictions])  # Model predictions
    # Compute C-index considering censoring
    c_index = concordance_index_censored(event_indicator, labels, predicted_times)
    assert isinstance(c_index, tuple)
    results_metrics["c-index"] = {
        "score": float(c_index[0]),
        "concordant": int(c_index[1]),
        "discordant": int(c_index[2]),
        "tied_risk": int(c_index[3]),
        "tied_time": int(c_index[4]),
    }
    averages["c-index"].append(float(c_index[0]))

    try:
        c_index = concordance_index_ipcw(survival_train, survival_test, predicted_times)
        assert isinstance(c_index, tuple)
        results_metrics["c-index-ipcw"] = {
            "score": float(c_index[0]),
            "concordant": int(c_index[1]),
            "discordant": int(c_index[2]),
            "tied_risk": int(c_index[3]),
            "tied_time": int(c_index[4]),
        }
        averages["c-index-ipcw"].append(float(c_index[0]))

        # print(f"Fold {folds[i]} Test set C-index - with sksurv --> ipcw: {c_index_sksurv_ipcw[0]:.4f}")
    except Exception as e:
        print(f"An error occurred on concordance_ipcw calculation: {e}")

    true_survival = labels
    pred_survival = np.array(predictions)
    c_index_lifelines_censored = concordance_index(true_survival, pred_survival, event_observed=event_indicator)

    print(f"Test set C-index - with lifelines --> censored: {c_index_lifelines_censored:.4f}")
    results_metrics["c-index-lifeline"] = c_index_lifelines_censored
    averages["c-index-lifeline"].append(c_index_lifelines_censored)
    c_index_lifelines_all_data_not_censored = concordance_index(true_survival, pred_survival, event_observed=None)

    print(
        f"Test set C-index - with lifelines --> AllData_notCensored: {c_index_lifelines_all_data_not_censored:.4f}")
    results_metrics["c-index-lifeline_not_censored"] = c_index_lifelines_all_data_not_censored
    averages["c-index-lifeline_not_censored"].append(c_index_lifelines_all_data_not_censored)

    try:
        # Time points must be within follow-up window
        tmin = df_eval["time_to_os"].min()
        tmax = df_eval["time_to_os"].max()
        time_points = np.arange(tmin, tmax, 365)

        # build risk scores from predicted survival days
        risk_scores = -np.array(pred_survival)

        # repeat scores over all time points
        pred_prob_matrix = np.tile(risk_scores[:, None], (1, len(time_points)))

        cum_auc = cumulative_dynamic_auc(
            survival_train,
            survival_test,
            pred_prob_matrix,
            time_points
        )

        auc = cum_auc[1]
        if not np.isnan(auc):
            averages["AUC"].append(round(auc, 4))
            results_metrics["AUC"] = round(auc, 4)

        print(f"AUC: {auc:.4f}")

    except Exception as e:
        print(f"AUC error: {e}")

    results_metrics["fold_key_scores"] = averages
    write_json(result_file, results_metrics)


def post_process_llama_70_results(filename):
    data = load_json(filename)
    print(data.keys())
    clean_predicted = []
    for pred in data["predicted"]:
        pred = pred[:5]
        clean_pred = []
        for x in pred:
            if x.isnumeric():
                clean_pred.append(x)
            else:
                break
        # clean_pred = [x for x in pred if x.isnumeric()]
        clean_pred = "".join(clean_pred)
        clean_predicted.append(clean_pred)
    data["dirty_predicted"] = data["predicted"]
    data["predicted"] = clean_predicted
    write_json(filename, data)


def change_sign(filename):
    data = load_json(filename)
    data["predicted"] = [-x for x in data["predicted"]]
    write_json(filename, data)


def get_metrics_all_files(folder):
    base_data_dir = "Folds_Dimitris"
    files = [x for x in os.listdir(folder) if x.endswith(".json") and "metrics" not in x]
    base_metrics = {}
    bests = {"brier": {}, "c_index": {}}
    for file in files:
        fold = "_".join(file.replace(".json", "").split("_")[-2:])
        name = "_".join(file.replace(".json", "").split("_")[:-2])

        train_csv = os.path.join(base_data_dir, f"{fold}_train_data.csv")
        test_csv = os.path.join(base_data_dir, f"{fold}_test_data.csv")
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        event_indicator = df_test["os"].values
        event_indicator = event_indicator.astype(bool)
        survival_train = Surv.from_dataframe("os", "time_to_os", df_train)
        survival_test = Surv.from_dataframe("os", "time_to_os", df_test)

        if fold not in base_metrics:
            base_metrics[fold] = {}
        if fold not in bests["brier"]:
            bests["brier"][fold] = {}
        if fold not in bests["c_index"]:
            bests["c_index"][fold] = {}
        data = load_json(os.path.join(folder, file))
        base_metrics[fold][name] = metrics(data.get("predicted"), data.get("real"))
        brier_score_value = brier_score_calculation(data.get("predicted"), survival_train, survival_test).get("average")
        base_metrics[fold][name]["brier_score"] = brier_score_value
        if bests["brier"][fold].get("score") is None or bests["brier"][fold].get("score") > brier_score_value:
            bests["brier"][fold]["score"] = brier_score_value
            bests["brier"][fold]["name"] = file
        c_index_value = concordance_index(data.get("real"), data.get("predicted"), event_observed=event_indicator)
        base_metrics[fold][name]["c_index"] = c_index_value
        if bests["c_index"][fold].get("score") is None or bests["c_index"][fold].get("score") < c_index_value:
            bests["c_index"][fold]["score"] = c_index_value
            bests["c_index"][fold]["name"] = file

    base_metrics["bests"] = bests
    write_json(os.path.join(folder, "metrics.json"), base_metrics)


if __name__ == '__main__':
    pass
    # eval_cox_files("ml_method_results/cox")
    # compute_metrics(base_pred_dir="ml_method_results/tree_results", result_file="tree_5_fold_results.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="tree_results_3", result_file="tree_5_fold_results_4.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="ml_method_results/random_forest", result_file="random_forest_5_fold_results.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="ml_method_results/gradient_boosting", result_file="gradient_boosting_5_fold_results.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="model_results/llama_results", result_file="med_llama_fold_0.json", base_pred_file_name="llama_results_")
    # compute_metrics(base_pred_dir="model_results/t5-base_new_prompt_results", result_file="t5-base_fold_0.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="model_results/med_llama_results", result_file="med_llama_fold_0.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="llama_results_old_prompt", result_file="llama_results_old_prompt_fold_0.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="model_results/t5-base_new_prompt_results", result_file="t5_new_prompt_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="t5-small_5_folds", result_file="t5-small_results_5_folds.json", base_pred_file_name="t5-small_finetuned_")
    # compute_metrics(base_pred_dir="model_results/flan_t5-large_results", result_file="flan_t5-large_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="model_results/t5-large_results", result_file="t5-large_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="t5-base_2_epochs_results", result_file="t5-base_2_epochs_results.json", base_pred_file_name="t5-base_finetuned_")
    # post_process_llama_70_results(os.path.join("llama-70B_results", "llama_results_fold_4.json"))
    # compute_metrics(base_pred_dir="llama-70B_results", result_file="Llama70_results_5_folds.json", base_pred_file_name="llama_results_")

    # change_sign("biobert/biobert_fold_0.json")
    # compute_metrics(base_pred_dir="biobert", result_file="biobert_results_5_folds.json", base_pred_file_name="biobert_results_")
    # compute_metrics(base_pred_dir="biobert_2_epochs", result_file="biobert_2_epochs_results_5_folds.json", base_pred_file_name="biobert_results_")
    # post_process_llama_70_results(os.path.join("medlama_3-8B-v2", "medllama_results_fold_4.json"))
    # compute_metrics(base_pred_dir="medlama_3-8B-v2", result_file="medlama_3-8B-v2_results_5_folds.json", base_pred_file_name="medllama_results_")

    # compute_metrics(base_pred_dir="t5_small_mae", result_file="t5_small_mae_2_epoch_results_5_folds.json", base_pred_file_name="t5-small_results_")
    # compute_metrics(base_pred_dir="t5-base_smoot_mae", result_file="t5-base_mae_2_epoch_results_5_folds.json", base_pred_file_name="t5-base_results_")
    # compute_metrics(base_pred_dir="t5-base_survival_2_epoch", result_file="t5-base_mae_2_epoch_results_5_folds.json", base_pred_file_name="t5-base_results_")
    # compute_metrics(base_pred_dir="qwen2", result_file="Qwen2_results_5_folds.json", base_pred_file_name="qwen_results_")
    # compute_metrics(base_pred_dir="t5-large_mae", result_file="t5-large_mae_results_5_folds.json", base_pred_file_name="t5-large_results_")
    # get_metrics_all_files("t5-small_att_pool")
    # compute_metrics(base_pred_dir="t5-base_att_pool", result_file="t5-base_att_pooling_1_epoch_results_5_folds.json", base_pred_file_name="t5-base_epoch_0_step_6000_results_")
    # compute_metrics(base_pred_dir="t5-small_att_pool", result_file="t5-small_att_pooling_1_epoch_results_5_folds.json", base_pred_file_name="t5-small_epoch_0_step_6000_results_")
    # compute_metrics(base_pred_dir="t5-small_mean_pool", result_file="t5-small_mean_pool_1_epoch_results_5_folds.json", base_pred_file_name="t5-small_epoch_0_step_6000_results_")
    # compute_metrics(base_pred_dir="t5-base_mean_pool", result_file="t5-base_mean_pool_1_epoch_results_5_folds.json", base_pred_file_name="t5-base_epoch_0_step_6000_results_")
    # compute_metrics(base_pred_dir="ensemble/avg", result_file="ens_avg_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="ensemble/robust_median", result_file="ens_robust_median_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="ensemble/ens_trim_mean", result_file="ens_trim_mean_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="ensemble/c-index_weighted", result_file="ens_c-index_weighted_results_5_folds.json", base_pred_file_name="")
    # compute_metrics(base_pred_dir="real_data", result_file="real_data/result.json", base_pred_file_name="biobert_results_real_data_")
    # single_compute_metrics(base_pred_dir="real_data", pred_file_name="cox_real_data_result.json", result_file="cox_real_data_result_metrics.json")
    single_compute_metrics(base_pred_dir="real_data", pred_file_name="gb_real_data_results.json",
                           result_file="gb_real_data_result_metrics.json")
    # single_compute_metrics(base_pred_dir="new_cox", pred_file_name="fold_4_cox_result.json",
    #                        result_file="new_cox/fold_4-results.json")

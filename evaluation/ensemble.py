import os
from operator import indexOf

import numpy as np
from scipy.stats import trim_mean
from files_utilities import *

folds = [f"fold_{x}" for x in range(5)]
folders = ["biobert_2_epochs", "t5-base_2_epochs", "t5-base_att_pool_1_epoch"]
base_dir = "ensemble"


def get_data():
    data = load_json("ensemble/ens_data.json")
    data["metrics"] = {}
    for folder in folders:
        metrics = load_json(f"ensemble/{folder}.json")
        print(metrics.keys())
        data["metrics"][folder] = {"c-index": metrics["fold_key_scores"]["c-index"],
                                   "mae": metrics["fold_key_scores"]["mae"]}
        for fold in folds:
            if fold not in data:
                data[fold] = {}
            print(os.path.join(base_dir, folder, fold)+".json")
            res = load_json(os.path.join(base_dir, folder, fold)+".json")
            if "real" not in data[fold]:
                data[fold]["real"] = res["real"]
                data[fold]["os"] = res["os"]
            if "predicted" not in data[fold]:
                data[fold]["predicted"] = {}
            data[fold]["predicted"][folder] = [int(x) for x in res["predicted"]]
    write_json("ensemble/ens_data.json", data)


def average_results():
    data = load_json("ensemble/ens_data.json")
    for fold in data:
        res = data[fold]["predicted"]
        models = list(res.keys())
        ens_avg = []
        for i, pred in enumerate(res[models[0]]):
            avg = res[models[0]][i] + res[models[1]][i] + res[models[2]][i]
            avg = round(avg / 3)
            ens_avg.append(avg)
        data[fold]["ens_avg"] = ens_avg
    write_json("ensemble/ens_data.json", data)


def closest_pair_average(p1, p2, p3):
    d12 = abs(p1 - p2)
    d13 = abs(p1 - p3)
    d23 = abs(p2 - p3)

    if d12 <= d13 and d12 <= d23:
        return round((p1 + p2) / 2)
    elif d13 <= d12 and d13 <= d23:
        return round((p1 + p3) / 2)
    else:
        return round((p2 + p3) / 2)


def median_ensemble():
    data = load_json("ensemble/ens_data.json")
    for fold in data:
        if fold not in folds:
            continue
        res = data[fold]["predicted"]
        models = list(res.keys())
        ensemble_preds = np.median(np.stack([res[models[0]], res[models[1]], res[models[2]]]), axis=0)
        data[fold]["ens_robust_median"] = ensemble_preds.round().astype(int).tolist()

        stacked = np.stack([res[models[0]], res[models[1]], res[models[2]]])
        trimmed = trim_mean(stacked, proportiontocut=0.1, axis=0)
        data[fold]["ens_trim_mean"] = np.round(trimmed).astype(int).tolist()

        data[fold]["ens_closest_pair"] = [
            closest_pair_average(a, b, c)
            for a, b, c in zip(res[models[0]], res[models[1]], res[models[2]])
        ]
        maes = [data["metrics"][models[x]]["mae"][indexOf(folds, fold)] for x in range(3)]
        data[fold]["mae_weighted"] = weighted_ensemble([res[models[0]], res[models[1]], res[models[2]]], maes, mode='mae')

        c_index = [data["metrics"][models[x]]["c-index"][indexOf(folds, fold)] for x in range(3)]
        data[fold]["c-index_weighted"] = weighted_ensemble([res[models[0]], res[models[1]], res[models[2]]], c_index, mode='cindex')

    write_json("ensemble/ens_data.json", data)


def write_data(out_dir):
    os.makedirs(os.path.join(base_dir, out_dir), exist_ok=True)
    data = load_json("ensemble/ens_data.json")
    for fold in data:
        if fold not in folds:
            continue
        res = {"real": data[fold]["real"], "os": data[fold]["os"], "predicted": data[fold][out_dir]}
        write_json(os.path.join(base_dir, out_dir, fold+".json"), res)


def weighted_ensemble(preds_list, metrics, mode='mae'):
    """
    Compute a weighted ensemble of predictions.

    Args:
        preds_list (list of lists): Each sublist contains predictions from a model.
        metrics (list of floats): Performance scores for each model (MAE or C-index).
        mode (str): 'mae' (lower is better) or 'cindex' (higher is better).

    Returns:
        list of ints: Final ensemble predictions as integers.
    """
    if len(preds_list) != len(metrics):
        raise ValueError("Number of prediction sets and metrics must match.")

    preds_array = np.array(preds_list)

    if mode == 'mae':
        weights = 1 / np.array(metrics)
    elif mode == 'cindex':
        weights = np.array(metrics)
    else:
        raise ValueError("Mode must be 'mae' or 'cindex'.")

    weights /= weights.sum()  # Normalize

    weighted_preds = np.average(preds_array, axis=0, weights=weights)
    return weighted_preds.round().astype(int).tolist()


if __name__ == '__main__':
    pass
    get_data()
    average_results()
    write_data("ens_avg")
    median_ensemble()
    write_data("ens_robust_median")
    write_data("ens_trim_mean")
    write_data("ens_closest_pair")
    write_data("c-index_weighted")


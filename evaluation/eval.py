import os

from lifelines.utils import concordance_index
import numpy as np
from files_utilities import load_json, write_json


def eval_results(res, real, predicted, event):
    response = {}
    actual_times = np.array([int(x) for x in res[real]])  # True survival times
    predicted_times = np.array([int(x) for x in res[predicted]])  # Model predictions
    event_observed = [x == "1" or x == 1 for x in res[event]]  # 1 if the event (death) occurred, 0 if censored

    c_index = concordance_index(actual_times, predicted_times, event_observed)
    print(f"Concordance Index (C-Index) with Censoring on entire test set:  {c_index:.4f}")
    response["full"] = c_index

    actual_times = np.array(res[real])
    predicted_times = np.array(res[predicted])
    event_observed = np.array([x for x in res[event]])

    c_index = concordance_index(actual_times, predicted_times)
    print(f"Concordance Index (C-Index) without Censoring on entire test set: {c_index:.4f}")
    response["no_censoring"] = c_index

    r = []
    p = []

    for i, _ in enumerate(res["real"]):
        if res["os"][i] == "0":
            r.append(res["real"][i])
            p.append(res["predicted"][i])

    c_index = concordance_index(r, p)
    print(f"Concordance Index (C-Index) without Censoring and only on alive ones: {c_index:.4f}")
    response["alive_only"] = c_index
    return response


def eval_results_new(res, real, predicted, event):
    response = {}

    # Convert to numeric
    actual_times = np.array(res[real], dtype=float)
    predicted_times = np.array(res[predicted], dtype=float)
    event_observed = np.array(res[event], dtype=int)

    # Check there is variation
    if event_observed.sum() == 0 or event_observed.sum() == len(event_observed):
        print("Warning: all events are censored or all observed. Skipping censored concordance.")
        c_index = concordance_index(actual_times, predicted_times)  # no censoring
        response["full"] = c_index
    else:
        c_index = concordance_index(actual_times, predicted_times, event_observed)
        response["full"] = c_index
    print(f"Concordance Index (C-Index) with Censoring: {c_index:.4f}")

    # No censoring
    c_index = concordance_index(actual_times, predicted_times)
    print(f"Concordance Index (C-Index) without Censoring: {c_index:.4f}")
    response["no_censoring"] = c_index

    # Alive only (event==0)
    mask = event_observed == 0
    if mask.sum() == 0:
        print("No alive patients in the test set for this calculation.")
        response["alive_only"] = None
    else:
        c_index = concordance_index(actual_times[mask], predicted_times[mask])
        print(f"Concordance Index (C-Index) on alive only: {c_index:.4f}")
        response["alive_only"] = c_index

    return response


def write_comparable(res, namefile="compare_view.json"):
    new_res = []
    for i, a in enumerate(res["real"]):
        row = {"real": res["real"][i], "predicted": res["predicted"][i], "os": res["os"][i]}
        new_res.append(row)
    write_json(namefile, new_res)


def eval_all_folds(fold):
    ref_data = load_json("llama_results.json")
    total_eval_results = {"full": [], "no_censoring": [], "alive_only": []}
    for file in os.listdir(fold):
        print(file)
        if not file.endswith(".json"):
            continue
        res = load_json(os.path.join(fold, file))
        res = [res[x] for x in res]
        new_res = {"predicted": res, "os": ref_data["os"], "real": ref_data["real"]}
        # print(len(res), len(ref_data["os"]), len(ref_data["real"]))
        eval_res = eval_results(new_res, "real", "predicted", "os")
        total_eval_results["full"].append(eval_res["full"])
        total_eval_results["no_censoring"].append(eval_res["no_censoring"])
        total_eval_results["alive_only"].append(eval_res["alive_only"])
    averages = {"full": sum(total_eval_results["full"]) / len(total_eval_results["full"]),
                "no_censoring": sum(total_eval_results["no_censoring"]) / len(total_eval_results["no_censoring"]),
                "alive_only": sum(total_eval_results["alive_only"]) / len(total_eval_results["alive_only"]),
                }
    write_json("total_result_" + fold + ".json", {"average": averages, "results": total_eval_results})

    print(f"Average C-Index (5 fold) with Censoring on entire test set:  {averages["full"]:.4f}")
    print(f"Average C-Index (5 fold) without Censoring on entire test set:  {averages["no_censoring"]:.4f}")
    print(f"Average C-Index (5 fold) without Censoring and only on alive ones  {averages["alive_only"]:.4f}")


if __name__ == '__main__':
    # results = load_json("cox/fold_0_cox_result.json")
    # results = [results[x] for x in results]
    # data = load_json("llama_results.json")
    # new_results = {"predicted": results, "os": data["os"], "real": data["real"]}
    # write_comparable(new_results, "compare_view_cox_fold_0.json")
    # eval_results(new_results, "real", "predicted", "os")
    # eval_all_folds("cox")
    data = load_json("fold_0_cox_result.json")
    eval_results_new(data, "real", "predicted", "os")

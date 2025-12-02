from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
import numpy as np
from files_utilities import load_json, write_json


def eval_results(results, real, predicted, event, n=12000):
    # Extract actual and predicted survival times
    actual_times = np.array([int(x) for x in results["real"][:n]])  # True survival times
    predicted_times = np.array([-int(x) for x in results["predicted"][:n]])  # Model predictions
    event_observed = [x == 1 for x in results["os"][:n]]  # 1 if the event (death) occurred, 0 if censored
    print(type(event_observed[0]))
    # Compute C-index considering censoring
    c_index = concordance_index_censored(event_observed, actual_times, predicted_times)
    print(f"Concordance Index (concordance_index_censored from sksurv) - all test set:", c_index)



    # Extract actual and predicted survival times
    actual_times = np.array(results["real"])  # True survival times
    predicted_times = np.array(results["predicted"])  # Model predictions
    event_observed = np.array([x for x in results["os"]])  # 1 if the event (death) occurred, 0 if censored

    c_index = concordance_index(actual_times, predicted_times, event_observed)
    print(f"Concordance Index (C-Index) (using 'os' data) -entire test set:  {c_index:.4f}")

    # Compute C-index considering censoring
    c_index = concordance_index(actual_times, predicted_times)
    print(f"Concordance Index (C-Index) (without using 'os' data) - entire test set: {c_index:.4f}")

    r = []
    p = []

    for i, _ in enumerate(results["real"]):
        if results["os"][i] == 0:
            r.append(results["real"][i])
            p.append(results["predicted"][i])

    c_index = concordance_index(r, p)
    print(f"Concordance Index (C-Index) (without using 'os' data) only sample with 'os' == 0: {c_index:.4f}")

    # res = []
    # for i, a in enumerate(results["real"]):
    #     res.append(row)
    #
    # write_json("vista.json", res)


if __name__ == '__main__':
    results = load_json("biobert_results_real_data_3000.json")
    print(len(results["os"]))
    # results = load_json("t5-large_results/fold_0.json")
    eval_results(results, "real", "predicted", "os")

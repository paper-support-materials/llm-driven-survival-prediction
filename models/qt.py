import os
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from datasets import load_from_disk

datasets_path = "datasets"


def prepare_and_fit(fold):
    data = load_from_disk(os.path.join(datasets_path, f"train{fold}"))
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    times = qt.fit_transform(np.array(data["time_to_os"]).reshape(-1, 1)).squeeze()
    print(times)
    joblib.dump(qt, f"qt_{fold}.joblib")
    # to reload
    # qt = joblib.load("filename.joblib")


if __name__ == '__main__':
    folds = [f"fold_{x}" for x in range(5)]
    for fold in folds:
        prepare_and_fit(fold)


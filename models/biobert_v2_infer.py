import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset as Torch_dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from lifelines.utils import concordance_index
from datasets import Dataset, load_from_disk
from sklearn.preprocessing import QuantileTransformer
import torch.nn.functional as F
import pandas as pd
from glossary import *

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Load BioBERT tokenizer
MODEL_NAME = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

folds = [f"fold_{x}" for x in range(5)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_real_data_dataset(filename="real_data.csv"):
    base_data_dir = "csv_data/"
    cols_to_keep = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]

    # renamed for clarity, we’ll align these with model expectations
    cols_needed_names = [
        "gesl", "tumsoort", "topo_sublok", "later", "morf", "gedrag", "diffgrad",
        "er_stat", "pr_stat", "her2_stat", "leeft", "incjr", "time_to_os", "os"
    ]

    csv_data = os.path.join(base_data_dir, filename)
    df = pd.read_csv(csv_data, sep=";")
    print(df.columns)

    # safety check because humans never name columns properly
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input CSV: {missing_cols}")

    df = df[cols_to_keep].copy()
    print(df)
    df.columns = cols_needed_names
    print(df)
    # Quantile-transform time_to_os
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    df["time_to_os_transformed"] = qt.fit_transform(df[["time_to_os"]]).squeeze()
    print(df)
    # Convert to Hugging Face dataset
    data_hf = Dataset.from_pandas(df)
    os.makedirs("datasets", exist_ok=True)
    data_hf.save_to_disk("datasets/real_data")
    joblib.dump(qt, "datasets/qt_real_data.joblib")

    print("Dataset saved to datasets/real_data")
    print("QuantileTransformer saved to datasets/qt_real_data.joblib")
    print(data_hf)


def build_prompt(example):
    """
    Construct a descriptive string prompt from the row's data,
    mapping coded values to human-readable text.
    """
    gender_str = gender_map.get(example["gesl"], f"Unknown(gender={example['gesl']})")
    tumor_type_str = tumor_type_map.get(example["tumsoort"], f"Unknown(tumsoort={example['tumsoort']})")
    behavior_str = behavior_map.get(example["gedrag"], f"Unknown(gedrag={example['gedrag']})")
    diff_str = diffgrad_map.get(example["diffgrad"], f"Unknown(diff={example['diffgrad']})")
    later_key = str(example["later"])
    later_str = later_map.get(later_key, f"Unknown(later={example['later']})")
    topo_sub_str = topo_sublok_map.get(example["topo_sublok"], f"Unknown(topo_sublok={example['topo_sublok']})")
    morph_str = morphology_map.get(example["morf"], f"Unknown(morf={example['morf']})")
    er_str = hr_stat_map.get(example["er_stat"], f"Unknown(ER={example['er_stat']})")
    pr_str = hr_stat_map.get(example["pr_stat"], f"Unknown(PR={example['pr_stat']})")
    her2_str = her2_stat_map.get(example["her2_stat"], f"Unknown(HER2={example['her2_stat']})")
    # Handle cases where survival status may be missing (in inference)
    if "os" in example:
        status_str = "Alive" if example["os"] == 0 else "Deceased"
        status_line = f"Patient status: {status_str}."
    else:
        status_line = "Patient status: Unknown (predict survival time based on patient data)."

    prompt_lines = [
        f"Predict the survival time in days based on the following patient information:",
        f"Patient age: {int(example['leeft'])} years old.",
        f"Gender: {gender_str}.",
        f"Incident year: {int(example['incjr'])}.",
        f"Tumor type: {tumor_type_str}, morphology: {morph_str}, behavior: {behavior_str}.",
        f"Location: {later_str} breast, sublocation: {topo_sub_str}.",
        f"Differentiation grade: {diff_str}.",
        f"ER status: {er_str}, PR status: {pr_str}, HER2 status: {her2_str}.",
        status_line  # Adaptively includes status or a guiding note if missing
    ]
    prompt_text = "\n".join(prompt_lines)
    prompt_text += "\nSurvival time in days: "
    return prompt_text


def preprocess_function(example):
    """
    Build the prompt for an example, tokenize it, and attach the label.
    """
    prompt_text = build_prompt(example)
    time = example["time_to_os"]
    event = example["os"]

    return {"text": prompt_text, "os": event, "time_to_os": time}


# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


# Create dataset
class SurvivalDataset(Torch_dataset):
    def __init__(self, texts, times, events):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["time"] = self.times[idx]
        item["event"] = self.events[idx]
        return item


class BioBERTSurvivalModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(BioBERTSurvivalModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # Single output

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        risk_score = self.regressor(cls_embedding)
        return risk_score.view(-1)


def mae_loss(predicted_survival, actual_survival):
    return F.l1_loss(predicted_survival, actual_survival)


def evaluate_model(model, dataloader, result_name, qt):
    model.eval()
    pred_times, true_times, true_events = [], [], []

    print("Evaluating model")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            times = batch["time"].cpu().numpy()
            events = batch["event"].cpu().numpy()
            predicted_times = model(input_ids, attention_mask).cpu().numpy()

            # Convert predicted times back to original scale
            predicted_times = qt.inverse_transform(predicted_times.reshape(-1, 1)).squeeze()

            pred_times.extend(predicted_times)
            true_times.extend(times)
            true_events.extend(events)

    results = {
        "predicted": [round(x) for x in pred_times],
        "os": [int(x) for x in true_events],
        "real": [round(x) for x in true_times]
    }

    with open(f"biobert_results_{result_name}.json", "w") as f:
        json.dump(results, f)

    # Compute Concordance Index
    c_index = concordance_index(true_times, np.array(pred_times), true_events)
    print(f"Concordance Index: {c_index:.4f}")


def get_dataloader(dataset):
    dataset = dataset.map(preprocess_function, batched=False)
    texts = [x["text"] for x in dataset]
    times = [int(x["time_to_os"]) for x in dataset]
    events = [int(x["os"]) for x in dataset]
    # Create dataset
    infer_dataset = SurvivalDataset(texts, times, events)
    dataloader = DataLoader(infer_dataset, batch_size=2, shuffle=False)
    return dataloader


def main(model_path=None, dataset=None, result_name="real_data"):

    if (model_path is not None and os.path.isfile(model_path)
            and dataset is not None and os.path.isdir(f"datasets/{dataset}")):
        dataset = load_from_disk(f"datasets/{dataset}")
        dataloader = get_dataloader(dataset)
        qt = joblib.load("datasets/qt_full_data.joblib")

        model = BioBERTSurvivalModel()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        # Run evaluation
        evaluate_model(model, dataloader, result_name, qt)
    else:
        print("Nothing to do", os.path.isfile(model_path), os.path.isdir(f"datasets/test{dataset}"))


if __name__ == '__main__':
    get_real_data_dataset()
    main(model_path="biobert_2/biobert_survival_all_data.pth", dataset="real_data")

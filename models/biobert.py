import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from datasets import load_from_disk
from glossary import *

from huggingface_hub import login
login()

# Load BioBERT tokenizer
MODEL_NAME = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# **Load Dataset (Example: Yelp Reviews)**
dataset = load_from_disk("datasets/trainfold_0")
test_dataset = load_from_disk("datasets/testfold_0")


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
    prompt_text += "\nfSurvival time in days: "
    return prompt_text


def preprocess_function(example):
    """
    Build the prompt for an example, tokenize it, and attach the label.
    """
    prompt_text = build_prompt(example)
    time = example["time_to_os"]
    event = example["os"]

    return {"text": prompt_text, "os": event, "time_to_os": time}


dataset = dataset.map(preprocess_function, batched=False)
test_dataset = test_dataset.map(preprocess_function, batched=False)

texts = [x["text"] for x in dataset]
times = [x["time_to_os"] for x in dataset]
events = [x["os"] for x in dataset]

test_texts = [x["text"] for x in test_dataset]
test_times = [x["time_to_os"] for x in test_dataset]
test_events = [x["os"] for x in test_dataset]

# # Example dataset (replace with real data)
# texts = ["Patient has hypertension and diabetes.", "History of heart disease.", "No prior conditions."]
# times = [1000, 500, 1500]  # Survival times (days)
# events = [1, 1, 0]  # 1 = event occurred, 0 = censored


# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


# Create dataset
class SurvivalDataset(Dataset):
    def __init__(self, texts, times, events):
        self.encodings = tokenize_function(texts)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # ✅ Fix tensor copy warning
        item["time"] = self.times[idx]
        item["event"] = self.events[idx]
        return item


# Create train and test datasets
train_dataset = SurvivalDataset(texts, times, events)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)

test_dataset = SurvivalDataset(texts, times, events)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class BioBERTSurvivalModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(BioBERTSurvivalModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # Single output

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        risk_score = self.regressor(cls_embedding)
        return risk_score.view(-1)  # ✅ Ensures correct shape


def cox_loss(survival_time, risk_score, event_indicator):
    sorted_idx = torch.argsort(survival_time, descending=True)
    risk_score = risk_score[sorted_idx].squeeze()  # ✅ Ensure it's 1D
    event_indicator = event_indicator[sorted_idx]

    log_risk = torch.logsumexp(risk_score, dim=0)
    loss = -torch.sum(risk_score * event_indicator - log_risk)
    return loss


def cox_loss(survival_time, risk_score, event_indicator):
    # Sort survival times and corresponding risk scores in descending order
    sorted_idx = torch.argsort(survival_time, descending=True)
    survival_time = survival_time[sorted_idx]
    risk_score = risk_score[sorted_idx]
    event_indicator = event_indicator[sorted_idx]

    # Calculate the cumulative sum of exponentials of the risk scores
    exp_risk = torch.exp(risk_score)
    cum_exp_risk = torch.cumsum(exp_risk, dim=0)  # cumulative sum of exp(risk_score)

    # Apply log to the cumulative sum (log-sum-exp trick)
    log_risk = torch.log(cum_exp_risk)

    # Compute the Cox proportional hazards loss
    loss = -torch.sum(event_indicator * (risk_score - log_risk))

    return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = BioBERTSurvivalModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    logging_steps = 1000
    step = 0

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        step += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        times = batch["time"].to(device)
        events = batch["event"].to(device)

        risk_score = model(input_ids, attention_mask)
        loss = cox_loss(times, risk_score, events)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % logging_steps == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss / (step + 1):.4f}")

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    pred_risks, true_times, true_events = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            times = batch["time"].cpu().numpy()
            events = batch["event"].cpu().numpy()

            risk_score = model(input_ids, attention_mask).cpu().numpy()

            pred_risks.extend(risk_score.flatten())
            true_times.extend(times)
            true_events.extend(events)

    # Compute Concordance Index

    results = {"predicted": list(-1 * np.array(pred_risks)), "os": list(true_events), "real": true_times}

    c_index = concordance_index(true_times, -1 * np.array(pred_risks), true_events)
    print(f"Concordance Index: {c_index:.4f}")
    with open("biobert_fold_0.json", "w") as f:
        json.dump(results, f)


# Run evaluation
evaluate_model(model, test_dataloader)


torch.save(model.state_dict(), "biobert_survival_fold_0.pth")

# To load the model
# model = BioBERTSurvivalModel()
# model.load_state_dict(torch.load("biobert_survival_fold_0.pth"))
# model.to(device)

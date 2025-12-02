import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from lifelines.utils import concordance_index
from datasets import load_from_disk
from sklearn.preprocessing import QuantileTransformer
import torch.nn.functional as F
from glossary import *

from huggingface_hub import login
login()

# Load BioBERT tokenizer
MODEL_NAME = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds = [f"fold_{x}" for x in range(5)]


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


# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


# Create dataset
class SurvivalDataset(Dataset):
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


def evaluate_model(model, dataloader: DataLoader, qt, fold):
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
            times_original = qt.inverse_transform(times.reshape(-1, 1)).squeeze()

            pred_times.extend(predicted_times)
            true_times.extend(times_original)
            true_events.extend(events)

    results = {
        "predicted": [round(x) for x in pred_times],
        "os": [int(x) for x in true_events],
        "real": [round(x) for x in true_times]
    }

    with open(f"biobert_results_{fold}.json", "w") as f:
        json.dump(results, f)

    # Compute Concordance Index
    c_index = concordance_index(true_times, np.array(pred_times), true_events)
    print(f"Concordance Index: {c_index:.4f}")


def train_on(dataset_path="datasets/full_data"):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(preprocess_function, batched=False)
    texts = [x["text"] for x in dataset]
    times = [int(x["time_to_os"]) for x in dataset]
    events = [int(x["os"]) for x in dataset]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    times = qt.fit_transform(np.array(times).reshape(-1, 1)).squeeze()
    train_dataset = SurvivalDataset(texts, times, events)
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=False)

    # Initialize model
    model = BioBERTSurvivalModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-6)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
    num_epochs = 2

    loss_fn = mae_loss

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        logging_steps = 2000
        step = 0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            times = batch["time"].to(device)
            events = batch["event"].to(device)

            predicted_times = model(input_ids, attention_mask)

            loss = loss_fn(predicted_times, times)

            loss.backward()
            optimizer.step()

            # Step scheduler every 5000 steps
            if step % 5000 == 0 and step > 0:
                scheduler.step()
                print(f"Learning rate reduced to {scheduler.get_last_lr()[0]:.6f}")
                torch.save(model.state_dict(), f"new_biobert/biobert_survival_step_{step}_all_data.pth")

            total_loss += loss.item()

            if step % logging_steps == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss / (step + 1):.4f}")

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

    torch.save(model.state_dict(), f"new_biobert/biobert_survival_all_data.pth")


def main():
    for fold in folds:
        # **Load Dataset**
        dataset = load_from_disk(f"datasets/train{fold}")
        test_dataset = load_from_disk(f"datasets/test{fold}")

        dataset = dataset.map(preprocess_function, batched=False)
        test_dataset = test_dataset.map(preprocess_function, batched=False)

        texts = [x["text"] for x in dataset]
        times = [int(x["time_to_os"]) for x in dataset]
        events = [int(x["os"]) for x in dataset]

        test_texts = [x["text"] for x in test_dataset]
        test_times = [int(x["time_to_os"]) for x in test_dataset]
        test_events = [int(x["os"]) for x in test_dataset]

        # Initialize a QuantileTransformer for survival time transformation
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        times = qt.fit_transform(np.array(times).reshape(-1, 1)).squeeze()
        test_times = qt.transform(np.array(test_times).reshape(-1, 1)).squeeze()

        # Create train and test datasets
        train_dataset = SurvivalDataset(texts, times, events)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)

        test_dataset = SurvivalDataset(test_texts, test_times, test_events)
        test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        # Initialize model
        model = BioBERTSurvivalModel().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-6)
        scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
        num_epochs = 2

        loss_fn = mae_loss

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            logging_steps = 2000
            step = 0

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                step += 1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                times = batch["time"].to(device)
                events = batch["event"].to(device)

                predicted_times = model(input_ids, attention_mask)

                loss = loss_fn(predicted_times, times)

                loss.backward()
                optimizer.step()

                # Step scheduler every 5000 steps
                if step % 5000 == 0 and step > 0:
                    scheduler.step()
                    print(f"Learning rate reduced to {scheduler.get_last_lr()[0]:.6f}")
                    torch.save(model.state_dict(), f"biobert_survival_step_{step}_{fold}.pth")

                total_loss += loss.item()

                if step % logging_steps == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss / (step + 1):.4f}")

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

        torch.save(model.state_dict(), f"biobert_survival_{fold}.pth")

        # Run evaluation
        evaluate_model(model, test_dataloader, qt, fold)


# To load the model
# model = BioBERTSurvivalModel()
# model.load_state_dict(torch.load("biobert_survival_fold_0.pth"))
# model.to(device)

train_on()

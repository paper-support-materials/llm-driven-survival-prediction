import joblib
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

from glossary import *
from huggingface_hub import login

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import T5Model, AutoTokenizer

from tqdm import tqdm
import numpy as np
import json
from sklearn.preprocessing import QuantileTransformer
from lifelines.utils import concordance_index
from transformers import get_linear_schedule_with_warmup

login()
MODEL_NAME = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_text(file__name, content):
    with open(file__name, "a", encoding="utf-8") as text_file:
        print(content, file=text_file)


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
def tokenize_function(texts_):
    return tokenizer(texts_, padding="max_length", truncation=True, max_length=512, return_tensors="pt")


# Create dataset
class SurvivalDataset(Dataset):
    def __init__(self, texts_, times_, events_):
        self.encodings = tokenizer(texts_, padding="max_length", truncation=True, max_length=512)
        self.times = torch.tensor(times_, dtype=torch.float32)
        self.events = torch.tensor(events_, dtype=torch.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["time"] = self.times[idx]
        item["event"] = self.events[idx]
        return item


# # Define the T5 model for survival prediction
# class T5SurvivalModel(nn.Module):
#     def __init__(self, model_name=f"{MODEL_NAME}"):
#         super(T5SurvivalModel, self).__init__()
#         self.t5 = T5Model.from_pretrained(model_name)  # Load only the encoder
#         self.regressor = nn.Linear(self.t5.config.d_model, 1)  # Single output for survival prediction

#     def forward(self, input_ids, attention_mask):
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)

#         outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding

#         risk_score = self.regressor(cls_embedding)
#         return risk_score.view(-1)

    
# # Define the T5 model for survival prediction
# class T5SurvivalModel(nn.Module):
#     def __init__(self, model_name="t5-base"):  # You can set a different T5 model here
#         super(T5SurvivalModel, self).__init__()
#         self.t5 = T5Model.from_pretrained(model_name)  # Load only the encoder (T5 model)
#         self.regressor = nn.Linear(self.t5.config.d_model, 1)  # Regression head to predict survival

#     def forward(self, input_ids, attention_mask):
#         # Forward pass through the T5 encoder
#         outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)

#         # Pooled output (mean pooling across the sequence length)
#         pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over the sequence dimension

#         # Pass the pooled output through the regression head
#         risk_score = self.regressor(pooled_output)

#         # Return the predicted survival time as a scalar value
#         return risk_score.view(-1)  # Ensure the output is a 1D tensor (for regression tasks)

    
class T5SurvivalModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(T5SurvivalModel, self).__init__()
        self.t5 = T5Model.from_pretrained(model_name)  # Load only the encoder
        hidden_size = self.t5.config.d_model  # Hidden size of T5 embeddings
        
        # Attention-based pooling layer
        self.attention_weights = nn.Linear(hidden_size, 1)  # Linear layer to compute attention scores
        self.regression_head = nn.Linear(hidden_size, 1)  # Regression layer for survival prediction

    def forward(self, input_ids, attention_mask):
        outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Compute attention scores (batch_size, seq_len, 1)
        attn_scores = self.attention_weights(token_embeddings)  
        attn_scores = torch.softmax(attn_scores, dim=1)  # Normalize scores across sequence length

        # Weighted sum of token embeddings
        pooled_output = torch.sum(attn_scores * token_embeddings, dim=1)  # (batch_size, hidden_size)

        # Predict survival time
        risk_score = self.regression_head(pooled_output)
        return risk_score.view(-1)  # Ensure correct shape for regression


# Define MAE loss
def mae_loss(predicted_survival, actual_survival):
    return F.l1_loss(predicted_survival, actual_survival)
    # return F.smooth_l1_loss(predicted_survival, actual_survival)
    # return F.huber_loss(predicted_survival, actual_survival, delta=0.5)


# Model evaluation function
def evaluate_model(model, dataloader: DataLoader, qt, fold, epoch=None, step=None):
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
            # predicted_times = qt.inverse_transform(predicted_times.reshape(-1, 1)).squeeze()
            predicted_times = qt.inverse_transform(predicted_times.reshape(-1, 1))
            predicted_times = predicted_times.flatten()
            times_original = qt.inverse_transform(times.reshape(-1, 1)).squeeze()

            # pred_times.extend(predicted_times)
            pred_times.extend(predicted_times.tolist())
            # true_times.extend(times_original)
            # true_events.extend(events)
            true_times.extend(times_original.tolist() if times_original.ndim > 0 else [times_original])
            true_events.extend(events.tolist() if events.ndim > 0 else [events])

    results = {
        "predicted": [round(x) for x in pred_times],
        "os": [int(x) for x in true_events],
        "real": [round(x) for x in true_times]
    }

    result_file = f"{MODEL_NAME}_epoch_{epoch}_step_{step}_results_{fold}.json" if epoch is not None and step is not None else f"{MODEL_NAME}_results_{fold}.json"

    with open(result_file, "w") as f:
        json.dump(results, f)

    # Compute Concordance Index
    c_index = concordance_index(true_times, np.array(pred_times), true_events)
    print(f"Concordance Index: {c_index:.4f}")
    content_text = f"Concordance Index: {c_index:.4f} - " + result_file
    write_text(f"{MODEL_NAME}_c-index.txt",  content_text)


def training_loop(train_data, test_data, fold):
    # **Load Dataset**
    dataset = load_from_disk(f"datasets/{train_data}")
    test_dataset = load_from_disk(f"datasets/{test_data}")

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
    joblib.dump(qt, "datasets/t5_qt_full_data.joblib")

    # Create train and test datasets
    train_dataset = SurvivalDataset(texts, times, events)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    test_dataset = SurvivalDataset(test_texts, test_times, test_events)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = T5SurvivalModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    num_training_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    loss_fn = mae_loss

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        logging_steps = 100
        step = 0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            times = batch["time"].to(device)

            predicted_times = model(input_ids, attention_mask)

            loss = loss_fn(predicted_times, times)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 2000 == 0 and 0 < step < num_training_steps - 5:
                torch.save(model.state_dict(), f"t5_base_ap/{MODEL_NAME}_survival_epoch_{epoch}_step_{step}.pth")
                evaluate_model(model, test_dataloader, qt, fold, epoch, step)

            total_loss += loss.item()

            if step % logging_steps == 0:
                content_text = f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss / (step + 1):.4f}"
                content_text += f" - Learning rate: {optimizer.param_groups[0]['lr']}"
                write_text(f"{MODEL_NAME}_training.txt",  content_text)

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

    torch.save(model.state_dict(), f"t5_base_ap/{MODEL_NAME}_survival.pth")

    # Run evaluation
    evaluate_model(model, test_dataloader, qt, fold)


if __name__ == '__main__':
    training_loop("full_data", "testfold_0", "fold_0")

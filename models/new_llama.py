import os
from pathlib import Path
from typing import Any

import datasets
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from glossary import *

from huggingface_hub import login
login()


# 0. Prepare dataset
def prepare_dataset(fold, tokenizer) -> datasets.Dataset:
    base_prompt = """
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """

    eos_token = tokenizer.eos_token  # Must add EOS_TOKEN

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

        # 🔹 Handle cases where survival status may be missing (in inference)
        if "os" in example:
            status_str = "Alive" if example["os"] == 0 else "Deceased"
            status_line = f"Patient status: {status_str}."
        else:
            status_line = "Patient status: Unknown (predict survival time based on patient data)."

        prompt_lines = [
            f"Patient age: {int(example['leeft'])} years old.",
            f"Gender: {gender_str}.",
            f"Incident year: {int(example['incjr'])}.",
            f"Tumor type: {tumor_type_str}, morphology: {morph_str}, behavior: {behavior_str}.",
            f"Location: {later_str} breast, sublocation: {topo_sub_str}.",
            f"Differentiation grade: {diff_str}.",
            f"ER status: {er_str}, PR status: {pr_str}, HER2 status: {her2_str}.",
            status_line  # Adaptively includes status or a guiding note if missing
        ]

        # 🔹 Instruction explicitly tells the model what to do in all scenarios
        instruction = (
            "Predict the number of days the patient has survived or is expected to survive based on the provided medical data.\n\n"
            "- If the patient is marked as 'Deceased', predict the actual survival duration in days.\n"
            "- If the patient is marked as 'Alive', estimate the expected survival time based on similar cases.\n"
            "- If the patient status is not provided, use all available data to infer the most probable survival duration."
        )

        input_text = "\n".join(prompt_lines)
        # response = str(example["time_to_os"])

        return {"instruction": instruction, "input": input_text, "labels": example["time_to_os"]}

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        # outputs = examples["response"]
        texts = []
        for instruction, input_text in zip(instructions, inputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = base_prompt.format(instruction, input_text) + eos_token
            texts.append(text)
        return {"text": texts}

    # **Load Dataset**
    raw_dataset = load_from_disk(f"./datasets/train{fold}")

    new_dataset = []
    for sample in raw_dataset:
        row = build_prompt(sample)
        new_dataset.append(row)

    dataset = Dataset.from_list(new_dataset)
    print(dataset)

    dataset = dataset.map(formatting_prompts_func, batched=True, )
    print(dataset[0])
    return dataset


# 1. Load tokenizer
def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# 2. Create regression model with mean pooling
class LlamaForRegression(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        return self.regression_head(pooled_output)

    # Add this method to prevent errors when LoRA calls it
    def prepare_inputs_for_generation(self, *args, **kwargs):
        # You can just pass an empty function here if it's not needed for regression
        return {}


# 3. Load and wrap model
def build_model(model_name, hidden_size=4096):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    model = LlamaForRegression(base_model, hidden_size)
    model.base_model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model


# 4. Apply LoRA
def apply_lora(model):
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"]
    )
    return get_peft_model(model, lora_config)


# 5. Preprocess dataset
def preprocess_dataset(folder_path, tokenizer, qt=None):
    data = prepare_dataset(folder_path, tokenizer)

    if qt is None:
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        times = qt.fit_transform(np.array(data["labels"]).reshape(-1, 1)).squeeze()
        data = data.add_column("transformed", times)
    else:
        s = np.array(data["labels"]).reshape(-1, 1)
        data = data.add_column("transformed", qt.transform(s).flatten())

    def tokenize_fn(examples):
        tokens = tokenizer(examples["text"], padding=True, truncation=True)
        tokens["labels"] = examples["transformed"]
        return tokens

    tokenized = data.map(tokenize_fn, batched=True)
    return tokenized, qt


# 6. MAE loss
def mae_loss(predicted, target):
    return F.l1_loss(predicted, target)


# 7. Custom trainer
class RegressionTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Any = None
    ) -> tuple[Any, dict]:

        labels = inputs["labels"]
        outputs = model(**inputs).squeeze(-1)

        # MAE loss
        loss = F.l1_loss(outputs, labels)

        # Match the required return type
        return (loss, {"loss": loss, "predictions": outputs}) if return_outputs else (loss, {})


# 8. Train
def train_on_folder(folder_path, output_dir, model_name="meta-llama/Llama-3.2-3B-Instruct", qt=None):
    tokenizer = load_tokenizer(model_name)
    model = build_model(model_name)
    model = apply_lora(model)
    tokenized_data, qt = preprocess_dataset(folder_path, tokenizer, qt)  # plain Dataset

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        save_steps=2000,
        save_total_limit=2,
        num_train_epochs=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        report_to="none",
        evaluation_strategy="no",
        save_strategy="steps",
        label_names=["labels"]
    )

    trainer = RegressionTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_data
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(qt, os.path.join(output_dir, "qt.joblib"))


if __name__ == '__main__':
    # my_tokenizer = load_tokenizer("meta-llama/Llama-3.2-3B-Instruct")
    # print(my_tokenizer.model_max_length)
    # folds = [f"fold_{x}" for x in range(5)]
    # fold = folds[0]
    # tokenized, qt = preprocess_dataset(folder_path=fold, tokenizer=my_tokenizer)
    # print(tokenized[0])
    # joblib.dump(qt, "quantile_transformer.joblib")
    # qt = joblib.load("quantile_transformer.joblib")

    base_output_dir = Path("output")  # Relative path for saving models

    # Create list of fold directories (relative to the current directory)
    folds = [f"fold_{x}" for x in range(5)]

    for fold in folds:
        output_dir = base_output_dir / fold  # Relative path to save the model for each fold

        print(f"🔁 Training on {fold}... -> {output_dir}")

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train the model for this fold
        train_on_folder(
            folder_path=str(fold),
            output_dir=str(output_dir)  # Output the model in the corresponding folder
        )

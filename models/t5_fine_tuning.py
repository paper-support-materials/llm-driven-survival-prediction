import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from glossary import *

from huggingface_hub import login
login()


# **Load T5-Base Model**
MODEL_NAME = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

# **Load Tokenizer**
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# **Load Dataset (Example: Yelp Reviews)**
dataset = load_from_disk("datasets/trainfold_0")


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
    tokenized = tokenizer(prompt_text, truncation=True, padding="max_length", max_length=256)
    tokenized_target = tokenizer(str(example["time_to_os"]), truncation=True, padding="max_length", max_length=5)
    tokenized["labels"] = tokenized_target["input_ids"]
    return tokenized


def train():
    tokenized_dataset = dataset.map(preprocess_function, batched=False)
    train_dataset = tokenized_dataset

    # **Training Arguments**
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        max_steps=100,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        # load_best_model_at_end=True,
    )

    # **Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset
    )

    # **Train the model**
    trainer.train()

    trainer.save_model("./t5_finetuned")
    tokenizer.save_pretrained("./t5_finetuned")


if __name__ == '__main__':
    pass

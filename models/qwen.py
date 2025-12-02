from unsloth import FastLanguageModel
import torch
from glossary import *
from datasets import load_from_disk
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

fold = "fold_4"

max_seq_length = 2048 # Choose any!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0, # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


base_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


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
    response = str(example["time_to_os"])

    return {"instruction": instruction, "input": input_text, "response": response}


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["response"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = base_prompt.format(instruction, input_text, output) + EOS_TOKEN
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


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"qwen_outputs_{fold}",
        report_to="none",
        # resume_from_checkpoint = True,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained(f"Qwen2.5-7B-Instruct_2_epochs_new_prompt_{fold}")
tokenizer.save_pretrained(f"Qwen2.5-7B-Instruct_2_epochs_new_prompt_{fold}")
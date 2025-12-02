import torch
import json
from tqdm import tqdm
from glossary import *
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk


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


def predict_score(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the score (you may need extra parsing for clean output)
    predicted_score = generated_text.replace(prompt, "").strip()

    return predicted_score


def infer(test_file, result_file):
    test_dataset = load_from_disk(test_file)
    results = {"predicted": [], "real": [], "os": []}
    i = 0
    for row in tqdm(test_dataset):
        i += 1
        text = build_prompt(row)
        # print(text)
        predicted_score = predict_score(text)
        results["predicted"].append(predicted_score)
        results["real"].append(row["time_to_os"])
        results["os"].append(row["os"])
        # print(f"Predicted Score: {predicted_score} Real score: {row['time_to_os']} Status: {row['os']}")
        # print(row)
        if i % 50 == 0:
            with open(result_file, "w") as f:
                json.dump(results, f)

    with open(result_file, "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    # ** Load T5 Model **
    MODEL_NAME = "models/t5-base_new_prompt_model_ft1_fold_0"
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

    # ** Load Tokenizer **
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # ** Get Results **
    infer("./datasets/testfold_0", "./t5-base_new_results/t5-base_results_fold_0.json")

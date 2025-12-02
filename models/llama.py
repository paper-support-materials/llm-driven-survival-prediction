from tqdm import tqdm

from datasets import load_from_disk, Dataset
from glossary import *

base_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "<eos>"  # tokenizer.eos_token # Must add EOS_TOKEN


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


if __name__ == '__main__':

    # **Load Dataset**
    raw_dataset = load_from_disk("./datasets/trainfold_0")

    # new_dataset = raw_dataset.map(build_prompt, batched=False)
    # print(new_dataset)

    # new_dataset = []
    # for sample in tqdm(raw_dataset):
    #     row = build_prompt(sample)
    #     new_dataset.append(row)

    # dataset = Dataset.from_list(new_dataset)
    # print(dataset)

    # dataset = new_dataset.map(formatting_prompts_func, batched=True, )
    counter = 0
    somma = 0
    max_sur = 0
    min_sur = 65535
    dead_dataset = []
    for sample in tqdm(raw_dataset):
        if sample["time_to_os"] < min_sur:
            min_sur = sample["time_to_os"]
        if sample["time_to_os"] > max_sur:
            max_sur = sample["time_to_os"]
        if sample["os"] == 1:
            dead_dataset.append(sample)
            counter += 1
            somma += sample["time_to_os"]

    print("conteggio:", counter)
    print("media:", somma/counter)
    print("max:", max_sur)
    print("min:", min_sur)

    dead_dataset = Dataset.from_list(dead_dataset)
    dead_dataset.save_to_disk("fold_0_dead")
    print(dead_dataset)

    # print(dataset[0])

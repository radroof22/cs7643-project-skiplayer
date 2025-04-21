from datasets import load_dataset
import json
import os

# Load ARC Challenge test set
dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

formatted_data = []

for example in dataset:
    question = example["question"]  # ✅ plain string
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    answer = example["answerKey"]

    # Create a multiple-choice formatted input
    input_text = question + " " + " ".join(f"({label}) {choice}" for label, choice in zip(labels, choices))

    formatted_data.append({
        "input": input_text,
        "target": answer
    })

# Save the data to custom_datasets/arc_test.json
os.makedirs("custom_datasets", exist_ok=True)
with open("custom_datasets/arc_test.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print(f"Saved {len(formatted_data)} examples to custom_datasets/arc_test.json")
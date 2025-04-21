from datasets import load_dataset
import json
import os

dataset = load_dataset("mbpp", split="test")

formatted_data = []
for example in dataset:
    formatted_data.append({
        "input": example["text"],
        "target": example["code"],
    })

# Ensure the folder exists
os.makedirs("custom_datasets", exist_ok=True)

with open("custom_datasets/mbpp_test.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print(f"Saved {len(formatted_data)} examples to custom_datasets/mbpp_test.json")

from datasets import load_dataset
import json
import os

# Load the test split of MBPP
dataset = load_dataset("mbpp", split="test")

# Ensure the folder exists
os.makedirs("custom_datasets", exist_ok=True)

# Save in JSONL format
with open("custom_datasets/mbpp_test.jsonl", "w") as f:
    for example in dataset:
        json.dump({
            "input": example["text"],
            "target": example["code"],
        }, f)
        f.write("\n")  # write each JSON object on a new line

print(f"Saved {len(dataset)} examples to custom_datasets/mbpp_test.jsonl")
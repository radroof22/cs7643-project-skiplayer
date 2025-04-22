from datasets import load_dataset
import json

# Load BoolQ dataset
dataset = load_dataset("boolq", split="validation")

# Convert to prompt/answer format
formatted = []
for example in dataset:
    prompt = f"Question: {example['question']} Context: {example['passage']} Answer:"
    answer = "Yes" if example['answer'] else "No"
    formatted.append({"prompt": prompt, "answer": answer})

# Save to custom_datasets directory
output_path = "custom_datasets/boolq_test.json"
with open(output_path, "w") as f:
    for item in formatted:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(formatted)} examples to {output_path}")
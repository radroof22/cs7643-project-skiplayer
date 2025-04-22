# How to add a dataset
1. Add the JSON file of test cases to `custom_datasets/`
2. Identify the **prompt_field** and **response_field** for the JSON file
3. Write down command and save in this file. Template should be

```bash
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path PATH_JSON_FILE --prompt_field PROMPT_FIELD --response_field RESPONSE_FIELD
```

# GSM8K

```bash
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/gsm8k_test.json --prompt_field question --response_field answer
```

# MMLU

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/mmlu_test.jsonl --prompt_field prompt --response_field answer
```

# BoolQ

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/boolq_test.json --prompt_field prompt --response_field answer
```

# MBPP

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/mbpp_test.json --prompt_field input --response_field target
```

# NQ-Open

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/nq_open_val.jsonl --prompt_field prompt --response_field answer
```

# RACE

```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl     --num_samples 100     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs --data_path ../custom_datasets/race_test.jsonl --prompt_field prompt --response_field answer
```

# ARC
```
torchrun benchmark.py --model facebook/layerskip-llama2-7B --dataset custom_jsonl --num_samples 100 --generation_strategy self_speculative --exit_layer 8 --num_speculations 6 --output_dir ./logs --data_path custom_datasets/arc_test.jsonl --prompt_field input --response_field target
```

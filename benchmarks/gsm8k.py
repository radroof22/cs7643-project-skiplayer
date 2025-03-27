from copy import deepcopy
import warnings
import os

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from tqdm import tqdm

MODEL_NAME = "facebook/layerskip-llama3.2-1B"
SAMPLE_SIZE = 20

class GSM8KBenchmark:
    def __init__(self, model_name="facebook/layerskip-llama3.2-1B", device=None, early_exit: int = 4):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_model(early_exit)

    def load_model(self, early_exit):
        print(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        self.generation_config = self.model.generation_config
        weights_memo = {id(w): w for w in self.model.parameters()}
        self.assistant_model = deepcopy(self.model, memo=weights_memo) 
        self.assistant_model.model.layers = self.assistant_model.model.layers[:early_exit] 
        del self.assistant_model.model.layers[early_exit:]
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_prompt(self, question):
        return f"Solve the following math problem:\n\n{question}\n\nAnswer:"

    def evaluate_question(self, question, answer):
        prompt = self.prepare_prompt(question)
        # print(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, generation_config=self.generation_config, 
                assistant_model=self.assistant_model, max_new_tokens=512
            )
        
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_answer = generated_text.split("Answer:")[-1].strip()
        print(f"Model Answer: {generated_answer}")
        actual_answer = answer.split("####")[1]
        print(f"Actual Answer: {actual_answer}")
        return int(answer.lower() in generated_text.lower())

    def run_benchmark(self, sample_size=None):
        print("\nLoading GSM8K dataset...")
        gsm8k = load_dataset('gsm8k', 'main')
        
        # Access the 'test' split and apply shuffle/select on it
        gsm8k_test = gsm8k['test']
        gsm8k_test = gsm8k_test.shuffle(seed=42).select(range(min(sample_size, len(gsm8k_test)))) if sample_size else gsm8k_test

        correct_predictions = 0
        total_questions = len(gsm8k_test)

        print(f"\nEvaluating {total_questions} questions from GSM8K")
        for example in tqdm(gsm8k_test):
            question, answer = example['question'], example['answer']
            correct_predictions += self.evaluate_question(question, answer)

        accuracy = correct_predictions / total_questions if total_questions > 0 else 0
        print(f"\nAccuracy on GSM8K: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    logging.set_verbosity_error()
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    benchmark = GSM8KBenchmark(model_name=MODEL_NAME)
    benchmark.run_benchmark(sample_size=SAMPLE_SIZE)

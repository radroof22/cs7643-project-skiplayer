from copy import deepcopy
import warnings
import os

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from tqdm import tqdm

import torch
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "facebook/layerskip-llama3.2-1B"
SUBJECTS = ["sociology"]
SAMPLE_SIZE = 3000
BATCH_SIZE = 8

class MMLUBenchmark:
    def __init__(self, model_name="facebook/layerskip-llama3.2-1B", device=None, early_exit: int = 4):
        """
        Initialize MMLU Benchmarking class.
        
        Args:
            model_name (str): Name of the model checkpoint to use.
            device (str, optional): Device to use ("cuda" or "cpu"). Defaults to GPU if available.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_model(early_exit)

    def load_model(self, early_exit):
        """Loads the model and tokenizer."""
        print(f"Loading model and tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        self.generation_config = self.model.generation_config
        weights_memo = {id(w): w for w in self.model.parameters()}
        self.assistant_model = deepcopy(self.model, memo=weights_memo) # Clone main model with shared weights
        self.assistant_model.model.layers = self.assistant_model.model.layers[:early_exit] # Apply early exit
        del self.assistant_model.model.layers[early_exit:]

        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_prompt(self, question, choices):
        """
        Prepare the input prompt for multiple-choice questions.
        
        Args:
            question (str): The test question.
            choices (list): List of multiple-choice options.
        
        Returns:
            str: Formatted prompt for model input.
        """
        choice_labels = ['A', 'B', 'C', 'D']
        formatted_choices = '\n'.join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
        return f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"

    def prepare_prompts(self, questions, choices):
        choice_labels = ['A', 'B', 'C', 'D']
        prompts = []
        for question, choice_set in zip(questions, choices):
            formatted_choices = '\n'.join([f"{label}. {choice}" for label, choice in zip(choice_labels, choice_set)])
            prompt = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer:"
            prompts.append(prompt)
        return prompts

    def evaluate_batch(self, questions, choices):
        prompts = self.prepare_prompts(questions, choices)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, generation_config=self.generation_config, assistant_model=self.assistant_model, max_new_tokens=10
            )
        
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = []
        for text in generated_texts:
            answer_line = next((line for line in text.splitlines() if line.startswith("Answer")), None)
            if answer_line:
                llm_answer = answer_line.split(":")[1].strip()
                for i, label in enumerate(['A', 'B', 'C', 'D']):
                    if label in llm_answer:
                        predictions.append(i)
                        break
                else:
                    predictions.append(np.random.randint(0, len(choices[0])))
            else:
                predictions.append(np.random.randint(0, len(choices[0])))
        return predictions

    def evaluate_question(self, question, choices):
        """
        Evaluate a single multiple-choice question.
        
        Args:
            question (str): The test question.
            choices (list): List of multiple-choice options.
        
        Returns:
            int: Index of the predicted answer (0=A, 1=B, 2=C, 3=D)
        """
        prompt = self.prepare_prompt(question, choices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            # outputs = self.model.generate(
            #     inputs.input_ids, 
            #     max_new_tokens=10,
            #     num_return_sequences=1,
            #     do_sample=False
            # )
            outputs = self.model.generate(**inputs, generation_config=self.generation_config, assistant_model=self.assistant_model, max_new_tokens=512)


        # Decode output
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # extract solution
        llm_answer = [line for line in generated_text.splitlines() if line.startswith("Answer")][0].split(":")[1]
        # Simple heuristic to extract the answer
        for i, label in enumerate(['A', 'B', 'C', 'D']):
            if label in llm_answer:
                return i
        
        # Default to random choice if no answer found
        return np.random.randint(0, len(choices))

    def run_benchmark(self, subjects=None, sample_size=None, batch_size=1):
        """
        Run the MMLU benchmark.
        
        Args:
            subjects (list, optional): List of subjects to evaluate. Defaults to all.
            sample_size (int, optional): Number of questions per subject. Defaults to full dataset.
        
        Returns:
            dict: Results containing accuracy for each subject and overall accuracy.
        """
        print("\nLoading MMLU dataset...")
        mmlu = load_dataset("cais/mmlu", "all")

        # Default to all subjects if none specified
        if subjects is None:
            subjects = mmlu['test'].features['subject'].names

        self.results = {}

        for subject in subjects:
            # Filter dataset for specific subject
            subject_data = mmlu['test'].filter(lambda x: x['subject'] == subject)

            # Limit dataset size if needed
            if sample_size:
                subject_data = subject_data.select(range(min(sample_size, len(subject_data))))

            correct_predictions = 0
            total_questions = len(subject_data)

            print(f"\nEvaluating subject: {subject} (size: {total_questions})")
            for example in tqdm(subject_data):
                question, choices, ground_truth = example['question'], example['choices'], example['answer']
                predicted_answer = self.evaluate_question(question, choices)

                if predicted_answer == ground_truth:
                    correct_predictions += 1

            # Store results
            accuracy = correct_predictions / total_questions if total_questions > 0 else 0
            self.results[subject] = {
                'total_questions': total_questions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy
            }
    
    def print_results(self):
        print(f"\n--- MMLU Benchmark Results: {self.model_name} ---")
        overall_total, overall_correct = 0, 0
        for subject, metrics in self.results.items():
            print(f"\n{subject}:")
            print(f"  Total Questions: {metrics['total_questions']}")
            print(f"  Correct Predictions: {metrics['correct_predictions']}")
            print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
            overall_total += metrics["total_questions"]
            overall_correct += metrics["correct_predictions"]
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

if __name__ == "__main__":
    ## SUPRESS CONSOLE
    # Configure warnings and console prints
    logging.set_verbosity_error()
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # Suppress torch warnings about attention masks
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    ## RUN BENCHMARK
    benchmark = MMLUBenchmark(model_name=MODEL_NAME)
    results = benchmark.run_benchmark(subjects=SUBJECTS, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE)
    benchmark.print_results()

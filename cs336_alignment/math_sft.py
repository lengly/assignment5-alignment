from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
import random
from typing import List, Dict, Any
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
import tqdm
import wandb
import numpy
import json
import os
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
import queue
import time
import math
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import (
    format_prompt_with_r1_zero, 
    load_gsm8k_data,
    extract_answer_from_response,
    extract_answer_from_gsm8k_answer,
)

def set_random_seed(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def adjust_learning_rate(optimizer, step, total_steps, initial_lr, warmup_steps=100, min_lr_factor=0.1):
    """
    Manually adjust learning rate with warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to adjust
        step: Current training step
        total_steps: Total number of training steps
        initial_lr: Initial learning rate
        warmup_steps: Number of warmup steps
        min_lr_factor: Minimum learning rate as a factor of initial_lr
    
    Returns:
        current_lr: Current learning rate
    """
    min_lr = initial_lr * min_lr_factor
    
    if step < warmup_steps:
        # Linear warmup: from 0.1 * lr to lr
        warmup_factor = step / warmup_steps
        current_lr = initial_lr * warmup_factor
    else:
        # Cosine decay: from lr to min_lr
        decay_steps = total_steps - warmup_steps
        current_step = step - warmup_steps
        progress = current_step / decay_steps
        # Cosine decay formula: min_lr + (initial_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
    
    # Set learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    return current_lr

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
            the tokenized prompt and output strings. Then the returned dictionary should have the
            following keys:
            - input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            - labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input ids, i.e., the input ids without the first token.
            - response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
                1): a mask on the response tokens in the labels.
    """
    prompt_and_output_lens = []

    input_ids = []
    labels = []
    response_mask = []
    prompt_tokens = []
    output_tokens = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens.append(tokenizer.encode(prompt))
        output_tokens.append(tokenizer.encode(output))
        prompt_and_output_lens.append(len(prompt_tokens[-1]) + len(output_tokens[-1]))
    max_len = max(prompt_and_output_lens)
    for i in range(len(prompt_and_output_lens)):
        if prompt_and_output_lens[i] < max_len:
            input_ids.append(prompt_tokens[i] + output_tokens[i] + [tokenizer.pad_token_id] * (max_len - prompt_and_output_lens[i] - 1))
        else:
            input_ids.append(prompt_tokens[i] + output_tokens[i][:-1])
        labels.append(prompt_tokens[i][1:] + output_tokens[i] + [tokenizer.pad_token_id] * (max_len - prompt_and_output_lens[i]))
        response_mask.append([0] * (len(prompt_tokens[i]) - 1) + [1] * len(output_tokens[i]) + [0] * (max_len - prompt_and_output_lens[i]))
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor(response_mask)
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get the conditional log-probs of the response given the prompt,
    and optionally the entropy of the next token predictions.
    
    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length)
        labels: torch.Tensor of shape (batch_size, sequence_length)
        return_token_entropy: bool, whether to return token entropy
        
    Returns:
        dict[str, torch.Tensor]: dictionary containing log_probs and optionally entropy
    """
    logits = model(input_ids=input_ids).logits  # (batch_size, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": target_log_probs}
    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy        
    return result # (batch_size, sequence_length)

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Normalize a tensor by summing over a specified dimension and dividing by a constant.
    
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
        dimensions.
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    """
    return (tensor * mask).sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int = 1,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        policy_log_probs: torch.Tensor (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
        response_mask: torch.Tensor (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps: int Number of microbatches per optimizer step.
        normalize_constant: float The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        - loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        - metadata Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    """
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss = loss / gradient_accumulation_steps / policy_log_probs.shape[0]
    loss.backward()
    return loss, {}

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: callable,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_samples: int = 5,
) -> dict[str, any]:
    """
    Generate responses from the model and log comprehensive information.
    
    Args:
        model: PreTrainedModel, the model to generate from
        tokenizer: PreTrainedTokenizerBase, tokenizer for the model
        prompts: list[str], input prompts
        ground_truths: list[str], ground truth answers
        reward_fn: callable, function that computes rewards (prompt, response, ground_truth) -> dict
        max_new_tokens: int, maximum number of tokens to generate
        temperature: float, sampling temperature
        top_p: float, top-p sampling parameter
        num_samples: int, number of samples to generate per prompt
        
    Returns:
        dict[str, any]: comprehensive logging information
    """
    
    # Sample prompts if we have too many
    if len(prompts) > num_samples:
        indices = random.sample(range(len(prompts)), num_samples)
        sampled_prompts = [prompts[i] for i in indices]
        sampled_ground_truths = [ground_truths[i] for i in indices]
    else:
        sampled_prompts = prompts
        sampled_ground_truths = ground_truths
    
    model.eval()
    all_generations = []
    
    with torch.no_grad():
        for prompt, ground_truth in zip(sampled_prompts, sampled_ground_truths):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Compute rewards
            reward_info = reward_fn(prompt, response, ground_truth)
            
            # Compute token entropy for the response
            response_inputs = tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=1024)
            response_logits = model(response_inputs['input_ids']).logits
            
            # Get entropy for response tokens only
            prompt_length = inputs['input_ids'].shape[1]
            response_logits = response_logits[:, prompt_length-1:-1, :]  # -1 because of shift
            response_entropy = compute_entropy(response_logits).mean().item()
            
            generation_info = {
                'prompt': prompt,
                'response': response,
                'ground_truth': ground_truth,
                'reward_info': reward_info,
                'avg_token_entropy': response_entropy,
                'response_length': len(response.split()),
                'is_correct': reward_info.get('answer_reward', 0) > 0.5,  # Assuming binary correctness
            }
            
            all_generations.append(generation_info)
    
    # Compute aggregate statistics
    response_lengths = [gen['response_length'] for gen in all_generations]
    correct_lengths = [gen['response_length'] for gen in all_generations if gen['is_correct']]
    incorrect_lengths = [gen['response_length'] for gen in all_generations if not gen['is_correct']]
    
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    avg_correct_length = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0
    avg_incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0
    
    # Compute average rewards
    total_rewards = [gen['reward_info'].get('reward', 0) for gen in all_generations]
    format_rewards = [gen['reward_info'].get('format_reward', 0) for gen in all_generations]
    answer_rewards = [gen['reward_info'].get('answer_reward', 0) for gen in all_generations]
    avg_token_entropies = [gen['avg_token_entropy'] for gen in all_generations]
    
    avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    avg_format_reward = sum(format_rewards) / len(format_rewards) if format_rewards else 0
    avg_answer_reward = sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0
    avg_entropy = sum(avg_token_entropies) / len(avg_token_entropies) if avg_token_entropies else 0
    
    # Compile final logging information
    logging_info = {
        'generations': all_generations,
        'statistics': {
            'avg_response_length': avg_response_length,
            'avg_correct_response_length': avg_correct_length,
            'avg_incorrect_response_length': avg_incorrect_length,
            'avg_total_reward': avg_total_reward,
            'avg_format_reward': avg_format_reward,
            'avg_answer_reward': avg_answer_reward,
            'avg_token_entropy': avg_entropy,
            'num_correct': len(correct_lengths),
            'num_incorrect': len(incorrect_lengths),
            'accuracy': len(correct_lengths) / len(all_generations) if all_generations else 0,
        }
    }
    
    return logging_info

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on a GPu separate from the policy.
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_init_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init_wandb():
    wandb.init(project="math-sft", name="math-sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

class GSM8KDataset(Dataset):
    """GSM8K dataset class"""
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append({
                    'question': item['question'],
                    'answer': item['answer']
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['question'], item['answer']

def async_evaluate_model(model, tokenizer, eval_data, device, vllm_model, eval_queue, eval_step):
    """Asynchronous evaluation function that runs in a separate thread"""
    try:
        print(f"Starting evaluation at step {eval_step}...")
        accuracy = evaluate_model(model, tokenizer, eval_data, device, vllm_model)
        eval_queue.put((eval_step, accuracy))
        print(f"Evaluation completed at step {eval_step}, accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Evaluation failed at step {eval_step}: {e}")
        eval_queue.put((eval_step, None))

def evaluate_model(model, tokenizer, eval_data, device, vllm_model):
    """Evaluate model performance using vLLM on separate GPU"""

    # Prepare evaluation data
    prompts = []
    ground_truths = []
    
    for item in eval_data:
        question = item['question']
        ground_truth = item['answer']
        
        # Format prompt
        prompt = format_prompt_with_r1_zero(question)
        prompts.append(prompt)
        # Extract the numeric answer from the ground truth (same as math_baseline.py)
        ground_truths.append(extract_answer_from_gsm8k_answer(ground_truth))
    
    print(f"Evaluating on {len(eval_data)} validation samples using vLLM...")
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # Stop when the model completes its answer
        include_stop_str_in_output=True,
    )
    
    # Generate responses using vLLM
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # Extract responses and calculate rewards
    predictions = []
    answer_rewards = []
    
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        ground_truth = ground_truths[i]
        
        # Extract answers using existing functions
        predicted_answer = extract_answer_from_response(response)
        
        # Calculate reward using r1_zero_reward_fn
        reward_result = r1_zero_reward_fn(response, ground_truth, fast=True)
        answer_reward = reward_result["answer_reward"]
        
        predictions.append(predicted_answer)
        answer_rewards.append(answer_reward)
        
        # Print progress every 100 samples
        if (i + 1) % 500 == 0:
            current_accuracy = sum(answer_rewards) / len(answer_rewards)
            print(f"Evaluated {i + 1}/{len(eval_data)} samples, current accuracy: {current_accuracy:.4f}")
    
    # Calculate final accuracy using reward-based method (same as math_baseline.py)
    accuracy = sum(answer_rewards) / len(answer_rewards) if answer_rewards else 0.0
    print(f"Final evaluation accuracy: {accuracy:.4f}")
    
    return accuracy

def prepare_training_data(train_data_path, num_samples, tokenizer):
    """Prepare training data by loading and tokenizing"""
    print("Loading training data...")
    train_data = load_gsm8k_data(train_data_path)
    
    # Sample data if specified
    if num_samples is not None and len(train_data) > num_samples:
        train_data = train_data[:num_samples]
    
    print(f"Training on {len(train_data)} samples")
    
    # Prepare training data
    prompts = []
    responses = []
    for item in train_data:
        question = item['question']
        answer = item['answer']
        
        # Convert answer to the expected format for training
        reasoning_part = answer.split("####")[0]
        final_answer = extract_answer_from_gsm8k_answer(answer)
        formatted_response = f" {reasoning_part} </think> <answer> {final_answer} </answer>"
        
        prompts.append(format_prompt_with_r1_zero(question))
        responses.append(formatted_response)
    
    # Tokenize data
    tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return tokenized_data, train_data

def create_dataloader(tokenized_data, batch_size):
    """Create data loader from tokenized data"""
    dataset = torch.utils.data.TensorDataset(
        tokenized_data['input_ids'],
        tokenized_data['labels'],
        tokenized_data['response_mask']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def log_evaluation_result(eval_step, accuracy, num_samples, lr, batch_size, train_data_len):
    """Log evaluation result to wandb"""
    wandb.log({
        "eval_step": eval_step,
        "eval/accuracy": accuracy,
        "config/samples": num_samples if num_samples else train_data_len,
        "config/learning_rate": lr,
        "config/batch_size": batch_size
    })

def save_best_model(model, tokenizer, accuracy, best_accuracy, num_samples, lr, batch_size, save_dir, train_data_len):
    """Save model if it's the best so far"""
    if accuracy > best_accuracy:
        best_config = {
            'samples': num_samples if num_samples else train_data_len,
            'lr': lr,
            'batch_size': batch_size,
            'accuracy': accuracy
        }
        
        model_save_path = os.path.join(save_dir, f"best_model_samples_{num_samples}_lr_{lr}_bs_{batch_size}")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Saved best model to {model_save_path}")
        
        return accuracy, best_config
    return best_accuracy, None

def reset_model_for_next_config(model_name, device, num_samples, num_samples_list, lr, learning_rates, batch_size, batch_sizes):
    """Reset model weights for next configuration"""
    if num_samples != num_samples_list[-1] or lr != learning_rates[-1] or batch_size != batch_sizes[-1]:
        print("Reloading model for next configuration...")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = model.to(device)
        return model
    return None

def log_training_step(total_steps, loss, current_lr, num_samples, lr, batch_size, train_data_len, warmup_steps=100):
    """Log training step to wandb"""
    is_warmup = total_steps < warmup_steps
    
    wandb.log({
        "train_step": total_steps,
        "train/loss": loss.item(),
        "train/learning_rate": current_lr,
        "train/is_warmup": is_warmup,
        "train/warmup_progress": min(total_steps / warmup_steps, 1.0) if is_warmup else 1.0,
        "config/samples": num_samples if num_samples else train_data_len,
        "config/learning_rate": lr,
        "config/batch_size": batch_size,
        "config/warmup_steps": warmup_steps
    })

def process_evaluation_results(eval_queue, model, tokenizer, num_samples, lr, batch_size, save_dir, best_accuracy, train_data_len):
    """Process evaluation results from queue"""
    try:
        while not eval_queue.empty():
            eval_step, accuracy = eval_queue.get_nowait()
            if accuracy is not None:
                print(f"Received evaluation result at step {eval_step}: accuracy = {accuracy:.4f}")
                log_evaluation_result(eval_step, accuracy, num_samples, lr, batch_size, train_data_len)
                best_accuracy, best_config = save_best_model(
                    model, tokenizer, accuracy, best_accuracy, num_samples, lr, batch_size, save_dir, train_data_len
                )
                if best_config:
                    return best_accuracy, best_config
            else:
                print(f"Evaluation at step {eval_step} failed")
    except queue.Empty:
        pass
    return best_accuracy, None

def train_sft_model(
    model_name="Qwen/Qwen2.5-Math-1.5B",
    train_data_path="data/gsm8k/train.jsonl",
    eval_data_path="data/gsm8k/test.jsonl",
    num_samples_list=[None],  # None means use all data
    learning_rates=[1e-5, 2e-5, 5e-5],
    batch_sizes=[16, 32, 64],
    gradient_accumulation_steps=2,
    max_epochs=1,
    device="cuda:0",
    vllm_device="cuda:1",
    save_dir="sft_models"
):
    """Main training function for SFT on math problems"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    vllm_model = init_vllm(model_name, vllm_device, seed=42)
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_gsm8k_data(eval_data_path)
    
    # Initialize wandb
    init_wandb()
    
    best_accuracy = 0.0
    best_config = None
    
    # Iterate through different configurations
    for num_samples in num_samples_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                if batch_size > 16:
                    actual_batch_size = 16
                    gradient_accumulation_steps = batch_size // 16
                else:
                    actual_batch_size = batch_size
                    gradient_accumulation_steps = 1
                print(f"\n=== Training with config: samples={num_samples}, lr={lr}, batch_size={batch_size} ===")
                
                # Load training data
                tokenized_data, train_data = prepare_training_data(train_data_path, num_samples, tokenizer)
                
                # Create data loader
                dataloader = create_dataloader(tokenized_data, actual_batch_size)
                
                # Optimizer and manual learning rate adjustment
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                current_lr = lr
                
                # Calculate total training steps
                total_training_steps = len(dataloader) * max_epochs // gradient_accumulation_steps
                warmup_steps = total_training_steps*0.05
                
                # Training loop
                model.train()
                total_steps = 0
                
                # Initialize evaluation queue and thread
                eval_queue = queue.Queue()
                eval_thread = None
                
                # Perform initial evaluation
                print("Performing initial evaluation...")
                load_policy_init_vllm_instance(model, vllm_model)
                initial_accuracy = evaluate_model(model, tokenizer, eval_data, device, vllm_model)
                log_evaluation_result(0, initial_accuracy, num_samples, lr, batch_size, len(train_data))
                best_accuracy, best_config = save_best_model(
                    model, tokenizer, initial_accuracy, best_accuracy, num_samples, lr, batch_size, save_dir, len(train_data)
                )
                if best_config:
                    print(f"Saved initial best model to {os.path.join(save_dir, f'best_model_samples_{num_samples}_lr_{lr}_bs_{batch_size}')}")
                
                print(f"Initial accuracy: {initial_accuracy:.4f}")
                
                max_steps = len(dataloader) * max_epochs
                for epoch in range(max_epochs):
                    epoch_loss = 0.0
                    
                    for batch_idx, (input_ids, labels, response_mask) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
                        input_ids = input_ids.to(device)
                        labels = labels.to(device)
                        response_mask = response_mask.to(device)
                        
                        response_log_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                        log_probs = response_log_probs['log_probs']
                        loss, _ = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps)
                        
                        total_steps += 1
                        # Backward pass
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            # Add gradient clipping to prevent gradient explosion
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Manually adjust learning rate
                            current_lr = adjust_learning_rate(
                                optimizer, 
                                total_steps, 
                                total_training_steps, 
                                lr, 
                                warmup_steps
                            )
                        
                        epoch_loss += loss.item()
                        
                        # Check if evaluation should be started
                        if total_steps % 200 == 0 or total_steps == max_steps - 1:
                            # If previous evaluation thread is still running, wait for it to complete
                            if eval_thread and eval_thread.is_alive():
                                print(f"Waiting for previous evaluation to complete...")
                                eval_thread.join()
                            
                            load_policy_init_vllm_instance(model, vllm_model)
                            # Start new evaluation thread
                            eval_thread = threading.Thread(
                                target=async_evaluate_model,
                                args=(model, tokenizer, eval_data, device, vllm_model, eval_queue, total_steps)
                            )
                            eval_thread.daemon = True  # Set as daemon thread, automatically ends when main program ends
                            eval_thread.start()
                            print(f"Started evaluation thread at step {total_steps}")
                        
                        # Check evaluation results
                        best_accuracy, best_config = process_evaluation_results(eval_queue, model, tokenizer, num_samples, lr, batch_size, save_dir, best_accuracy, len(train_data))
                        
                        # Log to wandb
                        if total_steps % 10 == 0:
                            log_training_step(total_steps, loss, current_lr, num_samples, lr, batch_size, len(train_data), warmup_steps)
                        
                        if batch_idx % 50 == 0:
                            warmup_status = "WARMUP" if total_steps < warmup_steps else "TRAINING"
                            print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}, {warmup_status}")
                    
                    print(f"Epoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")
                
                # Wait for final evaluation to complete
                if eval_thread and eval_thread.is_alive():
                    print("Waiting for final evaluation to complete...")
                    eval_thread.join()
                
                # Process final evaluation results
                best_accuracy, best_config = process_evaluation_results(eval_queue, model, tokenizer, num_samples, lr, batch_size, save_dir, best_accuracy, len(train_data))
                
                # Reset model weights for next configuration
                new_model = reset_model_for_next_config(model_name, device, num_samples, num_samples_list, lr, learning_rates, batch_size, batch_sizes)
                if new_model is not None:
                    model = new_model
    
    print(f"\n=== Training Complete ===")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best config: {best_config}")
    
    return best_accuracy, best_config

if __name__ == "__main__":
    set_random_seed()
    # Run training
    best_acc, best_cfg = train_sft_model()
    print(f"Final best accuracy: {best_acc:.4f}")
    print(f"Best configuration: {best_cfg}")

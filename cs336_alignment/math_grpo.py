from typing import Callable, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import os
import csv
from datetime import datetime

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cs336_alignment.math_baseline import (
    format_prompt_with_r1_zero, 
    load_gsm8k_data,
    extract_answer_from_response,
    extract_answer_from_gsm8k_answer,
    evaluate_vllm,
)
from cs336_alignment.math_sft import (
    set_random_seed,
    init_vllm,
    load_policy_init_vllm_instance,
    tokenize_prompt_and_output,
    get_response_log_probs,
    adjust_learning_rate,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

class CSVLogger:
    """CSV logger for training and validation metrics"""
    
    def __init__(self, train_csv_path, val_csv_path):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        
        # Initialize training CSV
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_reward', 'train_format_reward', 'train_answer_reward', 
                           'train_loss', 'gradient_norm', 'token_entropy', 'clip_fraction', 'learning_rate'])
        
        # Initialize validation CSV
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'val_reward', 'val_format_reward', 'val_answer_reward'])
    
    def log_train_step(self, step, train_reward, train_format_reward, train_answer_reward, 
                      train_loss, gradient_norm, token_entropy, clip_fraction, learning_rate):
        """Log training metrics for each step"""
        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, train_reward, train_format_reward, train_answer_reward,
                           train_loss, gradient_norm, token_entropy, clip_fraction, learning_rate])
    
    def log_val_step(self, step, val_reward, val_format_reward, val_answer_reward):
        """Log validation metrics after evaluation"""
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, val_reward, val_format_reward, val_answer_reward])

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    reward_infos = []
    raw_rewards = []
    n_prompts_per_rollout_batch = len(rollout_responses) // group_size
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, ground_truth)
        reward_infos.append(reward_info)
        raw_rewards.append(reward_info['reward'])
    raw_rewards = torch.tensor(raw_rewards).reshape(n_prompts_per_rollout_batch, group_size)
    advantages = raw_rewards - raw_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = advantages / (raw_rewards.std(dim=1, keepdim=True) + advantage_eps)
    advantages = advantages.reshape(-1)
    raw_rewards = raw_rewards.reshape(-1)
    return advantages, raw_rewards, reward_infos

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> torch.Tensor:
    relative_probs = (policy_log_probs - old_log_probs).exp()
    ans = - torch.min(relative_probs * advantages, torch.clamp(relative_probs, 1 - cliprange, 1 + cliprange) * advantages)
    clipped_count = ((relative_probs > 1 + cliprange) | (relative_probs < 1 - cliprange)).sum().item()
    return ans, {"clipped_count": clipped_count}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), None
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), None
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss, loss_info = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, loss_info

class DataSampler:
    """Simple data sampler for training data with shuffle and sequential access"""
    
    def __init__(self, prompts, ground_truths):
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.current_idx = 0
        self.shuffle_data()
    
    def shuffle_data(self):
        """Shuffle the data and reset current index"""
        indices = list(range(len(self.prompts)))
        random.shuffle(indices)
        self.shuffled_prompts = [self.prompts[i] for i in indices]
        self.shuffled_ground_truths = [self.ground_truths[i] for i in indices]
        self.current_idx = 0
    
    def get_batch(self, batch_size):
        """Get next batch of data, reshuffle if needed"""
        start_idx = self.current_idx
        end_idx = start_idx + batch_size
        
        # If we run out of data, reshuffle and start over
        if end_idx > len(self.shuffled_prompts):
            self.shuffle_data()
            start_idx = 0
            end_idx = batch_size
        
        batch_prompts = self.shuffled_prompts[start_idx:end_idx]
        batch_ground_truths = self.shuffled_ground_truths[start_idx:end_idx]
        self.current_idx = end_idx
        
        return batch_prompts, batch_ground_truths

def evaluate_model_grpo(model, tokenizer, eval_prompts, eval_ground_truths, device, vllm_model):
    """Evaluate model performance using vLLM on separate GPU for GRPO"""
    print(f"Evaluating on {len(eval_prompts)} validation samples using vLLM...")
    # Set sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]
    )
    eval_sampling_params.include_stop_str_in_output = True
    
    # Evaluate using the imported evaluate_vllm function
    eval_results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=eval_prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=eval_ground_truths
    )
    return eval_results

def grpo_train_loop(
    policy: nn.Module,
    tokenizer,
    train_data_path: str,
    val_data_path: str,
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    vllm_device: str = "cuda:1",
    device: str = "cuda:0",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 32,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    val_every_n_steps: int = 10,
    gpu_memory_utilization: float = 0.85,
    seed: int = 42,
) -> dict[str, list]:
    """
    Complete GRPO training loop implementation for GSM8K math problems.
    
    Args:
        policy: The policy model to train
        tokenizer: Tokenizer for the model
        train_data_path: Path to training data file
        val_data_path: Path to validation data file
        model_name: Model name for vLLM initialization
        vllm_device: Device for vLLM engine
        n_grpo_steps: Number of GRPO training steps
        learning_rate: Learning rate for optimizer
        advantage_eps: Epsilon for advantage normalization
        rollout_batch_size: Number of rollouts per batch
        group_size: Number of rollouts per group for normalization
        sampling_temperature: Temperature for sampling
        sampling_min_tokens: Minimum tokens to generate
        sampling_max_tokens: Maximum tokens to generate
        epochs_per_rollout_batch: Number of epochs per rollout batch
        train_batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        loss_type: Type of policy gradient loss
        use_std_normalization: Whether to normalize advantages by std
        cliprange: Clip range for GRPO-Clip loss
        val_every_n_steps: Validate every N steps
        gpu_memory_utilization: GPU memory utilization for vLLM
        seed: Random seed for reproducibility
        
    Returns:
        dict: Training history with losses, rewards, etc.
    """
    
    # Sanity check asserts
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    # Initialize vLLM engine
    print(f"Initializing vLLM engine on {vllm_device}...")
    vllm_engine = init_vllm(model_name, vllm_device, seed, gpu_memory_utilization)
    
    # Load training and validation data
    print("Loading training and validation data...")
    
    # Load training data
    train_data = load_gsm8k_data(train_data_path)
    train_prompts = []
    train_ground_truths = []
    for item in train_data:
        question = item['question']
        answer = item['answer']
        train_prompts.append(format_prompt_with_r1_zero(question))
        train_ground_truths.append(extract_answer_from_gsm8k_answer(answer))
    
    # Load validation data
    val_data = load_gsm8k_data(val_data_path)
    val_prompts = []
    val_ground_truths = []
    for item in val_data:
        question = item['question']
        answer = item['answer']
        val_prompts.append(format_prompt_with_r1_zero(question))
        val_ground_truths.append(extract_answer_from_gsm8k_answer(answer))
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    # Calculate total training steps for learning rate scheduling
    total_training_steps = n_grpo_steps * epochs_per_rollout_batch
    warmup_steps = int(total_training_steps * 0.05)  # 5% warmup
    
    # Create data sampler for training
    train_sampler = DataSampler(train_prompts, train_ground_truths)
    
    # Training history
    history = {
        'train_losses': [],
        'train_rewards': [],
        'train_format_rewards': [],
        'train_answer_rewards': [],
        'val_rewards': [],
        'val_format_rewards': [],
        'val_answer_rewards': [],
        'gradient_norms': [],
        'token_entropies': [],
        'clip_fractions': [],
        'learning_rates': [],
    }
    
    # Initialize CSV logger
    train_csv_path = f"training_logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    val_csv_path = f"training_logs/validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("training_logs", exist_ok=True)
    csv_logger = CSVLogger(train_csv_path, val_csv_path)

    # Training loop
    print('Starting training loop...')
    for step in tqdm(range(n_grpo_steps), desc="GRPO Training"):
        policy.train()
        
        # Get next batch of data using the sampler
        rollout_prompts, rollout_ground_truths = train_sampler.get_batch(n_prompts_per_rollout_batch)
        
        # Load current policy weights into vLLM
        load_policy_init_vllm_instance(policy, vllm_engine)
        
        # Generate multiple responses per prompt using vLLM's n parameter
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            min_tokens=sampling_min_tokens,
            max_tokens=sampling_max_tokens,
            stop=['</answer>'],
            n=group_size  # Generate n responses per prompt
        )
        
        # Generate with progress bar disabled
        outputs = vllm_engine.generate(rollout_prompts, sampling_params, use_tqdm=False)
        
        # Extract all responses and ground truths
        repeated_rollout_prompts = []
        repeated_rollout_responses = []
        repeated_ground_truths = []
        for i, output in enumerate(outputs):
            # Each output contains n responses for one prompt
            for j in range(group_size):
                response = output.outputs[j].text
                repeated_rollout_responses.append(response)
                repeated_rollout_prompts.append(rollout_prompts[i])
                repeated_ground_truths.append(rollout_ground_truths[i])
        assert len(repeated_rollout_responses) == len(repeated_ground_truths) == rollout_batch_size

        # Compute rewards and advantages
        advantages, raw_rewards, reward_infos = compute_group_normalized_rewards(
            r1_zero_reward_fn, repeated_rollout_responses, repeated_ground_truths, 
            group_size, advantage_eps, use_std_normalization
        )
        advantages = advantages.unsqueeze(-1).to(device)
        raw_rewards = raw_rewards.unsqueeze(-1).to(device)

        # Tokenize rollout data
        tokenized_data = tokenize_prompt_and_output(repeated_rollout_prompts, repeated_rollout_responses, tokenizer)
        input_ids = tokenized_data['input_ids'].to(device)
        labels = tokenized_data['labels'].to(device)
        response_mask = tokenized_data['response_mask'].to(device)
        
        # Compute old log probs if using GRPO-Clip (off-policy)
        old_log_probs = None
        if loss_type == "grpo_clip" and epochs_per_rollout_batch > 1:            
            with torch.no_grad():
                old_log_probs = get_response_log_probs(
                    policy, 
                    input_ids, 
                    labels
                )['log_probs']
        
        # Multiple epochs per rollout batch (off-policy)
        for epoch in range(epochs_per_rollout_batch):         
            # Process in microbatches
            optimizer.zero_grad()
            
            token_entropy_list = []
            loss_batch = 0.0
            for microbatch_idx in range(n_microbatches_per_rollout_batch):
                start_idx = microbatch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size

                microbatch_input_ids = input_ids[start_idx:end_idx]
                microbatch_labels = labels[start_idx:end_idx]
                microbatch_response_mask = response_mask[start_idx:end_idx]
                microbatch_advantages = advantages[start_idx:end_idx]
                microbatch_raw_rewards = raw_rewards[start_idx:end_idx]
                
                if old_log_probs is not None:
                    microbatch_old_log_probs = old_log_probs[start_idx:end_idx]
                else:
                    microbatch_old_log_probs = None
                
                # Forward pass
                response_log_probs_result = get_response_log_probs(policy, microbatch_input_ids, microbatch_labels, return_token_entropy=True)
                log_probs = response_log_probs_result['log_probs']
                token_entropy = response_log_probs_result['token_entropy']
                token_entropy_list.append(token_entropy.mean().item())
                # Compute loss
                loss, loss_info = grpo_microbatch_train_step(
                    log_probs,
                    microbatch_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    microbatch_raw_rewards,
                    microbatch_advantages,
                    microbatch_old_log_probs,
                    cliprange
                )
                loss_batch += loss.item() / n_microbatches_per_rollout_batch
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Adjust learning rate
            current_lr = adjust_learning_rate(
                optimizer, 
                step, 
                total_training_steps, 
                learning_rate, 
                warmup_steps
            )
            
            # Logging
            history['train_losses'].append(loss_batch)
            history['train_rewards'].append(raw_rewards.mean().item())
            history['train_format_rewards'].append(
                np.mean([info['format_reward'] for info in reward_infos])
            )
            history['train_answer_rewards'].append(
                np.mean([info['answer_reward'] for info in reward_infos])
            )
            
            # Compute gradient norm
            total_norm = 0
            for p in policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            history['gradient_norms'].append(total_norm)
            
            # Record token entropy
            history['token_entropies'].append(np.mean(token_entropy_list))
            
            # Log clip fraction if using GRPO-Clip
            if loss_type == "grpo_clip" and loss_info is not None:
                clip_fraction = loss_info['clipped_count'] / log_probs.numel()
                history['clip_fractions'].append(clip_fraction)
            else:
                history['clip_fractions'].append(0.0)
            
            # Record learning rate
            history['learning_rates'].append(current_lr)
            
            # Log training metrics to CSV for each step
            csv_logger.log_train_step(
                step,
                history['train_rewards'][-1],
                history['train_format_rewards'][-1],
                history['train_answer_rewards'][-1],
                history['train_losses'][-1],
                history['gradient_norms'][-1],
                history['token_entropies'][-1],
                history['clip_fractions'][-1],
                history['learning_rates'][-1]
            )
        
        # Check if evaluation should be started
        if step % val_every_n_steps == 0 or step == n_grpo_steps - 1:
            print(f"Starting evaluation at step {step}...")
            
            # Load current policy weights into vLLM for evaluation
            load_policy_init_vllm_instance(policy, vllm_engine)
            
            # Perform synchronous evaluation
            eval_results = evaluate_model_grpo(policy, tokenizer, val_prompts, val_ground_truths, device, vllm_engine)
            
            # Log validation metrics
            history['val_rewards'].append(eval_results['metrics']['avg_reward'])
            history['val_format_rewards'].append(eval_results['metrics']['format_accuracy'])
            history['val_answer_rewards'].append(eval_results['metrics']['accuracy'])
            
            # Log validation metrics to CSV
            csv_logger.log_val_step(
                step,
                eval_results['metrics']['avg_reward'],
                eval_results['metrics']['format_accuracy'],
                eval_results['metrics']['accuracy']
            )
            
            print(f"Step {step}: Val Reward: {eval_results['metrics']['avg_reward']:.4f}, "
                  f"Val Format: {eval_results['metrics']['format_accuracy']:.4f}, "
                  f"Val Answer: {eval_results['metrics']['accuracy']:.4f}")
        
        # Print training progress
        if step % val_every_n_steps == 0 or step == n_grpo_steps - 1:
            print(f"Step {step}: Train Reward: {history['train_rewards'][-1]:.4f}, LR: {current_lr:.2e}")
            if history['val_rewards']:
                print(f"Step {step}: Val Reward: {history['val_rewards'][-1]:.4f}")
    
    return history


def main():
    set_random_seed(42)
    # Set device
    device = "cuda:0"
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    
    # Data paths
    train_data_path = "data/gsm8k/train.jsonl"
    val_data_path = "data/gsm8k/test.jsonl"
    
    # Run GRPO training
    print("Starting GRPO training...")
    history = grpo_train_loop(
        policy=model,
        tokenizer=tokenizer,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        model_name=model_name,
    )
    
    print("Training completed!")
    print(f"Final train reward: {history['train_rewards'][-1]:.4f}")
    print(f"Final val reward: {history['val_rewards'][-1]:.4f}")
    print(f"Final learning rate: {history['learning_rates'][-1]:.2e}")
    
    return history


if __name__ == "__main__":
    main()


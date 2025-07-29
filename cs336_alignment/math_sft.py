from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
import random
from typing import List, Dict, Any

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
    gradient_accumulation_steps: int,
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
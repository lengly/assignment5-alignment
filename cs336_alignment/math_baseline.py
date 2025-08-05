import json
import re
from typing import Callable, List, Dict, Any
from vllm import LLM, SamplingParams
import os
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_gsm8k_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load GSM8k dataset
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_prompt_with_r1_zero(question: str) -> str:
    """
    Format question using r1_zero prompt template from file
    """
    prompt_file = "/workspace/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Could not find prompt file {prompt_file}, using fallback template")
        prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    
    return prompt_template.format(question=question)

def extract_answer_from_response(response: str) -> str:
    """
    Extract answer from model response
    """
    # Try to extract answer from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # If no <answer> tags found, try to extract the last number
    numbers = re.findall(r'\d+', response)
    if numbers:
        return numbers[-1]
    
    return response.strip()

def extract_answer_from_gsm8k_answer(answer: str) -> str:
    """
    Extract final answer from GSM8k answer
    """
    # GSM8k answer format is usually "#### number"
    match = re.search(r'####\s*(\d+)', answer)
    if match:
        return match.group(1)
    return answer.strip()

def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate accuracy
    """
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        if pred.strip() == gt.strip():
            correct += 1
    
    return correct / total if total > 0 else 0.0

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    
    Returns:
        Dict containing evaluation results and metrics
    """
    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Extract generated texts
    generated_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    
    # Calculate reward scores
    rewards = []
    format_rewards = []
    answer_rewards = []
    
    if ground_truths:
        # Use r1_zero_reward_fn if ground truths are provided
        for generated_text, ground_truth in zip(generated_texts, ground_truths):
            reward_result = reward_fn(generated_text, ground_truth, fast=True)
            rewards.append(reward_result["reward"])
            format_rewards.append(reward_result["format_reward"])
            answer_rewards.append(reward_result["answer_reward"])
    else:
        # Use provided reward function
        for prompt, generated_text in zip(prompts, generated_texts):
            score = reward_fn(prompt, generated_text)
            if isinstance(score, dict):
                rewards.append(score.get("reward", 0.0))
                format_rewards.append(score.get("format_reward", 0.0))
                answer_rewards.append(score.get("answer_reward", 0.0))
            else:
                rewards.append(score)
                format_rewards.append(score)
                answer_rewards.append(score)
    
    # Calculate metrics
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    accuracy = 0.0
    format_accuracy = 0.0
    
    if ground_truths and answer_rewards:
        accuracy = sum(answer_rewards) / len(answer_rewards)
        format_accuracy = sum(format_rewards) / len(format_rewards) if format_rewards else 0.0
    
    # Print results
    print(f"Generated {len(generated_texts)} responses")
    print(f"Average reward: {avg_reward:.4f}")
    if ground_truths:
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Format accuracy: {format_accuracy:.4f} ({format_accuracy*100:.2f}%)")
    
    # Prepare return results
    evaluation_results = {
        "generated_texts": generated_texts,
        "rewards": rewards,
        "format_rewards": format_rewards,
        "answer_rewards": answer_rewards,
        "metrics": {
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "format_accuracy": format_accuracy,
            "total_examples": len(generated_texts)
        }
    }
    
    if ground_truths:
        evaluation_results["ground_truths"] = ground_truths
    
    # Save results to file
    results = {
        "prompts": prompts,
        "generated_texts": generated_texts,
        "rewards": rewards,
        "format_rewards": format_rewards,
        "answer_rewards": answer_rewards
    }
    
    if ground_truths:
        results["ground_truths"] = ground_truths
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Evaluation results saved to evaluation_results.json")
    
    return evaluation_results

def main():
    """
    Main function: Evaluate Qwen 2.5 Math 1.5B zero-shot performance on GSM8k
    """
    # 1. Load GSM8k validation data
    gsm8k_file = "/workspace/assignment5-alignment/data/gsm8k/test.jsonl"
    print(f"Loading GSM8k data from: {gsm8k_file}")
    
    if not os.path.exists(gsm8k_file):
        print(f"Error: File not found {gsm8k_file}")
        return
    
    gsm8k_data = load_gsm8k_data(gsm8k_file)
    print(f"Loaded {len(gsm8k_data)} test samples")
    
    # 2. Format prompts
    print("Formatting prompts...")
    formatted_prompts = []
    ground_truth_answers = []
    
    for example in gsm8k_data:
        question = example["question"]
        answer = example["answer"]
        
        formatted_prompt = format_prompt_with_r1_zero(question)
        formatted_prompts.append(formatted_prompt)
        ground_truth_answers.append(extract_answer_from_gsm8k_answer(answer))
    
    # 3. Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,  # Sample with temperature 1.0
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"]  # Stop when the model completes its answer
    )
    sampling_params.include_stop_str_in_output = True
    
    # 4. Load model
    print("Loading Qwen 2.5 Math 1.5B model...")
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B", dtype="bfloat16", enable_prefix_caching=True, gpu_memory_utilization=0.85)
    
    # 5. Use evaluate_vllm function for evaluation
    print("Evaluating model using evaluate_vllm function...")
    evaluation_results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=formatted_prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truth_answers
    )
    
    # Extract results from evaluation_results
    generated_texts = evaluation_results["generated_texts"]
    rewards = evaluation_results["rewards"]
    format_rewards = evaluation_results["format_rewards"]
    answer_rewards = evaluation_results["answer_rewards"]
    metrics = evaluation_results["metrics"]
    
    # Extract predicted answers from generated texts
    predicted_answers = []
    for generated_text in generated_texts:
        predicted_answer = extract_answer_from_response(generated_text)
        predicted_answers.append(predicted_answer)
    
    # 6. Serialize results to disk
    print("Saving results...")
    results = {
        "model": "Qwen/Qwen2.5-Math-1.5B",
        "dataset": "GSM8k",
        "evaluation_type": "zero_shot",
        "total_examples": len(gsm8k_data),
        "accuracy": metrics["accuracy"],
        "format_accuracy": metrics["format_accuracy"],
        "average_reward": metrics["avg_reward"],
        "examples": []
    }
    
    for i, (example, prompt, generated_text, predicted_answer, ground_truth, reward, format_reward, answer_reward) in enumerate(
        zip(gsm8k_data, formatted_prompts, generated_texts, predicted_answers, ground_truth_answers, rewards, format_rewards, answer_rewards)
    ):
        result_example = {
            "id": i,
            "question": example["question"],
            "ground_truth_answer": example["answer"],
            "ground_truth_extracted": ground_truth,
            "formatted_prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "reward": reward,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "is_correct": answer_reward == 1.0
        }
        results["examples"].append(result_example)
    
    # Save detailed results
    output_file = "gsm8k_zero_shot_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    
    # Save summary results for subsequent analysis
    summary_results = {
        "model": "Qwen/Qwen2.5-Math-1.5B",
        "dataset": "GSM8k",
        "evaluation_type": "zero_shot",
        "total_examples": len(gsm8k_data),
        "accuracy": metrics["accuracy"],
        "format_accuracy": metrics["format_accuracy"],
        "average_reward": metrics["avg_reward"],
        "correct_predictions": sum(1 for r in answer_rewards if r == 1.0),
        "incorrect_predictions": sum(1 for r in answer_rewards if r == 0.0),
        "correct_format": sum(1 for r in format_rewards if r == 1.0),
        "incorrect_format": sum(1 for r in format_rewards if r == 0.0)
    }
    
    summary_file = "gsm8k_zero_shot_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to: {summary_file}")
    print("Evaluation completed!")

def analyze_generation_categories(results_file: str = "gsm8k_zero_shot_results.json") -> Dict[str, Any]:
    """
    Analyze model generation categories
    
    Args:
        results_file: Path to the JSON file containing evaluation results
        
    Returns:
        Dictionary containing statistics for each category
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {results_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Cannot parse JSON file {results_file}")
        return {}
    
    # Initialize counters
    category_counts = {
        "correct_format_and_answer": 0,  # format reward 1 and answer reward 1
        "correct_format_wrong_answer": 0,  # format reward 1 and answer reward 0
        "wrong_format_and_answer": 0,  # format reward 0 and answer reward 0
        "wrong_format_correct_answer": 0  # format reward 0 and answer reward 1 (additional category)
    }
    
    # Analyze each sample
    for example in results.get("examples", []):
        format_reward = example.get("format_reward", 0)
        answer_reward = example.get("answer_reward", 0)
        
        if format_reward == 1.0 and answer_reward == 1.0:
            category_counts["correct_format_and_answer"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            category_counts["correct_format_wrong_answer"] += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            category_counts["wrong_format_and_answer"] += 1
        elif format_reward == 0.0 and answer_reward == 1.0:
            category_counts["wrong_format_correct_answer"] += 1
    
    total_examples = len(results.get("examples", []))
    
    # Calculate percentages
    percentages = {}
    for category, count in category_counts.items():
        percentages[category] = (count / total_examples * 100) if total_examples > 0 else 0
    
    # Prepare results
    analysis_results = {
        "total_examples": total_examples,
        "category_counts": category_counts,
        "percentages": percentages,
        "summary": {
            "correct_format_and_answer": f"{category_counts['correct_format_and_answer']} ({percentages['correct_format_and_answer']:.2f}%)",
            "correct_format_wrong_answer": f"{category_counts['correct_format_wrong_answer']} ({percentages['correct_format_wrong_answer']:.2f}%)",
            "wrong_format_and_answer": f"{category_counts['wrong_format_and_answer']} ({percentages['wrong_format_and_answer']:.2f}%)",
            "wrong_format_correct_answer": f"{category_counts['wrong_format_correct_answer']} ({percentages['wrong_format_correct_answer']:.2f}%)"
        }
    }
    
    # Print results
    print(f"\n=== Model Generation Category Analysis ===")
    print(f"Total examples: {total_examples}")
    print(f"\nCategory statistics:")
    print(f"(1) Correct format and answer (format=1, answer=1): {analysis_results['summary']['correct_format_and_answer']}")
    print(f"(2) Correct format but wrong answer (format=1, answer=0): {analysis_results['summary']['correct_format_wrong_answer']}")
    print(f"(3) Wrong format and answer (format=0, answer=0): {analysis_results['summary']['wrong_format_and_answer']}")
    print(f"(4) Wrong format but correct answer (format=0, answer=1): {analysis_results['summary']['wrong_format_correct_answer']}")
    
    # Save analysis results
    analysis_file = "generation_categories_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAnalysis results saved to: {analysis_file}")
    
    return analysis_results

if __name__ == "__main__":
    main()
    analyze_generation_categories()
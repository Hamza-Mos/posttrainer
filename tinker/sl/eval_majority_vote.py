"""Evaluate SFT model with majority voting (N samples per problem).

Samples N completions per problem at temperature 0.7, extracts answers,
and takes the majority vote. Reduces variance significantly.

Usage: python eval_majority_vote.py <sampler_weights_path> [num_samples]
"""

import json
import re
import sys
from collections import Counter
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(completion: str) -> str | None:
    """Extract the answer from a completion."""
    boxed_matches = _BOXED_RE.findall(completion)
    if boxed_matches:
        return boxed_matches[-1].strip()
    num_matches = _NUM_RE.findall(completion.replace(",", ""))
    if num_matches:
        return num_matches[-1].strip()
    return None


def check_correct(answer: str, ground_truth: str) -> bool:
    if answer is None:
        return False
    expected = _normalize_number(ground_truth)
    predicted = _normalize_number(answer)
    if expected is not None and predicted is not None:
        return abs(predicted - expected) < 1e-6
    return answer.strip() == ground_truth.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_majority_vote.py <sampler_weights_path> [num_samples]")
        sys.exit(1)

    sampler_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else TEMPERATURE
    print(f"Model: {MODEL}")
    print(f"Sampler weights: {sampler_path}")
    print(f"Samples per problem: {num_samples}")
    print(f"Temperature: {temperature}")

    # Load eval prompts
    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                eval_prompts.append(json.loads(line))
    print(f"Eval problems: {len(eval_prompts)}")

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=MODEL, model_path=sampler_path
    )

    stop_sequences = []
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)
    for stop_tok in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if stop_tok not in stop_sequences:
            stop_sequences.append(stop_tok)

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        stop=stop_sequences if stop_sequences else None,
    )

    # Submit all sampling requests (N samples per problem)
    futures = []
    for item in eval_prompts:
        messages = [{"role": "user", "content": item["prompt"]}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        model_input = types.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=token_ids)]
        )
        future = sampling_client.sample(
            prompt=model_input, num_samples=num_samples, sampling_params=sampling_params
        )
        futures.append((future, item))

    # Collect results with majority voting
    correct_greedy = 0  # Would-be greedy (first sample)
    correct_majority = 0
    correct_any = 0  # If ANY sample got it right

    for i, (future, item) in enumerate(futures):
        result = future.result()
        answers = []
        any_correct = False

        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            answer = extract_answer(text)
            if answer is not None:
                answers.append(answer)
            if check_correct(answer, item["ground_truth"]):
                any_correct = True

        # First sample (pseudo-greedy)
        if answers and check_correct(answers[0], item["ground_truth"]):
            correct_greedy += 1

        # Majority vote: normalize answers and pick most common
        if answers:
            # Normalize to numbers where possible
            normalized = []
            for a in answers:
                n = _normalize_number(a)
                if n is not None:
                    normalized.append(f"{n:.6f}")
                else:
                    normalized.append(a.strip())

            majority_answer = Counter(normalized).most_common(1)[0][0]
            if check_correct(majority_answer, item["ground_truth"]):
                correct_majority += 1

        if any_correct:
            correct_any += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_prompts)}] "
                  f"greedy: {correct_greedy/(i+1):.1%} | "
                  f"majority: {correct_majority/(i+1):.1%} | "
                  f"any: {correct_any/(i+1):.1%}")

    n = len(eval_prompts)
    print(f"\n{'='*60}")
    print(f"RESULTS: Majority Vote ({num_samples} samples, temp={TEMPERATURE})")
    print(f"{'='*60}")
    print(f"greedy_accuracy: {correct_greedy/n:.4f} ({correct_greedy}/{n})")
    print(f"majority_accuracy: {correct_majority/n:.4f} ({correct_majority}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")
    print(f"\nThree-way comparison (majority vote):")
    print(f"  Base Qwen3-8B:     ~15%")
    print(f"  SFT majority vote: {correct_majority/n:.1%}")
    print(f"  RL (exp 7):        83.13%")


if __name__ == "__main__":
    main()

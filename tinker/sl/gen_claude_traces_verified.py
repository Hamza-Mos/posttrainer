"""Generate Claude reasoning traces with answer verification against ground truth.

Only keeps traces where Claude's boxed answer matches the MATH dataset's answer.
This eliminates incorrect reasoning from the training data.

Run: python gen_claude_traces_verified.py
"""

import anthropic
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load API key
env_path = Path(__file__).parent / "../../.env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 10
TARGET = 800  # Generate more since we'll filter (~70% pass rate → ~550)

SYSTEM_PROMPT = """You are a math tutor solving problems step by step. Your solutions must:

1. Show EVERY intermediate step - never skip steps or say "clearly" or "obviously"
2. Explain WHY you're doing each step
3. Self-verify your answer: after finding it, check by substitution or estimation
4. End with the final answer in \\boxed{answer} format (numeric answer only inside the box)

Example format:
I need to find the value of x where 2x + 3 = 11.

First, I'll isolate x by subtracting 3 from both sides:
2x + 3 - 3 = 11 - 3
2x = 8

Now I'll divide both sides by 2:
x = 8/2 = 4

Let me verify: 2(4) + 3 = 8 + 3 = 11 ✓

\\boxed{4}"""


def find_boxed(text: str) -> list[tuple[int, int, str]]:
    """Find all \\boxed{...} spans with content, handling nested braces."""
    spans = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        j = idx + 6
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    content = text[idx+7:j]
                    spans.append((idx, j + 1, content))
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return spans


def normalize_number(s: str) -> float | None:
    """Try to parse a string as a number."""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected (numeric comparison)."""
    pred_num = normalize_number(predicted)
    exp_num = normalize_number(expected)
    if pred_num is not None and exp_num is not None:
        return abs(pred_num - exp_num) < 1e-6
    # String comparison as fallback
    return predicted.strip() == expected.strip()


def generate_trace(client, problem: str) -> str | None:
    """Generate a reasoning trace, return raw text or None."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Solve this problem step by step:\n\n{problem}"}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


def main():
    print("Loading MATH problems with ground truth...")

    # Load eval/RL problems to exclude
    eval_problems = set()
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                eval_problems.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Load MATH dataset WITH ground truth answers
    from datasets import load_dataset
    import random

    subjects = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    all_problems = []
    for subj in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
        for ex in ds:
            try:
                level_num = int(ex['level'].replace('Level ', ''))
            except ValueError:
                continue
            if level_num < 2:
                continue
            problem_norm = re.sub(r'\s+', ' ', ex['problem'].strip())
            if problem_norm in eval_problems:
                continue

            # Extract ground truth answer from solution
            boxed = find_boxed(ex['solution'])
            if not boxed:
                continue
            ground_truth = boxed[-1][2]  # content of last \boxed{}

            all_problems.append({
                "problem": ex["problem"],
                "level": level_num,
                "ground_truth": ground_truth,
            })

    random.seed(42)
    random.shuffle(all_problems)

    # Select with 40/60 mix
    level_23 = [p for p in all_problems if p["level"] <= 3]
    level_45 = [p for p in all_problems if p["level"] >= 4]
    n_23 = int(TARGET * 0.4)
    n_45 = TARGET - n_23
    selected = level_23[:n_23] + level_45[:n_45]
    random.shuffle(selected)
    print(f"Selected {len(selected)} problems for trace generation")

    # Generate and verify traces
    client = anthropic.Anthropic()
    verified_results = []
    failed = 0
    wrong_answer = 0
    no_boxed = 0

    print(f"Generating and verifying traces with {MODEL}...")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {}
        for item in selected:
            future = executor.submit(generate_trace, client, item["problem"])
            futures[future] = item

        for i, future in enumerate(as_completed(futures)):
            item = futures[future]
            solution = future.result()

            if solution is None:
                failed += 1
            else:
                boxed = find_boxed(solution)
                if not boxed:
                    no_boxed += 1
                else:
                    predicted = boxed[-1][2]
                    if answers_match(predicted, item["ground_truth"]):
                        # Format with <think> blocks
                        start, end, _ = boxed[-1]
                        boxed_str = solution[start:end]
                        reasoning = solution[:start].rstrip()
                        formatted = f"<think>\n{reasoning}\n</think>\n\nThe answer is ${boxed_str}$."
                        verified_results.append({
                            "prompt": item["problem"],
                            "response": formatted,
                        })
                    else:
                        wrong_answer += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(selected)}] verified: {len(verified_results)}, "
                      f"wrong: {wrong_answer}, no_boxed: {no_boxed}, failed: {failed}")

    total_attempted = len(selected)
    print(f"\nResults:")
    print(f"  Verified correct: {len(verified_results)}/{total_attempted} ({len(verified_results)/total_attempted:.1%})")
    print(f"  Wrong answer: {wrong_answer}/{total_attempted} ({wrong_answer/total_attempted:.1%})")
    print(f"  No boxed: {no_boxed}/{total_attempted} ({no_boxed/total_attempted:.1%})")
    print(f"  API failed: {failed}/{total_attempted} ({failed/total_attempted:.1%})")

    # Write data.jsonl
    with open("data.jsonl", "w") as f:
        for item in verified_results:
            f.write(json.dumps(item) + "\n")

    print(f"\nWrote {len(verified_results)} verified examples to data.jsonl")

    # Show samples
    for item in verified_results[:2]:
        print(f"\nPROMPT: {item['prompt'][:100]}")
        print(f"RESPONSE: {item['response'][:500]}")
        print("---")


if __name__ == "__main__":
    main()

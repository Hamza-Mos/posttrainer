"""Generate high-quality reasoning traces using Claude API.

Takes MATH problems and generates verbose, step-by-step solutions with
self-verification, formatted for Qwen3-8B's <think> blocks.

Run: python gen_claude_traces.py
Requires: ANTHROPIC_API_KEY in ../../.env
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
TARGET = 700  # Generate 700 traces (expect ~600 after failures)

SYSTEM_PROMPT = """You are a math tutor solving problems step by step. Your solutions must:

1. Show EVERY intermediate step - never skip steps or say "clearly" or "obviously"
2. Explain WHY you're doing each step
3. Self-verify your answer: after finding it, check by substitution or estimation
4. End with the final answer in \\boxed{answer} format

Example format:
I need to find the value of x where 2x + 3 = 11.

First, I'll isolate x by subtracting 3 from both sides:
2x + 3 - 3 = 11 - 3
2x = 8

Now I'll divide both sides by 2:
x = 8/2 = 4

Let me verify: 2(4) + 3 = 8 + 3 = 11 ✓

\\boxed{4}"""


def generate_trace(client, problem: str, ground_truth: str = None) -> dict | None:
    """Generate a reasoning trace for a single problem."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Solve this problem step by step:\n\n{problem}"}],
        )
        solution = response.content[0].text

        # Verify it has \boxed{} format
        if "\\boxed" not in solution:
            return None

        # Format with <think> blocks
        # Find the last \boxed{...}
        boxed_spans = find_boxed(solution)
        if not boxed_spans:
            return None

        start, end = boxed_spans[-1]
        boxed_str = solution[start:end]
        reasoning = solution[:start].rstrip()

        formatted = f"<think>\n{reasoning}\n</think>\n\nThe answer is ${boxed_str}$."

        return {"prompt": problem, "response": formatted}
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def find_boxed(text: str) -> list[tuple[int, int]]:
    """Find all \\boxed{...} spans, handling nested braces."""
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
                    spans.append((idx, j + 1))
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return spans


def main():
    # Load the raw MATH problems (before think-formatting)
    # First regenerate from gen_data.py to get raw solutions
    print("Loading MATH problems...")

    # Load eval/RL problems to exclude
    eval_problems = set()
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                eval_problems.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Load MATH dataset
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
            all_problems.append({
                "problem": ex["problem"],
                "level": level_num,
                "solution": ex["solution"],
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
    print(f"Selected {len(selected)} problems ({n_23} level 2-3, {n_45} level 4-5)")

    # Generate traces with Claude
    client = anthropic.Anthropic()
    results = []
    failed = 0

    print(f"Generating traces with {MODEL}...")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {}
        for item in selected:
            future = executor.submit(generate_trace, client, item["problem"])
            futures[future] = item

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(selected)}] generated: {len(results)}, failed: {failed}")

    print(f"\nGenerated {len(results)} traces ({failed} failed)")

    # Write data.jsonl
    with open("data.jsonl", "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(results)} examples to data.jsonl")

    # Show samples
    for item in results[:2]:
        print(f"\nPROMPT: {item['prompt'][:100]}")
        print(f"RESPONSE: {item['response'][:500]}")
        print("---")


if __name__ == "__main__":
    main()

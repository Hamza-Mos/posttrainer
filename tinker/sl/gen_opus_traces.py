"""Generate high-quality math reasoning traces using Claude Opus.

Generates traces on MATH problems, verifies answers against ground truth,
outputs verified traces in data.jsonl format.

Usage: python gen_opus_traces.py [num_problems]
"""

import json
import os
import re
import sys
import random
from datasets import load_dataset

# Use Anthropic API
import anthropic

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MODEL = "claude-opus-4-20250514"

SYSTEM = """You are solving a math competition problem. Think through it step by step.
Show ALL intermediate steps. Verify your answer. End with \\boxed{answer}."""


def find_boxed(text):
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
                    spans.append(text[idx+7:j])
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return spans


def normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def answers_match(predicted, expected):
    pred_num = normalize_number(predicted)
    exp_num = normalize_number(expected)
    if pred_num is not None and exp_num is not None:
        return abs(pred_num - exp_num) < 1e-6
    return predicted.strip() == expected.strip()


def main():
    client = anthropic.Anthropic()

    # Load existing training prompts to exclude
    existing_prompts = set()
    if os.path.exists("data.jsonl"):
        with open("data.jsonl") as f:
            for line in f:
                item = json.loads(line.strip())
                existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    # Exclude RL eval prompts
    for path in ["../../tinker/rl/eval_prompts.jsonl", "../../tinker/rl/prompts.jsonl"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    existing_prompts.add(re.sub(r'\s+', ' ', item["prompt"].strip()))

    print(f"Excluding {len(existing_prompts)} existing prompts", flush=True)

    # Load MATH problems
    subjects = ['algebra', 'counting_and_probability', 'intermediate_algebra',
                'number_theory', 'prealgebra', 'precalculus']
    problems = []
    for subj in subjects:
        ds = load_dataset('EleutherAI/hendrycks_math', subj, split='train')
        for ex in ds:
            try:
                level_num = int(ex['level'].replace('Level ', ''))
            except ValueError:
                continue
            if level_num < 3:  # Focus on harder problems
                continue
            pn = re.sub(r'\s+', ' ', ex['problem'].strip())
            if pn in existing_prompts:
                continue
            boxed = find_boxed(ex['solution'])
            if not boxed:
                continue
            problems.append({
                "problem": ex["problem"],
                "level": level_num,
                "ground_truth": boxed[-1],
            })

    random.seed(456)
    random.shuffle(problems)
    selected = problems[:int(TARGET * 1.5)]
    print(f"Selected {len(selected)} problems for Claude Opus traces", flush=True)

    traces = []
    wrong = 0
    errors = 0

    for i, item in enumerate(selected):
        if len(traces) >= TARGET:
            break
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                messages=[{"role": "user", "content": item["problem"]}],
            )
            text = response.content[0].text
            boxed = find_boxed(text)
            if boxed and answers_match(boxed[-1], item["ground_truth"]):
                # Format as Qwen3-8B think-block style
                formatted = f"<think>\n{text}\n</think>\n\n\\boxed{{{boxed[-1]}}}"
                traces.append({
                    "prompt": item["problem"],
                    "response": formatted,
                })
            else:
                wrong += 1
        except Exception as e:
            errors += 1
            print(f"  Error: {e}", file=sys.stderr, flush=True)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(selected)}] verified: {len(traces)}, wrong: {wrong}, errors: {errors}", flush=True)

    print(f"\nGenerated {len(traces)} verified Opus traces")
    print(f"Wrong: {wrong}, Errors: {errors}")

    # Save to separate file (don't modify data.jsonl directly)
    outfile = "opus_traces.jsonl"
    with open(outfile, "w") as f:
        for item in traces:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {outfile}")

    # Show samples
    for item in traces[:2]:
        print(f"\nPROMPT: {item['prompt'][:100]}")
        print(f"RESPONSE: {item['response'][:400]}")
        print("---")


if __name__ == "__main__":
    main()

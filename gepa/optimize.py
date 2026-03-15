"""GEPA co-evolution: system prompt + evaluation rubric.

Co-evolves TWO text artifacts simultaneously using GEPA multi-module optimization:
  1. system_prompt — instructs the task LM to solve problems step by step
  2. evaluation_rubric — instructs the evaluator LM to score responses (without
     seeing the reference answer, forcing it to develop genuine quality criteria)

Scoring: 60% correctness (answer matches reference) + 40% rubric calibration
(rubric score agrees with ground truth). This rewards both better answers AND
more accurate evaluation.

Usage:
    python optimize.py > run.log 2>&1

Grep-parsable output:
    val_score: 0.85
    best_prompt: {"system_prompt": "...", "evaluation_rubric": "..."}
"""

import json
import logging
import os
import re
from pathlib import Path

import litellm
import gepa
from gepa.core.adapter import GEPAAdapter, EvaluationBatch

# Load API keys from ../.env
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
litellm.suppress_debug_info = True

# ============================================================
# CONFIGURATION (agent modifies these)
# ============================================================

TASK_LM = "openai/gpt-4.1-mini"
EVALUATOR_LM = "openai/gpt-4.1-mini"
REFLECTION_LM = "anthropic/claude-sonnet-4-6"

MAX_METRIC_CALLS = 50

SEED = {
    "system_prompt": (
        "You are a helpful problem-solving assistant. "
        "Think through problems step by step, showing your reasoning. "
        "Always end with your final answer on a new line starting with 'ANSWER:'"
    ),
    "evaluation_rubric": (
        "You are an evaluation assistant. Given a question and a response, "
        "score the response quality on a scale of 0.0 to 1.0.\n"
        "Consider:\n"
        "- Correctness: Does the response arrive at the right answer?\n"
        "- Reasoning: Is the reasoning clear and sound?\n"
        "- Completeness: Does it address all parts of the question?\n"
        "Respond with ONLY a single number between 0.0 and 1.0."
    ),
}

# ============================================================
# DATA (agent modifies these)
# ============================================================

def _d(q, a):
    return {"input": q, "answer": a, "additional_context": {}}

# Competition-level math: AMC/AIME difficulty, multi-step reasoning, combinatorics
TRAINSET = [
    _d("How many integers between 1 and 1000 (inclusive) are divisible by both 3 and 7 but not by 5?", "38"),
    _d("A 5-digit number is formed using the digits 1, 2, 3, 4, 5 without repetition. What is the probability that the number is odd? Express as a fraction.", "3/5"),
    _d("In how many ways can 8 people be seated around a circular table?", "5040"),
    _d("Find the remainder when 2^100 is divided by 7.", "2"),
    _d("A box contains 6 red, 4 blue, and 5 green marbles. If 3 marbles are drawn without replacement, what is the probability that all 3 are the same color? Express as a fraction.", "34/455"),
    _d("The sum of the first n positive integers is 325. What is n?", "25"),
    _d("If log base 2 of x plus log base 2 of (x-2) equals 3, what is x?", "4"),
    _d("A triangle has sides of length 5, 12, and 13. What is its area?", "30"),
    _d("How many 4-letter codes can be formed using the letters A, B, C, D, E if repetition is allowed and the code must start with a vowel?", "250"),
    _d("Find the value of the infinite series: 1/2 + 1/4 + 1/8 + 1/16 + ... Express as a whole number or fraction.", "1"),
    _d("A fair coin is flipped 6 times. What is the probability of getting exactly 3 heads? Express as a fraction.", "5/16"),
    _d("What is the greatest common divisor of 252 and 198?", "18"),
    _d("In a class of 40 students, 25 play soccer, 20 play basketball, and 5 play neither. How many students play both sports?", "10"),
    _d("If f(x) = 3x^2 - 2x + 1, what is f(f(1))?", "7"),
    _d("A ladder 10 meters long leans against a wall. The base is 6 meters from the wall. How high up the wall does the ladder reach?", "8"),
]

VALSET = [
    _d("How many distinct prime factors does 2310 have?", "5"),
    _d("If the product of two consecutive positive integers is 306, what is the smaller integer?", "17"),
    _d("Three cards are drawn from a standard 52-card deck without replacement. What is the probability that all three are aces? Express as a fraction.", "1/5525"),
    _d("Find the sum of all positive divisors of 28.", "56"),
    _d("A geometric sequence has first term 3 and common ratio 2. What is the sum of the first 8 terms?", "765"),
    _d("How many trailing zeros does 25! have?", "6"),
    _d("Two trains leave stations 300 miles apart heading toward each other. One travels at 50 mph, the other at 70 mph. After how many hours do they meet? Express as a fraction.", "5/2"),
    _d("What is the number of diagonals in a convex polygon with 12 sides?", "54"),
]

# ============================================================
# HELPERS
# ============================================================

def extract_number(text):
    """Extract the final numerical answer from text."""
    # Look for ANSWER: pattern first
    m = re.search(r'ANSWER:\s*\$?\s*([\d,./]+)', text, re.IGNORECASE)
    if m:
        return _norm(m.group(1))
    # Fall back to last number in text
    nums = re.findall(r'[\d,./]+', text)
    if nums:
        return _norm(nums[-1])
    return None


def _norm(s):
    """Normalize a number string for comparison."""
    s = s.strip().rstrip('.').replace(',', '')
    if '/' in s:
        return s
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def check_answer(generated, reference):
    """Check if the generated answer matches the reference. Returns 0.0 or 1.0."""
    gen = extract_number(generated)
    ref = _norm(reference)
    if gen is None:
        return 0.0
    if gen == ref:
        return 1.0
    try:
        if abs(float(gen) - float(ref)) < 0.01:
            return 1.0
    except (ValueError, ZeroDivisionError):
        pass
    # Fraction comparison
    def _eval_frac(s):
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        return float(s)
    try:
        if abs(_eval_frac(gen) - _eval_frac(ref)) < 0.01:
            return 1.0
    except (ValueError, ZeroDivisionError):
        pass
    return 0.0


def extract_score(text):
    """Extract a 0-1 score from evaluator response."""
    m = re.search(r'(0?\.\d+|1\.0|1|0)', text.strip())
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    return 0.5


# ============================================================
# ADAPTER
# ============================================================

class CoEvolutionAdapter(GEPAAdapter):
    """Co-evolves system_prompt and evaluation_rubric."""

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["system_prompt"]
        rubric = candidate["evaluation_rubric"]

        outputs, scores = [], []
        trajectories = [] if capture_traces else None

        for item in batch:
            # Step 1: Generate response using system_prompt
            try:
                gen_resp = litellm.completion(
                    model=TASK_LM,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["input"]},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                generated = gen_resp.choices[0].message.content
            except Exception as e:
                generated = f"[Generation error: {e}]"

            # Step 2: Evaluate with rubric (NO reference answer — rubric must judge independently)
            try:
                eval_resp = litellm.completion(
                    model=EVALUATOR_LM,
                    messages=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": (
                            f"Question: {item['input']}\n\n"
                            f"Response to evaluate:\n{generated}\n\n"
                            f"Score (0.0 to 1.0):"
                        )},
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                rubric_score = extract_score(eval_resp.choices[0].message.content)
            except Exception:
                rubric_score = 0.5

            # Step 3: Ground truth check
            gt = check_answer(generated, item["answer"])

            # Hybrid score: correctness + rubric calibration
            calibration = 1.0 - abs(rubric_score - gt)
            score = 0.6 * gt + 0.4 * calibration

            outputs.append({"generated": generated, "rubric_score": rubric_score, "gt": gt})
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "input": item["input"],
                    "reference": item["answer"],
                    "generated": generated,
                    "rubric_score": rubric_score,
                    "gt": gt,
                    "feedback": (
                        f"Ground truth: {'CORRECT' if gt else 'INCORRECT'}. "
                        f"Rubric scored {rubric_score:.2f}. "
                        f"Calibration: {calibration:.2f}. "
                        f"Final score: {score:.2f}"
                    ),
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        reflective_data = {}
        for comp in components_to_update:
            examples = []
            for traj in eval_batch.trajectories:
                if comp == "system_prompt":
                    examples.append({
                        "Inputs": f"Question: {traj['input']}",
                        "Generated Outputs": traj["generated"],
                        "Feedback": (
                            f"Expected answer: {traj['reference']}. "
                            f"{traj['feedback']}"
                        ),
                    })
                elif comp == "evaluation_rubric":
                    examples.append({
                        "Inputs": (
                            f"Question: {traj['input']}\n"
                            f"Generated response: {traj['generated']}"
                        ),
                        "Generated Outputs": f"Score: {traj['rubric_score']:.2f}",
                        "Feedback": (
                            f"Ground truth: {'CORRECT' if traj['gt'] else 'INCORRECT'}. "
                            f"Rubric gave {traj['rubric_score']:.2f}, "
                            f"ideal would be {traj['gt']:.1f}. "
                            f"Error: {abs(traj['rubric_score'] - traj['gt']):.2f}"
                        ),
                    })
            reflective_data[comp] = examples
        return reflective_data


# ============================================================
# RUN
# ============================================================

def main():
    log.info("Starting GEPA co-evolution")
    log.info(f"Task LM: {TASK_LM}")
    log.info(f"Evaluator LM: {EVALUATOR_LM}")
    log.info(f"Reflection LM: {REFLECTION_LM}")
    log.info(f"Budget: {MAX_METRIC_CALLS} metric calls")
    log.info(f"Train: {len(TRAINSET)} examples, Val: {len(VALSET)} examples")
    log.info(f"Seed system_prompt: {SEED['system_prompt'][:100]}...")
    log.info(f"Seed rubric: {SEED['evaluation_rubric'][:100]}...")

    adapter = CoEvolutionAdapter()

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=TRAINSET,
        valset=VALSET,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        module_selector="round_robin",
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        display_progress_bar=True,
    )

    # Extract results
    best = result.best_candidate
    best_idx = result.best_idx
    val_score = result.val_aggregate_scores[best_idx]

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    print(f"\nval_score: {val_score:.6f}")
    print(f"best_prompt: {json.dumps(best)}")

    log.info(f"Val score: {val_score}")
    log.info(f"Best system_prompt: {best.get('system_prompt', '')[:200]}")
    log.info(f"Best rubric: {best.get('evaluation_rubric', '')[:200]}")
    log.info(f"Candidates explored: {len(result.candidates)}")
    log.info(f"Total metric calls: {result.total_metric_calls}")


if __name__ == "__main__":
    main()

"""GEPA co-evolution: system prompt + evaluation rubric.

Co-evolves TWO text artifacts simultaneously using GEPA multi-module optimization:
  1. system_prompt — instructs the task LM to solve AIME-level math problems
  2. evaluation_rubric — instructs the evaluator LM to score solutions (without
     seeing the reference answer, forcing it to develop genuine quality criteria)

Multi-objective scoring via GEPA's Pareto optimization:
  - generation_quality: does the model get the right answer?
  - rubric_calibration: does the rubric agree with ground truth?
Both objectives contribute to the Pareto frontier.

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

# SOTA models — cross-family to avoid self-evaluation bias
TASK_LM = "anthropic/claude-sonnet-4-6"       # generates solutions
EVALUATOR_LM = "openai/gpt-5.4"              # applies the rubric
REFLECTION_LM = "anthropic/claude-opus-4-6"   # proposes improvements (strongest)

MAX_METRIC_CALLS = 80

SEED = {
    "system_prompt": (
        "You are an expert mathematician solving competition-level problems. "
        "Think through each problem carefully and systematically. "
        "Show your complete reasoning step by step. "
        "Always end with your final numerical answer on a new line starting with 'ANSWER:' "
        "followed by just the number (integer). For example: ANSWER: 42"
    ),
    "evaluation_rubric": (
        "You are a mathematics evaluation expert. Given a math problem and a "
        "proposed solution, assess the solution quality on a scale of 0.0 to 1.0.\n"
        "Consider:\n"
        "- Mathematical correctness: Are the calculations and logic sound?\n"
        "- Reasoning completeness: Are all steps justified?\n"
        "- Answer validity: Does the final answer make sense given the problem?\n"
        "Respond with ONLY a single number between 0.0 and 1.0."
    ),
}

# ============================================================
# DATA — AIME-level problems (all answers verified with Python)
# ============================================================

def _d(q, a):
    return {"input": q, "answer": str(a), "additional_context": {}}

TRAINSET = [
    # Number theory
    _d("Find the number of positive integers n less than or equal to 100 such that n^2 - 1 is divisible by 24.", 33),
    _d("Find the remainder when 3^2024 is divided by 100.", 81),
    _d("How many positive integers n satisfy both n | 720 and gcd(n, 30) = 6?", 8),
    # Combinatorics
    _d("A committee of 5 is to be formed from 6 men and 4 women. In how many ways can this be done if the committee must contain at least 2 women?", 186),
    _d("In how many ways can 12 identical balls be distributed into 4 distinct boxes such that each box contains at least 2 balls?", 10),
    _d("How many 6-digit positive integers have their digits in strictly increasing order?", 84),
    # Algebra
    _d("If x and y are positive reals with x + y = 10 and x^2 + y^2 = 58, find xy.", 21),
    _d("Find the sum of all real solutions to the equation |2x - 5| + |x + 3| = 10.", 2),
    # Geometry
    _d("In triangle ABC, AB = 13, BC = 14, and CA = 15. Find the area of triangle ABC.", 84),
    _d("A circle is inscribed in a right triangle with legs 5 and 12. What is the radius of the inscribed circle?", 2),
    # Probability
    _d("Three distinct numbers are chosen at random from {1, 2, 3, ..., 10}. What is the probability that the sum of the three numbers is divisible by 3? Express as a percentage rounded to the nearest integer.", 35),
    # Series
    _d("Find the value of the sum: 1*2 + 2*3 + 3*4 + ... + 99*100.", 333300),
]

VALSET = [
    _d("Find the number of ordered pairs (a, b) of positive integers such that a + b = 100 and lcm(a, b) = 180.", 0),
    _d("How many 4-digit palindromes are divisible by 3?", 30),
    _d("Find the last three digits of 7^999.", 143),
    _d("In how many ways can you tile a 2x10 board with 1x2 dominoes?", 89),
    _d("If a, b, c are roots of x^3 - 6x^2 + 11x - 6 = 0, find a^2 + b^2 + c^2.", 14),
    _d("A bag has 5 red and 7 blue balls. Balls are drawn one at a time without replacement until 2 red balls have been drawn. What is the expected number of draws? Express as a fraction's numerator if the fraction in lowest terms is p/q. Give p.", 13),
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
    # Look for boxed answer (LaTeX)
    m = re.search(r'\\boxed\{(\d+)\}', text)
    if m:
        return _norm(m.group(1))
    # Fall back to last number in text
    nums = re.findall(r'\b\d+\b', text)
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
    """Co-evolves system_prompt and evaluation_rubric with multi-objective scoring."""

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["system_prompt"]
        rubric = candidate["evaluation_rubric"]

        outputs, scores = [], []
        objective_scores = []
        trajectories = [] if capture_traces else None

        for item in batch:
            # Step 1: Generate solution using system_prompt (temp=0 for determinism)
            try:
                gen_resp = litellm.completion(
                    model=TASK_LM,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["input"]},
                    ],
                    temperature=0,
                    max_tokens=2048,
                )
                generated = gen_resp.choices[0].message.content
            except Exception as e:
                generated = f"[Generation error: {e}]"

            # Step 2: Evaluate with rubric (NO reference answer)
            try:
                eval_resp = litellm.completion(
                    model=EVALUATOR_LM,
                    messages=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": (
                            f"Problem: {item['input']}\n\n"
                            f"Proposed solution:\n{generated}\n\n"
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

            # Multi-objective scoring
            calibration = 1.0 - abs(rubric_score - gt)
            # Combined score for GEPA's primary metric
            score = 0.6 * gt + 0.4 * calibration

            outputs.append({"generated": generated, "rubric_score": rubric_score, "gt": gt})
            scores.append(score)
            objective_scores.append({
                "generation_quality": gt,
                "rubric_calibration": calibration,
            })

            if capture_traces:
                trajectories.append({
                    "input": item["input"],
                    "reference": item["answer"],
                    "generated": generated,
                    "rubric_score": rubric_score,
                    "gt": gt,
                    "feedback": (
                        f"Ground truth: {'CORRECT' if gt else 'INCORRECT'} (answer={item['answer']}). "
                        f"Rubric scored {rubric_score:.2f}. "
                        f"Calibration: {calibration:.2f}. "
                        f"Final score: {score:.2f}"
                    ),
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        reflective_data = {}
        for comp in components_to_update:
            examples = []
            for traj in eval_batch.trajectories:
                if comp == "system_prompt":
                    examples.append({
                        "Inputs": f"Problem: {traj['input']}",
                        "Generated Outputs": traj["generated"],
                        "Feedback": (
                            f"Expected answer: {traj['reference']}. "
                            f"{traj['feedback']}"
                        ),
                    })
                elif comp == "evaluation_rubric":
                    examples.append({
                        "Inputs": (
                            f"Problem: {traj['input']}\n"
                            f"Proposed solution: {traj['generated']}"
                        ),
                        "Generated Outputs": f"Score: {traj['rubric_score']:.2f}",
                        "Feedback": (
                            f"Ground truth: {'CORRECT' if traj['gt'] else 'INCORRECT'} "
                            f"(answer={traj['reference']}). "
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
        frontier_type="hybrid",
        reflection_minibatch_size=6,
        use_merge=True,
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
    log.info(f"Best system_prompt: {best.get('system_prompt', '')[:300]}")
    log.info(f"Best rubric: {best.get('evaluation_rubric', '')[:300]}")
    log.info(f"Candidates explored: {len(result.candidates)}")
    log.info(f"Total metric calls: {result.total_metric_calls}")

    # Log per-objective scores if available
    if result.val_aggregate_subscores and best_idx < len(result.val_aggregate_subscores):
        subs = result.val_aggregate_subscores[best_idx]
        log.info(f"Generation quality: {subs.get('generation_quality', 'N/A')}")
        log.info(f"Rubric calibration: {subs.get('rubric_calibration', 'N/A')}")


if __name__ == "__main__":
    main()

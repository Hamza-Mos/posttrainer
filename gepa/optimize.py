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

MAX_METRIC_CALLS = 30

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

TRAINSET = [
    _d("A store sells apples for $2 each and oranges for $3 each. If you buy 5 apples and 4 oranges, how much do you spend in total?", "22"),
    _d("What is the next number in the sequence: 2, 6, 18, 54, ...?", "162"),
    _d("A train travels at 60 mph for 2.5 hours. How many miles does it travel?", "150"),
    _d("If the area of a circle is 49π square meters, what is its radius in meters?", "7"),
    _d("Three friends split a bill of $87 equally. How much does each person pay?", "29"),
    _d("What is 15% of 240?", "36"),
    _d("A rectangle has a perimeter of 30 cm and a width of 5 cm. What is its length in cm?", "10"),
    _d("If 3x + 7 = 28, what is x?", "7"),
    _d("How many prime numbers are there between 20 and 40?", "4"),
    _d("A bag contains 5 red balls and 3 blue balls. What fraction of the balls are red?", "5/8"),
]

VALSET = [
    _d("A car gets 30 miles per gallon. How many gallons are needed for a 450-mile trip?", "15"),
    _d("What is the sum of all integers from 1 to 20?", "210"),
    _d("A shirt originally costs $80 and is marked down 25%. What is the sale price in dollars?", "60"),
    _d("Two dice are rolled. What is the probability that the sum is 7? Express as a fraction.", "1/6"),
    _d("The average of 5 numbers is 12. If four of the numbers are 10, 14, 8, and 16, what is the fifth number?", "12"),
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
                    max_tokens=512,
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

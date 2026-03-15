"""GEPA prompt optimization script.

Optimizes a system prompt (or any text artifact) using GEPA's evolutionary
Pareto search with LLM reflection. The agent modifies this file to change:
  - TASK_LM: the model being optimized
  - REFLECTION_LM: the model doing reflection (should be stronger)
  - SEED: the initial prompt to optimize
  - METRIC: the evaluation function
  - TRAINSET / VALSET: the evaluation data
  - MAX_METRIC_CALLS: budget (number of evaluations)

Usage:
    export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY
    python optimize.py > run.log 2>&1

Grep-parsable output:
    val_score: 0.85
    best_prompt: <the optimized prompt>
"""

import json
import logging
import gepa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (agent modifies these)
# ============================================================

# Models
TASK_LM = "openai/gpt-4.1-mini"        # model being prompted
REFLECTION_LM = "openai/gpt-4.1-mini"  # model proposing improvements

# Budget
MAX_METRIC_CALLS = 30  # total evaluations (start small, increase once working)

# Seed prompt to optimize
SEED = {
    "system_prompt": (
        "You are a helpful math tutor. "
        "Solve the problem step by step, showing your work clearly. "
        "Put your final answer after ###"
    )
}

# ============================================================
# DATA (agent modifies these)
# ============================================================

def _d(input, answer):
    """Create a data instance with required fields."""
    return {"input": input, "answer": answer, "additional_context": {}}

TRAINSET = [
    _d("What is 15 * 13?", "### 195"),
    _d("If a rectangle has length 7 and width 4, what is its area?", "### 28"),
    _d("What is 144 / 12?", "### 12"),
    _d("What is 2^8?", "### 256"),
    _d("What is the sum of the first 10 positive integers?", "### 55"),
    _d("If x + 5 = 12, what is x?", "### 7"),
    _d("What is 37 + 68?", "### 105"),
    _d("What is 1000 - 387?", "### 613"),
    _d("What is 25% of 200?", "### 50"),
    _d("What is the square root of 169?", "### 13"),
]

VALSET = [
    _d("What is 17 * 19?", "### 323"),
    _d("What is 3^5?", "### 243"),
    _d("If 2x - 3 = 11, what is x?", "### 7"),
    _d("What is 15% of 80?", "### 12"),
    _d("What is the sum of 99 and 87?", "### 186"),
]

# ============================================================
# RUN
# ============================================================

def main():
    log.info("Starting GEPA optimization")
    log.info(f"Task LM: {TASK_LM}")
    log.info(f"Reflection LM: {REFLECTION_LM}")
    log.info(f"Budget: {MAX_METRIC_CALLS} metric calls")
    log.info(f"Train: {len(TRAINSET)} examples, Val: {len(VALSET)} examples")
    log.info(f"Seed prompt: {SEED['system_prompt'][:100]}...")

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=TRAINSET,
        valset=VALSET,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )

    # Extract results
    best = result.best_candidate
    best_idx = result.best_idx
    val_score = result.val_aggregate_scores[best_idx]

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    # Grep-parsable output
    print(f"\nval_score: {val_score:.6f}")
    print(f"best_prompt: {json.dumps(best)}")

    # Human-readable
    log.info(f"Val score: {val_score}")
    log.info(f"Best prompt: {best.get('system_prompt', str(best))}")
    log.info(f"Candidates explored: {len(result.candidates)}")
    log.info(f"Total metric calls: {result.total_metric_calls}")
    log.info("Optimization completed.")


if __name__ == "__main__":
    main()

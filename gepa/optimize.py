"""GEPA co-evolution: system prompt + evaluation rubric for technical writing.

Co-evolves TWO text artifacts simultaneously:
  1. system_prompt — guides how sonnet writes technical analyses
  2. evaluation_rubric — evolves criteria for assessing writing quality

Ground truth: a fixed expert rubric (applied by gpt-5.4) provides the
reference quality signal. The evolved rubric tries to match the expert.
Score = 50% output quality (expert) + 50% rubric calibration (agreement).

Usage:
    python optimize.py > run.log 2>&1
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
# CONFIGURATION
# ============================================================

TASK_LM = "openai/gpt-5.4"                    # generates analyses
EVALUATOR_LM = "openai/gpt-5.4"              # applies both rubrics
REFLECTION_LM = "anthropic/claude-opus-4-6"   # proposes improvements

MAX_METRIC_CALLS = 500

# Fixed expert rubric — multi-criteria for fine-grained ground truth (never evolved)
EXPERT_RUBRIC = (
    "You are an expert technical writing evaluator. Rate the response on 5 criteria.\n"
    "For EACH criterion, give an integer score from 1 to 10.\n\n"
    "1. ACCURACY (1-10): Are technical claims factually correct? No hallucinations?\n"
    "2. DEPTH (1-10): Does it explain mechanisms/tradeoffs beyond surface level?\n"
    "3. STRUCTURE (1-10): Well-organized with clear progression of ideas?\n"
    "4. ACTIONABILITY (1-10): Concrete insights someone could act on?\n"
    "5. CONCISENESS (1-10): Appropriately detailed without unnecessary verbosity?\n\n"
    "Respond with EXACTLY 5 numbers separated by commas. Example: 8,7,9,6,8"
)

# Load seed from file if available, otherwise use default
_seed_path = Path(__file__).parent / "seed.json"
if _seed_path.exists():
    SEED = json.loads(_seed_path.read_text())
    log.info(f"Loaded seed from {_seed_path} ({len(SEED.get('system_prompt',''))} + {len(SEED.get('evaluation_rubric',''))} chars)")
else:
    SEED = {
        "system_prompt": (
            "You are a technical writing assistant. When asked to explain or analyze "
            "a technical topic, provide a clear, well-structured response. "
            "Be accurate, insightful, and concise."
        ),
        "evaluation_rubric": (
            "You are an evaluation assistant. Rate the response on 5 criteria.\n"
            "For EACH criterion, give an integer score from 1 to 10.\n\n"
            "1. CORRECTNESS (1-10): Are the technical details accurate?\n"
            "2. CLARITY (1-10): Is the explanation easy to follow?\n"
            "3. DEPTH (1-10): Does it go beyond surface-level?\n"
            "4. USEFULNESS (1-10): Does it provide actionable insights?\n"
            "5. EFFICIENCY (1-10): Is it appropriately concise?\n\n"
            "Respond with EXACTLY 5 numbers separated by commas. Example: 8,7,9,6,8"
        ),
    }

# ============================================================
# DATA — diverse technical analysis prompts
# ============================================================

def _d(q):
    return {"input": q, "answer": "", "additional_context": {}}

TRAINSET = [
    _d("Explain how database connection pooling works, when to use it, and common pitfalls."),
    _d("What are the tradeoffs between microservices and monolithic architectures? When should you choose each?"),
    _d("Explain how TLS 1.3 handshake differs from TLS 1.2 and why the changes were made."),
    _d("What is the CAP theorem and how do real-world distributed databases handle it?"),
    _d("Explain the difference between optimistic and pessimistic concurrency control. Give concrete examples."),
    _d("How does a B-tree index work in a database? Why is it preferred over a hash index for range queries?"),
    _d("What are the key differences between REST and GraphQL APIs? When would you choose one over the other?"),
    _d("Explain how garbage collection works in the JVM, including the different collector types and when to use each."),
    _d("What is eventual consistency? Give a real-world example where it causes problems and how to mitigate them."),
    _d("Explain the concept of backpressure in distributed systems and common strategies for handling it."),
    _d("How does consistent hashing work and why is it important for distributed caching?"),
    _d("What are the tradeoffs of using event sourcing vs traditional CRUD for application state management?"),
]

VALSET = [
    _d("Explain how the Linux kernel's OOM killer works and how to tune it for production servers."),
    _d("What are the security implications of JWT tokens vs session-based authentication?"),
    _d("Explain the differences between TCP congestion control algorithms (Reno, CUBIC, BBR)."),
    _d("How does write-ahead logging (WAL) work in databases and why is it critical for durability?"),
    _d("What are the tradeoffs between row-oriented and column-oriented databases?"),
    _d("Explain how container networking works in Kubernetes, including the CNI plugin architecture."),
]

# ============================================================
# HELPERS
# ============================================================

def parse_multi_score(text):
    """Parse 5 comma-separated integers (1-10) into a 0-1 average."""
    nums = re.findall(r'\b(\d+)\b', text.strip())
    if len(nums) >= 5:
        scores = [min(max(int(n), 1), 10) for n in nums[:5]]
        return sum(scores) / 50.0  # normalize to 0-1
    # Fallback: try single score
    m = re.search(r'(0?\.\d+|1\.0|1|0)', text.strip())
    if m:
        return min(max(float(m.group(1)), 0.0), 1.0)
    return 0.5


def evaluate_with_rubric(rubric_text, question, response):
    """Score a response using a multi-criteria rubric via the evaluator LM."""
    try:
        resp = litellm.completion(
            model=EVALUATOR_LM,
            messages=[
                {"role": "system", "content": rubric_text},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Response to evaluate:\n{response}\n\n"
                    f"Scores (5 integers, comma-separated):"
                )},
            ],
            # gpt-5.4 only supports temperature=1 (default)
            max_tokens=30,
        )
        return parse_multi_score(resp.choices[0].message.content)
    except Exception as e:
        log.warning(f"Evaluation failed: {e}")
        return 0.5


# ============================================================
# ADAPTER
# ============================================================

class CoEvolutionAdapter(GEPAAdapter):
    """Co-evolves system_prompt and evaluation_rubric for technical writing."""

    def evaluate(self, batch, candidate, capture_traces=False):
        system_prompt = candidate["system_prompt"]
        rubric = candidate["evaluation_rubric"]

        outputs, scores = [], []
        objective_scores = []
        trajectories = [] if capture_traces else None

        for item in batch:
            # Step 1: Generate response using evolved system_prompt
            try:
                gen_resp = litellm.completion(
                    model=TASK_LM,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["input"]},
                    ],
                    # gpt-5.4 only supports temperature=1
                    max_tokens=1200,
                )
                generated = gen_resp.choices[0].message.content
            except Exception as e:
                generated = f"[Generation error: {e}]"

            # Step 2: Score with evolved rubric
            rubric_score = evaluate_with_rubric(rubric, item["input"], generated)

            # Step 3: Score with fixed expert rubric (ground truth)
            expert_score = evaluate_with_rubric(EXPERT_RUBRIC, item["input"], generated)

            # Scoring: output quality + rubric calibration
            calibration = 1.0 - abs(rubric_score - expert_score)
            score = 0.5 * expert_score + 0.5 * calibration

            outputs.append({
                "generated": generated[:500],
                "rubric_score": rubric_score,
                "expert_score": expert_score,
            })
            scores.append(score)
            objective_scores.append({
                "output_quality": expert_score,
                "rubric_calibration": calibration,
            })

            if capture_traces:
                trajectories.append({
                    "input": item["input"],
                    "generated": generated,
                    "rubric_score": rubric_score,
                    "expert_score": expert_score,
                    "feedback": (
                        f"Expert score: {expert_score:.2f}. "
                        f"Rubric score: {rubric_score:.2f}. "
                        f"Calibration: {calibration:.2f}. "
                        f"Final: {score:.2f}"
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
                        "Inputs": f"Topic: {traj['input']}",
                        "Generated Outputs": traj["generated"][:1000],
                        "Feedback": (
                            f"Expert quality score: {traj['expert_score']:.3f}/1.0 "
                            f"(average of accuracy, depth, structure, actionability, conciseness "
                            f"each rated 1-10). {traj['feedback']}"
                        ),
                    })
                elif comp == "evaluation_rubric":
                    examples.append({
                        "Inputs": (
                            f"Topic: {traj['input']}\n"
                            f"Response (truncated): {traj['generated'][:500]}"
                        ),
                        "Generated Outputs": f"Rubric avg: {traj['rubric_score']:.3f}",
                        "Feedback": (
                            f"Expert avg: {traj['expert_score']:.3f}. "
                            f"Your rubric avg: {traj['rubric_score']:.3f}. "
                            f"Error: {abs(traj['rubric_score'] - traj['expert_score']):.3f}. "
                            f"Improve criteria to better capture quality dimensions."
                        ),
                    })
            reflective_data[comp] = examples
        return reflective_data


# ============================================================
# RUN
# ============================================================

def main():
    log.info("Starting GEPA co-evolution (technical writing)")
    log.info(f"Task LM: {TASK_LM}")
    log.info(f"Evaluator LM: {EVALUATOR_LM}")
    log.info(f"Reflection LM: {REFLECTION_LM}")
    log.info(f"Budget: {MAX_METRIC_CALLS} metric calls")
    log.info(f"Train: {len(TRAINSET)} examples, Val: {len(VALSET)} examples")

    adapter = CoEvolutionAdapter()

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=TRAINSET,
        valset=VALSET,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        module_selector="all",
        candidate_selection_strategy="epsilon_greedy",
        frontier_type="hybrid",
        reflection_minibatch_size=4,
        use_merge=True,
        skip_perfect_score=False,
        cache_evaluation=True,
        display_progress_bar=True,
    )

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

    if result.val_aggregate_subscores and best_idx < len(result.val_aggregate_subscores):
        subs = result.val_aggregate_subscores[best_idx]
        log.info(f"Output quality: {subs.get('output_quality', 'N/A')}")
        log.info(f"Rubric calibration: {subs.get('rubric_calibration', 'N/A')}")


if __name__ == "__main__":
    main()

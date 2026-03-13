# gepa

Autonomous prompt optimization using [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto). GEPA evolves text artifacts (prompts, rubrics, agent architectures) using LLM reflection and Pareto-efficient search. It outperforms GRPO by 6% on average while using 35x fewer rollouts.

## 1. Task Description

**What to optimize**: Evaluation rubric that scores code review quality (0-1). The rubric itself is the text artifact being evolved — not a prompt for a model, but criteria for judging whether a code review is thorough, actionable, and catches real bugs vs. rubber-stamp/vague.

**Task LM**: `openai/gpt-5-mini` (applies the rubric to score reviews)

**Evaluation data**: 100 GitHub PR reviews labeled binary (50 good, 50 bad). Good = catches bugs, provides actionable suggestions, references specific lines. Bad = "LGTM", vague comments, incorrect suggestions. Train: 70, Val: 30. Metric: discrimination accuracy (% correctly classified as good/bad using the evolved rubric).

## 2. Setup

1. **Create a worktree** (never work on main):
   ```bash
   REPO=$(git rev-parse --show-toplevel)
   git worktree add "$REPO/../<experiment-name>" -b experiment/<experiment-name>
   cd "$REPO/../<experiment-name>/gepa"
   ```
   Each worktree is an isolated copy — multiple experiments can run in parallel in separate terminals.
2. **Read the files**:
   - `../rules.md` — hard constraints
   - `optimize.py` — the script you modify (models, data, seed prompt, budget)
3. **Set API key**: `export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`)
4. **Install GEPA**: `pip install gepa`

## 3. The Loop

```
LOOP FOREVER:
  1. Reconstruct state: read results.tsv + ../lab context + ../lab failures
  2. Decide what to try + form hypothesis — WHY will this improve val_score?
     Priority: evaluation data > seed prompt > model choice > GEPA config
       ../lab hypothesis "what" -m "why"
  3. Modify optimize.py → git commit → ../lab experiment <H_ID>
  4. Run: python optimize.py > run.log 2>&1
  5. Extract: grep '^val_score:' run.log
  6. Read the best_prompt in run.log (verify it makes sense, not degenerate)
  7. Log to results.tsv + ../lab result <E_ID> -v keep|discard|crash \
       --metrics '{"val_score": X}' --mechanism-confirmed (or --mechanism-refuted) \
       --theory-revision "what I learned"
  8. If val_score improved → KEEP. If not → git reset --hard HEAD~1.
  NEVER STOP
```

**Research discipline:** Before every experiment, state WHY (`--mechanism`). After every result, confirm or refute.

**NEVER STOP**: Do NOT ask "should I continue?". The human expects you to work *indefinitely* until manually stopped. If you run out of ideas, think harder — try different seed prompts, better evaluation data, different GEPA parameters (merge, frontier type, batch size).

**Logging results:** Append to `results.tsv` (tab-separated). Columns: `commit	val_score	status	description`. Example:
```
a1b2c3d	0.850000	keep	better seed prompt for math
b2c3d4e	0.720000	discard	switched to gpt-4.1-nano (too weak)
```

## 4. What You Can Modify

**optimize.py** — everything is fair game:
- `SEED` — the starting prompt/artifact to optimize
- `TRAINSET` / `VALSET` — evaluation data (more diverse = better GEPA performance)
- `TASK_LM` — which model is being prompted
- `REFLECTION_LM` — which model proposes improvements (stronger = better)
- `MAX_METRIC_CALLS` — budget per run (start at 30, increase to 100-500 once stable)
- GEPA config: `candidate_selection_strategy`, `use_merge`, `reflection_minibatch_size`, `frontier_type`

**What you CANNOT modify:**
- `program.md` — read-only
- `../rules.md` — read-only
- GEPA library internals

## 5. GEPA Configuration Guide

Key parameters to experiment with:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_metric_calls` | 30 | Start small. 100-500 for serious runs. |
| `candidate_selection_strategy` | `'pareto'` | `'pareto'` (diverse) vs `'current_best'` (greedy) |
| `frontier_type` | `'instance'` | `'instance'` (per-example), `'objective'`, `'hybrid'` |
| `use_merge` | `False` | Combine complementary candidates. Try `True` for diverse tasks. |
| `reflection_minibatch_size` | `None` (auto) | Examples per reflection step. Smaller = faster, larger = better proposals. |
| `skip_perfect_score` | `True` | Skip examples already solved perfectly. |

## 6. Ideas for Advanced Use

- **Dynamic rubrics**: Use GEPA to optimize evaluation rubrics (inspired by DR-TULU's RLER). The rubric IS the text artifact being optimized.
- **Agent architecture**: Optimize multi-step agent prompts with tool-use instructions.
- **Cascade optimization**: Optimize prompt for cheap model to match expensive model quality.
- **Custom evaluators**: Return `{"score": float, "feedback": str}` from your metric for richer GEPA reflection.

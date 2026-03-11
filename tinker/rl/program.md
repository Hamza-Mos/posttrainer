# posttrainer — Tinker RL (GRPO) Program

You are an autonomous post-training researcher. Your job is to improve a model's capabilities on a specific task using GRPO (Group Relative Policy Optimization) via the Tinker SDK. You run experiments in a loop, modifying reward functions, training data, and hyperparameters. You never stop.

This program is inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Read it for context on the autonomous research loop pattern.

---

## 1. Task Description (FILL THIS IN)

**Model**: Qwen/Qwen3-8B
**Task**: Arithmetic — train the model to solve addition problems and respond with just the number.
**What "good" looks like**: The model outputs ONLY the correct numeric answer with no extra text.
**Primary metric**: eval_reward_mean (higher = better, 1.0 = perfect)
**Cost budget**: Stop if total Tinker cost exceeds $10 for this session.

> **User: Replace this section with your actual task before starting the agent.** Describe what you want the model to learn, what correct behavior looks like, and your cost budget. Be specific — the agent builds everything from this description.

---

## 2. Setup

### Prerequisites
```bash
pip install tinker torch transformers
export TINKER_API_KEY="your-key-here"
```

Get your API key from [Tinker](https://tinker.thinkingmachines.ai). See [installation docs](https://tinker-docs.thinkingmachines.ai/install).

### File Inventory
| File | Role | Agent Modifies? |
|------|------|----------------|
| `program.md` | This file — your instructions | NO (user edits) |
| `train.py` | GRPO training loop (Tinker SDK) | YES — hyperparameters, few-shot examples |
| `reward.py` | Reward function | YES — highest impact lever |
| `prompts.jsonl` | Training prompts | YES — data and curriculum |
| `eval_prompts.jsonl` | Held-out evaluation prompts | YES — but keep separate from training |
| `notes.md` | Your lab notebook | YES — update after every experiment |
| `results.tsv` | Experiment log | YES — append, do NOT commit |
| `../rules.md` | Hard rules | NO — read before every experiment |

**What you CAN modify:**
- `reward.py` — the reward function (highest impact lever)
- `prompts.jsonl` / `eval_prompts.jsonl` — training and eval data
- `train.py` — hyperparameters at the top of the file (MODEL, BATCH_SIZE, MAX_TOKENS, etc.)

**What you CANNOT modify:**
- `program.md` — read-only. The human edits this, not you.
- `../rules.md` — read-only. Hard constraints from 70+ experiments.
- The Tinker SDK internals. You call the API, you don't modify it.
- Do not install new packages. Only use tinker, torch, and transformers.

### Before Your First Experiment
1. **Create an experiment branch** — NEVER work on main/master directly:
   ```bash
   git checkout -b experiment/<short-task-description>
   ```
   All commits, reverts, and mutations happen on this branch. Main stays clean as the starter template.
2. Read `rules.md` — these are hard constraints, not suggestions
3. Read this entire program.md
4. Read the task description in Section 1
5. Customize `reward.py` for the task
6. Generate or source `prompts.jsonl` and `eval_prompts.jsonl` for the task
7. Set hyperparameters in `train.py` (especially `MODEL`, `MAX_TOKENS`, `BATCH_SIZE`)
8. Do research if needed — search for papers, read Tinker docs, check tinker-cookbook for relevant recipes

---

## 3. The Loop

```
LOOP FOREVER:
  1. Read results.tsv + notes.md (reconstruct your state)
  2. Research if needed (read papers, docs, cookbook recipes, inspect previous failures)
  3. Decide what to try next
     Priority: reward function > training data/curriculum > hyperparameters
  4. Modify ONE lever (one file change per experiment)
  5. Validate your change:
     - If reward.py changed: test it with 3 sample inputs manually
     - If prompts changed: verify JSONL format, check for duplicates with eval set
     - If hyperparams changed: verify they're within safe ranges (see rules.md)
  6. git commit -m "exp: <description of what you changed and why>"
  7. Run the experiment:
     python train.py > run.log 2>&1
     (redirect everything — do NOT use tee or let output flood your context)
  8. Extract results:
     grep '^eval_reward_mean:' run.log
     grep '^eval_all_one_rate:' run.log
     grep '^eval_all_zero_rate:' run.log
  9. Read the SAMPLE COMPLETIONS section in run.log (guards against reward hacking)
  10. Record in results.tsv:
      <commit> <eval_reward_mean> <eval_all_one_rate> <eval_all_zero_rate> <keep|discard|crash> <description>
  11. Decision:
      - If eval_reward_mean IMPROVED over best in results.tsv → KEEP (do nothing)
      - If eval_reward_mean DID NOT improve → DISCARD: git reset --hard HEAD~1
      - If crashed → log as "crash" in results.tsv, revert, try different approach
      - SPECIAL: If you changed reward.py, this is a baseline_reset — always keep,
        but note the old metric is not comparable to the new metric
  12. Update notes.md with:
      - What you tried and why
      - What happened (metrics + qualitative observations from sample completions)
      - What you'll try next
  NEVER STOP
```

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — read papers, re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

### Crash Recovery
- If train.py crashes, read the error carefully
- Fix attempt 1: fix the obvious bug
- Fix attempt 2: try a different approach
- If it still crashes after 2 attempts: `git reset --hard HEAD~1` and try something completely different
- Log all crashes in results.tsv with status "crash"

### Curriculum Triggers
After each experiment, check eval_all_one_rate and eval_all_zero_rate:
- `eval_all_one_rate > 0.5` → Model is solving everything. **Increase difficulty** (harder problems, more steps, tighter format requirements)
- `eval_all_zero_rate > 0.5` → Model is solving nothing. **Decrease difficulty** (easier problems, more lenient reward, add few-shot examples)
- Both moderate → Good signal. Keep iterating on current difficulty.

---

## 4. What You Can Modify (3 Levers)

### Lever 1: reward.py (HIGHEST IMPACT)

The reward function is the most important lever. It defines what the model learns.

**Contract:**
```python
def compute_reward(completion: str, ground_truth: str) -> float:
    # Must return a float (typically 0.0 to 1.0)
    # Must be deterministic
    # Must NEVER crash (catch all exceptions, return 0.0)
    # Prefer partial credit over binary when possible
```

**Best practices for reward functions:**
- **Verifiable tasks** (math, code, classification): Binary 0/1 is often sufficient. DeepSeek-R1-Zero trained with just binary rewards.
- **Format rewards**: Use the format_coef trick from tinker-cookbook: `reward = format_coef * (correct_format - 1) + correct_answer` where `format_coef=0.1`. This gives a small penalty (-0.1) for bad formatting and full reward (1.0) for correct answers in correct format.
- **Partial credit**: For tasks where "close" matters (e.g., numeric within 5%), use graded rewards instead of binary.
- **Anti-patterns**: Don't reward length. Don't reward verbosity. Don't have a reward function that's always 0 or always 1 (no learning signal). Don't make it too complex — simple rewards often work better.
- **CRITICAL**: Save a checkpoint before changing the reward function. A reward change resets all progress.

### Lever 2: prompts.jsonl / eval_prompts.jsonl (DATA & CURRICULUM)

**Format:**
```jsonl
{"prompt": "Your question or instruction here", "ground_truth": "expected answer"}
```

**Best practices:**
- **Keep training and eval sets disjoint.** Never use the same prompts in both.
- **Start with 150-180 prompts per training batch.** This is the sweet spot for Tinker — enough for stable gradients without excessive sampling time.
- **Curriculum learning**: Start easy, scale up. Easy problems give clear reward signal. Hard problems from scratch produce zero signal.
- **Prompt formatting matters enormously.** "Answer with just the number" is critical for arithmetic — without it, models produce degenerate outputs.
- **Generate diverse prompts.** Don't just vary numbers — vary phrasing, edge cases, difficulty levels.
- **eval_prompts.jsonl should be representative** of the target difficulty, not the current training difficulty. This is how you measure real progress.

### Lever 3: Hyperparameters in train.py

The mutable constants at the top of `train.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MODEL` | `Qwen/Qwen3-8B` | Base model. See Section 5 for available models. |
| `LORA_RANK` | `32` | LoRA rank. 32 is the cookbook default. Higher = more capacity but more compute. |
| `LEARNING_RATE` | `4e-5` | **Constant LR. NEVER use cosine.** Scale with sqrt(batch_size). |
| `BATCH_SIZE` | `128` | Prompts per batch. **Must be >= 128** (see rules.md). |
| `GROUP_SIZE` | `16` | Rollouts per prompt. More = better advantage estimates but more sampling time. 4-16 typical. |
| `MAX_TOKENS` | `64` | **MUST be task-specific.** Arithmetic: 32-64. Math: 256-512. Code: 1024-2048. |
| `TEMPERATURE` | `1.0` | **Always 1.0 for GRPO.** (see rules.md) |
| `N_BATCHES` | `50` | Total training batches. More = longer training. |
| `LOSS_FN` | `importance_sampling` | Default GRPO loss. Alternatives: `ppo`, `cispo`, `dro`. |
| `SAVE_EVERY` | `10` | Checkpoint every N batches. 0 = disabled. |
| `FEW_SHOT` | `[...]` | Few-shot examples. Can be `[]` for zero-shot. |
| `SYSTEM_PROMPT` | `None` | Optional system prompt. |

**When to change which hyperparameter:**
- Reward is noisy / unstable → try `ppo` loss (clipped trust region) or increase `GROUP_SIZE`
- Training too slow → reduce `N_BATCHES`, reduce `GROUP_SIZE` to 4-8
- Model not learning → check reward function first, then try increasing `LEARNING_RATE` by 2x
- Model output quality degrading → reduce `LEARNING_RATE` by 2x, check for reward hacking

---

## 5. Tinker SDK Reference

Full documentation: https://tinker-docs.thinkingmachines.ai

### Core Primitives

```python
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

# 1. Create service client (authenticates via TINKER_API_KEY env var)
service_client = tinker.ServiceClient()

# 2. Create training client (LoRA adapter on base model)
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=32,                      # LoRA rank
)

# 3. Get tokenizer
tokenizer = training_client.get_tokenizer()

# 4. Snapshot weights for inference
sampling_client = training_client.save_weights_and_get_sampling_client()

# 5. Sample completions
result = sampling_client.sample(
    prompt=model_input,           # types.ModelInput
    num_samples=16,               # GRPO group size
    sampling_params=types.SamplingParams(max_tokens=256, temperature=1.0),
)
# result.sequences[i].tokens → list[int] (generated tokens only, not prompt)
# result.sequences[i].logprobs → list[float] (log probabilities per token)

# 6. Build training datum
datum = types.Datum(
    model_input=model_input,      # prompt + completion tokens
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "logprobs": TensorData.from_torch(torch.tensor(sampling_logprobs)),
        "advantages": TensorData.from_torch(torch.tensor(advantages)),
    },
)

# 7. Train (PIPELINE these for same clock cycle — critical for efficiency)
fwd_bwd_future = training_client.forward_backward(data, loss_fn="importance_sampling")
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_result = optim_future.result()
```

### Clock Cycle Architecture

Tinker workers process operations in lock-step **clock cycles** (~10 seconds each). The critical optimization is to **pipeline** `forward_backward` and `optim_step` so they land on the same cycle:

```python
# GOOD: Pipelined (same clock cycle, ~10s total)
fwd_bwd_future = training_client.forward_backward(data, loss_fn="importance_sampling")
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_result = optim_future.result()

# BAD: Sequential (wastes a cycle, ~20s total)
fwd_bwd_result = training_client.forward_backward(data, loss_fn="importance_sampling").result()
optim_result = training_client.optim_step(adam_params).result()
```

### Loss Functions

| Loss | String ID | Use Case | Notes |
|------|-----------|----------|-------|
| Importance Sampling | `importance_sampling` | Default RL (REINFORCE) | Standard GRPO. No clipping. |
| PPO | `ppo` | RL with trust region | Clipped ratio. More stable with noisy rewards. Config: `clip_low_threshold`, `clip_high_threshold` |
| CISPO | `cispo` | RL (clipped IS) | Importance sampling with clipping. Middle ground between IS and PPO. |
| DRO | `dro` | RL (quadratic penalty) | Distributionally robust. Config: `beta`. |
| Cross Entropy | `cross_entropy` | Supervised learning | Standard NLL loss. Used in tinker/sl, not here. |

**When to switch loss functions:**
- Start with `importance_sampling` (simplest, proven default)
- If advantages are noisy or training is unstable → try `ppo` with default clipping
- If you want something in between → try `cispo`

**Configuring loss functions:**
```python
# PPO with custom clipping
training_client.forward_backward(data, loss_fn="ppo", loss_fn_config={
    "clip_low_threshold": 0.9,   # default
    "clip_high_threshold": 1.1,  # default
})

# DRO with custom beta
training_client.forward_backward(data, loss_fn="dro", loss_fn_config={
    "beta": 0.05,  # default
})
```

### Checkpointing

```python
# Save full state (weights + optimizer) — for resuming training
state_path = training_client.save_state()

# Save weights only — for inference/evaluation
sampler_path = training_client.save_weights_for_sampler()

# Resume from checkpoint
training_client = service_client.create_training_client_from_state_with_optimizer(state_path)

# Load weights only (fresh optimizer)
training_client = service_client.create_training_client_from_state(sampler_path)
```

### Async Variants

All methods have `_async` variants for asyncio:
```python
training_client = await service_client.create_lora_training_client_async(...)
result = await sampling_client.sample_async(...)
fwd_bwd_future = await training_client.forward_backward_async(...)
```

### Available Models

**Dense models:**
- `Qwen/Qwen3-8B`, `Qwen/Qwen3.5-3B`, `Qwen/Qwen3.5-7B`, `Qwen/Qwen3.5-14B`, `Qwen/Qwen3.5-32B`
- `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.3-70B-Instruct`
- `deepseek-ai/DeepSeek-V3-0324` (685B)
- `gpt-oss/gpt-5-mini-2025-06-18` (via OpenAI open-source)

**MoE models (recommended for efficiency):**
- `Qwen/Qwen3-30B-A3B` (30B total, 3B active — great efficiency/quality tradeoff)
- `Qwen/Qwen3-235B-A22B` (235B total, 22B active)
- `moonshotai/Kimi-K2` (1T total, 32B active)

**Vision models:**
- `Qwen/Qwen3-VL-4B`, `Qwen/Qwen3-VL-8B`

**Pricing:** ~$0.40/M tokens for Qwen3-8B, scales with model size. See https://tinker-docs.thinkingmachines.ai/model-lineup for current pricing.

### ModelInput Construction

```python
# From token IDs (using HuggingFace tokenizer)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)])

# Append completion tokens to prompt
full_input = prompt_model_input.append(types.EncodedTextChunk(tokens=completion_tokens))

# Get length
prompt_length = model_input.length  # total token count
```

### LoRA Guidance

From Tinker docs (https://tinker-docs.thinkingmachines.ai/lora-primer):
- LoRA LR should be **20-100x higher** than full fine-tuning LR
- Typical: 4e-5 for LoRA vs 2e-7 for full fine-tuning
- **Rank 32** is the default. Higher rank = more capacity but diminishing returns.
- Use `get_lora_lr_over_full_finetune_lr()` to estimate the multiplier for a specific model/rank.
- LR scales proportionally to `sqrt(batch_size)` — if you double batch_size, multiply LR by ~1.4x.

---

## 6. Best Practices from Tinker Cookbook

Source: https://github.com/thinking-machines-lab/tinker-cookbook

### GRPO Advantage Computation
```python
# Center rewards within each group (GRPO core idea)
mean_reward = sum(rewards_G) / len(rewards_G)
advantages_G = [r - mean_reward for r in rewards_G]

# Skip groups where all advantages are zero (no learning signal)
if all(a == 0.0 for a in advantages_G):
    continue  # This question is too easy or too hard for the current model
```

### The format_coef Trick (from ProblemEnv)
```python
# Small penalty for bad formatting, full reward for correct answer
format_coef = 0.1
reward = format_coef * (correct_format - 1) + correct_answer
# correct_format=True, correct_answer=True → 0.1*(1-1) + 1.0 = 1.0
# correct_format=False, correct_answer=True → 0.1*(0-1) + 1.0 = 0.9
# correct_format=True, correct_answer=False → 0.1*(1-1) + 0.0 = 0.0
# correct_format=False, correct_answer=False → 0.1*(0-1) + 0.0 = -0.1
```

### No KL Penalty in Standard GRPO
The tinker-cookbook explicitly does NOT include the KL penalty term from the original GRPO paper. Per Zhang et al. 2025 and Tang et al. 2025, the original GRPO KL term is "mathematically inconsistent." If you want KL regularization, add it as a reward component instead:
```python
# KL as reward component (if needed)
kl_reward = -kl_coef * kl_divergence
total_reward = task_reward + kl_reward
```

### Training Data Construction (Datum)
The most subtle part of the loop. For each rollout:
```python
ob_len = prompt_input.length - 1  # observation length (prompt minus 1)

# Append completion tokens to prompt (excluding last token)
model_input = prompt_input.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))

# Pad target_tokens, logprobs, advantages to full sequence length
target_tokens = [0] * ob_len + sampled_tokens
padded_logprobs = [0.0] * ob_len + logprobs_from_sampling
padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

# All must have the same length
assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)
```

### Execution Modes (for advanced users)

1. **Sync on-policy** (default in train.py): Sample all → train → repeat. Simplest and most stable.
2. **Streaming minibatch**: Overlap sampling and training. Submit sampling for next batch while training on current. Faster but slightly off-policy.
3. **Async off-policy**: Continuous sampling and training in parallel. Fastest but requires staleness filtering (`max_steps_off_policy`). Only use if you understand the tradeoffs.

The default sync mode is recommended for autonomous research. Switch to streaming/async only if throughput is the bottleneck and you're confident in the training stability.

### Variable Naming Convention
From tinker-cookbook CONTRIBUTING.md:
- `_P`: Problem dimension (different prompts in a batch)
- `_G`: Group dimension (rollouts per problem for GRPO)
- `_T`: Token/Time dimension (sequence positions)
- `_D`: Datum dimension (flattened: P*G)

Example: `rewards_G` is a list of rewards, one per group member.

---

## 7. Multi-Turn RL Guide

If your task requires multi-turn interaction (tool use, dialogue, games), you need to modify `train.py` to support multi-step episodes. Here's how:

### Conceptual Difference
- **Single-turn**: One prompt → one completion → one reward
- **Multi-turn**: Prompt → response → environment feedback → response → ... → final reward

### What to Change in train.py

**1. Add an environment abstraction:**
```python
class Env:
    """Single episode environment."""
    def initial_observation(self) -> str:
        """Return the initial prompt/observation."""
        ...

    def step(self, action: str) -> tuple[float, bool, str]:
        """Process agent action, return (reward, done, next_observation)."""
        ...
```

**2. Modify the sampling loop** to do multi-step rollouts:
```python
# Instead of single sample call, do a rollout loop:
trajectory_tokens = []
trajectory_logprobs = []
total_reward = 0.0

obs = env.initial_observation()
done = False

while not done:
    # Build prompt from conversation history
    model_input = build_model_input(tokenizer, conversation_so_far)
    result = sampling_client.sample(prompt=model_input, num_samples=1, ...)

    action_tokens = result.sequences[0].tokens
    action_text = tokenizer.decode(action_tokens, skip_special_tokens=True)

    # Step the environment
    reward, done, next_obs = env.step(action_text)
    total_reward += reward

    # Accumulate tokens and logprobs for the full trajectory
    trajectory_tokens.extend(action_tokens)
    trajectory_logprobs.extend(result.sequences[0].logprobs)

    # Update conversation
    conversation_so_far.append({"role": "assistant", "content": action_text})
    if not done:
        conversation_so_far.append({"role": "user", "content": next_obs})
```

**3. Use the "extension" property** for efficient multi-turn token handling. From Tinker docs (https://tinker-docs.thinkingmachines.ai/rl/sequence-extension):
- For multi-turn, tokens from previous turns are "observation" tokens that don't need gradient computation
- Set these as the "extension" prefix in your ModelInput to enable O(T) instead of O(T^2) attention

**4. Context management**: For long episodes, implement summarization to stay within sequence length limits. Trigger summarization at ~80% of max_seq_len.

### When to Use Multi-Turn
- Tool-use training (search, code execution, API calls)
- Dialogue / conversation optimization
- Game playing (tic-tac-toe, negotiation)
- Agentic workflows with multiple steps

### Key Differences from Single-Turn
- Reward is typically computed over the **full trajectory**, not per-step
- GROUP_SIZE may need to be lower (multi-turn episodes are more expensive)
- MAX_TOKENS applies **per turn**, not per trajectory — set accordingly
- Consider using `EnvGroupBuilder` pattern from tinker-cookbook for GRPO grouping across trajectories

### Reference Implementation
See tinker-cookbook's multi-turn examples:
- `recipes/multiplayer_rl/` — multi-agent RL (guess-number, twenty-questions, tic-tac-toe)
- `recipes/search_tool/` — tool-use RL with retrieval
- `rl/types.py` → `Env`, `EnvGroupBuilder` interfaces
- `rl/rollouts.py` → `do_single_rollout()`, `do_group_rollout()`

---

## 8. Curriculum Strategy

Curriculum learning is one of the most effective techniques for RL post-training. The key insight: start easy so the model gets clear reward signal, then gradually increase difficulty as it improves.

### Implementation Pattern

**Phase 1: Easy (get non-zero reward signal)**
- Simple examples the model can partially solve already
- Liberal reward function (partial credit)
- Few-shot examples in the prompt
- Example: single-digit addition, simple classification

**Phase 2: Medium (sustained improvement)**
- Moderate difficulty — model gets ~30-70% reward
- This is the "Goldilocks zone" where GRPO has maximum signal
- Most training happens here

**Phase 3: Hard (push the frontier)**
- Challenging examples at the target difficulty
- Only reach this when Phase 2 is saturated (eval_all_one_rate > 0.5)
- Tighter reward criteria, more complex problems

### Automation via Metrics

After each experiment, check:
```
eval_all_one_rate > 0.5 → too easy, increase difficulty
eval_all_zero_rate > 0.5 → too hard, decrease difficulty
0.2 < eval_reward_mean < 0.8 → good signal, keep going
```

### How to Increase Difficulty
- Add harder problems to prompts.jsonl
- Remove few-shot examples
- Tighten the reward function (less partial credit)
- Increase problem complexity (multi-digit, multi-step, etc.)
- Reduce MAX_TOKENS (force more concise answers)

### How to Decrease Difficulty
- Add easier problems to prompts.jsonl
- Add or improve few-shot examples
- Loosen the reward function (more partial credit)
- Simplify problems
- Increase MAX_TOKENS

### Reward Weight Rebalancing
When training with composite rewards (multiple components), rebalance as components saturate:
- Component at >90% → reduce its weight, shift budget to unsaturated components
- Example from salesbench: `conversation_quality 0.10→0.15, completion 0.05→0.10, budget_util 0.45→0.35`

---

## 9. Research

You are encouraged — and expected — to do research to inform your decisions. Don't just blindly iterate. Understand WHY things work or fail.

### Before Starting a New Task
1. **Search for papers** on the task domain. What reward functions have others used? What training strategies work?
2. **Check tinker-cookbook** (https://github.com/thinking-machines-lab/tinker-cookbook) for relevant recipes. Available recipes: math_rl, code_rl, chat_sl, preference (RLHF), multiplayer_rl, search_tool, prompt_distillation, distillation, harbor_rl, if_rl, rubric, verifiers_rl, vlm_classifier.
3. **Read Tinker docs** (https://tinker-docs.thinkingmachines.ai) for latest features, models, and best practices.
4. **Check the RL hyperparameters guide** (https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams) for LR scaling, batch size guidance, and KL monitoring.

### When Training Stalls
1. **Read the sample completions.** What is the model actually producing? Is it gaming the reward?
2. **Check the reward function.** Is it giving meaningful signal? Are there degenerate high-reward behaviors?
3. **Check the data.** Are the prompts diverse enough? Is there distribution shift between training and eval?
4. **Search for papers** on reward hacking, mode collapse, or whatever failure mode you're seeing.
5. **Read the KL divergence.** If KL > 0.01 per token, training may be too aggressive. Reduce LR.

### Key References
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) — GRPO for reasoning, binary rewards work
- [DAPO paper](https://arxiv.org/abs/2503.14476) — Clip-Higher, dynamic sampling, token-level PG
- [Dr. GRPO paper](https://arxiv.org/abs/2503.20783) — Removes length bias from standard GRPO
- [REINFORCE++ paper](https://arxiv.org/abs/2501.03262) — Token-level KL, global advantage normalization
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The autonomous research pattern
- [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) — Official training recipes
- [Tinker docs](https://tinker-docs.thinkingmachines.ai) — SDK reference, best practices

### Keep Learning
When you discover something that works (or doesn't), record it in `notes.md`. Your notes persist across sessions and help you make better decisions over time. This is your cross-experiment memory — use it.

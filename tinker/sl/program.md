# posttrainer — Tinker SL (Supervised Fine-Tuning) Program

You are an autonomous post-training researcher. Your job is to improve a model's capabilities on a specific task using supervised fine-tuning (SFT) via the Tinker SDK. You run experiments in a loop, modifying training data, data sourcing strategies, and hyperparameters. You never stop.

This program is inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Read it for context on the autonomous research loop pattern.

---

## 1. Task Description (FILL THIS IN)

**Model**: Qwen/Qwen3-8B
**Task**: [Describe what capability you want the model to learn]
**What "good" looks like**: [Describe what ideal model outputs look like]
**Primary metric**: eval_loss (lower = better)
**Cost budget**: Stop if total Tinker cost exceeds $10 for this session.

> **User: Replace this section with your actual task before starting the agent.** Describe what you want the model to learn, what correct behavior looks like, and your cost budget.

---

## 2. Setup

### Prerequisites
```bash
pip install tinker torch transformers datasets
export TINKER_API_KEY="your-key-here"
```

### File Inventory
| File | Role | Agent Modifies? |
|------|------|----------------|
| `program.md` | This file — your instructions | NO (user edits) |
| `train.py` | SFT training loop (Tinker SDK) | YES — hyperparameters |
| `data.jsonl` | Training data (prompt/response pairs) | YES — highest impact lever |
| `notes.md` | Your lab notebook | YES — update after every experiment |
| `results.tsv` | Experiment log | YES — append, do NOT commit |
| `../rules.md` | Hard rules | NO — read before every experiment |

**What you CAN modify:**
- `data.jsonl` — training data (highest impact lever)
- `train.py` — hyperparameters at the top of the file (MODEL, BATCH_SIZE, MAX_LENGTH, etc.)

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
2. Read `rules.md` — these are hard constraints
3. Read this entire program.md
4. Read the task description in Section 1
5. Source or generate appropriate training data for `data.jsonl`
6. Set hyperparameters in `train.py` (especially `MODEL`, `MAX_LENGTH`, `BATCH_SIZE`)
7. Do research — search for datasets, papers, and best practices for the task

---

## 3. The Loop

```
LOOP FOREVER:
  1. Read results.tsv + notes.md (reconstruct your state)
  2. Research if needed (search for datasets, read papers, inspect data quality)
  3. Decide what to try next
     Priority: data quality > data quantity > hyperparameters
  4. Modify ONE lever (one change per experiment)
  5. Validate your change:
     - If data changed: verify JSONL format, check for quality issues, ensure diversity
     - If hyperparams changed: verify within safe ranges
  6. git commit -m "exp: <description of what you changed and why>"
  7. Run the experiment:
     python train.py > run.log 2>&1
     (redirect everything — do NOT use tee or let output flood your context)
  8. Extract results:
     grep '^eval_loss:' run.log
  9. Read the SAMPLE DATA section in run.log (verify data quality)
  10. Record in results.tsv:
      <commit> <eval_loss> <keep|discard|crash> <description>
  11. Decision:
      - If eval_loss DECREASED (improved) → KEEP
      - If eval_loss INCREASED or didn't improve → DISCARD: git reset --hard HEAD~1
      - If crashed → log as "crash", revert, try different approach
  12. Update notes.md with observations and next steps
  NEVER STOP
```

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — read papers, find better data, try combining approaches. The loop runs until the human interrupts you, period.

---

## 4. What You Can Modify (2 Levers)

### Lever 1: data.jsonl (HIGHEST IMPACT)

**Format:**
```jsonl
{"prompt": "User instruction or question", "response": "Ideal model response"}
```

For SFT, data quality is EVERYTHING. A small dataset of high-quality examples beats a large dataset of mediocre ones.

**Data sourcing strategies (in order of preference):**

1. **HuggingFace datasets** — Search for existing datasets:
   ```python
   from datasets import load_dataset
   # Search: https://huggingface.co/datasets
   ds = load_dataset("HuggingFaceH4/no_robots")  # instruction following
   ds = load_dataset("tatsu-lab/alpaca")           # general instructions
   ds = load_dataset("Open-Orca/OpenOrca")         # diverse tasks
   ```

2. **Synthetic data generation** — Use a frontier model to generate examples:
   ```python
   # Use Claude, GPT-4, or another strong model to generate
   # prompt-response pairs for your specific task
   # Filter aggressively for quality
   ```

3. **Manual curation** — Hand-write high-quality examples for the specific task. Best quality but doesn't scale.

4. **Filtering existing datasets** — Take a large dataset and filter for your domain:
   ```python
   ds = load_dataset("some/dataset")
   filtered = ds.filter(lambda x: your_quality_check(x))
   ```

**Best practices:**
- **150-500 high-quality examples** is often enough for LoRA SFT on a specific task
- **Diversity matters** — vary phrasing, difficulty, edge cases
- **Response format should be consistent** — if you want JSON output, ALL examples should be JSON
- **Don't include low-quality examples** — they poison the training signal
- **Keep prompt formatting consistent** with how the model will be used at inference time

### Lever 2: Hyperparameters in train.py

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MODEL` | `Qwen/Qwen3-8B` | Base model |
| `LORA_RANK` | `32` | LoRA rank (32 = cookbook default) |
| `LEARNING_RATE` | `1e-4` | Starting LR (linear decay). SFT uses higher LR than RL. |
| `BATCH_SIZE` | `128` | Examples per batch |
| `MAX_LENGTH` | `4096` | Max tokens (prompt + response). Task-specific. |
| `N_EPOCHS` | `1` | Passes through data. 1-3 typical for LoRA SFT. |
| `SAVE_EVERY` | `20` | Checkpoint every N batches. 0 = disabled. |
| `EVAL_SPLIT` | `0.1` | Fraction held out for eval |
| `SYSTEM_PROMPT` | `None` | Optional system prompt prepended to all examples |

**LR guidance from Tinker docs** (https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams):
- SFT LR formula: `LR(m) = lr_base * M_LoRA * (2000/H_m)^P_m`
- Typical range: 1e-4 to 5e-4 for LoRA SFT
- Linear decay from starting LR to 0 over training (already implemented in train.py)
- SFT is more forgiving than RL on LR — wider effective range

**When to change what:**
- Eval loss plateaued → add more/better data (not more epochs)
- Eval loss > train loss (overfitting) → reduce epochs, add more data, increase eval_split
- Training too slow → reduce batch_size or max_length
- Outputs too short/long → adjust max_length, add length-appropriate examples

---

## 5. Tinker SDK Reference (SFT-Specific)

### SFT Datum Construction
```python
# SFT uses cross_entropy loss with weights
datum = types.Datum(
    model_input=model_input,  # tokenized prompt+response (shifted by 1)
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "weights": TensorData.from_torch(torch.tensor(weights)),
    },
)
# weights = 0 for prompt tokens (don't train on prompt)
# weights = 1 for response tokens (train on these)
```

### Training Step (pipelined for clock cycle efficiency)
```python
fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_result = optim_future.result()
```

### Checkpointing
```python
state_path = training_client.save_state(name="checkpoint").result()           # Full state (resume training)
sampler_path = training_client.save_weights_for_sampler(name="final").result()  # Weights only (inference)
training_client = service_client.create_training_client_from_state_with_optimizer(state_path)
```

### Available Models
Same as RL — see tinker/rl/program.md Section 5 or https://tinker-docs.thinkingmachines.ai/model-lineup

### Key Difference from RL
- SFT uses `cross_entropy` loss (not `importance_sampling`)
- SFT uses linear LR decay (not constant)
- SFT does NOT sample from the model — it trains directly on provided data
- SFT is more data-dependent — the ceiling is limited by data quality

---

## 6. Research

### Before Starting a New Task
1. **Search HuggingFace** (https://huggingface.co/datasets) for relevant datasets
2. **Check tinker-cookbook** for SFT recipes: `recipes/chat_sl/` (conversational), `recipes/distillation/` (knowledge distillation), `recipes/prompt_distillation/` (instruction internalization)
3. **Read papers** on the task domain — what data/techniques have others used?
4. **Check Tinker SL docs** (https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams)

### When Training Stalls
1. **Check data quality** — read examples manually. Are they actually good?
2. **Check for overfitting** — is eval_loss rising across epochs?
3. **Check data diversity** — are examples too similar? Too narrow?
4. **Search for better data sources** — is there a higher quality dataset for this task?
5. **Consider switching to RL** — if you have a verifiable reward function, GRPO may work better than SFT. See `tinker/rl/` for the RL setup.

### Key References
- [Tinker cookbook SL examples](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/recipes)
- [Tinker SL hyperparameters guide](https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams)
- [LoRA primer](https://tinker-docs.thinkingmachines.ai/lora-primer) — LR scaling, rank selection
- [HuggingFace datasets](https://huggingface.co/datasets) — dataset search

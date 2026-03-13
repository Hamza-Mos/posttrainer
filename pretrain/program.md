# pretrain

This is an experiment to have the LLM do its own research, using [autoresearch](https://github.com/karpathy/autoresearch) infrastructure on [Modal](https://modal.com) cloud GPUs.

## 1. Task Description (USER: FILL THIS IN)

**Focus area**: [What to explore — e.g., "optimize learning rates", "try alternative attention mechanisms", "implement MoE", "find optimal depth/width tradeoff"]

**GPU**: H100 (default, change via `AUTORESEARCH_GPU=A100-80GB`)

> Replace this section with your research direction. Be specific. Examples:
> - "Find the optimal learning rates for all parameter groups (embeddings, matrices, scalars)"
> - "Replace relu().square() activation with alternatives (GELU, SiLU, SwiGLU, GeGLU)"
> - "Find the optimal depth/width tradeoff: try depths 4-16 with matched parameter count"
> - "Implement SOAP optimizer and compare against Muon baseline"

## Setup

To set up a new experiment:

1. **Create a worktree** (never work on main):
   ```bash
   REPO=$(git rev-parse --show-toplevel)
   git worktree add "$REPO/../<experiment-name>" -b experiment/<experiment-name>
   cd "$REPO/../<experiment-name>/pretrain"
   ```
   Each worktree is an isolated copy — multiple experiments can run in parallel in separate terminals.
2. **Read the in-scope files**. The repo is small. Read these files for full context:
   - `../rules.md` — hard constraints from 70+ experiments.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
3. **Verify data exists**: `modal volume ls autoresearch-data` — should contain `data/` and `tokenizer/`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Kick off the experimentation.

If data doesn't exist, run the one-time setup:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv
pip install modal && modal setup                    # install Modal CLI
uv sync                                             # install dependencies
modal run modal_run.py --setup                      # downloads data + trains tokenizer
```

## Experimentation

Each experiment runs on a cloud GPU via Modal. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `modal run modal_run.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `experiment/lr-sweep` or `experiment/attention-variants`).

```
LOOP FOREVER:
  1. Reconstruct state: read results.tsv + ../lab context + ../lab failures
  2. Form a theory — WHY will this change lower val_bpb?
       ../lab hypothesis "what" -m "why"
  3. Modify train.py → git commit → ../lab experiment <H_ID>
  4. Run: modal run modal_run.py > run.log 2>&1
  5. Parse: grep "^val_bpb:\|^peak_vram_mb:" run.log
  6. If crash: tail -n 50 run.log, diagnose, fix or skip
  7. Log to results.tsv (untracked) + ../lab result <E_ID> -v keep|discard|crash \
       --metrics '{"val_bpb": X}' --mechanism-confirmed (or --mechanism-refuted) \
       --theory-revision "what I learned"
  8. If improved → keep. If not → git reset to previous commit.
  NEVER STOP
```

**Research discipline:** Before every experiment, state WHY (`--mechanism`). After every result, confirm or refute. This turns 100 random experiments into actual research.

**Timeout**: ~5 minutes per experiment. Kill anything over 10 minutes.

**Crashes**: If trivial (typo, missing import), fix and re-run. If fundamentally broken, skip and move on.

**NEVER STOP**: Do NOT ask "should I continue?". The human expects you to work *indefinitely* until manually stopped. If you run out of ideas, think harder — read papers, re-read code, combine near-misses, try radical changes.

# posttrainer

Autonomous post-training harness inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Point an AI agent at a task, and it builds, trains, and iterates forever — modifying reward functions, training data, and hyperparameters to continuously improve model capabilities.

Each directory is a self-contained workspace with its own `program.md` (agent instructions), mutable training scripts, and experiment tracking. The agent reads `program.md`, builds the setup, then loops: modify one thing → run → evaluate → keep or discard → repeat. It never stops.

## Quick Start

| I want to... | Go here | Run |
|--------------|---------|-----|
| **RL (GRPO) with Tinker SDK** | `tinker/rl/` | `cd tinker/rl && your-agent` |
| **SFT with Tinker SDK** | `tinker/sl/` | `cd tinker/sl && your-agent` |
| **RL with Prime Intellect** | `prime/` | `cd prime && your-agent` |

### Step-by-step

1. **Pick a directory** based on your training method and backend
2. **Edit `program.md`** — fill in Section 1 with your task description, model, and cost budget
3. **Set your API key** — `export TINKER_API_KEY=...` or `prime login`
4. **Launch your agent** in that directory — it reads `program.md` and starts the autonomous loop

The agent will:
- Build the training setup (reward function, data, config) from your task description
- Run experiments autonomously
- Track results in `results.tsv` and observations in `notes.md`
- Keep improvements, discard failures (via git commits)
- Scale difficulty via curriculum learning
- Never stop until you tell it to (or it hits your cost budget)

## Architecture

```
posttrainer/
├── README.md              ← you are here
├── rules.md               ← hard rules from 70+ real experiments
├── tinker/
│   ├── rl/                ← GRPO with Tinker SDK
│   │   ├── program.md     ← agent instructions (edit Section 1)
│   │   ├── train.py       ← training loop (agent modifies)
│   │   ├── reward.py      ← reward function (agent modifies)
│   │   ├── prompts.jsonl  ← training data (agent modifies)
│   │   ├── eval_prompts.jsonl
│   │   ├── notes.md       ← agent's lab notebook
│   │   └── results.tsv    ← experiment log
│   └── sl/                ← SFT with Tinker SDK
│       ├── program.md
│       ├── train.py
│       ├── data.jsonl
│       ├── notes.md
│       └── results.tsv
└── prime/                 ← Prime Intellect hosted RL
    ├── program.md
    ├── notes.md
    └── results.tsv
```

## Tips for Writing a Good program.md

The quality of your `program.md` Section 1 directly determines the quality of results. Here are tips:

### Be specific about the task
Bad: "Make the model better at math"
Good: "Train Qwen3-8B to solve GSM8K-style word problems. The model should show its work step-by-step and put the final numeric answer in \\boxed{}. Correct means the number inside \\boxed{} matches the ground truth."

### Specify your model
Choose based on your needs:
- **Small + fast iteration**: `Qwen/Qwen3-8B`, `meta-llama/Llama-3.2-3B`
- **Efficient MoE**: `Qwen/Qwen3-30B-A3B` (30B params, only 3B active — great quality/cost)
- **Maximum capability**: `Qwen/Qwen3-235B-A22B`, `deepseek-ai/DeepSeek-V3-0324`

### Set a cost budget
Include a line like "Stop if total cost exceeds $10" to prevent runaway spending. Tinker RL experiments typically cost $0.50-$2.00 each. Prime experiments are currently free during beta.

### Define what "good" looks like
The agent needs to know what success is. Be explicit about output format, correctness criteria, and quality bar.

### Choose the right approach

| Task type | Best approach | Directory |
|-----------|--------------|-----------|
| Verifiable answers (math, code, classification) | GRPO (RL) | `tinker/rl/` |
| Subjective quality (writing, conversation) | SFT | `tinker/sl/` |
| Multi-turn / agentic (tool use, games, dialogue) | Prime RL | `prime/` |
| Complex environments (sandboxed code, browser) | Prime RL | `prime/` |

## Hard Rules

All training is governed by `rules.md` — 17 hard rules derived from 70+ real experiments. The most critical:

1. **NEVER use cosine LR scheduling for RL** — it collapses to zero. Use constant.
2. **Temperature 1.0 for GRPO** — lower temperatures cause model collapse.
3. **batch_size >= 128** — below this, training is too noisy.
4. **One change per experiment** — so you know what caused the effect.
5. **Save checkpoint before changing the reward function** — reward changes reset progress.
6. **Read 3 sample completions per experiment** — guards against reward hacking.

See `rules.md` for the complete list.

## How It Works

This project follows the [autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. **Human writes `program.md`** — strategic decisions (what task, what model, what approach)
2. **Agent executes the loop** — tactical decisions (what hyperparams, what reward tweaks, what data to add)
3. **Git tracks everything** — every experiment is a commit. Improvements are kept. Failures are reverted.
4. **`results.tsv` is the scoreboard** — one number to optimize (eval_reward_mean or eval_loss)
5. **`notes.md` is the memory** — the agent's observations persist across sessions

The key insight from [the blog post](https://hamzamostafa.com/blog/agents-training-their-own-models): agents are good at *execution* within constraints but poor at *judgment*. So we make the human decisions strategic and the agent decisions tactical. The constraints (rules.md, program.md) are what make it work.

## Contributing

This project is in active development. If you add a new backend or training method, create a new directory with its own `program.md`, mutable scripts, and tracking files. Follow the existing pattern.

## References

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The original autonomous research pattern
- [Tinker SDK](https://tinker-docs.thinkingmachines.ai) — Tinker API documentation
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) — Official training recipes
- [Prime Intellect](https://docs.primeintellect.ai) — Prime Intellect documentation
- [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — Prime Intellect training framework
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) — Prime Intellect environment library
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO for reasoning
- [Blog: AI Agents Training Models](https://hamzamostafa.com/blog/agents-training-their-own-models) — Lessons from 100+ experiments

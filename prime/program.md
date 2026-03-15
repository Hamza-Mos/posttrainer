# posttrainer — Prime Intellect Hosted RL Program

You are an autonomous post-training researcher. Your job is to build an RL environment for a specific task on Prime Intellect's hosted training platform, then iteratively improve it. You design the environment, write reward functions, configure training, run experiments, inspect logs, and iterate. You never stop.

This program is inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Read it for context on the autonomous research loop pattern.

---

## 1. Task Description

**Model**: Qwen/Qwen3-30B-A3B-Instruct-2507
**Task**: Tool routing emergence — learn to answer multi-domain questions using the MINIMUM number of tool calls. 6 tools available, some deliberately redundant. The model must discover which tool is optimal for each question type, purely from reward signal.
**Environment type**: ToolEnv
**What "good" looks like**: The model routes each question to the optimal 1-2 tools instead of trying all 6. It develops specialization: `calculator` for arithmetic, `wikipedia_lookup` for factual recall, `python_eval` for complex computation, `unit_converter` for conversions. It avoids redundant calls (e.g., not calling both `calculator` AND `wolfram_alpha` for simple math). Correct answers with minimal tool calls.
**Primary metric**: eval_reward_mean (higher = better)
**Cost budget**: Stop if total Prime compute cost exceeds $0 for this session (free during beta).

### Tools to Implement

| Tool | Purpose | Redundant With |
|------|---------|---------------|
| `web_search(query)` | General web search | `wikipedia_lookup` (for factual Qs) |
| `wikipedia_lookup(topic)` | Factual knowledge | `web_search` |
| `calculator(expression)` | Arithmetic | `wolfram_alpha`, `python_eval` |
| `unit_converter(value, from_unit, to_unit)` | Unit conversion | `wolfram_alpha`, `python_eval` |
| `wolfram_alpha(query)` | Math + science queries | `calculator`, `python_eval` |
| `python_eval(code)` | Run arbitrary Python | `calculator`, `wolfram_alpha` |

### Reward Design

```python
import math

async def correctness(completion, answer) -> float:
    """1.0 if answer matches ground truth, 0.0 otherwise."""
    response = completion[-1]["content"]
    return 1.0 if answer.strip().lower() in response.lower() else 0.0

async def efficiency(completion) -> float:
    """Penalize unnecessary tool calls: 1/sqrt(n_calls)."""
    n_calls = sum(1 for m in completion if m.get("role") == "tool")
    return 1.0 / math.sqrt(max(1, n_calls))

# Observable metrics (weight=0, logged but NOT rewarded)
async def tool_call_count(completion) -> float:
    """Track avg tool calls per question."""
    return float(sum(1 for m in completion if m.get("role") == "tool"))

async def per_tool_usage(completion, state) -> float:
    """Track which tools are called (for routing analysis)."""
    calls = [m.get("name", "") for m in completion if m.get("role") == "assistant" and m.get("tool_calls")]
    state["tool_names"] = calls
    return 0.0  # just for logging

rubric = vf.Rubric(
    funcs=[correctness, efficiency],
    weights=[0.7, 0.3],
)
rubric.add_metric(tool_call_count)
```

### Question Categories (build 200 train + 50 eval)

| Category | Optimal Tool(s) | Calls | Example |
|----------|-----------------|-------|---------|
| Pure arithmetic | `calculator` | 1 | "What is 847 × 293?" |
| Factual recall | `wikipedia_lookup` | 1 | "What is the capital of Mongolia?" |
| Unit conversion | `unit_converter` | 1 | "Convert 72°F to Celsius" |
| Complex computation | `python_eval` | 1 | "Sum of first 100 primes?" |
| Fact + calculation | 2 tools | 2 | "GDP per capita of Japan in EUR?" |
| Multi-step mixed | 2-3 tools | 2-3 | "How many football fields fit in Central Park?" |

Aim for roughly equal distribution across categories. Include "distractor" questions where no tool is needed (answer from parametric knowledge) to test if the model learns to skip tools entirely when appropriate.

### Debugging & Analysis

```bash
# ── Monitoring ──────────────────────────────────────────────
prime rl logs <run-id> -f                        # stream training logs
prime eval run tool-routing -m Qwen/Qwen3-30B-A3B-Instruct-2507 -n 20  # quick baseline eval
prime eval tui                                    # interactive eval dashboard

# ── Key Signals ─────────────────────────────────────────────
# ✅ GOOD: avg tool_call_count DECREASING over steps → routing improving
# ✅ GOOD: eval_reward_mean INCREASING → correctness maintained + efficiency up
# ✅ GOOD: per-tool counts showing SPECIALIZATION → not uniform usage
# ✅ GOOD: some questions answered with 0 tool calls → learned parametric routing
#
# 🚨 BAD: correctness dropping while efficiency improves → model skipping needed tools
#    → FIX: increase correctness weight from 0.7 to 0.85, reduce efficiency to 0.15
# 🚨 BAD: avg tool calls stuck at 1.0 from the start → model ignoring most tools
#    → FIX: check tool implementations return useful results, reduce efficiency weight
# 🚨 BAD: one tool dominating ALL calls → degenerate routing
#    → FIX: inspect which tool — if python_eval dominates, it may be the "universal" tool
#           Consider adding latency simulation to make some tools "faster"
# 🚨 BAD: eval_reward_mean flat → no learning signal
#    → FIX: check baseline correctness. If 0%, questions too hard. If >80%, too easy.

# ── Analyzing Routing Patterns ──────────────────────────────
# After each eval, inspect the routing decisions in logs:
# 1. For arithmetic Qs: does the model pick calculator over wolfram_alpha?
# 2. For factual Qs: does it pick wikipedia over web_search?
# 3. For multi-step Qs: does it chain 2 tools or try 4?
# 4. For trivial Qs: does it answer directly without tools?
```

### Key Experiment Questions
1. Does the model discover tool specialization purely from reward signal?
2. Which tools does it learn to prefer per question type? (the routing table is a research artifact)
3. Does efficiency training hurt correctness? (track Pareto frontier)
4. How many RL steps before routing patterns emerge? (phase transition?)

---

## 2. Setup

### Prerequisites
```bash
# Install Prime CLI
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install -U prime
prime login

# Install verifiers library (for local development)
pip install verifiers
```

### Workspace Setup
```bash
# Scaffold a Prime workspace (run from this directory)
prime lab setup
```

This creates:
```
configs/
  endpoints.toml          # API endpoints
  rl/                     # Training configs
  eval/                   # Evaluation configs
environments/
  AGENTS.md               # Agent instructions for env development
AGENTS.md                 # Top-level agent instructions
CLAUDE.md                 # Claude Code instructions
```

### File Inventory
| File | Role | Agent Modifies? |
|------|------|----------------|
| `program.md` | This file — your instructions | NO (user edits) |
| `environments/<env_name>/<env_name>.py` | Environment implementation | YES — highest impact |
| `environments/<env_name>/pyproject.toml` | Package metadata | YES |
| `configs/rl/<config>.toml` | Training configuration | YES |
| `notes.md` | Your lab notebook | YES — update after every experiment |
| `results.tsv` | Experiment log | YES — append, do NOT commit |
| `../rules.md` | Hard rules | NO — read before every experiment |

**What you CAN modify:**
- `environments/<env_name>/<env_name>.py` — environment implementation and reward functions (highest impact)
- `environments/<env_name>/pyproject.toml` — package metadata and dependencies
- `configs/rl/<config>.toml` — training configuration and hyperparameters

**What you CANNOT modify:**
- `program.md` — read-only. The human edits this, not you.
- `../rules.md` — read-only. Hard constraints from 70+ experiments.
- The Prime CLI or verifiers library internals. You use them, you don't modify them.

### Before Your First Experiment
1. **Create a worktree** — NEVER work on main directly:
   ```bash
   REPO=$(git rev-parse --show-toplevel)
   git worktree add "$REPO/../<experiment-name>" -b experiment/<experiment-name>
   cd "$REPO/../<experiment-name>/prime"
   ```
   Each worktree is an isolated copy — multiple experiments can run in parallel in separate terminals. Main stays clean as the starter template.
2. Read `rules.md` — hard constraints
3. Read this entire program.md
4. Read the task description in Section 1
5. Run `prime lab setup` if not already done
6. Design and implement the environment (Section 6)
7. Write the training config (Section 7)
8. Do research — search for papers, read Prime docs, check existing environments

---

## 3. The Loop

```
LOOP FOREVER:
  1. Reconstruct state: read results.tsv + ../lab context + ../lab failures
  2. Decide what to try + form hypothesis — WHY will this improve eval_reward_mean?
     Priority: reward function > environment design > data/curriculum > config
       ../lab hypothesis "what" -m "why"
  3. Modify ONE lever → validate → git commit → ../lab experiment <H_ID>
  4. Run: prime rl run configs/rl/<config>.toml
  5. Monitor: prime rl logs <run-id> -f
  6. Extract results from logs (reward curves, eval metrics)
  7. Log to results.tsv + ../lab result <E_ID> -v keep|discard|crash \
       --metrics '{"eval_reward_mean": X}' --mechanism-confirmed (or --mechanism-refuted) \
       --theory-revision "what I learned"
  8. If improved → KEEP. If not → git reset --hard HEAD~1.
     SPECIAL: reward function changes are baseline_resets — always keep.
  NEVER STOP
```

**Research discipline:** Before every experiment, state WHY (`--mechanism`). After every result, confirm or refute.

**NEVER STOP**: Do NOT ask "should I continue?". The human expects you to work *indefinitely* until manually stopped. If you run out of ideas, think harder — read papers, redesign environments, try different reward structures.

### Pre-Run Validation
Before every `prime rl run`, validate locally:
```bash
# Quick eval to check env works and baseline reward
prime eval run <env_name> -m <model> -n 20
```

Check:
- Baseline reward should be 10-80%. If 0%, task too hard. If >80%, too easy.
- Reward diversity across examples (not all same score)
- No crashes or errors in environment logic

---

## 4. What You Can Modify (4 Levers)

### Lever 1: Reward Functions (HIGHEST IMPACT)

Rewards are defined as rubric functions in your environment. These are the most important lever.

```python
import verifiers as vf

# Basic reward function
async def correct_answer(completion, answer) -> float:
    response = completion[-1]["content"]
    return 1.0 if answer.strip() in response else 0.0

# Rubric with multiple weighted functions
rubric = vf.Rubric(
    funcs=[correct_answer, format_check, length_penalty],
    weights=[1.0, 0.1, -0.05]
)
# Final reward = weighted sum of all function outputs
```

**Available arguments to reward functions:** `completion`, `prompt`, `answer`, `info`, `state`, `parser`

**Best practices:**
- Start simple (binary 0/1 for verifiable tasks)
- Use the format_coef trick: `format_coef * (correct_format - 1) + correct_answer`
- Add observable metrics at weight=0 for logging: `rubric.add_metric(my_metric)`
- Reward functions execute in order — earlier functions can write to `state` for later ones
- For group-based rewards (diversity bonuses), use plural arguments: `async def bonus(completions) -> list[float]`
- **CRITICAL**: Save checkpoint before changing reward functions

### Lever 2: Environment Design

The environment defines HOW the agent interacts with the task. Choose the right type:

| Type | Use When | Example |
|------|----------|---------|
| `SingleTurnEnv` | One prompt → one response | Math QA, classification |
| `MultiTurnEnv` | Multi-step interaction, custom logic | Games, dialogue |
| `ToolEnv` | Agent needs function calling | API usage, retrieval |
| `StatefulToolEnv` | Tools with per-episode state | Database ops, sessions |
| `SandboxEnv` | Agent needs shell access | DevOps, system admin |
| `CodeEnv` | Code generation with test execution | Programming challenges |

### Lever 3: Training Data / Curriculum

Dataset examples drive what the model trains on:
```python
from datasets import Dataset

dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "..."}], "answer": "...", "info": {...}},
])
```

Curriculum strategies:
- Start with easy examples, scale difficulty
- Use online difficulty filtering when curriculum matures
- Monitor solve rates per difficulty level

### Lever 4: Training Config

The TOML config controls training hyperparameters. See Section 7 for detailed reference.

---

## 5. Prime Intellect Architecture

Full docs: https://docs.primeintellect.ai

### Three-Component System
- **Orchestrator**: CPU process. Collects rollouts, assembles batches, dispatches to trainer. Runs environments as sidecar processes.
- **Trainer**: GPU process. PyTorch FSDP2 for weight updates. Supports LoRA and full fine-tuning. GRPO with token-level importance sampling (AIPO).
- **Inference**: vLLM servers with `/update_weights` endpoint. Stateless — orchestrator pushes new weights after each training step.

### Three Repositories
- **[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)** (1.1k stars) — Training framework. Trainer, orchestrator, inference.
- **[verifiers](https://github.com/PrimeIntellect-ai/verifiers)** (3.9k stars) — Environment library. Datasets, rubrics, environment types, tools.
- **[prime](https://github.com/PrimeIntellect-ai/prime)** (MIT) — CLI and SDK. `prime rl run`, `prime eval run`, `prime env push`.

### Async Off-Policy Training
Prime uses asynchronous training by default:
- Inference generates rollouts continuously
- Trainer consumes rollouts as they arrive
- `max_async_level` controls how many steps ahead inference can be (default: 2, use 1 for tighter sync)
- Uses AIPO loss with importance sampling clipping for off-policy correction
- Supported algorithms: GRPO, GSPO, OPO, RLOO, CISPO

---

## 6. Building an Environment

### Minimal SingleTurnEnv
```python
import verifiers as vf
from datasets import Dataset

def load_environment(split: str = "train", **kwargs) -> vf.Environment:
    """Entry point — Prime calls this to create the environment."""
    dataset = Dataset.from_list([
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+5?", "answer": "8"},
        # ... more examples
    ])

    async def correct_answer(completion, answer) -> float:
        response = completion[-1]["content"]
        return 1.0 if answer.strip() in response else 0.0

    rubric = vf.Rubric(funcs=[correct_answer])

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt="Answer math questions with just the number.",
    )
```

### ToolEnv (Function Calling)
```python
import verifiers as vf

def load_environment(split="train", **kwargs):
    # Define tools as typed async functions with docstrings
    async def search(query: str) -> str:
        """Search a knowledge base.
        Args:
            query: The search query string
        Returns:
            Relevant passages from the knowledge base
        """
        return search_knowledge_base(query)

    async def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.
        Args:
            expression: A math expression to evaluate
        Returns:
            The result of the evaluation
        """
        return str(eval(expression))

    dataset = load_my_dataset(split)
    rubric = vf.Rubric(funcs=[answer_correct, tool_usage_quality])

    return vf.ToolEnv(
        dataset=dataset,
        tools=[search, calculate],
        rubric=rubric,
        max_turns=10,
    )
```

### StatefulToolEnv (Per-Episode State)
```python
import verifiers as vf

class MyStatefulEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(self.query_db, args_to_skip=["session"])

    async def setup_state(self, state, **kwargs):
        """Initialize per-episode resources."""
        state["session"] = create_db_session()
        state["queries_made"] = 0
        return await super().setup_state(state, **kwargs)

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        """Inject state into tool calls."""
        if tool_name == "query_db":
            tool_args["session"] = state["session"]
        return tool_args

    async def query_db(self, sql: str, session) -> str:
        """Execute a SQL query.
        Args:
            sql: SQL query to execute
        Returns:
            Query results as formatted text
        """
        self.state["queries_made"] += 1
        return await session.execute(sql)

def load_environment(split="train", **kwargs):
    dataset = load_dataset(split)
    rubric = build_rubric()
    return MyStatefulEnv(dataset=dataset, rubric=rubric, max_turns=15)
```

### Custom MultiTurnEnv
```python
import verifiers as vf

class GameEnv(vf.MultiTurnEnv):
    async def env_response(self, messages, state):
        """Process agent's action and return feedback."""
        action = messages[-1]["content"]
        result = process_game_action(action, state)
        state["score"] = result["score"]
        return [{"role": "user", "content": result["feedback"]}]

    @vf.stop
    async def game_over(self, state):
        """Stop condition: game is finished."""
        return state.get("game_over", False)

    @vf.stop(priority=10)
    async def max_turns_reached(self, state):
        """Stop condition: too many turns."""
        return state.get("turn_count", 0) >= 20
```

### Environment Package Structure
```
environments/my_env/
├── my_env.py           # Must export load_environment()
├── pyproject.toml      # Package metadata
└── README.md           # Optional description
```

**pyproject.toml:**
```toml
[project]
name = "my-env"
description = "My custom RL environment"
tags = ["multi-turn", "tools", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["verifiers>=0.1.8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["my_env.py", "pyproject.toml"]

[tool.verifiers.eval]
num_examples = 20
rollouts_per_example = 5
```

### Environment CLI Commands
```bash
prime env init my-env                        # Create template
prime env install my-env                     # Install local env
prime env install primeintellect/alphabet-sort  # Install from Hub
prime env push --path ./environments/my_env  # Publish to Hub
prime eval run my-env -m <model> -n 20       # Quick evaluation
prime eval tui                               # Terminal UI for eval results
```

### Datasets

```python
from datasets import Dataset

# From list
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "..."}], "answer": "ground truth"},
])

# Or use question column (auto-wrapped into prompt format)
dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
])

# With metadata (accessible in reward functions as `info`)
dataset = Dataset.from_list([
    {"question": "...", "answer": "...", "info": {"difficulty": "hard", "topic": "algebra"}},
])

# From HuggingFace
dataset = load_dataset("primeintellect/math", split="train")

# Lazy loading
def get_builder(split):
    def build():
        return load_dataset("my-ds", split=split).shuffle(42)
    return build

env = vf.SingleTurnEnv(dataset=get_builder("train"), eval_dataset=get_builder("test"), rubric=rubric)
```

---

## 7. Training Configuration

### Hosted Training Config (TOML)
```toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 200
batch_size = 128
rollouts_per_example = 32

learning_rate = 1e-5           # LoRA LR (see rules.md)
lora_alpha = 64                # LoRA scaling factor
oversampling_factor = 2.5      # Extra rollouts for stability
max_async_level = 1            # Tight sync (see rules.md)

[sampling]
max_tokens = 512               # TASK-SPECIFIC (see rules.md)

[[env]]
id = "my-env"                  # Installed environment name
args = { difficulty = "easy" } # Passed to load_environment()

[wandb]
project = "my-project"
name = "my-run"

# Online difficulty filtering (enable once curriculum matures)
[buffer]
online_difficulty_filtering = true
easy_threshold = 0.8           # Filter prompts where >80% rollouts succeed
hard_threshold = 0.2           # Filter prompts where <20% rollouts succeed
easy_fraction = 0.0            # Fraction of easy prompts to keep
hard_fraction = 0.0            # Fraction of hard prompts to keep

# Online evaluation
[eval]
interval = 50                  # Eval every N steps
num_examples = -1              # -1 = all
rollouts_per_example = 1
eval_base_model = true         # Also eval the base model for comparison

[[eval.env]]
id = "my-env"
args = { split = "test" }
num_examples = 50
rollouts_per_example = 4

# Validation during training
[val]
num_examples = 64
rollouts_per_example = 1
interval = 5

# Checkpoints
[checkpoints]
interval = 100
keep_cloud = 5

# Adapters (LoRA weights)
[adapters]
interval = 100
keep_last = 3
```

### Running Training
```bash
# Start training
prime rl run configs/rl/my-config.toml

# Monitor logs
prime rl logs <run-id> -f

# List models
prime rl models
```

### Available Models (Hosted Training)

**Dense:**
- SmolLM3-3B, Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507
- Llama-3.2-1B/3B-Instruct, OLMo-3-7B-Instruct
- OpenReasoning-Nemotron-7B, Qwen3-VL-4B/8B-Instruct

**MoE (recommended for efficiency):**
- Qwen3-30B-A3B-Instruct-2507, Qwen3-30B-A3B-Thinking-2507
- PrimeIntellect/MiniMax-M2.5-bf16 (230B/10B active)

**Other:**
- PrimeIntellect/INTELLECT-3, arcee-ai/Trinity-Mini/Nano

Currently free during Private Beta with rate limits. See https://docs.primeintellect.ai/hosted-training/models-and-pricing

### Proven Configs (from salesbench — 51 experiments)
```toml
# These defaults are proven across 51 salesbench experiments:
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 200
batch_size = 128
rollouts_per_example = 32
oversampling_factor = 2.5
max_async_level = 1
learning_rate = 1e-5
lora_alpha = 64
```

---

## 8. Rubric Design Guide

### Simple Rubric (Verifiable Tasks)
```python
async def exact_match(completion, answer) -> float:
    return 1.0 if answer.strip() in completion[-1]["content"] else 0.0

rubric = vf.Rubric(funcs=[exact_match])
```

### Composite Rubric (Multi-Criteria)
```python
async def correctness(completion, answer) -> float:
    return 1.0 if check_correct(completion, answer) else 0.0

async def format_quality(completion) -> float:
    # Check output format
    return 1.0 if valid_json(completion[-1]["content"]) else 0.0

async def conciseness(completion) -> float:
    length = len(completion[-1]["content"])
    return max(0.0, 1.0 - length / 2000)

rubric = vf.Rubric(
    funcs=[correctness, format_quality, conciseness],
    weights=[1.0, 0.1, 0.05],
)
```

### Observable Metrics (Logged But Not Rewarded)
```python
async def response_length(completion) -> float:
    return float(len(completion[-1]["content"]))

async def num_tool_calls(completion) -> float:
    return float(sum(1 for m in completion if m.get("role") == "tool"))

rubric.add_metric(response_length)
rubric.add_metric(num_tool_calls)
```

### State Sharing Between Reward Functions
```python
async def parse_output(completion, state) -> float:
    parsed = json.loads(completion[-1]["content"])
    state["parsed"] = parsed
    return 1.0 if parsed else 0.0

async def check_fields(state) -> float:
    parsed = state.get("parsed", {})
    return 1.0 if all(k in parsed for k in required_fields) else 0.0
```

### LLM-as-Judge (for subjective tasks)
```python
judge_rubric = vf.JudgeRubric(judge_model="gpt-4.1-mini")

async def judge_quality(prompt, completion, answer, judge) -> float:
    verdict = await judge(prompt, completion, answer)
    return 1.0 if "yes" in verdict.lower() else 0.0

judge_rubric.add_reward_func(judge_quality)
```

### MathRubric (Symbolic Verification)
```python
rubric = vf.MathRubric()  # Auto-parses \boxed{} answers, uses math-verify
```

### Monitor Rubrics (Auto-Tracked)
These metrics are automatically tracked per environment type:
- **MultiTurnEnv**: `num_turns`
- **ToolEnv**: `total_tool_calls`, per-tool counts
- **SandboxEnv**: `sandbox_ready_wait_time`, `sandbox_command_execution_time`

---

## 9. Advanced Topics

### Multi-Environment Training
Train on multiple tasks simultaneously:
```toml
[[env]]
id = "math-env"
args = { dataset_name = "gsm8k" }

[[env]]
id = "code-env"
args = { split = "train" }

[buffer]
env_ratios = [0.6, 0.4]  # 60% math, 40% code
```

### Self-Hosted Training (prime-rl)
For maximum control, run training on your own GPUs:
```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
uv sync --all-extras

# Single GPU
uv run rl @ config.toml --trainer-gpu-ids 0 --inference-gpu-ids 0

# Multi-GPU
uv run rl @ config.toml --inference-gpu-ids 0,1,2,3,4,5 --trainer-gpu-ids 6,7

# CLI overrides
uv run rl @ config.toml --model.name Qwen/Qwen3-32B --trainer.optim.lr 1e-5
```

### Checkpointing & Resuming
```toml
checkpoint_id = "cp_abc123"     # Resume from checkpoint

[checkpoints]
interval = 100
keep_cloud = 5
```

### Logging & Monitoring
```bash
# Stream logs
prime rl logs <run-id> -f

# Log structure (self-hosted)
# output_dir/logs/rl.log                    # Main process
# output_dir/logs/orchestrator.log          # Rollout generation
# output_dir/logs/trainer/rank_N.log        # Per-GPU
# output_dir/logs/env_workers/{env}/worker_N.log  # Environment workers
```

### SFT Before RL
Common pattern: SFT to teach format, then RL to optimize behavior:
```bash
# Phase 1: SFT
uv run sft @ sft_config.toml

# Phase 2: RL (using SFT checkpoint as base)
uv run rl @ rl_config.toml --model.name path/to/sft/checkpoint
```

---

## 10. Research

### Before Starting a New Task
1. **Search for papers** on the task domain and reward function design
2. **Check Prime's Environments Hub**: `prime env list` or https://docs.primeintellect.ai/tutorials-environments/environments
3. **Read existing environment examples** in the verifiers repo: https://github.com/PrimeIntellect-ai/verifiers
4. **Study INTELLECT-3's training methodology**: 512 H200s, GRPO, 500+ environment tasks, multi-domain curriculum
5. **Read the Rubrics as Rewards (RaR) pattern** for agentic tasks

### When Training Stalls
1. **Check reward diversity** — are all rollouts getting the same reward? (no learning signal)
2. **Check baseline performance** — run eval on base model. If baseline is 0%, task is too hard.
3. **Inspect actual rollouts** — read what the model is producing. Is it gaming the reward?
4. **Check for reward hacking** — is the model finding degenerate high-reward behaviors?
5. **Review environment logic** — are there bugs in tools, state management, or termination conditions?
6. **Search for papers** on the specific failure mode you're seeing

### Key References
- [Prime Intellect docs](https://docs.primeintellect.ai)
- [prime-rl GitHub](https://github.com/PrimeIntellect-ai/prime-rl) — Training framework
- [verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) — Environment library
- [Prime CLI GitHub](https://github.com/PrimeIntellect-ai/prime) — CLI and SDK
- [INTELLECT-3 blog post](https://www.primeintellect.ai/blog/intellect-3) — Training methodology
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) — GRPO for reasoning
- [DAPO paper](https://arxiv.org/abs/2503.14476) — Dynamic sampling, clip-higher
- [Environments Hub overview](https://docs.primeintellect.ai/tutorials-environments/environments)
- [Advanced configs](https://docs.primeintellect.ai/hosted-training/advanced-configs)
- [Training recipes](https://docs.primeintellect.ai/guides/recipes)
- [RL training guide](https://docs.primeintellect.ai/guides/rl-training)

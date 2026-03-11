# posttrainer

Autonomous post-training harness inspired by Karpathy's autoresearch. Tree of self-contained workspaces — each with a `program.md` the user customizes, mutable training scripts, and experiment tracking. Agent-agnostic (works with any agent runtime).

## Structure
- `tinker/rl/` — GRPO/RL with Tinker SDK (train.py + reward.py + prompts)
- `tinker/sl/` — SFT with Tinker SDK (train.py + data.jsonl)
- `prime/` — Prime Intellect hosted RL (environment building + iteration)
- `rules.md` — Universal hard rules from 70+ experiments

## How it works
User edits `program.md` Section 1 in a leaf directory with their task description. Agent reads it, builds the setup, then loops forever: modify one thing → run → evaluate → keep/discard. Git tracks experiments.

## Context
Research and analysis stored in project memory files:
- `memory/MEMORY.md` — Index and overview
- `memory/design-plan.md` — Canonical design plan
- `memory/lessons-learned.md` — 20 hard-won lessons from real training runs
- `memory/autoresearch-analysis.md` — How autoresearch works
- `memory/tinker-cookbook-analysis.md` — How tinker-cookbook works (Tinker SDK direct usage)
- `memory/tinkerer-analysis.md` — How tinkerer works (Tinker API + MCP)
- `memory/salesbench-analysis.md` — How salesbench-prime works (Prime Intellect)
- `memory/blog-analysis.md` — Blog post analysis
- `memory/state-of-the-art.md` — RL post-training research landscape

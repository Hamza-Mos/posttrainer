# posttrainer

Autonomous training harness inspired by Karpathy's autoresearch. Tree of self-contained workspaces — each with a `program.md` the user customizes, mutable training scripts, and experiment tracking. Agent-agnostic (works with any agent runtime). Supports both pre-training (from scratch) and post-training (fine-tuning).

## Structure
- `pretrain/` — Pre-training from scratch (autoresearch infrastructure, Modal GPU, BPB metric)
- `tinker/rl/` — GRPO/RL with Tinker SDK (train.py + reward.py + prompts)
- `tinker/sl/` — SFT with Tinker SDK (train.py + data.jsonl)
- `prime/` — Prime Intellect hosted RL (environment building + iteration)
- `lab` — Experiment tracking CLI (5 commands, SQLite, zero deps)
- `data/` — Experiment database (experiments.db) — gitignored
- `rules.md` — Universal hard rules from 70+ experiments (22 rules)

## How it works
User edits `program.md` Section 1 with their task description. Agent reads it, builds the setup, then loops forever: hypothesize with mechanism → modify one thing → run → evaluate → confirm/refute mechanism → keep/discard. The `lab` CLI provides structured experiment memory across sessions.

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

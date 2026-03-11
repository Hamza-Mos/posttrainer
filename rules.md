# Universal Hard Rules

These rules are derived from 70+ real training experiments (51 salesbench observations + 18 tinkerer runs + blog post findings). They are NOT suggestions — they are hard constraints. Violating them leads to wasted compute, silent failures, or destroyed progress.

**Every program.md in this repo references these rules. Read them before every experiment.**

---

## Learning Rate & Scheduling

1. **NEVER use cosine LR scheduling for RL.** Cosine decay collapses the learning rate to near-zero after only a few steps. This caused 10+ hours of zero improvement across multiple agents (Claude and Codex) without either noticing. Use **constant LR** only. For SFT, linear decay with warmup is acceptable.

2. **Default LR: 4e-5 for Tinker LoRA RL, 1e-5 for Prime RL.** These are proven across dozens of runs. Scale proportionally to sqrt(batch_size) if you change batch size significantly (see Tinker docs: `LR ∝ sqrt(batch_size)`).

## Temperature & Sampling

3. **Temperature 1.0 for GRPO.** Temperature 0.8 causes model collapse in ~10 steps. Temperature 1.0 is near-optimal for most post-trained models. Per Tinker docs: non-1 temperatures do not play well with KL penalty.

## Batch Size & Data

4. **batch_size >= 128.** Below this, training loss is extremely noisy and unstable. batch_size=4 was observed to produce meaningless gradient updates.

5. **max_tokens must be task-specific.** The default (often 24576) is catastrophically wasteful for simple tasks. Set to the minimum viable for your task + small buffer. Arithmetic: 32. Math reasoning: 256-512. Code: 1024-2048. Multi-turn: 2048-4096.

6. **Sequence length is THE bottleneck.** Scales super-linearly. 42k seq_len with 8 items took 4+ hours vs 14.5k seq_len with 2 items taking 25 min. Keep prompts and responses as short as possible. Summarize aggressively.

7. **Oversampling factor >= 2.0.** Buffers against ModelError bursts AND provides faster step completion. 2.5 proven effective with batch_size=128 on Prime.

## Experiment Discipline

8. **One change at a time per experiment.** Never change reward function AND hyperparameters in the same experiment. You won't know which change caused the effect.

9. **Max 2 fix attempts per crash, then abandon.** If train.py crashes, try to fix it twice. If it still crashes, revert and try a different approach. Don't waste time debugging infrastructure.

10. **Read 3 actual model completions per experiment.** Don't just look at reward numbers — read real outputs. This guards against reward hacking where the model finds degenerate high-reward behaviors.

## Reward Function Safety

11. **Save checkpoint before changing the reward function.** Changing the reward function mid-training resets all progress. The model was optimized for the OLD reward. Always save a checkpoint first and log the change as a baseline_reset in results.tsv.

12. **Reward stability > reward perfection.** Don't change the reward function frivolously. A mediocre but stable reward is better than a "perfect" reward that you keep tweaking. Codex changed the reward function 3x in one session, destroying all progress each time.

## Curriculum

13. **Start easy, scale difficulty gradually.**
    - When `eval_all_one_rate > 0.5` (model solving everything) → increase difficulty
    - When `eval_all_zero_rate > 0.5` (model solving nothing) → decrease difficulty
    - Never jump more than one difficulty level at a time
    - Starting too hard produces zero learning signal

## Model Selection

14. **Prefer non-thinking models for training.** Thinking models (e.g., INTELLECT-3.1) waste 80%+ of sequence length on think tokens. Use instruction-tuned non-thinking variants (e.g., Qwen3-30B-A3B-Instruct) unless your task specifically requires chain-of-thought.

## Async & Infrastructure

15. **max_async_level=1 for tighter sync on Prime.** Avoids off-policy lag and checkpoint wait escalation.

16. **W&B causes 8.8s event loop lag.** Disable during latency-sensitive training if needed.

17. **Use ThreadPoolExecutor for blocking API calls in async contexts.** Synchronous API calls in asyncio loops cause deadlock at step 2. Wrap with ThreadPoolExecutor + sync client.

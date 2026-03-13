"""
GRPO Training Loop for Tinker SDK

Autonomous post-training harness. The agent modifies reward.py, prompts.jsonl,
eval_prompts.jsonl, and the hyperparameters below. Based on tinker-cookbook's
rl_loop.py pattern (https://github.com/thinking-machines-lab/tinker-cookbook).

Run:   python train.py > run.log 2>&1
Check: grep '^eval_reward_mean:' run.log

Variable naming convention (from tinker-cookbook CONTRIBUTING.md):
    _P: Problem dimension (different prompts in a batch)
    _G: Group dimension (rollouts per problem for GRPO variance reduction)
    _T: Token/Time dimension (sequence positions)
    _D: Datum dimension (training examples after flattening P*G)

Dependencies: pip install tinker torch transformers
"""

import json
import logging
import sys
import time
from concurrent.futures import Future

import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

from reward import compute_reward

# ============================================================================
# MUTABLE HYPERPARAMETERS — Agent modifies these
# ============================================================================
MODEL = "Qwen/Qwen3-8B"                    # Base model to fine-tune
LORA_RANK = 32                              # LoRA rank (32 = cookbook default)
LEARNING_RATE = 4e-5                        # Constant LR (NEVER use cosine, see rules.md)
BATCH_SIZE = 128                            # Prompts per training batch (>= 128, see rules.md)
GROUP_SIZE = 32                             # Doubled for better advantage estimates on hard problems
MAX_TOKENS = 1024                           # Max response tokens — longer chains for hard MATH
TEMPERATURE = 1.0                           # Sampling temperature (1.0 for GRPO, see rules.md)
N_BATCHES = 100                             # Total training batches — long run for emergent behavior
SAVE_EVERY = 10                             # Checkpoint every N batches (0 = disabled)
LOSS_FN = "ppo"                             # PPO: clipped trust region for stable gradients

# Few-shot examples prepended to every prompt (set to [] for zero-shot)
# EXPERIMENT: Zero scaffolding — no few-shot, no system prompt, no CoT instructions.
# Testing if reasoning emerges from pure RL signal (DeepSeek-R1-Zero style).
FEW_SHOT = []

# System prompt (set to None to skip)
# EXPERIMENT: No system prompt — pure RL signal only.
SYSTEM_PROMPT = None

# ============================================================================
# FIXED — Do not modify unless you know what you're doing
# ============================================================================
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
LOSS_EXPLOSION_THRESHOLD = 100.0  # Abort if loss exceeds this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_prompts(path: str) -> list[dict]:
    """Load prompts from JSONL file. Each line: {"prompt": "...", "ground_truth": "..."}"""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def build_model_input(tokenizer, prompt_text: str) -> types.ModelInput:
    """Build a ModelInput from a prompt string using the model's chat template."""
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(FEW_SHOT)
    messages.append({"role": "user", "content": prompt_text})

    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=token_ids)])


def run_eval(
    sampling_client,
    tokenizer,
    eval_prompts: list[dict],
    sampling_params: types.SamplingParams,
) -> dict:
    """Run evaluation on held-out prompts. Returns metrics dict."""
    eval_rewards_P = []
    eval_all_one = 0
    eval_all_zero = 0
    sample_completions = []

    # Submit all eval sampling requests
    eval_futures: list[tuple[Future, dict]] = []
    for item in eval_prompts:
        model_input = build_model_input(tokenizer, item["prompt"])
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=GROUP_SIZE,
            sampling_params=sampling_params,
        )
        eval_futures.append((future, item))

    # Collect results
    for future, item in eval_futures:
        result = future.result()
        rewards_G = []
        for seq in result.sequences:
            text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            reward = compute_reward(text, item.get("ground_truth", ""))
            rewards_G.append(reward)
            if len(sample_completions) < 3:
                sample_completions.append((item["prompt"], text, reward))

        mean_r = sum(rewards_G) / len(rewards_G)
        eval_rewards_P.append(mean_r)
        if all(r >= 1.0 for r in rewards_G):
            eval_all_one += 1
        if all(r <= 0.0 for r in rewards_G):
            eval_all_zero += 1

    eval_reward_mean = sum(eval_rewards_P) / len(eval_rewards_P) if eval_rewards_P else 0.0
    eval_all_one_rate = eval_all_one / len(eval_prompts) if eval_prompts else 0.0
    eval_all_zero_rate = eval_all_zero / len(eval_prompts) if eval_prompts else 0.0

    return {
        "eval_reward_mean": eval_reward_mean,
        "eval_all_one_rate": eval_all_one_rate,
        "eval_all_zero_rate": eval_all_zero_rate,
        "sample_completions": sample_completions,
    }


def main():
    logger.info("=" * 60)
    logger.info("POSTTRAINER — GRPO Training Loop")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL} | LoRA rank: {LORA_RANK}")
    logger.info(f"LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Group: {GROUP_SIZE}")
    logger.info(f"Max tokens: {MAX_TOKENS} | Temperature: {TEMPERATURE}")
    logger.info(f"Loss function: {LOSS_FN}")

    # Load data
    train_prompts = load_prompts("prompts.jsonl")
    eval_prompts = load_prompts("eval_prompts.jsonl")
    logger.info(f"Loaded {len(train_prompts)} training prompts, {len(eval_prompts)} eval prompts")

    if len(train_prompts) < BATCH_SIZE:
        logger.warning(
            f"Only {len(train_prompts)} prompts but BATCH_SIZE={BATCH_SIZE}. "
            f"Will use {len(train_prompts)} prompts per batch."
        )

    # Setup Tinker clients
    logger.info("Initializing Tinker service client...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=MODEL, rank=LORA_RANK
    )
    logger.info("Training client created.")

    # Get tokenizer
    logger.info(f"Loading tokenizer for {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Training params (constant — NEVER use cosine scheduling)
    adam_params = types.AdamParams(
        learning_rate=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS
    )
    # Build stop sequences from tokenizer (ensures model stops at EOS, not MAX_TOKENS)
    stop_sequences = []
    if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)
    # Common chat template stop tokens
    for stop_tok in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if stop_tok not in stop_sequences:
            stop_sequences.append(stop_tok)

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_sequences if stop_sequences else None,
    )

    # Compute number of batches
    effective_batch = min(BATCH_SIZE, len(train_prompts))
    if N_BATCHES > 0:
        n_batches = N_BATCHES  # User-specified: allow multi-epoch (wraps around data)
    else:
        n_batches = max(1, len(train_prompts) // effective_batch)  # Auto: one epoch
    logger.info(f"Training for {n_batches} batches ({effective_batch} prompts/batch)")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    for batch_idx in range(n_batches):
        t_start = time.time()

        # Checkpoint
        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            state_path = training_client.save_state(name=f"batch_{batch_idx:06d}").result()
            logger.info(f"Checkpoint saved: {state_path}")

        # Get batch (wrap around if needed)
        batch_start = (batch_idx * effective_batch) % len(train_prompts)
        batch_end = batch_start + effective_batch
        if batch_end <= len(train_prompts):
            batch = train_prompts[batch_start:batch_end]
        else:
            batch = train_prompts[batch_start:] + train_prompts[:batch_end - len(train_prompts)]

        # Snapshot weights for on-policy sampling
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Submit all sampling requests (non-blocking)
        futures_P: list[Future] = []
        prompts_P: list[types.ModelInput] = []
        for item in batch:
            model_input = build_model_input(tokenizer, item["prompt"])
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=GROUP_SIZE,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        # Collect rewards and build training data
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        n_skipped = 0

        for future, prompt_input, item in zip(futures_P, prompts_P, batch):
            sample_result = future.result()
            ground_truth = item.get("ground_truth", "")

            rewards_G: list[float] = []
            tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []

            for seq in sample_result.sequences:
                if not seq.tokens:  # skip empty sequences (e.g., first token was stop)
                    continue
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                reward = compute_reward(text, ground_truth)
                rewards_G.append(reward)
                tokens_G_T.append(seq.tokens)
                assert seq.logprobs is not None, "Sampling must return logprobs"
                logprobs_G_T.append(seq.logprobs)

            # GRPO: center rewards within group
            if not rewards_G:  # all sequences were empty
                n_skipped += 1
                continue
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            # Skip uniform-reward groups (no learning signal)
            if all(a == 0.0 for a in advantages_G):
                n_skipped += 1
                continue

            # Build training datums
            ob_len = prompt_input.length - 1
            for tokens, logprobs, advantage in zip(tokens_G_T, logprobs_G_T, advantages_G):
                model_input = prompt_input.append(
                    types.EncodedTextChunk(tokens=tokens[:-1])
                )
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (
                    model_input.length - ob_len
                )

                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                ), (
                    f"Length mismatch: input={model_input.length}, "
                    f"targets={len(target_tokens)}, logprobs={len(padded_logprobs)}, "
                    f"advantages={len(padded_advantages)}"
                )

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(padded_logprobs)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(padded_advantages)
                        ),
                    },
                )
                datums_D.append(datum)

        # Training step (pipelined for same clock cycle — see rules.md)
        if datums_D:
            fwd_bwd_future = training_client.forward_backward(
                datums_D, loss_fn=LOSS_FN
            )
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            # Check for loss explosion
            metrics = fwd_bwd_result.metrics if fwd_bwd_result.metrics else {}
            if optim_result.metrics:
                metrics.update(optim_result.metrics)
            for key in ["loss", "train_loss", "total_loss"]:
                loss_val = metrics.get(key)
                if loss_val is not None and loss_val > LOSS_EXPLOSION_THRESHOLD:
                    logger.error(
                        f"LOSS EXPLOSION detected: {key}={loss_val:.4f} > {LOSS_EXPLOSION_THRESHOLD}. Aborting."
                    )
                    sys.exit(1)
        else:
            logger.warning(f"Batch {batch_idx}: No training datums (all groups uniform). Skipping.")

        # Log training metrics
        train_reward = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        elapsed = time.time() - t_start
        logger.info(
            f"batch {batch_idx + 1}/{n_batches} | "
            f"train_reward_mean: {train_reward:.4f} | "
            f"datums: {len(datums_D)} | skipped: {n_skipped}/{len(batch)} | "
            f"time: {elapsed:.1f}s"
        )
        # Grep-parsable training metric
        print(f"train_reward_mean: {train_reward:.6f}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    sampling_client = training_client.save_weights_and_get_sampling_client()
    eval_results = run_eval(sampling_client, tokenizer, eval_prompts, sampling_params)

    # Grep-parsable results (the agent greps for these)
    print(f"\neval_reward_mean: {eval_results['eval_reward_mean']:.6f}")
    print(f"eval_all_one_rate: {eval_results['eval_all_one_rate']:.6f}")
    print(f"eval_all_zero_rate: {eval_results['eval_all_zero_rate']:.6f}")

    # Sample completions for quality check (agent MUST read these — see rules.md #10)
    print("\n--- SAMPLE COMPLETIONS (read these to check for reward hacking) ---")
    for prompt, completion, reward in eval_results["sample_completions"]:
        print(f"PROMPT: {prompt}")
        print(f"COMPLETION: {completion}")
        print(f"REWARD: {reward}")
        print("---")

    # Save final checkpoint
    state_path = training_client.save_state(name="final").result()
    sampler_path = training_client.save_weights_for_sampler(name="final").result()
    logger.info(f"Final checkpoint (state): {state_path}")
    logger.info(f"Final checkpoint (sampler): {sampler_path}")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()

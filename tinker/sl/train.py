"""
Supervised Fine-Tuning (SFT) Loop for Tinker SDK

Autonomous post-training harness. The agent modifies data.jsonl and the
hyperparameters below. Based on tinker-cookbook's sl_loop.py pattern
(https://github.com/thinking-machines-lab/tinker-cookbook).

Run:   python train.py > run.log 2>&1
Check: grep '^eval_loss:' run.log

Dependencies: pip install tinker torch transformers
"""

import json
import logging
import math
import random
import sys
import time
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer

# ============================================================================
# MUTABLE HYPERPARAMETERS — Agent modifies these
# ============================================================================
MODEL = "Qwen/Qwen3-8B"                    # Base model to fine-tune
LORA_RANK = 32                              # LoRA rank
LEARNING_RATE = 3e-4                        # Starting LR (linear decay from this)
BATCH_SIZE = 128                            # Examples per training batch
MAX_LENGTH = 2048                           # Max sequence length (prompt + response)
N_EPOCHS = 10                               # Number of passes through the data
SAVE_EVERY = 20                             # Checkpoint every N batches (0 = disabled)
EVAL_SPLIT = 0.1                            # Fraction of data held out for eval

# System prompt prepended to all examples (set to None to skip)
SYSTEM_PROMPT = None

# ============================================================================
# FIXED — Do not modify unless you know what you're doing
# ============================================================================
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_data(path: str) -> list[dict]:
    """Load SFT data from JSONL. Each line: {"prompt": "...", "response": "..."}"""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_sft_datum(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> types.Datum | None:
    """Build a supervised learning Datum from a prompt-response pair."""
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})

    # Tokenize the full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    # Tokenize just the prompt (to find where response starts)
    prompt_messages = messages[:-1]  # exclude assistant response
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )

    # Truncate if needed
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]

    prompt_len = min(len(prompt_tokens), len(full_tokens))

    # Build target tokens and weights
    # Weight = 0 for prompt tokens (don't train on prompt)
    # Weight = 1 for response tokens (train on these)
    target_tokens = full_tokens[:]
    weights = [0.0] * prompt_len + [1.0] * (len(full_tokens) - prompt_len)

    # Skip if no response tokens
    if sum(weights) == 0:
        return None

    model_input = types.ModelInput(
        chunks=[types.EncodedTextChunk(tokens=full_tokens[:-1])]
    )
    target_tokens = full_tokens[1:]  # shifted by 1 for next-token prediction
    weights = weights[1:]  # align weights with targets

    # Ensure lengths match
    if model_input.length != len(target_tokens) or model_input.length != len(weights):
        return None

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
        },
    )


def compute_mean_nll(loss_fn_outputs: list, datums: list[types.Datum]) -> float:
    """Compute mean negative log-likelihood from forward_backward outputs."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0
    for output, datum in zip(loss_fn_outputs, datums):
        logprobs = output["logprobs"] if isinstance(output, dict) else output.logprobs
        weights = datum.loss_fn_inputs["weights"]
        if isinstance(logprobs, TensorData):
            logprobs = logprobs.to_torch()
        if isinstance(weights, TensorData):
            weights = weights.to_torch()
        logprobs_t = torch.as_tensor(logprobs, dtype=torch.float32)
        weights_t = torch.as_tensor(weights, dtype=torch.float32)
        total_weighted_logprobs += logprobs_t.dot(weights_t).item()
        total_weights += weights_t.sum().item()
    return float(-total_weighted_logprobs / total_weights) if total_weights > 0 else 0.0


def main():
    logger.info("=" * 60)
    logger.info("POSTTRAINER — SFT Training Loop")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL} | LoRA rank: {LORA_RANK}")
    logger.info(f"LR: {LEARNING_RATE} (linear decay) | Batch: {BATCH_SIZE}")
    logger.info(f"Max length: {MAX_LENGTH} | Epochs: {N_EPOCHS}")

    # Load data
    all_data = load_data("data.jsonl")
    logger.info(f"Loaded {len(all_data)} examples from data.jsonl")

    if len(all_data) == 0:
        logger.error("No data found in data.jsonl. Exiting.")
        sys.exit(1)

    # Shuffle before splitting to avoid biased eval set
    random.seed(42)
    all_data_shuffled = all_data[:]
    random.shuffle(all_data_shuffled)

    # Split into train/eval
    n_eval = max(1, int(len(all_data_shuffled) * EVAL_SPLIT))
    eval_data = all_data_shuffled[:n_eval]
    train_data = all_data_shuffled[n_eval:]
    if not train_data:
        logger.error("No training data after eval split. Add more examples to data.jsonl.")
        sys.exit(1)
    logger.info(f"Train: {len(train_data)} examples | Eval: {len(eval_data)} examples")

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

    # Compute total steps for LR schedule
    effective_batch = min(BATCH_SIZE, len(train_data))
    n_batches_per_epoch = max(1, math.ceil(len(train_data) / effective_batch))
    total_steps = n_batches_per_epoch * N_EPOCHS
    logger.info(f"Training for {total_steps} steps ({n_batches_per_epoch} batches/epoch x {N_EPOCHS} epochs)")

    # Pre-tokenize eval data
    logger.info("Tokenizing eval data...")
    eval_datums = []
    for item in eval_data:
        datum = build_sft_datum(tokenizer, item["prompt"], item["response"], MAX_LENGTH)
        if datum is not None:
            eval_datums.append(datum)
    logger.info(f"Eval datums: {len(eval_datums)}")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    global_step = 0

    for epoch in range(N_EPOCHS):
        logger.info(f"--- Epoch {epoch + 1}/{N_EPOCHS} ---")

        # Shuffle training data each epoch (deterministic per epoch for reproducibility)
        shuffled_train = train_data[:]
        random.seed(42 + epoch)
        random.shuffle(shuffled_train)

        for batch_idx in range(n_batches_per_epoch):
            t_start = time.time()

            # Checkpoint
            if SAVE_EVERY > 0 and global_step > 0 and global_step % SAVE_EVERY == 0:
                state_path = training_client.save_state(name=f"step_{global_step:06d}").result()
                logger.info(f"Checkpoint saved: {state_path}")

            # Linear LR decay
            lr_mult = max(0.0, 1.0 - global_step / total_steps)
            current_lr = LEARNING_RATE * lr_mult
            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                eps=ADAM_EPS,
            )

            # Get batch
            batch_start = batch_idx * effective_batch
            batch_end = min(batch_start + effective_batch, len(shuffled_train))
            batch_items = shuffled_train[batch_start:batch_end]

            # Build datums
            datums = []
            for item in batch_items:
                datum = build_sft_datum(
                    tokenizer, item["prompt"], item["response"], MAX_LENGTH
                )
                if datum is not None:
                    datums.append(datum)

            if not datums:
                logger.warning(f"Step {global_step}: No valid datums. Skipping.")
                continue

            # Training step (pipelined)
            fwd_bwd_future = training_client.forward_backward(
                datums, loss_fn="cross_entropy"
            )
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            elapsed = time.time() - t_start
            logger.info(
                f"step {global_step + 1}/{total_steps} | "
                f"lr: {current_lr:.2e} | "
                f"datums: {len(datums)} | "
                f"time: {elapsed:.1f}s"
            )
            print(f"train_step: {global_step}")

            global_step += 1

        # Per-epoch eval to detect overfitting
        if eval_datums:
            epoch_eval = training_client.forward(eval_datums, loss_fn="cross_entropy").result()
            epoch_eval_loss = compute_mean_nll(epoch_eval.loss_fn_outputs, eval_datums)
            print(f"epoch_{epoch + 1}_eval_loss: {epoch_eval_loss:.6f}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    if eval_datums:
        # Run eval forward pass (no backward, no optimizer step)
        eval_fwd_result = training_client.forward(eval_datums, loss_fn="cross_entropy").result()

        # Compute eval loss from logprobs
        eval_loss = compute_mean_nll(eval_fwd_result.loss_fn_outputs, eval_datums)

        print(f"\neval_loss: {eval_loss:.6f}")
        print(f"eval_examples: {len(eval_datums)}")
    else:
        logger.warning("No eval datums available.")
        print("\neval_loss: nan")

    # Print sample data for quality check
    print("\n--- SAMPLE DATA (verify these are high quality) ---")
    for item in eval_data[:3]:
        print(f"PROMPT: {item['prompt'][:200]}")
        print(f"RESPONSE: {item['response'][:200]}")
        print("---")

    # Save final checkpoint
    state_path = training_client.save_state(name="final").result()
    sampler_path = training_client.save_weights_for_sampler(name="final").result()
    logger.info(f"Final checkpoint (state): {state_path}")
    logger.info(f"Final checkpoint (sampler): {sampler_path}")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()

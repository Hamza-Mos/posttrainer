"""
Reward function for GRPO training.

CONTRACT:
    compute_reward(completion: str, ground_truth: str) -> float
    - Must return a float (typically 0.0 to 1.0)
    - Must be deterministic
    - Must NEVER crash (catch all exceptions, return 0.0)
    - Prefer partial credit over binary when possible

The agent modifies this file to match the task described in program.md.
This starter implements arithmetic answer checking as an example.
"""

import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Check if the completion contains the correct arithmetic answer.

    Returns:
        1.0 if correct answer found
        0.0 otherwise
    """
    try:
        # Extract first number from completion (handles "42!", "(42)", "**42**", etc.)
        match = _NUM_RE.search(completion.replace(",", ""))
        if not match:
            return 0.0

        predicted = float(match.group())
        expected = float(ground_truth.strip().replace(",", ""))

        return 1.0 if predicted == expected else 0.0
    except Exception:
        return 0.0

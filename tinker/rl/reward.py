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


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Check if the completion contains the correct arithmetic answer.

    Returns:
        1.0 if correct answer found
        0.0 otherwise
    """
    try:
        # Extract the answer from the completion (take first number-like token)
        completion = completion.strip()
        # Try to find a number in the completion
        tokens = completion.replace(",", "").split()
        predicted = None
        for token in tokens:
            try:
                predicted = int(token)
                break
            except ValueError:
                try:
                    predicted = float(token)
                    break
                except ValueError:
                    continue

        if predicted is None:
            return 0.0

        expected = float(ground_truth.strip().replace(",", ""))

        # Exact match for integers
        if float(predicted) == expected:
            return 1.0

        return 0.0
    except Exception:
        return 0.0

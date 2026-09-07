"""
inference.py
============
Prediction utilities for DiSMEC PTQ.
"""


def save_predictions(predictions, path, num_samples, topk):
    """
    Save top-k predictions in XMC format.

    Args:
        predictions: list of strings, each = space-separated label:score pairs
        path:        output file path
        num_samples: number of test samples
        topk:        k value
    """
    with open(path, 'w') as f:
        f.write(f"{num_samples} {topk}\n")
        for line in predictions:
            f.write(line + "\n")

"""
data_loader.py
==============
Unified sparse data loader for DiSMEC PTQ.
Works for all datasets: EURLex, Wiki10, AmazonCat-13K,
Amazon-670K, Delicious-200K, Amazon-3M.

Key design:
- Returns scipy CSR matrix (never dense) — safe for large feature spaces
- Handles 1-indexed features (DiSMEC format) → 0-indexed internally
- load_sparse_weights handles bias column correctly
"""

import numpy as np
from scipy.sparse import csr_matrix


def load_sparse_data(path):
    """
    Load XMC test/train data in sparse CSR format.

    Args:
        path: path to LibSVM-format file with header line:
              N_samples  N_features  N_labels

    Returns:
        X:            scipy CSR matrix (N_samples x N_features), float32
        num_features: number of features (from header)
    """
    rows_data    = []
    rows_indices = []
    rows_indptr  = [0]
    num_features = 0

    with open(path) as f:
        header = f.readline().strip().split()
        num_features = int(header[1])

        for line in f:
            for item in line.split():
                if ':' not in item:
                    continue
                idx_str, val_str = item.split(':', 1)
                idx = int(idx_str) - 1   # 1-indexed → 0-indexed
                if 0 <= idx < num_features:
                    rows_indices.append(idx)
                    rows_data.append(float(val_str))
            rows_indptr.append(len(rows_indices))

    X = csr_matrix(
        (rows_data, rows_indices, rows_indptr),
        shape=(len(rows_indptr) - 1, num_features),
        dtype=np.float32,
    )
    return X, num_features


def load_sparse_weights(path, num_features):
    """
    Load DiSMEC weight file in sparse CSR format.
    Each row = one label's linear classifier weights.

    Args:
        path:         path to DiSMEC SparseTXT weights file
        num_features: total feature count including bias column

    Returns:
        W: scipy CSR matrix (N_labels x num_features), float32
    """
    data    = []
    indices = []
    indptr  = [0]

    with open(path) as f:
        for line in f:
            for item in line.split():
                if ':' not in item:
                    continue
                idx_str, val_str = item.split(':', 1)
                idx = int(idx_str) - 1   # 1-indexed → 0-indexed
                if 0 <= idx < num_features:
                    indices.append(idx)
                    data.append(float(val_str))
            indptr.append(len(indices))

    W = csr_matrix(
        (data, indices, indptr),
        shape=(len(indptr) - 1, num_features),
        dtype=np.float32,
    )
    return W

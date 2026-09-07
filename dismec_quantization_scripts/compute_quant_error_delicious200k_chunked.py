"""
compute_quant_error_delicious200k_chunked.py
================================================
Streaming/chunked version. Two previous fixes (vectorized numpy instead
of Python lists, then process isolation via a fresh subprocess per
dataset) BOTH still hit the same ~140GB OOM ceiling on delicious200k's
full 205,443 rows. That rules out "leaked memory across configs/datasets"
as the cause -- it means even ONE config's full FP32 + dequantized
matrices, held simultaneously, exceed available RAM for this dataset
specifically (782,586 features is large enough that even efficient
numpy arrays don't fit at full scale).

FIX: process rows in chunks (default 20,000 rows at a time). For each
chunk:
  - read only that chunk's lines from the FP32 weights file
  - slice only that chunk's nonzeros from the NPZ arrays (via indptr)
  - compute SQNR/RMSE/cosine-sim contributions for just this chunk,
    accumulate into running totals (never hold the full matrix)
  - accumulate this chunk's rows into a small, pre-sized rank-change
    score buffer (L x n_queries, ~40MB total for delicious200k -- this
    was NEVER the expensive part; the full sparse matrices were)

This trades some speed (more, smaller csr_matrix constructions) for a
hard bound on peak memory: at any moment, only chunk_size rows of FP32
and dequant data are live, regardless of total dataset size.

Usage:
    python3 compute_quant_error_delicious200k_chunked.py \
        --chunk-size 20000
    (all other args default to the same paths as before)
"""

import numpy as np
import os, sys, json, time, argparse, gc
from scipy.sparse import csr_matrix

ALL_CONFIGS = [
    'int8_row_sym',    'int8_row_asym',    'int8_row_sym_clip',
    'int8_group_sym',  'int8_group_sym_clip', 'int8_row_sym_clip_mixed',
    'int8_group_sym_clip_act', 'int8_group_sym_clip_act_intinfer',
    'int8_group_sym_clip_act_intinfer_bias',
    'int4_row_sym',    'int4_row_asym',    'int4_row_sym_clip',
    'int4_group_sym',  'int4_group_sym_clip', 'int4_row_sym_clip_mixed',
    'int4_group_sym_clip_act', 'int4_group_sym_clip_act_intinfer',
    'int4_group_sym_clip_act_intinfer_bias',
]

DEFAULT_WEIGHTS    = "/DATA2/rudra1/dismec_quantization/dismecpp/delicious_bow_baseline.model.weights-0-205442"
DEFAULT_MODELS_DIR = "/DATA2/rudra1/dismec_quantization/dismecpp/results/delicious200k/models"
DEFAULT_OUT_JSON    = "/DATA2/rudra1/dismec_quantization/dismecpp/results/quant_error_dismec_delicious200k.json"
DEFAULT_MAX_ROWS   = 205443
DEFAULT_CHUNK_SIZE = 20000


def count_fp32_rows_and_features(path, max_rows):
    """First pass: just get row count and feature dimension, cheap."""
    nrows = 0
    num_features = 0
    with open(path) as f:
        for ri, line in enumerate(f):
            if max_rows and ri >= max_rows:
                break
            nrows = ri + 1
            for part in line.strip().split():
                if ':' not in part:
                    continue
                fid = int(part.split(':', 1)[0])
                if fid >= 0 and fid + 1 > num_features:
                    num_features = fid + 1
    return nrows, num_features


def load_fp32_chunk(path, row_start, row_end, num_features):
    """Reads only lines [row_start, row_end) from the FP32 file."""
    rows, cols, vals = [], [], []
    with open(path) as f:
        for ri, line in enumerate(f):
            if ri < row_start:
                continue
            if ri >= row_end:
                break
            for part in line.strip().split():
                if ':' not in part:
                    continue
                fid_str, val_str = part.split(':', 1)
                fid = int(fid_str)
                if fid < 0:
                    continue
                rows.append(ri - row_start)
                cols.append(fid)
                vals.append(float(val_str))
    n = row_end - row_start
    return csr_matrix((np.array(vals, dtype=np.float32),
                        (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
                       shape=(n, num_features), dtype=np.float32)


def load_dequant_chunk(npz_data, row_start, row_end, num_features):
    """Slices only [row_start,row_end) rows' nonzeros from already-loaded
    npz arrays (indices/indptr/data/scales/zero_points), for delicious200k
    (0-based, no index_offset)."""
    indices, indptr, data_raw, symmetric, group_size, scales, zero_points = npz_data

    s_ptr = int(indptr[row_start])
    e_ptr = int(indptr[row_end])
    if e_ptr == s_ptr:
        return csr_matrix((row_end - row_start, num_features), dtype=np.float32)

    chunk_indices = indices[s_ptr:e_ptr]
    if not symmetric:
        chunk_data = (data_raw[s_ptr:e_ptr].astype(np.int32) & 0xFF).astype(np.float64)
    else:
        chunk_data = data_raw[s_ptr:e_ptr].astype(np.float64)

    nnz = e_ptr - s_ptr
    local_indptr = indptr[row_start:row_end + 1] - s_ptr
    dq_rows = np.searchsorted(local_indptr[1:], np.arange(nnz), side='right')
    dq_cols = chunk_indices.astype(np.int64)

    if group_size <= 0:
        sc_per_nz = scales[row_start:row_end][dq_rows].astype(np.float64)
        zp_per_nz = (zero_points[row_start:row_end][dq_rows].astype(np.float64)
                     if zero_points is not None else 0.0)
    else:
        # Need global group offsets -- recompute group_ptr once outside
        # and pass in group id per absolute row; simpler: fall back to
        # per-row-in-chunk group math using ABSOLUTE row indices.
        row_lengths_full = np.diff(indptr)
        groups_per_row_full = (row_lengths_full + group_size - 1) // group_size
        group_ptr_full = np.zeros(len(indptr), dtype=np.int64)
        np.cumsum(groups_per_row_full, out=group_ptr_full[1:])

        abs_rows = dq_rows + row_start
        row_start_for_nz = indptr[abs_rows]
        pos_in_row = (np.arange(nnz) + s_ptr) - row_start_for_nz
        group_in_row = pos_in_row // group_size
        scale_idx = group_ptr_full[abs_rows] + group_in_row

        sc_per_nz = scales[scale_idx].astype(np.float64)
        zp_per_nz = (zero_points[scale_idx].astype(np.float64)
                     if zero_points is not None else 0.0)

    dq_vals = sc_per_nz * chunk_data if symmetric else (chunk_data * sc_per_nz) + zp_per_nz

    n = row_end - row_start
    return csr_matrix((dq_vals.astype(np.float32), (dq_rows.astype(np.int64), dq_cols)),
                       shape=(n, num_features), dtype=np.float32)


def run_config_chunked(config, npz_path, weights_path, L, num_features,
                        chunk_size, n_queries=50, k=5):
    with np.load(npz_path, allow_pickle=True) as npz:
        indices    = np.array(npz['indices'])
        indptr     = np.array(npz['indptr'])
        symmetric  = int(npz['symmetric'][0])
        group_size = int(npz['group_size'][0])
        data_raw   = np.array(npz['data'])
        scales     = np.array(npz['scales'])
        zero_points = np.array(npz['zero_points']) if 'zero_points' in npz else None

    npz_data = (indices, indptr, data_raw, symmetric, group_size, scales, zero_points)

    # Fixed random query projection (same seed as before, for comparability)
    np.random.seed(42)
    X = np.random.randn(num_features, n_queries).astype(np.float32)
    X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-8

    sum_signal_sq = 0.0
    sum_noise_sq = 0.0
    cos_sim_sum = 0.0
    cos_sim_n = 0

    Sf_all = np.zeros((L, n_queries), dtype=np.float32)
    Sq_all = np.zeros((L, n_queries), dtype=np.float32)

    for row_start in range(0, L, chunk_size):
        row_end = min(L, row_start + chunk_size)

        Af = load_fp32_chunk(weights_path, row_start, row_end, num_features)
        Aq = load_dequant_chunk(npz_data, row_start, row_end, num_features)

        Af_m = Af.multiply(Aq != 0)
        Aq_m = Aq.multiply(Af != 0)
        err = Af_m - Aq_m

        sum_signal_sq += float(np.sum(Af_m.data.astype(np.float64) ** 2))
        sum_noise_sq += float(np.sum(err.data.astype(np.float64) ** 2))

        nf = np.sqrt(np.array(Af_m.power(2).sum(axis=1))).flatten() + 1e-8
        nq = np.sqrt(np.array(Aq_m.power(2).sum(axis=1))).flatten() + 1e-8
        dots = Af_m.multiply(Aq_m).sum(axis=1).A1
        cos_sim_sum += float(np.sum(dots / (nf * nq)))
        cos_sim_n += (row_end - row_start)

        Sf_all[row_start:row_end, :] = (Af @ X)
        Sq_all[row_start:row_end, :] = (Aq @ X)

        del Af, Aq, Af_m, Aq_m, err
        gc.collect()

    nr, nc = L, num_features
    sp_ = sum_signal_sq / (nr * nc)
    np_ = sum_noise_sq / (nr * nc)
    if sp_ == 0 or np_ == 0:
        sqnr = 0.0
    else:
        sqnr = float(10 * np.log10(sp_ / np_)) if np_ > 0 else float('inf')
    rmse = float(np.sqrt(np_))
    cos_sim = cos_sim_sum / cos_sim_n if cos_sim_n else 0.0

    kk = min(k, L)
    rc = 0
    for qi in range(n_queries):
        top_f = set(np.argpartition(Sf_all[:, qi], -kk)[-kk:])
        top_q = set(np.argpartition(Sq_all[:, qi], -kk)[-kk:])
        if top_f != top_q:
            rc += 1
    rank_change_pct = float(rc / n_queries * 100)

    return {
        'sqnr_db': round(sqnr, 3),
        'rmse': round(rmse, 6),
        'cosine_sim': round(cos_sim, 6),
        'rank_change_pct': round(rank_change_pct, 1),
        'n_rows': L,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--models-dir', default=DEFAULT_MODELS_DIR)
    ap.add_argument('--out-json', default=DEFAULT_OUT_JSON)
    ap.add_argument('--max-rows', type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument('--configs', nargs='+', default=ALL_CONFIGS)
    args = ap.parse_args()

    print(f"Counting rows/features (pass 1)...")
    L, num_features = count_fp32_rows_and_features(args.weights, args.max_rows)
    print(f"L={L} num_features={num_features} chunk_size={args.chunk_size}")

    results = {}
    print(f"{'Config':<45} {'SQNR(dB)':>9} {'RMSE':>10} {'CosSim':>9} {'RankChg%':>10}")
    print('-' * 86)

    for config in args.configs:
        npz_path = os.path.join(args.models_dir, f"{config}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {config:<41} — NPZ not found")
            results[config] = None
            continue

        t0 = time.time()
        m = run_config_chunked(config, npz_path, args.weights, L, num_features,
                                args.chunk_size)
        results[config] = m
        elapsed = time.time() - t0
        print(f"  {config:<43} {m['sqnr_db']:>9.2f} {m['rmse']:>10.6f} "
              f"{m['cosine_sim']:>9.6f} {m['rank_change_pct']:>9.1f}%  ({elapsed:.1f}s)")

        gc.collect()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump({'delicious200k': results}, f, indent=2, default=str)
    print(f"\nSaved: {args.out_json}")


if __name__ == '__main__':
    main()

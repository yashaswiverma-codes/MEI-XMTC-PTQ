"""
compute_quant_error.py  (memory-optimized, multi-dataset)
============================================================
Same logic as the previous version, with two fixes applied:

  1. MEMORY: dequantize_npz() and load_fp32_weights() rewritten to use
     vectorized numpy / pre-sized arrays instead of per-element Python
     list appends. This was causing OOM kills on large datasets
     (confirmed on delicious200k; the same risk applies to amazon670k
     and any other large dataset run through this generic script).

  2. INDEX OFFSET IS EXPLICIT, NOT ASSUMED: added --one-based / --zero-based
     (via datasets_config.json's "one_based" field, or --one-based CLI flag
     in single-dataset mode). Per run_pipeline.sh's ONE_BASED settings:
       eurlex, wiki10, amazoncat13k, amazon670k -> one_based = true
       delicious200k, amazon3m                   -> one_based = false
     Getting this wrong silently shifts every column index by 1 and
     corrupts every SQNR/RMSE/cosine-sim number without erroring —
     ALWAYS check the printed INDEX ALIGNMENT CHECK before trusting output.

Usage (single dataset):
  python3 compute_quant_error.py \
      --dataset    eurlex \
      --weights    /path/to/model.weights-0-3992 \
      --models-dir /path/to/results/eurlex/models \
      --out-json   /path/to/results/quant_error_eurlex.json \
      --one-based \
      --max-rows   0

Usage (all datasets):
  python3 compute_quant_error.py --datasets-config datasets_config.json
  (datasets_config.json: each entry needs "weights", "models_dir",
   "out_json", "one_based" (bool), and optionally "max_rows" (0=full))
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


# ============================================================
# LOAD FP32 WEIGHTS — two-pass, pre-allocated numpy arrays
# Indices kept exactly as they appear in the file (no shift here —
# any 0/1-based alignment happens on the NPZ side, see dequantize_npz)
# ============================================================
def load_fp32_weights(path, max_rows=0):
    print(f"  Loading FP32: {path}")

    total_nnz = 0
    nrows = 0
    num_features = 0
    with open(path) as f:
        for ri, line in enumerate(f):
            if max_rows and max_rows > 0 and ri >= max_rows:
                break
            nrows = ri + 1
            for part in line.strip().split():
                if ':' not in part:
                    continue
                fid = int(part.split(':', 1)[0])
                if fid < 0:
                    continue
                total_nnz += 1
                if fid + 1 > num_features:
                    num_features = fid + 1

    print(f"  Pass 1: nrows={nrows} num_features={num_features} nnz={total_nnz}")

    rows = np.empty(total_nnz, dtype=np.int64)
    cols = np.empty(total_nnz, dtype=np.int64)
    vals = np.empty(total_nnz, dtype=np.float32)

    idx = 0
    with open(path) as f:
        for ri, line in enumerate(f):
            if max_rows and max_rows > 0 and ri >= max_rows:
                break
            for part in line.strip().split():
                if ':' not in part:
                    continue
                fid_str, val_str = part.split(':', 1)
                fid = int(fid_str)
                if fid < 0:
                    continue
                rows[idx] = ri
                cols[idx] = fid
                vals[idx] = float(val_str)
                idx += 1

    W = csr_matrix((vals, (rows, cols)),
                   shape=(nrows, num_features), dtype=np.float32)
    print(f"  FP32 shape: {W.shape}  nnz={W.nnz}")
    return W


# ============================================================
# DEQUANTIZE NPZ — fully vectorized, no per-element Python loop
# index_offset: +1 for one_based=True datasets (matches FP32's 1-based
# indices), +0 for one_based=False (delicious200k, amazon3m style)
# ============================================================
def dequantize_npz(npz_path, max_rows=0, index_offset=1):
    # FIX: use np.load() as a context manager so the NpzFile is closed
    # and its decompressed-array cache released immediately after this
    # function returns, instead of lingering until GC gets to it. Over
    # 18 configs x 5 datasets (90+ calls), un-closed NpzFile objects
    # were very likely accumulating live in memory across the whole
    # run, which matches the observed symptom: RSS climbing steadily
    # dataset-over-dataset rather than spiking on one large dataset.
    with np.load(npz_path, allow_pickle=True) as npz:
        indices    = np.array(npz['indices'])
        indptr     = np.array(npz['indptr'])
        shape      = tuple(npz['shape'])
        symmetric  = int(npz['symmetric'][0])
        group_size = int(npz['group_size'][0])

        L_total = len(indptr) - 1
        L = min(L_total, max_rows) if (max_rows and max_rows > 0) else L_total

        end_ptr = int(indptr[L])
        indices = indices[:end_ptr]

        if not symmetric:
            data = (np.array(npz['data'][:end_ptr]).astype(np.int32) & 0xFF).astype(np.float64)
        else:
            data = np.array(npz['data'][:end_ptr]).astype(np.float64)

        scales      = np.array(npz['scales'])
        zero_points = np.array(npz['zero_points']) if 'zero_points' in npz else None
    # <-- npz is now closed here; everything needed was copied out above
    # via np.array(...), so no lazy references back into the closed
    # file remain.

    nnz = end_ptr
    nc = shape[1] + index_offset
    if nnz == 0:
        return csr_matrix((L, nc), dtype=np.float32)

    dq_rows = np.searchsorted(indptr[1:L + 1], np.arange(nnz), side='right')
    dq_cols = indices.astype(np.int64) + index_offset

    if group_size <= 0:
        sc_per_nz = scales[dq_rows].astype(np.float64)
        zp_per_nz = zero_points[dq_rows].astype(np.float64) if zero_points is not None else 0.0
    else:
        row_lengths = np.diff(indptr[:L + 1])
        row_start_for_nz = indptr[dq_rows]
        pos_in_row = np.arange(nnz) - row_start_for_nz
        group_in_row = pos_in_row // group_size

        groups_per_row = (row_lengths + group_size - 1) // group_size
        group_ptr = np.zeros(L + 1, dtype=np.int64)
        np.cumsum(groups_per_row, out=group_ptr[1:])

        scale_idx = group_ptr[dq_rows] + group_in_row
        sc_per_nz = scales[scale_idx].astype(np.float64)
        zp_per_nz = zero_points[scale_idx].astype(np.float64) if zero_points is not None else 0.0

    dq_vals = sc_per_nz * data if symmetric else (data * sc_per_nz) + zp_per_nz

    W_dq = csr_matrix(
        (dq_vals.astype(np.float32), (dq_rows.astype(np.int64), dq_cols)),
        shape=(L, nc), dtype=np.float32)
    return W_dq


# ============================================================
# COMPUTE METRICS (sparse-aligned) — unchanged
# ============================================================
def compute_metrics(W_fp32, W_dq):
    nr = min(W_fp32.shape[0], W_dq.shape[0])
    nc = min(W_fp32.shape[1], W_dq.shape[1])

    Af = W_fp32[:nr, :nc]
    Aq = W_dq[:nr,  :nc]

    Af_m = Af.multiply(Aq != 0)
    Aq_m = Aq.multiply(Af != 0)
    err  = Af_m - Aq_m

    sp_ = float(np.sum(Af_m.data ** 2)) / (nr * nc)
    np_ = float(np.sum(err.data   ** 2)) / (nr * nc)

    if sp_ == 0 or np_ == 0:
        sqnr    = 0.0
        rmse    = 0.0
        cos_sim = 0.0
    else:
        sqnr    = float(10 * np.log10(sp_ / np_)) if np_ > 0 else float('inf')
        rmse    = float(np.sqrt(np_))
        nf      = np.sqrt(np.array(Af_m.power(2).sum(axis=1))).flatten() + 1e-8
        nq      = np.sqrt(np.array(Aq_m.power(2).sum(axis=1))).flatten() + 1e-8
        dots    = Af_m.multiply(Aq_m).sum(axis=1).A1
        cos_sim = float(np.mean(dots / (nf * nq)))

    np.random.seed(42)
    n_q = 50
    X   = np.random.randn(nc, n_q).astype(np.float32)
    X  /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-8
    Sf  = Af @ X
    Sq  = Aq @ X
    k   = min(5, nr)
    rc  = sum(1 for qi in range(n_q)
              if set(np.argpartition(Sf[:, qi], -k)[-k:]) !=
                 set(np.argpartition(Sq[:, qi], -k)[-k:]))
    rank_change_pct = float(rc / n_q * 100)

    return {
        'sqnr_db':         round(sqnr, 3),
        'rmse':            round(rmse, 6),
        'cosine_sim':      round(cos_sim, 6),
        'rank_change_pct': round(rank_change_pct, 1),
        'n_rows':          nr,
    }


# ============================================================
# SINGLE-DATASET RUN
# ============================================================
def run_dataset(dataset, weights, models_dir, out_json, max_rows, configs, one_based=True):
    print(f"\n{'='*60}")
    print(f"DiSMEC Quant Error — {dataset.upper()}  (one_based={one_based}, max_rows={max_rows or 'FULL'})")
    print(f"{'='*60}")

    index_offset = 1 if one_based else 0
    W_fp32 = load_fp32_weights(weights, max_rows=max_rows)

    first_npz = os.path.join(models_dir, f"{configs[0]}.npz")
    if os.path.exists(first_npz):
        W_test = dequantize_npz(first_npz, max_rows=1, index_offset=index_offset)
        print(f"\n--- INDEX ALIGNMENT CHECK ---")
        print(f"  FP32 row-0 indices:   {list(W_fp32[0].indices[:8])}")
        print(f"  Dequant row-0 indices:{list(W_test[0].indices[:8])}")
        print(f"  (Should match for SQNR to be meaningful — if not, flip one_based)")
        print(f"-----------------------------\n")

    results = {}
    print(f"{'Config':<45} {'SQNR(dB)':>9} {'RMSE':>10} "
          f"{'CosSim':>9} {'RankChg%':>10}")
    print('-' * 86)

    for config in configs:
        npz_path = os.path.join(models_dir, f"{config}.npz")
        if not os.path.exists(npz_path):
            print(f"  SKIP {config:<41} — NPZ not found")
            results[config] = None
            continue

        t0   = time.time()
        W_dq = dequantize_npz(npz_path, max_rows=max_rows, index_offset=index_offset)
        m    = compute_metrics(W_fp32, W_dq)
        results[config] = m
        elapsed = time.time() - t0

        print(f"  {config:<43} {m['sqnr_db']:>9.2f} {m['rmse']:>10.6f} "
              f"{m['cosine_sim']:>9.6f} {m['rank_change_pct']:>9.1f}%"
              f"  ({elapsed:.1f}s)")

        del W_dq
        gc.collect()

    del W_fp32
    gc.collect()

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({dataset: results}, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--weights')
    parser.add_argument('--models-dir')
    parser.add_argument('--out-json')
    parser.add_argument('--max-rows', type=int, default=0,
                        help='0 = full dataset (default). Was 10000 in the old '
                             'version, which silently truncated large datasets.')
    parser.add_argument('--one-based', dest='one_based', action='store_true', default=True,
                        help='FP32 weights file uses 1-based feature indices (default)')
    parser.add_argument('--zero-based', dest='one_based', action='store_false',
                        help='FP32 weights file uses 0-based feature indices '
                             '(use this for delicious200k, amazon3m)')
    parser.add_argument('--configs', nargs='+', default=ALL_CONFIGS)
    parser.add_argument('--datasets-config',
                        help='JSON file listing all datasets to run (all-datasets mode)')
    parser.add_argument('--combined-out-json')
    args = parser.parse_args()

    if args.datasets_config:
        with open(args.datasets_config) as f:
            ds_cfg = json.load(f)

        combined = {}
        for dataset, cfg in ds_cfg.items():
            configs = cfg.get('configs', args.configs)
            res = run_dataset(
                dataset    = dataset,
                weights    = cfg['weights'],
                models_dir = cfg['models_dir'],
                out_json   = cfg['out_json'],
                max_rows   = cfg.get('max_rows', 0),
                configs    = configs,
                one_based  = cfg.get('one_based', True),
            )
            combined[dataset] = res

        if args.combined_out_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.combined_out_json)), exist_ok=True)
            with open(args.combined_out_json, 'w') as f:
                json.dump(combined, f, indent=2, default=str)
            print(f"\nSaved combined results: {args.combined_out_json}")
        return

    missing = [n for n in ('dataset', 'weights', 'models_dir', 'out_json')
               if getattr(args, n) is None]
    if missing:
        parser.error("Either provide --datasets-config, OR all of "
                     "--dataset --weights --models-dir --out-json.")

    run_dataset(
        dataset    = args.dataset,
        weights    = args.weights,
        models_dir = args.models_dir,
        out_json   = args.out_json,
        max_rows   = args.max_rows,
        configs    = args.configs,
        one_based  = args.one_based,
    )


if __name__ == '__main__':
    main()

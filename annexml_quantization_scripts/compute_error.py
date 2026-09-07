"""
compute_quant_error.py  (AnnexML, production, multi-dataset)

Computes standard quantization error metrics for every (dataset, config,
learner, cluster) combination, then aggregates to a per-dataset,
per-config summary.

Standard metrics reported:
  SQNR (dB)        — Signal-to-Quantization-Noise Ratio, primary metric
  RMSE             — reconstruction error
  Cosine Similarity— directional preservation (key for ANN retrieval)
  Rank Change %    — how often top-10 nearest neighbors change (aggregated,
                      computed once per dataset/config over the full model)

Key fixes vs the single-dataset / learner0-only version:
  1. All 5 datasets (eurlex, wiki10, amazoncat13k, amazon670k, delicious200k)
  2. Iterates over EVERY learner and EVERY cluster, not just learner0_cluster0
  3. Correct FP32 reference for learner{L}_cluster{C}.npz:
         model.learners[L].embeddings   (the FULL, unsliced embeddings
     matrix for that learner). Confirmed empirically: NPZ row counts are
     identical across every cluster belonging to a learner and match
     learner.num_data, so "cluster" in the filename is NOT a row-subset
     of the embeddings — every cluster file for a learner shares the same
     full FP32 reference. See resolve_fp32_cluster().
  4. Each learner{L}_cluster{C}.npz is treated as an independently
     quantized copy of that learner's FULL embeddings matrix (not a
     duplicate computation) — e.g. if inference routes different
     queries to different clusters, each cluster's quantized copy is
     real deployed signal. All cluster files are scored and
     row-count-weighted-averaged together; nothing is collapsed or
     discarded. Weighting is by n_rows for generality, though in
     practice n_rows is now equal across clusters of the same learner
     since all clusters share the same full FP32 reference.
  5. Indexing / shape sanity check printed per dataset before scoring
  6. Automatic file discovery via glob + regex, instead of a hardcoded
     learner0_cluster0.npz path
  7. Rank Change computed once per (dataset, config) against the full
     aggregated embedding matrix (all learners/clusters concatenated),
     not just learner 0
  8. Final per-dataset JSON + a combined all-dataset JSON + summary tables,
     matching the DiSMEC script's output format

Run:
    cd /DATA2/rudra1/AnnexML/quantization/intinference
    python3 compute_quant_error.py 2>&1 | tee quant_error_log.txt

Defaults now load the FULL test file per dataset (no 500-sample cap on the
raw file read), but rank-change queries default back to 200 samples — this
was reverted after hitting an OOM (7.35M aggregated embedding rows x 152,960
queries = 4TiB score matrix) on delicious200k. Raising --n-queries or
setting it to "use all" only makes sense on small datasets, or combined
with --max-cluster-rows-for-rank to bound the other dimension too.

Optional CLI (defaults to ALL 5 datasets, ALL test samples loaded, 200
rank-change queries, if omitted):
    python3 compute_quant_error.py --datasets eurlex wiki10
    python3 compute_quant_error.py --configs int8_row_sym int4_row_sym
    python3 compute_quant_error.py --max-test-samples 500   # cap test file load
    python3 compute_quant_error.py --n-queries 500          # raise rank-change queries
    python3 compute_quant_error.py --max-cluster-rows-for-rank 50000
"""

import numpy as np
import os, sys, re, glob, json, time, argparse, gc

sys.path.insert(0, '/DATA2/rudra1/AnnexML/quantization')
from annexml_data_loader import AnnexMLModel
from run_quantized_annexml import load_test_data, normalize_sparse

# ============================================================
# DATASET CONFIG — all 5 datasets
# ============================================================

BASE_MODEL_DIR = "/DATA2/rudra1/annexml_quantization/annexml"
BASE_RESULTS_DIR = "/DATA2/rudra1/annexml_quantization/annexml/results"
# Confirmed via `ls` on the actual filesystem — data root and filenames use
# inconsistent per-dataset conventions, so each path below is set explicitly
# rather than built from a single pattern.
BASE_DATA_DIR = "/DATA2/rudra1/annexml_quantization/annexml/data"

DATASETS = {
    "eurlex": {
        "model": f"{BASE_MODEL_DIR}/annexml-model-eurlex_baseline.bin",
        "test": f"{BASE_DATA_DIR}/eurlex/eurlex_test.txt",
        "results": f"{BASE_RESULTS_DIR}/eurlex",
    },
    "wiki10": {
        "model": f"{BASE_MODEL_DIR}/annexml-model-wiki10_baseline.bin",
        "test": f"{BASE_DATA_DIR}/wiki10/test.txt",
        "results": f"{BASE_RESULTS_DIR}/wiki10",
    },
    "amazoncat13k": {
        "model": f"{BASE_MODEL_DIR}/annexml-model-amazoncat13k_baseline.bin",
        "test": f"{BASE_DATA_DIR}/amazoncat13k/test_amazoncat13k.txt",
        "results": f"{BASE_RESULTS_DIR}/amazoncat13k",
    },
    "amazon670k": {
        "model": f"{BASE_MODEL_DIR}/annexml-model-amazon670k_baseline.bin",
        "test": f"{BASE_DATA_DIR}/amazon670k/Amazon670K_test.txt",
        "results": f"{BASE_RESULTS_DIR}/amazon670k",
    },
    "delicious200k": {
        "model": f"{BASE_MODEL_DIR}/annexml-model-delicious200k_baseline.bin",
        "test": f"{BASE_DATA_DIR}/deliciouslarge/deliciousLarge_test.txt",
        "results": f"{BASE_RESULTS_DIR}/delicious200k",
    },
}

# All 18 configs: (label, bits, is_group, npz_dirname)
CONFIGS = [
    ("int8_row_sym",                          8, False, "int8_row_sym"),
    ("int8_row_asym",                         8, False, "int8_row_asym"),
    ("int8_row_sym_clip",                     8, False, "int8_row_sym_clip"),
    ("int8_group_sym",                        8, True,  "int8_group_sym"),
    ("int8_group_sym_clip",                   8, True,  "int8_group_sym_clip"),
    ("int8_row_sym_clip_mixed",               8, False, "int8_row_sym_clip_mixed"),
    ("int8_group_sym_clip_act",               8, True,  "int8_group_sym_clip_act"),
    ("int8_group_sym_clip_act_intinfer",      8, True,  "int8_group_sym_clip_act_intinfer"),
    ("int8_group_sym_clip_act_intinfer_bias", 8, True,  "int8_group_sym_clip_act_intinfer_bias"),
    ("int4_row_sym",                          4, False, "int4_row_sym"),
    ("int4_row_asym",                         4, False, "int4_row_asym"),
    ("int4_row_sym_clip",                     4, False, "int4_row_sym_clip"),
    ("int4_group_sym",                        4, True,  "int4_group_sym"),
    ("int4_group_sym_clip",                   4, True,  "int4_group_sym_clip"),
    ("int4_row_sym_clip_mixed",               4, False, "int4_row_sym_clip_mixed"),
    ("int4_group_sym_clip_act",               4, True,  "int4_group_sym_clip_act"),
    ("int4_group_sym_clip_act_intinfer",      4, True,  "int4_group_sym_clip_act_intinfer"),
    ("int4_group_sym_clip_act_intinfer_bias", 4, True,  "int4_group_sym_clip_act_intinfer_bias"),
]

NPZ_NAME_RE = re.compile(r'learner(\d+)_cluster(\d+)\.npz$')


# ============================================================
# DEQUANTIZE NPZ -> float32 embeddings  (unchanged logic from
# the working single-cluster version — this part was already correct)
# ============================================================

def load_dequantized(npz_path, bits, is_group):
    if not os.path.exists(npz_path):
        return None

    # Use np.load() as a context manager so the NpzFile is closed and its
    # decompressed-array cache released immediately when this function
    # returns, instead of lingering until GC gets to it. Over 18 configs x
    # 5 datasets (90+ calls), un-closed NpzFile objects can accumulate live
    # in memory across the whole run — same leak class as in the DiSMEC
    # version of this script. Everything needed is copied out via
    # np.array()/.astype() below, so no lazy references back into the
    # closed file remain.
    with np.load(npz_path) as npz:
        if 'emb_data' not in npz:
            return None

        emb_data = np.array(npz['emb_data'])
        emb_nc   = int(npz['emb_ncols'][0])
        rows     = emb_data.shape[0]

        if is_group and 'emb_group_scales' in npz:
            gs_raw = np.array(npz['emb_group_scales']).astype(np.float32)
            ng     = int(npz['emb_num_groups'][0])
            gs     = gs_raw.reshape(-1, ng)
            gs32   = 32
            if bits == 4:
                E = np.zeros((rows, emb_nc), dtype=np.int8)
                for d in range(emb_nc):
                    byte = emb_data[:, d // 2]
                    nib  = (byte & 0x0F) if d % 2 == 0 else ((byte >> 4) & 0x0F)
                    sv   = nib.astype(np.int8)
                    sv[sv > 7] -= 16
                    E[:, d] = sv
            else:
                E = emb_data.astype(np.int8)
            E_dq = np.zeros((rows, emb_nc), dtype=np.float32)
            for g in range(ng):
                s = g * gs32
                e2 = min(emb_nc, s + gs32)
                E_dq[:, s:e2] = E[:, s:e2].astype(np.float32) * gs[:, g:g + 1]
            return E_dq

        if 'emb_scales' not in npz:
            return None

        scales = np.array(npz['emb_scales']).astype(np.float32)
        zp     = np.array(npz['emb_zp']).astype(np.float32) if 'emb_zp' in npz else np.zeros(len(scales), dtype=np.float32)
        mode_i = int(npz['mode_int'][0]) if 'mode_int' in npz else 0
        asym   = (mode_i != 0)

    # <-- npz is closed here; emb_data/scales/zp/mode_i were all copied
    # out above, so the rest of the function works purely off local arrays.

    actual_cols = emb_data.shape[1]
    is_packed4  = (actual_cols == (emb_nc + 1) // 2) and (actual_cols < emb_nc)

    if is_packed4 or bits == 4:
        E = np.zeros((rows, emb_nc), dtype=np.float32)
        src = emb_data
        for d in range(emb_nc):
            byte = src[:, d // 2].astype(np.uint8)
            nib  = (byte & 0x0F) if d % 2 == 0 else ((byte >> 4) & 0x0F)
            if asym:
                E[:, d] = nib.astype(np.float32) * scales + zp
            else:
                sv = nib.astype(np.int8)
                sv[sv > 7] -= 16
                E[:, d] = sv.astype(np.float32) * scales
        return E

    if asym:
        return emb_data.astype(np.float32) * scales[:, None] + zp[:, None]
    return emb_data.astype(np.float32) * scales[:, None]


# ============================================================
# Resolve the correct FP32 reference for learner{L}_cluster{C}.npz
#
# There are two candidate sources in the model:
#   (a) embeddings[cluster_assign[C]]  — data-point embeddings assigned
#       to this cluster (most likely correct — matches the ANN /
#       int8 inference nature of these files)
#   (b) dense w_mat_vec[C]             — per-feature embeddings for
#       this cluster (used for query embedding, less likely target)
#
# We pick whichever candidate's row count matches the NPZ row count.
# If neither matches, we fall back to (a) truncated/padded to the NPZ
# row count and print a loud warning so it's never silently wrong.
# ============================================================

def resolve_fp32_cluster(learner, cluster_id, npz_rows, warn_prefix=""):
    candidates = []

    # (a) FULL data-point embeddings for this learner, unsliced.
    #     Empirically confirmed: NPZ row counts are identical across every
    #     cluster of a learner (and match learner.num_data), so the
    #     "cluster" in learner{L}_cluster{C}.npz is NOT a row-subset of
    #     embeddings — the same full embeddings matrix is the correct
    #     reference for every cluster of a given learner.
    if learner.embeddings is not None and learner.embeddings.shape[0] > 0:
        candidates.append(("embeddings[full]", learner.embeddings))

    # (b) data-point embeddings assigned to this cluster (kept as a
    #     fallback candidate in case a future dataset really does
    #     partition rows per cluster)
    if cluster_id < len(learner.cluster_assign):
        idxs = learner.cluster_assign[cluster_id]
        if len(idxs) > 0:
            E_a = learner.embeddings[np.array(idxs, dtype=np.int64)]
            candidates.append(("embeddings[cluster_assign]", E_a))

    # (c) per-feature embeddings for this cluster (w_mat_vec)
    if cluster_id < len(learner.w_mat_vec) and len(learner.w_mat_vec[cluster_id]) > 0:
        entries = learner.w_mat_vec[cluster_id]
        E_c = np.zeros((len(entries), learner.embed_size), dtype=np.float32)
        for j, (fid, evec) in enumerate(entries):
            E_c[j, :len(evec)] = evec[:learner.embed_size]
        candidates.append(("w_mat_vec[cluster]", E_c))

    if not candidates:
        return None, None

    # Prefer an exact row-count match
    for name, E in candidates:
        if E.shape[0] == npz_rows:
            return E, name

    # No exact match — fall back to the first candidate, but warn loudly
    name, E = candidates[0]
    print(f"    {warn_prefix}WARNING: no FP32 source matched NPZ rows={npz_rows} "
          f"exactly (candidates: " +
          ", ".join(f"{n}={e.shape[0]}" for n, e in candidates) +
          f"). Falling back to '{name}' — verify this mapping before "
          f"trusting the metrics.")
    return E, name + " (UNVERIFIED)"


# ============================================================
# COMPUTE METRICS for one (learner, cluster) pair
# ============================================================

def compute_pair_metrics(E_fp32, E_quant):
    N = min(E_fp32.shape[0], E_quant.shape[0])
    E_f = E_fp32[:N].astype(np.float32)
    E_q = E_quant[:N].astype(np.float32)

    if E_f.shape[1] != E_q.shape[1]:
        min_cols = min(E_f.shape[1], E_q.shape[1])
        E_f = E_f[:, :min_cols]
        E_q = E_q[:, :min_cols]

    err = E_f - E_q

    signal_power = float(np.mean(E_f ** 2))
    noise_power  = float(np.mean(err ** 2))
    sqnr_db = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float('inf')

    rmse = float(np.sqrt(noise_power))

    nf = np.linalg.norm(E_f, axis=1, keepdims=True) + 1e-8
    nq = np.linalg.norm(E_q, axis=1, keepdims=True) + 1e-8
    cos_sim = float(np.mean(np.sum((E_f / nf) * (E_q / nq), axis=1)))

    return {
        "sqnr_db": sqnr_db,
        "rmse": rmse,
        "cosine_sim": cos_sim,
        "n_rows": N,
    }


def weighted_average(metric_dicts, keys):
    """Weighted (by n_rows) average of a list of per-cluster metric dicts."""
    total_w = sum(m["n_rows"] for m in metric_dicts)
    if total_w == 0:
        return {k: None for k in keys}
    out = {}
    for k in keys:
        out[k] = float(sum(m[k] * m["n_rows"] for m in metric_dicts) / total_w)
    return out


# ============================================================
# RANK CHANGE — computed once per (dataset, config) against the
# full aggregated embedding matrix (all learners/clusters concatenated)
# ============================================================

def compute_rank_change(E_fp32_all, E_quant_all, test_data, feat_map, embed_size,
                         n_queries=200, k=10):
    """n_queries caps how many test samples are used as rank-change queries.
    Default 200 — this is the query-count dimension of the score matrix
    (n_aggregated_embeddings x n_queries), so raising it scales memory
    linearly and can OOM on large datasets (e.g. amazon670k, delicious200k)
    once embeddings from multiple clusters are concatenated. n_queries=0
    means "use every loaded test sample" — only use that on datasets small
    enough for the resulting score matrix to fit in memory."""
    if test_data is None or feat_map is None or E_fp32_all.shape[0] == 0:
        return None, None

    z_vecs = []
    for labels, feats in test_data:
        fn = normalize_sparse(feats)
        z = np.zeros(embed_size, dtype=np.float32)
        for fid, val in fn:
            if fid in feat_map:
                z += val * feat_map[fid]
        nm = np.linalg.norm(z)
        if nm > 0:
            z_vecs.append(z / nm)
        if n_queries and n_queries > 0 and len(z_vecs) >= n_queries:
            break

    if not z_vecs:
        return None, None

    N = min(E_fp32_all.shape[0], E_quant_all.shape[0])
    E_f = E_fp32_all[:N]
    E_q = E_quant_all[:N]

    Z = np.array(z_vecs, dtype=np.float32)
    if Z.shape[1] != E_f.shape[1]:
        c = min(Z.shape[1], E_f.shape[1])
        Z, E_f, E_q = Z[:, :c], E_f[:, :c], E_q[:, :c]

    S_f = E_f @ Z.T
    S_q = E_q @ Z.T

    mean_score_diff = float(np.abs(S_f - S_q).mean())

    kk = min(k, N)
    rank_ch = 0
    for qi in range(len(z_vecs)):
        top_f = set(np.argpartition(S_f[:, qi], -kk)[-kk:])
        top_q = set(np.argpartition(S_q[:, qi], -kk)[-kk:])
        if top_f != top_q:
            rank_ch += 1
    rank_change_pct = float(rank_ch / len(z_vecs) * 100)

    return rank_change_pct, mean_score_diff


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(ds_name, cfg, configs, n_queries=200, max_test_samples=0,
                 rank_change_row_cap=0, out_json=None):
    print(f"\n{'=' * 65}")
    print(f"DATASET: {ds_name.upper()}")
    print(f"{'=' * 65}")

    if not os.path.exists(cfg["model"]):
        print(f"  SKIP — model not found: {cfg['model']}")
        return None

    print("  Loading model...")
    t0 = time.time()
    model = AnnexMLModel.load(cfg["model"])
    ES = model.embed_size
    print(f"  embed_size={ES}  learners={model.num_learner}  ({time.time() - t0:.1f}s)")

    test_data = None
    feat_map = None
    if os.path.exists(cfg["test"]):
        if max_test_samples and max_test_samples > 0:
            print(f"  Loading test data (first {max_test_samples} samples)...")
            test_data = load_test_data(cfg["test"])[:max_test_samples]
        else:
            print("  Loading test data (full test file)...")
            test_data = load_test_data(cfg["test"])
        print(f"  Loaded {len(test_data)} test samples")
        feat_map = {
            fid: np.array(evec, dtype=np.float32)
            for fid, evec in model.learners[0].w_mat_vec[0]
        } if model.learners and model.learners[0].w_mat_vec else None
    else:
        print(f"  WARNING: test file not found ({cfg['test']}) — rank change will be skipped")

    res_dir = cfg["results"]
    ds_results = {}
    metric_keys = ["sqnr_db", "rmse", "cosine_sim"]

    for label, bits, is_group, npz_dn in configs:
        config_dir = os.path.join(res_dir, "models", npz_dn)
        npz_files = sorted(glob.glob(os.path.join(config_dir, "learner*_cluster*.npz")))

        if not npz_files:
            print(f"  {'SKIP ' + label:<45} — no NPZ files found in {config_dir}")
            ds_results[label] = None
            continue

        per_cluster_metrics = []
        fp32_chunks, quant_chunks = [], []
        n_unverified = 0

        # Bound peak memory for the rank-change aggregate BEFORE
        # concatenation, not after. Every cluster of a learner shares the
        # same full FP32 embeddings matrix (per this script's own point 3),
        # so appending each cluster's full matrix and concatenating at the
        # end multiplies memory by n_clusters for no benefit — subsample
        # each cluster's contribution as we go instead.
        RANK_AGG_BUDGET = rank_change_row_cap if rank_change_row_cap else 200_000
        per_cluster_row_budget = max(1, RANK_AGG_BUDGET // max(1, len(npz_files)))

        t0 = time.time()
        for npz_path in npz_files:
            m = NPZ_NAME_RE.search(os.path.basename(npz_path))
            if not m:
                continue
            learner_id, cluster_id = int(m.group(1)), int(m.group(2))
            if learner_id >= len(model.learners):
                continue
            learner = model.learners[learner_id]

            E_q = load_dequantized(npz_path, bits, is_group)
            if E_q is None:
                continue

            E_f, source = resolve_fp32_cluster(
                learner, cluster_id, E_q.shape[0],
                warn_prefix=f"[{label} L{learner_id}C{cluster_id}] ")
            if E_f is None:
                continue
            if "UNVERIFIED" in source:
                n_unverified += 1

            pm = compute_pair_metrics(E_f, E_q)
            per_cluster_metrics.append(pm)

            # SQNR/RMSE/cosine-sim already scored above on the FULL
            # cluster (pm, unaffected by this cap). Only the rank-change
            # AGGREGATE buffer gets subsampled, to bound peak memory
            # instead of duplicating the full matrix once per cluster.
            n_rows_avail = pm["n_rows"]
            take = min(per_cluster_row_budget, n_rows_avail)
            if take < n_rows_avail:
                sel = np.random.RandomState(
                    hash((learner_id, cluster_id)) % (2**31)
                ).choice(n_rows_avail, take, replace=False)
                fp32_chunks.append(E_f[sel])
                quant_chunks.append(E_q[sel])
            else:
                fp32_chunks.append(E_f[:n_rows_avail])
                quant_chunks.append(E_q[:n_rows_avail])

        if not per_cluster_metrics:
            print(f"  {'SKIP ' + label:<45} — dequant/mapping failed for all clusters")
            ds_results[label] = None
            continue

        agg = weighted_average(per_cluster_metrics, metric_keys)
        agg["n_embeddings"] = int(sum(m["n_rows"] for m in per_cluster_metrics))
        agg["n_clusters_used"] = len(per_cluster_metrics)
        agg["n_clusters_unverified_mapping"] = n_unverified
        agg["bits"] = bits

        # Rank change on the aggregated embedding matrix
        E_fp32_all = np.concatenate(fp32_chunks, axis=0)
        E_quant_all = np.concatenate(quant_chunks, axis=0)
        if rank_change_row_cap and E_fp32_all.shape[0] > rank_change_row_cap:
            sel = np.random.RandomState(42).choice(
                E_fp32_all.shape[0], rank_change_row_cap, replace=False)
            E_fp32_all = E_fp32_all[sel]
            E_quant_all = E_quant_all[sel]

        rank_change_pct, mean_score_diff = compute_rank_change(
            E_fp32_all, E_quant_all, test_data, feat_map, ES, n_queries=n_queries)
        agg["rank_change_pct"] = round(rank_change_pct, 2) if rank_change_pct is not None else None
        agg["mean_score_diff"] = round(mean_score_diff, 6) if mean_score_diff is not None else None

        for k in metric_keys:
            agg[k] = round(agg[k], 6 if k != "sqnr_db" else 3)

        ds_results[label] = agg
        elapsed = time.time() - t0

        rchg = f"{agg['rank_change_pct']:.1f}%" if agg['rank_change_pct'] is not None else "N/A"
        uv = f"  [{n_unverified} unverified]" if n_unverified else ""
        print(f"  {label:<45} {agg['sqnr_db']:>9.2f} {agg['rmse']:>10.6f} "
              f"{agg['cosine_sim']:>9.6f} {rchg:>10}  "
              f"({len(per_cluster_metrics)} clusters, {elapsed:.1f}s){uv}")

        # Explicit cleanup per config, not just implicit overwrite on next
        # loop iteration — same discipline as the DiSMEC version, so peak
        # RSS doesn't ratchet up config-over-config within a dataset.
        del fp32_chunks, quant_chunks, E_fp32_all, E_quant_all, per_cluster_metrics
        gc.collect()

    # Free the loaded model before returning to the caller's dataset loop —
    # matches DiSMEC's "del W_fp32; gc.collect()" at the end of each dataset.
    del model
    gc.collect()

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump({ds_name: ds_results}, f, indent=2, default=str)
        print(f"  Saved (per-dataset): {out_json}")

    return ds_results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                         help='Subset of datasets to run (default: all 5)')
    parser.add_argument('--configs', nargs='+', default=[c[0] for c in CONFIGS],
                         help='Subset of config labels to run (default: all 18)')
    parser.add_argument('--n-queries', type=int, default=200,
                         help='Number of test queries for rank-change metric '
                              '(default 200; this is the dimension that '
                              'drives the score-matrix size — raising it or '
                              'setting 0/"all" can OOM on large datasets '
                              'once clusters are concatenated, use '
                              '--max-cluster-rows-for-rank alongside it if so)')
    parser.add_argument('--max-test-samples', type=int, default=0,
                         help='Max test-file lines to load per dataset '
                              '(0 = default = load the FULL test file)')
    parser.add_argument('--max-cluster-rows-for-rank', type=int, default=0,
                         help='Cap aggregated rows used for rank-change scoring '
                              '(0 = no cap). Use this for very large datasets '
                              '(e.g. amazon670k) to keep the score matmul tractable.')
    parser.add_argument('--out-json', default=None,
                         help='Combined output JSON path (default: alongside this script)')
    parser.add_argument('--datasets-config',
                         help='Optional JSON file overriding dataset paths for all-datasets '
                              'mode, keyed by dataset name. Each entry may set "model", '
                              '"test", "results" (any subset — unset keys fall back to the '
                              'hardcoded DATASETS dict) and optionally "out_json" for a '
                              'per-dataset output path. Datasets not listed in this file '
                              'still run using the hardcoded defaults.')
    args = parser.parse_args()

    selected_configs = [c for c in CONFIGS if c[0] in set(args.configs)]

    ds_overrides = {}
    if args.datasets_config:
        with open(args.datasets_config) as f:
            ds_overrides = json.load(f)

    ALL_RESULTS = {}
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset '{ds_name}', skipping. Known: {list(DATASETS.keys())}")
            continue

        # Merge hardcoded defaults with any per-dataset override from
        # --datasets-config, same idea as DiSMEC's datasets_config.json
        # but layered on top of (not replacing) the existing DATASETS dict.
        ds_cfg = {**DATASETS[ds_name], **ds_overrides.get(ds_name, {})}
        per_ds_out_json = ds_overrides.get(ds_name, {}).get("out_json")

        res = run_dataset(
            ds_name, ds_cfg, selected_configs,
            n_queries=args.n_queries,
            max_test_samples=args.max_test_samples,
            rank_change_row_cap=args.max_cluster_rows_for_rank,
            out_json=per_ds_out_json,
        )
        if res is not None:
            ALL_RESULTS[ds_name] = res

    out_path = args.out_json or "/DATA2/rudra1/AnnexML/quantization/intinference/quant_error_all_datasets.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # ---------------- Summary tables ----------------
    ds_keys = [d for d in DATASETS if d in ALL_RESULTS]

    print(f"\n{'=' * 65}")
    print("FINAL SUMMARY — SQNR (dB) per config per dataset")
    print("Higher SQNR = better quantization quality")
    print("INT8 typical: 40-50 dB | INT4 typical: 25-35 dB")
    print(f"{'=' * 65}")
    print(f"{'Config':<45} " + "  ".join(f"{d[:10]:>12}" for d in ds_keys))
    print("-" * (45 + 14 * len(ds_keys)))
    for label, bits, is_group, npz_dn in CONFIGS:
        row = f"{label:<45}"
        for ds in ds_keys:
            r = ALL_RESULTS.get(ds, {}).get(label)
            row += f"  {r['sqnr_db']:>12.2f}" if r else f"  {'N/A':>12}"
        print(row)

    print(f"\n{'=' * 65}")
    print("COSINE SIMILARITY per config per dataset")
    print(f"{'=' * 65}")
    print(f"{'Config':<45} " + "  ".join(f"{d[:10]:>12}" for d in ds_keys))
    print("-" * (45 + 14 * len(ds_keys)))
    for label, bits, is_group, npz_dn in CONFIGS:
        row = f"{label:<45}"
        for ds in ds_keys:
            r = ALL_RESULTS.get(ds, {}).get(label)
            row += f"  {r['cosine_sim']:>12.6f}" if r else f"  {'N/A':>12}"
        print(row)

    print(f"\n{'=' * 65}")
    print("RMSE per config per dataset")
    print(f"{'=' * 65}")
    print(f"{'Config':<45} " + "  ".join(f"{d[:10]:>12}" for d in ds_keys))
    print("-" * (45 + 14 * len(ds_keys)))
    for label, bits, is_group, npz_dn in CONFIGS:
        row = f"{label:<45}"
        for ds in ds_keys:
            r = ALL_RESULTS.get(ds, {}).get(label)
            row += f"  {r['rmse']:>12.6f}" if r else f"  {'N/A':>12}"
        print(row)

    print(f"\nAll done. Results saved to: {out_path}")


if __name__ == '__main__':
    main()
